import os
import time
import glob
import pickle
import logging
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

# Optional imports - will be checked before use
try:
    from astroquery.mast import Catalogs, Observations
    from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError
    HAVE_ASTROQUERY = True
except ImportError:
    HAVE_ASTROQUERY = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("primvs_tess_crossmatch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*distutils Version classes are deprecated.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")


class PrimvsTessCrossMatch:
    """
    Efficient cross-matching between PRIMVS catalog and TESS data
    with bulk downloading and parallel processing.
    """
    
    def __init__(
        self, 
        primvs_file, 
        output_dir='./output', 
        search_radius=30, 
        batch_size=10000,
        max_workers=8,
        tic_cache_file=None,
        vvv_bounds=None,
        cache_dir=None
    ):
        """
        Initialize the cross-match object.
        
        Parameters:
        -----------
        primvs_file : str
            Path to the PRIMVS FITS file
        output_dir : str
            Directory to save output files
        search_radius : float
            Search radius in arcseconds (default: 30)
        batch_size : int
            Number of sources to process in each batch for parallelization
        max_workers : int
            Maximum number of processes for parallel processing
        tic_cache_file : str, optional
            Path to a cached TIC catalog file (if available)
        vvv_bounds : dict, optional
            Dictionary with bounds for the VVV survey to limit TIC queries
            Format: {'ra_min': float, 'ra_max': float, 'dec_min': float, 'dec_max': float}
        cache_dir : str, optional
            Directory to cache downloaded data
        """
        if not HAVE_ASTROQUERY:
            raise ImportError("Required package 'astroquery' not found. Please install with 'pip install astroquery'")
            
        self.primvs_file = primvs_file
        self.output_dir = output_dir
        self.search_radius = search_radius * u.arcsec
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.tic_cache_file = tic_cache_file
        
        # VVV survey bounds (approximate)
        self.vvv_bounds = vvv_bounds or {
            'ra_min': 260.0, 'ra_max': 290.0,  # ~17h20m to ~19h20m
            'dec_min': -65.0, 'dec_max': -20.0 # Covering bulge and southern disk
        }
        
        # Create cache directory if specified
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results container
        self.results = []
        self.tic_catalog = None

        logger.info(f"Initialized cross-match with search radius: {search_radius} arcsec")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"VVV bounds: {self.vvv_bounds}")

    def get_primvs_file_info(self):
        """Get information about the PRIMVS FITS file and extract basic stats."""
        try:
            with fits.open(self.primvs_file, memmap=True) as hdul:
                # Get data from HDU 1, consistent with PRIMVS format
                data_hdu = 1
                num_rows = hdul[data_hdu].header.get('NAXIS2', 0)
                
                # Log available columns for debugging
                if len(hdul[data_hdu].columns.names) > 0:
                    columns = hdul[data_hdu].columns.names
                    logger.info(f"Available columns in PRIMVS file: {', '.join(columns[:min(10, len(columns))])}"
                                f"{'...' if len(columns) > 10 else ''}")
                
                # Check for required columns
                required_cols = ['ra', 'dec', 'sourceid']
                missing_cols = [col for col in required_cols if col not in hdul[data_hdu].columns.names]
                
                if missing_cols:
                    logger.error(f"Missing required columns in PRIMVS file: {', '.join(missing_cols)}")
                    raise ValueError(f"PRIMVS file missing required columns: {', '.join(missing_cols)}")
                
                # Get data bounds for pre-downloading TIC
                if num_rows > 0:
                    # Sample a subset to get approximate bounds
                    sample_size = min(10000, num_rows)
                    step = max(1, num_rows // sample_size)
                    sample_indices = np.arange(0, num_rows, step)
                    
                    # Safely extract and convert RA/Dec to float values
                    try:
                        ra_sample = hdul[data_hdu].data['ra'][sample_indices]
                        dec_sample = hdul[data_hdu].data['dec'][sample_indices]
                        
                        # Convert to numeric values, replacing non-numeric with NaN
                        ra_sample = np.array([float(val) if isinstance(val, (int, float, np.number)) 
                                            else np.nan for val in ra_sample])
                        dec_sample = np.array([float(val) if isinstance(val, (int, float, np.number)) 
                                             else np.nan for val in dec_sample])
                        
                        # Filter out invalid values
                        valid_mask = ~np.isnan(ra_sample) & ~np.isnan(dec_sample)
                        ra_sample = ra_sample[valid_mask]
                        dec_sample = dec_sample[valid_mask]
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error converting RA/Dec values: {str(e)}")
                        logger.warning("Trying alternate approach to read coordinates...")
                        
                        # Alternative approach: read data as a Table first
                        sample_table = Table(hdul[data_hdu].data[sample_indices])
                        
                        # Convert columns to numeric, forcing errors to become NaN
                        for colname in ['ra', 'dec']:
                            sample_table[colname] = sample_table[colname].astype(np.float64, copy=True)
                        
                        ra_sample = sample_table['ra'].data
                        dec_sample = sample_table['dec'].data
                        
                        # Filter out invalid values
                        valid_mask = np.isfinite(ra_sample) & np.isfinite(dec_sample)
                        ra_sample = ra_sample[valid_mask]
                        dec_sample = dec_sample[valid_mask]
                    
                    if len(ra_sample) > 0:
                        self.data_bounds = {
                            'ra_min': float(np.min(ra_sample)),
                            'ra_max': float(np.max(ra_sample)),
                            'dec_min': float(np.min(dec_sample)),
                            'dec_max': float(np.max(dec_sample))
                        }
                        
                        # Add a buffer for search radius
                        search_rad_deg = self.search_radius.to(u.deg).value
                        self.data_bounds['ra_min'] -= search_rad_deg / np.cos(np.radians(self.data_bounds['dec_min']))
                        self.data_bounds['ra_max'] += search_rad_deg / np.cos(np.radians(self.data_bounds['dec_max']))
                        self.data_bounds['dec_min'] -= search_rad_deg
                        self.data_bounds['dec_max'] += search_rad_deg
                        
                        logger.info(f"PRIMVS data bounds: {self.data_bounds}")
                    else:
                        self.data_bounds = self.vvv_bounds
                        logger.warning("Couldn't extract valid RA/Dec from sample. Using VVV bounds.")
                else:
                    self.data_bounds = self.vvv_bounds
                    logger.warning("Empty PRIMVS file. Using VVV bounds.")
                
                return num_rows
                
        except Exception as e:
            logger.error(f"Error reading PRIMVS file: {str(e)}")
            raise
    
    def download_tic_catalog(self):
        """
        Pre-download relevant portions of the TIC catalog in small, manageable chunks
        using parallel processing for speed.
        """
        # Check if we already have a cached catalog
        if self.tic_cache_file and os.path.exists(self.tic_cache_file):
            try:
                logger.info(f"Loading TIC catalog from cache: {self.tic_cache_file}")
                self.tic_catalog = Table.read(self.tic_cache_file)
                logger.info(f"Loaded {len(self.tic_catalog)} TIC sources from cache")
                return True
            except Exception as e:
                logger.warning(f"Error loading cached TIC catalog: {str(e)}")
                logger.warning("Will download fresh catalog")
        
        # Use data bounds or vvv bounds
        bounds = getattr(self, 'data_bounds', self.vvv_bounds)
        
        # Use manageable chunks that won't time out (based on previous test)
        ra_chunks = 15
        dec_chunks = 10
        
        ra_step = (bounds['ra_max'] - bounds['ra_min']) / ra_chunks
        dec_step = (bounds['dec_max'] - bounds['dec_min']) / dec_chunks
        
        # Create a list of all coordinates to download
        chunks_info = []
        for i in range(ra_chunks):
            ra_min = bounds['ra_min'] + i * ra_step
            ra_max = bounds['ra_min'] + (i + 1) * ra_step
            
            for j in range(dec_chunks):
                dec_min = bounds['dec_min'] + j * dec_step
                dec_max = bounds['dec_min'] + (j + 1) * dec_step
                
                # Create a central coordinate and radius for this chunk
                ra_center = (ra_min + ra_max) / 2.0
                dec_center = (dec_min + dec_max) / 2.0
                
                # Use a radius that won't time out (120 arcmin max)
                radius_deg = min(2.0, np.sqrt(((ra_max - ra_min) / 2.0)**2 + ((dec_max - dec_min) / 2.0)**2))
                radius_arcmin = min(120.0, radius_deg * 60.0)
                
                # Store chunk info
                chunks_info.append({
                    'idx': i*dec_chunks+j+1,
                    'ra_center': ra_center,
                    'dec_center': dec_center,
                    'radius_arcmin': radius_arcmin,
                    'total_chunks': ra_chunks * dec_chunks
                })
        
        logger.info(f"Downloading TIC catalog in {ra_chunks}x{dec_chunks} chunks using parallel processing")
        
        # Create a temporary directory for storing chunk results
        temp_dir = os.path.join(self.output_dir, "tic_chunks_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Function to download a single chunk in parallel
        def download_chunk(chunk_info):
            chunk_idx = chunk_info['idx']
            ra_center = chunk_info['ra_center']
            dec_center = chunk_info['dec_center']
            radius_arcmin = chunk_info['radius_arcmin']
            total_chunks = chunk_info['total_chunks']
            
            # Temporary file path for this chunk
            chunk_file = os.path.join(temp_dir, f"tic_chunk_{chunk_idx}.fits")
            
            # Skip if already downloaded
            if os.path.exists(chunk_file):
                try:
                    chunk_data = Table.read(chunk_file)
                    return {
                        'idx': chunk_idx,
                        'success': True,
                        'count': len(chunk_data),
                        'file': chunk_file,
                        'message': f"Loaded from cache: {len(chunk_data)} sources"
                    }
                except Exception:
                    # If reading fails, continue with download
                    pass
            
            # Prepare center coordinate for the cone search
            center_coord = SkyCoord(ra_center, dec_center, unit='deg')
            
            # Add retries for robustness
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Log which chunk is being processed
                    logger.info(f"Querying TIC chunk {chunk_idx}/{total_chunks}: RA={ra_center:.3f}, Dec={dec_center:.3f}, r={radius_arcmin:.1f} arcmin")
                    
                    # Query the TIC catalog
                    chunk_catalog = Catalogs.query_region(
                        coordinates=center_coord,
                        radius=radius_arcmin * u.arcmin,
                        catalog="TIC"
                    )
                    
                    if len(chunk_catalog) > 0:
                        # Save to temporary file
                        chunk_catalog.write(chunk_file, overwrite=True)
                        
                        return {
                            'idx': chunk_idx,
                            'success': True,
                            'count': len(chunk_catalog),
                            'file': chunk_file,
                            'message': f"Downloaded {len(chunk_catalog)} sources"
                        }
                    else:
                        return {
                            'idx': chunk_idx,
                            'success': True,
                            'count': 0,
                            'file': None,
                            'message': "No sources found"
                        }
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(5 * (retry + 1))
                    else:
                        return {
                            'idx': chunk_idx,
                            'success': False,
                            'count': 0,
                            'file': None,
                            'message': f"Failed after {max_retries} retries: {str(e)}"
                        }
        
        # Process chunks in parallel
        chunk_results = []
        successful_chunks = 0
        total_sources = 0
        
        # Use at most 12 workers for downloading to avoid overloading MAST
        download_workers = min(12, self.max_workers)
        logger.info(f"Using {download_workers} parallel workers for TIC catalog download")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = [executor.submit(download_chunk, chunk_info) for chunk_info in chunks_info]
            
            # Process results with a progress bar
            with tqdm(total=len(futures), desc="Downloading TIC chunks") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        chunk_results.append(result)
                        
                        if result['success']:
                            successful_chunks += 1
                            total_sources += result['count']
                            logger.info(f"Chunk {result['idx']}: {result['message']}")
                        else:
                            logger.warning(f"Chunk {result['idx']} failed: {result['message']}")
                    
                    except Exception as e:
                        logger.error(f"Error processing chunk result: {str(e)}")
                    
                    pbar.update(1)
        
        logger.info(f"Downloaded {successful_chunks}/{len(chunks_info)} chunks with {total_sources} total sources")
        
        # Combine all successful chunks
        if successful_chunks > 0:
            try:
                # Get paths to all successful chunk files
                chunk_files = [result['file'] for result in chunk_results 
                              if result['success'] and result['file'] is not None]
                
                if not chunk_files:
                    logger.warning("No valid chunk files found.")
                    return False
                
                # Read and combine all chunks
                all_chunks = []
                for chunk_file in chunk_files:
                    try:
                        chunk_data = Table.read(chunk_file)
                        if len(chunk_data) > 0:
                            all_chunks.append(chunk_data)
                    except Exception as e:
                        logger.warning(f"Error reading chunk file {chunk_file}: {str(e)}")
                
                if not all_chunks:
                    logger.warning("No valid chunks could be read.")
                    return False
                
                # Combine all chunks
                self.tic_catalog = Table.vstack(all_chunks)
                logger.info(f"Combined {len(self.tic_catalog)} TIC sources from {len(all_chunks)} chunks")
                
                # Save to cache if specified
                if self.tic_cache_file:
                    logger.info(f"Saving TIC catalog to cache: {self.tic_cache_file}")
                    self.tic_catalog.write(self.tic_cache_file, overwrite=True)
                
                # Clean up temporary directory
                for chunk_file in chunk_files:
                    try:
                        os.remove(chunk_file)
                    except Exception:
                        pass
                
                return True
            except Exception as e:
                logger.error(f"Error combining TIC chunks: {str(e)}")
                
        # If we get here with no chunks, try a different approach - individual cross-matching
        logger.warning("Bulk download failed. Will fall back to individual source cross-matching.")
        self.tic_catalog = None
        return False

    def process_batch(self, batch_idx, primvs_data):
        """
        Process a batch of PRIMVS sources against the TIC catalog.
        
        Parameters:
        -----------
        batch_idx : int
            Index of the batch being processed
        primvs_data : astropy.table.Table
            Table containing the PRIMVS data for this batch
            
        Returns:
        --------
        list
            List of match dictionaries for this batch
        """
        if len(primvs_data) == 0:
            return []
        
        logger.info(f"Processing batch {batch_idx} with {len(primvs_data)} PRIMVS sources")
        
        # Create SkyCoord objects for the PRIMVS sources
        try:
            # First, ensure the RA/Dec values are numeric
            for col in ['ra', 'dec']:
                if col in primvs_data.colnames:
                    # Try to convert to float64 - will set invalid values to NaN
                    try:
                        primvs_data[col] = primvs_data[col].astype(np.float64)
                    except Exception as e:
                        logger.warning(f"Error converting {col} to float: {str(e)}")
                        # Alternative approach for problematic data
                        temp_col = []
                        for val in primvs_data[col]:
                            try:
                                temp_col.append(float(val))
                            except (ValueError, TypeError):
                                temp_col.append(np.nan)
                        primvs_data[col] = temp_col
            
            # Filter out rows with invalid coordinates
            valid_mask = np.isfinite(primvs_data['ra']) & np.isfinite(primvs_data['dec'])
            valid_data = primvs_data[valid_mask]
            
            if len(valid_data) == 0:
                logger.warning(f"No valid coordinates in batch {batch_idx}. Skipping.")
                return []
                
            primvs_coords = SkyCoord(valid_data['ra'], valid_data['dec'], unit='deg')
            
        except Exception as e:
            logger.error(f"Error creating SkyCoord for PRIMVS batch {batch_idx}: {str(e)}")
            return []
        
        # If we have a pre-downloaded TIC catalog, use it
        if self.tic_catalog is not None:
            return self._process_with_local_tic(batch_idx, valid_data, primvs_coords)
        else:
            return self._process_with_remote_tic(batch_idx, valid_data, primvs_coords)
    
    def _process_with_local_tic(self, batch_idx, primvs_data, primvs_coords):
        """Process using the pre-downloaded TIC catalog"""
        batch_results = []
        tic_coords = SkyCoord(self.tic_catalog['ra'], self.tic_catalog['dec'], unit='deg')
        
        # Create a progress bar for individual sources
        source_iter = tqdm(
            enumerate(zip(primvs_data, primvs_coords)), 
            total=len(primvs_data),
            desc=f"Batch {batch_idx}",
            disable=None  # Only show in non-parallel mode
        )
        
        for i, (primvs_row, primvs_coord) in source_iter:
            # Find matches within search radius
            sep = primvs_coord.separation(tic_coords)
            matches_idx = np.where(sep < self.search_radius)[0]
            
            for match_idx in matches_idx:
                tic_row = self.tic_catalog[match_idx]
                
                match_info = self._create_match_info(
                    primvs_id=primvs_row['sourceid'], 
                    primvs_coord=primvs_coord,
                    primvs_data=primvs_row,
                    tic_data=tic_row,
                    separation=sep[match_idx],
                    match_count=len(matches_idx)
                )
                batch_results.append(match_info)
        
        logger.info(f"Batch {batch_idx} found {len(batch_results)} matches")
        return batch_results
    
    def _process_with_remote_tic(self, batch_idx, primvs_data, primvs_coords):
        """Process using remote TIC queries (slower but works without pre-download)"""
        batch_results = []
        
        # Create a progress bar for individual sources
        source_iter = tqdm(
            enumerate(zip(primvs_data, primvs_coords)), 
            total=len(primvs_data),
            desc=f"Batch {batch_idx}",
            disable=None  # Only show in non-parallel mode
        )
        
        for i, (primvs_row, primvs_coord) in source_iter:
            # Skip invalid coordinates (should be caught earlier, but just in case)
            if not np.isfinite(primvs_coord.ra.deg) or not np.isfinite(primvs_coord.dec.deg):
                continue
                
            # Query TIC around this coordinate
            max_retries = 3
            for retry in range(max_retries):
                try:
                    tic_results = Catalogs.query_region(
                        coordinates=primvs_coord,
                        radius=self.search_radius,
                        catalog="TIC"
                    )
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Error querying TIC for source {primvs_row['sourceid']}: {str(e)}. Retrying...")
                        time.sleep(1 * (retry + 1))
                    else:
                        logger.error(f"Failed to query TIC after {max_retries} retries for source {primvs_row['sourceid']}")
                        tic_results = None
            
            if tic_results is None or len(tic_results) == 0:
                continue
                
            # Create SkyCoord for TIC results
            tic_coords = SkyCoord(tic_results['ra'], tic_results['dec'], unit='deg')
            
            # Calculate separations
            seps = primvs_coord.separation(tic_coords)
            
            # Record all matches within search radius
            for j, (sep, tic_row) in enumerate(zip(seps, tic_results)):
                if sep < self.search_radius:
                    match_info = self._create_match_info(
                        primvs_id=primvs_row['sourceid'], 
                        primvs_coord=primvs_coord,
                        primvs_data=primvs_row,
                        tic_data=tic_row,
                        separation=sep,
                        match_count=len(tic_results)
                    )
                    batch_results.append(match_info)
                    
        logger.info(f"Batch {batch_idx} found {len(batch_results)} matches")
        return batch_results
    
    def _create_match_info(self, primvs_id, primvs_coord, primvs_data, tic_data, separation, match_count):
        """Create a standardized match info dictionary"""
        match_info = {
            'primvs_id': primvs_id,
            'primvs_ra': primvs_coord.ra.deg,
            'primvs_dec': primvs_coord.dec.deg,
            'tess_id': tic_data['ID'],
            'tess_ra': tic_data['ra'],
            'tess_dec': tic_data['dec'],
            'separation_arcsec': separation.arcsec,
            'match_count': match_count  # Number of TESS matches for this PRIMVS source
        }
        
        # Add Tmag if available
        if 'Tmag' in tic_data.colnames:
            match_info['tess_mag'] = tic_data['Tmag']
        else:
            match_info['tess_mag'] = np.nan
            
        # Add PRIMVS magnitude if available
        try:
            if 'mag_avg' in primvs_data.colnames:
                match_info['primvs_mag_avg'] = primvs_data['mag_avg']
            elif 'ks_mean_mag' in primvs_data.colnames:
                match_info['primvs_mag_avg'] = primvs_data['ks_mean_mag']
            else:
                match_info['primvs_mag_avg'] = np.nan
        except Exception:
            match_info['primvs_mag_avg'] = np.nan
            
        # Add period information if available
        try:
            if 'true_period' in primvs_data.colnames and primvs_data['true_period'] > 0:
                match_info['primvs_period'] = primvs_data['true_period']
                match_info['primvs_log_period'] = np.log10(primvs_data['true_period'])
            else:
                match_info['primvs_period'] = np.nan
                match_info['primvs_log_period'] = np.nan
        except Exception:
            match_info['primvs_period'] = np.nan
            match_info['primvs_log_period'] = np.nan
        
        # Add amplitude information if available
        try:
            if 'true_amplitude' in primvs_data.colnames:
                match_info['primvs_amplitude'] = primvs_data['true_amplitude']
            else:
                match_info['primvs_amplitude'] = np.nan
        except Exception:
            match_info['primvs_amplitude'] = np.nan
            
        # Add FAP information if available
        try:
            if 'best_fap' in primvs_data.colnames:
                match_info['primvs_fap'] = primvs_data['best_fap']
            else:
                match_info['primvs_fap'] = np.nan
        except Exception:
            match_info['primvs_fap'] = np.nan
            
        return match_info

    def run_crossmatch(self):
        """Run the full cross-matching process."""
        # Get information about the PRIMVS file
        try:
            self.total_sources = self.get_primvs_file_info()
            if self.total_sources == 0:
                logger.error("Failed to get PRIMVS file information or empty file. Aborting.")
                return False
        except Exception as e:
            logger.error(f"Error getting PRIMVS file info: {str(e)}")
            return False
        
        start_time = time.time()
        
        # Try to pre-download TIC catalog, but this may fail with timeouts
        tic_predownload = self.download_tic_catalog()
        if not tic_predownload:
            logger.warning("Pre-download of TIC catalog failed or was skipped.")
            logger.warning("Will use direct one-by-one query approach (slower but more robust).")
        
        # Calculate number of batches - use smaller batches for better checkpointing
        self.batch_size = min(5000, max(1000, self.total_sources // 1000))
        num_batches = (self.total_sources + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Starting cross-match with {num_batches} batches (batch size: {self.batch_size})")
        
        # Initialize intermediate results file
        interim_results_file = os.path.join(self.output_dir, "interim_results.csv")
        interim_header_written = False
        
        # Track completed batches to enable restart if needed
        completed_batches_file = os.path.join(self.output_dir, "completed_batches.pkl")
        completed_batches = set()
        
        if os.path.exists(completed_batches_file):
            try:
                with open(completed_batches_file, 'rb') as f:
                    completed_batches = pickle.load(f)
                logger.info(f"Loaded {len(completed_batches)} completed batches from previous run")
            except Exception as e:
                logger.warning(f"Error loading completed batches: {str(e)}")
        
        try:
            # Limit the number of workers to avoid memory issues
            effective_workers = min(self.max_workers, 24)  # Limit to 24 workers max for stability
            logger.info(f"Using {effective_workers} workers for parallel processing")
            
            # Create a ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                # Initialize batch futures dictionary
                futures = {}
                
                # Function to save completed batches
                def save_completed_batches():
                    with open(completed_batches_file, 'wb') as f:
                        pickle.dump(completed_batches, f)
                
                # Process sources in batches to avoid memory issues
                with fits.open(self.primvs_file, memmap=True) as hdul:
                    # Determine the HDU with data (typically 1 for PRIMVS)
                    data_hdu = 1
                    
                    # Process batches in parallel with dynamic scheduling
                    for batch_idx in range(num_batches):
                        if batch_idx in completed_batches:
                            logger.info(f"Skipping batch {batch_idx} (already completed)")
                            continue
                            
                        # Calculate batch range
                        start_idx = batch_idx * self.batch_size
                        end_idx = min((batch_idx + 1) * self.batch_size, self.total_sources)
                        
                        # Check if there's room for more futures
                        while len(futures) >= effective_workers * 2:
                            # Wait for some futures to complete
                            done, _ = concurrent.futures.wait(
                                futures.keys(), 
                                return_when=concurrent.futures.FIRST_COMPLETED,
                                timeout=10
                            )
                            
                            # Process completed futures
                            for future in done:
                                batch_idx_done = futures.pop(future)
                                try:
                                    batch_results = future.result()
                                    process_batch_results(batch_idx_done, batch_results)
                                except Exception as e:
                                    logger.error(f"Error processing batch {batch_idx_done}: {str(e)}")
                        
                        # Read batch data
                        try:
                            batch_data = Table(hdul[data_hdu].data[start_idx:end_idx])
                            
                            # Submit batch for processing
                            futures[executor.submit(self.process_batch, batch_idx, batch_data)] = batch_idx
                            logger.info(f"Submitted batch {batch_idx} for processing")
                        except Exception as e:
                            logger.error(f"Error reading batch {batch_idx} data: {str(e)}")
                            continue
                
                    # Helper function to process batch results
                    def process_batch_results(batch_idx, batch_results):
                        nonlocal interim_header_written
                        
                        # Save batch results to the intermediate file
                        if batch_results:
                            batch_df = pd.DataFrame(batch_results)
                            
                            # Write header only once
                            batch_df.to_csv(
                                interim_results_file, 
                                mode='a', 
                                header=not interim_header_written and not os.path.exists(interim_results_file),
                                index=False
                            )
                            
                            if not interim_header_written and os.path.exists(interim_results_file):
                                interim_header_written = True
                        
                        # Mark batch as completed
                        completed_batches.add(batch_idx)
                        
                        # Save completed batches periodically
                        if len(completed_batches) % 10 == 0:
                            save_completed_batches()
                    
                    # Process remaining futures with a progress bar
                    with tqdm(total=len(futures), desc="Processing batches") as pbar:
                        for future in as_completed(futures):
                            batch_idx = futures.pop(future)
                            try:
                                batch_results = future.result()
                                process_batch_results(batch_idx, batch_results)
                            except Exception as e:
                                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                            
                            # Update progress bar
                            pbar.update(1)
                
                # Final save of completed batches
                save_completed_batches()
        
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user.")
            # Save completed batches before exiting
            with open(completed_batches_file, 'wb') as f:
                pickle.dump(completed_batches, f)
            return False
        
        except Exception as e:
            logger.error(f"Error during cross-match process: {str(e)}")
            return False
        
        # Load all results from the intermediate file
        if os.path.exists(interim_results_file) and os.path.getsize(interim_results_file) > 0:
            try:
                self.matches_df = pd.read_csv(interim_results_file)
                logger.info(f"Loaded {len(self.matches_df)} matches from interim results file")
            except Exception as e:
                logger.error(f"Error loading interim results: {str(e)}")
                # Fallback to empty DataFrame
                self.matches_df = pd.DataFrame()
        else:
            self.matches_df = pd.DataFrame()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Cross-match completed in {elapsed_time:.2f} seconds")
        
        # Calculate statistics
        if not self.matches_df.empty:
            logger.info(f"Found {len(self.matches_df)} total matches")
            
            primvs_with_match = self.matches_df['primvs_id'].nunique()
            tess_matched = self.matches_df['tess_id'].nunique()
            logger.info(f"PRIMVS sources with at least one match: {primvs_with_match} ({100*primvs_with_match/self.total_sources:.2f}%)")
            logger.info(f"Unique TESS sources matched: {tess_matched}")
            
            # Sources with multiple matches
            primvs_multiple_matches = self.matches_df.groupby('primvs_id').size()
            primvs_multiple_matches = (primvs_multiple_matches > 1).sum()
            logger.info(f"PRIMVS sources with multiple TESS matches: {primvs_multiple_matches}")
            
            tess_multiple_matches = self.matches_df.groupby('tess_id').size()
            tess_multiple_matches = (tess_multiple_matches > 1).sum()
            logger.info(f"TESS sources matching multiple PRIMVS sources: {tess_multiple_matches}")
        
        return True

    def save_results(self, filename=None):
        """Save cross-match results to CSV and FITS files."""
        if not hasattr(self, 'matches_df') or self.matches_df.empty:
            logger.warning("No matches to save!")
            return False
        
        if filename is None:
            filename = 'primvs_tess_crossmatch'
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        self.matches_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV results to {csv_path}")
        
        # Save as FITS
        fits_path = os.path.join(self.output_dir, f"{filename}.fits")
        t = Table.from_pandas(self.matches_df)
        t.write(fits_path, overwrite=True)
        logger.info(f"Saved FITS results to {fits_path}")
        
        return True

    def fetch_tess_observations(self, max_sources=1000, min_confidence=0.7):
        """
        Fetch available TESS observations for matched sources
        
        Parameters:
        -----------
        max_sources : int
            Maximum number of sources to query
        min_confidence : float
            Minimum match confidence (based on separation)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with TESS observation information
        """
        if not hasattr(self, 'matches_df') or self.matches_df.empty:
            logger.warning("No matches to query observations for!")
            return None
        
        # Calculate a confidence score based on separation
        if 'separation_arcsec' in self.matches_df.columns:
            self.matches_df['match_confidence'] = 1 - (self.matches_df['separation_arcsec'] / self.search_radius.value)
        else:
            self.matches_df['match_confidence'] = 0.5  # Default confidence
        
        # Filter by confidence
        confident_matches = self.matches_df[self.matches_df['match_confidence'] >= min_confidence]
        confident_matches = confident_matches.sort_values('match_confidence', ascending=False)
        
        # Take the best matches for each TESS ID (to avoid redundant queries)
        best_matches = confident_matches.drop_duplicates(subset=['tess_id'], keep='first')
        
        # Limit to max_sources
        sample_tess_ids = best_matches['tess_id'].head(max_sources).tolist()
        
        logger.info(f"Querying TESS observations for {len(sample_tess_ids)} sources")
        
        all_observations = []
        
        # Query in batches to avoid timeouts
        batch_size = 20
        for i in range(0, len(sample_tess_ids), batch_size):
            batch_ids = sample_tess_ids[i:i+batch_size]
            logger.info(f"Querying batch {i//batch_size + 1} ({len(batch_ids)} sources)")
            
            # Query observations for each TIC ID in the batch
            for tic_id in tqdm(batch_ids, desc="Querying observations"):
                try:
                    # Query observations
                    obs = Observations.query_criteria(
                        target_name=f"TIC {tic_id}",
                        obs_collection="TESS"
                    )
                    
                    if len(obs) > 0:
                        logger.info(f"Found {len(obs)} observations for TIC {tic_id}")
                        
                        # Get observation details
                        for i in range(len(obs)):
                            # Basic observation info
                            obs_info = {
                                'tess_id': tic_id,
                                'obs_id': obs[i]['obsid'],
                                'target_name': obs[i]['target_name'],
                                'dataproduct_type': obs[i]['dataproduct_type'],
                            }
                            
                            # Add time information if available
                            for key in ['t_exptime', 's_ra', 's_dec', 't_min', 't_max']:
                                if key in obs[i].colnames:
                                    obs_info[key] = obs[i][key]
                                else:
                                    obs_info[key] = None
                            
                            # Add TESS-specific info if available
                            for key in ['target_classification', 'sequence_number', 'provenance_name']:
                                if key in obs[i].colnames:
                                    obs_info[key] = obs[i][key]
                                else:
                                    obs_info[key] = None
                            
                            all_observations.append(obs_info)
                    else:
                        logger.info(f"No observations found for TIC {tic_id}")
                        
                except Exception as e:
                    logger.warning(f"Error querying observations for TIC {tic_id}: {str(e)}")
                    continue
            
            # Sleep between batches to avoid overloading the server
            if i + batch_size < len(sample_tess_ids):
                time.sleep(5)
        
        if all_observations:
            obs_df = pd.DataFrame(all_observations)
            
            # Save observations
            obs_csv_path = os.path.join(self.output_dir, 'tess_observations.csv')
            obs_df.to_csv(obs_csv_path, index=False)
            logger.info(f"Saved {len(obs_df)} observations to {obs_csv_path}")
            
            return obs_df
        else:
            logger.warning("No observations found!")
            return None

    def get_tess_data_products(self, obs_df=None, product_types=None, max_products=100):
        """
        Get information about available TESS data products for matched sources
        
        Parameters:
        -----------
        obs_df : pandas.DataFrame, optional
            DataFrame with observation IDs (if None, will use result from fetch_tess_observations)
        product_types : list, optional
            List of product types to include (e.g., ['timeseries', 'lighthouse'])
        max_products : int
            Maximum number of products to retrieve
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with data product information
        """
        if obs_df is None:
            # Try to load saved observations
            obs_csv_path = os.path.join(self.output_dir, 'tess_observations.csv')
            if os.path.exists(obs_csv_path):
                try:
                    obs_df = pd.read_csv(obs_csv_path)
                except Exception as e:
                    logger.error(f"Error loading saved observations: {str(e)}")
                    return None
            else:
                logger.warning("No observations available. Run fetch_tess_observations first.")
                return None
        
        if len(obs_df) == 0:
            logger.warning("Empty observations DataFrame")
            return None
        
        # Default product types (focus on light curves)
        if product_types is None:
            product_types = ['lightcurve', 'timeseries']
        
        logger.info(f"Retrieving data products for {min(len(obs_df), max_products)} observations")
        
        all_products = []
        
        # Get unique observation IDs, limited to max_products
        obs_ids = obs_df['obs_id'].unique()[:max_products]
        
        # Query in batches
        batch_size = 10
        for i in range(0, len(obs_ids), batch_size):
            batch_ids = obs_ids[i:i+batch_size]
            logger.info(f"Querying batch {i//batch_size + 1} ({len(batch_ids)} observations)")
            
            for obs_id in tqdm(batch_ids, desc="Getting data products"):
                try:
                    # Get data products for this observation
                    products = Observations.get_product_list(obs_id)
                    
                    # Filter for desired product types
                    if product_types:
                        filtered_products = products[np.isin(products['productType'], product_types)]
                    else:
                        filtered_products = products
                    
                    if len(filtered_products) > 0:
                        logger.info(f"Found {len(filtered_products)} data products for observation {obs_id}")
                        
                        # Convert to dictionaries for easier handling
                        for j in range(len(filtered_products)):
                            product_info = {'obs_id': obs_id}
                            
                            # Add relevant columns
                            for key in filtered_products.colnames:
                                try:
                                    product_info[key] = filtered_products[j][key]
                                except:
                                    product_info[key] = None
                            
                            all_products.append(product_info)
                    else:
                        logger.info(f"No matching data products for observation {obs_id}")
                
                except Exception as e:
                    logger.warning(f"Error getting data products for observation {obs_id}: {str(e)}")
                    continue
            
            # Sleep between batches
            if i + batch_size < len(obs_ids):
                time.sleep(5)
        
        if all_products:
            products_df = pd.DataFrame(all_products)
            
            # Save data products info
            products_csv_path = os.path.join(self.output_dir, 'tess_data_products.csv')
            products_df.to_csv(products_csv_path, index=False)
            logger.info(f"Saved {len(products_df)} data product entries to {products_csv_path}")
            
            return products_df
        else:
            logger.warning("No data products found!")
            return None
    
    def download_sample_lightcurves(self, products_df=None, max_downloads=10, output_dir=None):
        """
        Download a sample of TESS light curves for analysis
        
        Parameters:
        -----------
        products_df : pandas.DataFrame, optional
            DataFrame with data product information
        max_downloads : int
            Maximum number of light curves to download
        output_dir : str, optional
            Directory to save light curves (defaults to 'lightcurves' in output_dir)
            
        Returns:
        --------
        list
            List of paths to downloaded files
        """
        if products_df is None:
            # Try to load saved product info
            products_csv_path = os.path.join(self.output_dir, 'tess_data_products.csv')
            if os.path.exists(products_csv_path):
                try:
                    products_df = pd.read_csv(products_csv_path)
                except Exception as e:
                    logger.error(f"Error loading saved product info: {str(e)}")
                    return None
            else:
                logger.warning("No product info available. Run get_tess_data_products first.")
                return None
        
        if len(products_df) == 0:
            logger.warning("Empty products DataFrame")
            return None
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'lightcurves')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter for light curve products
        lc_products = products_df[
            products_df['productType'].str.contains('lightcurve', case=False, na=False) | 
            products_df['productSubGroupDescription'].str.contains('light curve', case=False, na=False)
        ]
        
        if len(lc_products) == 0:
            logger.warning("No light curve products found")
            return None
        
        logger.info(f"Found {len(lc_products)} light curve products, downloading up to {max_downloads}")
        
        # Take a sample of light curves to download
        sample_products = lc_products.sample(min(max_downloads, len(lc_products)))
        
        downloaded_files = []
        
        for i, row in tqdm(sample_products.iterrows(), total=len(sample_products), desc="Downloading light curves"):
            try:
                # Create a unique filename from the observation ID and product type
                obs_id = row['obs_id']
                product_type = row['productType']
                filename = f"tess_{obs_id}_{product_type}.fits"
                output_path = os.path.join(output_dir, filename)
                
                # Download the data product
                manifest = Observations.download_products(
                    row['obsID'],
                    download_dir=output_dir,
                    cache=True
                )
                
                if manifest:
                    logger.info(f"Downloaded {len(manifest)} files for observation {obs_id}")
                    downloaded_files.extend(manifest['Local Path'].tolist())
                else:
                    logger.warning(f"Failed to download files for observation {obs_id}")
            
            except Exception as e:
                logger.warning(f"Error downloading light curve for observation {row['obs_id']}: {str(e)}")
                continue
        
        if downloaded_files:
            # Save list of downloaded files
            with open(os.path.join(self.output_dir, 'downloaded_files.txt'), 'w') as f:
                for file_path in downloaded_files:
                    f.write(f"{file_path}\n")
            
            logger.info(f"Downloaded {len(downloaded_files)} files")
            return downloaded_files
        else:
            logger.warning("No files downloaded!")
            return None

def main():
    """Main function to run the cross-match with hardcoded parameters."""
    import multiprocessing
    
    # Hardcoded parameters
    primvs_file = "/beegfs/car/njm/OUTPUT/PRIMVS.fits"  # Path to the PRIMVS FITS file
    output_dir = "/beegfs/car/njm/PRIMVS/tess_crossmatch"  # Output directory
    search_radius = 30.0  # Search radius in arcseconds
    tic_cache = "/beegfs/car/njm/PRIMVS/tic_cache.fits"  # Path to cache TIC catalog
    
    # Use all available cores
    max_workers = multiprocessing.cpu_count()
    logger.info(f"Using {max_workers} CPU cores for parallel processing")
    
    # Check if PRIMVS file exists
    if not os.path.exists(primvs_file):
        logger.error(f"PRIMVS file not found: {primvs_file}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize cross-match object
    crossmatch = PrimvsTessCrossMatch(
        primvs_file=primvs_file,
        output_dir=output_dir,
        search_radius=search_radius,
        max_workers=max_workers,
        tic_cache_file=tic_cache
    )
    
    try:
        # Run cross-match
        if crossmatch.run_crossmatch():
            # Save results
            crossmatch.save_results()
            logger.info("Cross-match process completed successfully!")
            return 0
        else:
            logger.error("Cross-match process failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Saving partial results...")
        if hasattr(crossmatch, 'matches_df') and not crossmatch.matches_df.empty:
            crossmatch.save_results(filename="primvs_tess_crossmatch_partial")
            logger.info("Partial results saved.")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        # Try to save partial results if available
        if hasattr(crossmatch, 'matches_df') and not crossmatch.matches_df.empty:
            crossmatch.save_results(filename="primvs_tess_crossmatch_error")
            logger.info("Results saved despite error.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    if 'sys' in globals():
        sys.exit(exit_code)
    else:
        exit(exit_code)

if __name__ == "__main__":
    sys.exit(main())