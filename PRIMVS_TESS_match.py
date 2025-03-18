import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astroquery.mast import Catalogs, Observations
import astropy.io.fits as fits
import os
from tqdm import tqdm
import time
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class CrossMatch:
    def __init__(
        self, 
        primvs_file, 
        output_dir='./output', 
        search_radius=30, 
        batch_size=100,
        max_workers=10
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
            Number of sources to process in each batch
        max_workers : int
            Maximum number of threads for parallel processing
        """
        self.primvs_file = primvs_file
        self.output_dir = output_dir
        self.search_radius = search_radius * u.arcsec
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results container
        self.results = []
        
        logger.info(f"Initialized cross-match with search radius: {search_radius} arcsec")
        logger.info(f"Output directory: {output_dir}")

    def get_primvs_file_info(self):
        """Get information about the PRIMVS FITS file without loading it all."""
        try:
            with fits.open(self.primvs_file, memmap=True) as hdul:
                # Get data from HDU 1, which is consistent with other PRIMVS scripts
                data_hdu = 1  # PRIMVS files consistently use extension 1 for data
                num_rows = hdul[data_hdu].header.get('NAXIS2', 0)
                logger.info(f"PRIMVS file contains {num_rows} sources in HDU {data_hdu}")
                
                # Also log available columns for debugging
                columns = hdul[data_hdu].columns.names
                logger.info(f"Available columns: {', '.join(columns[:min(10, len(columns))])}{'...' if len(columns) > 10 else ''}")
                
                return num_rows
        except Exception as e:
            logger.error(f"Error reading PRIMVS file info: {str(e)}")
            return 0
    
    def process_primvs_chunk(self, start_idx, end_idx):
        """Process a chunk of the PRIMVS catalog."""
        try:
            # Read only the specified rows from the FITS file
            with fits.open(self.primvs_file, memmap=True) as hdul:
                # Use HDU 1 for data, consistent with other PRIMVS scripts
                data_hdu = 1
                
                # Get the total number of rows
                total_rows = hdul[data_hdu].header.get('NAXIS2', 0)
                end_idx = min(end_idx, total_rows)
                
                # Make sure we're not asking for rows that don't exist
                if start_idx >= total_rows:
                    logger.warning(f"Start index {start_idx} exceeds total rows {total_rows}")
                    return []
                
                # Get column names
                col_names = hdul[data_hdu].columns.names
                
                # Define the columns we need
                needed_cols = ['sourceid', 'ra', 'dec']
                
                # Check if optional columns exist
                optional_cols = ['mag_avg', 'true_period']
                for col in optional_cols:
                    if col in col_names:
                        needed_cols.append(col)
                
                # Read the chunk rows as a slice directly - this follows your pattern
                # from your other PRIMVS scripts
                chunk_data = Table(hdul[data_hdu].data[start_idx:end_idx])
                
                # Filter to only include the columns we need
                chunk_data = chunk_data[needed_cols]
                
            # Log the chunk info
            logger.info(f"Processing chunk with {len(chunk_data)} sources ({start_idx} to {end_idx-1})")
            
            # Create SkyCoord object for this chunk
            chunk_coords = SkyCoord(
                chunk_data['ra'], 
                chunk_data['dec'], 
                unit=(u.degree, u.degree)
            )
            
            # Process each source in the chunk
            chunk_results = []
            for j, primvs_coord in enumerate(chunk_coords):
                primvs_idx = j
                primvs_id = chunk_data['sourceid'][primvs_idx]
                
                # Perform cone search around this coordinate
                try:
                    catalog_data = Catalogs.query_region(
                        coordinates=primvs_coord,
                        radius=self.search_radius,
                        catalog="TIC"
                    )
                    
                    # If no matches found, continue to next source
                    if len(catalog_data) == 0:
                        continue
                    
                    # Create SkyCoord object for TIC results
                    catalog_coords = SkyCoord(
                        catalog_data['ra'], 
                        catalog_data['dec'], 
                        unit=(u.degree, u.degree)
                    )
                    
                    # Calculate separations
                    separations = primvs_coord.separation(catalog_coords)
                    
                    # Record all matches within search radius
                    for k, sep in enumerate(separations):
                        if sep <= self.search_radius:
                            match_info = {
                                'primvs_id': primvs_id,
                                'primvs_ra': primvs_coord.ra.degree,
                                'primvs_dec': primvs_coord.dec.degree,
                                'tess_id': catalog_data[k]['ID'],
                                'tess_ra': catalog_data[k]['ra'],
                                'tess_dec': catalog_data[k]['dec'],
                                'tess_mag': catalog_data[k]['Tmag'],
                                'separation_arcsec': sep.arcsecond,
                                'batch_size': len(catalog_data)  # Number of TESS matches for this PRIMVS source
                            }
                            
                            # Add optional columns if available
                            if 'mag_avg' in chunk_data.colnames:
                                match_info['primvs_mag_avg'] = chunk_data['mag_avg'][primvs_idx]
                            else:
                                match_info['primvs_mag_avg'] = np.nan
                                
                            if 'true_period' in chunk_data.colnames and chunk_data['true_period'][primvs_idx] > 0:
                                match_info['primvs_log_period'] = np.log10(chunk_data['true_period'][primvs_idx])
                            else:
                                match_info['primvs_log_period'] = np.nan
                                
                            chunk_results.append(match_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing source {primvs_id}: {str(e)}")
                    continue
                    
            logger.info(f"Found {len(chunk_results)} matches in chunk {start_idx} to {end_idx-1}")
            return chunk_results
            
        except Exception as e:
            logger.error(f"Error processing chunk {start_idx} to {end_idx-1}: {str(e)}")
            return []

    def process_batch(self, batch_idx):
        """
        Process a batch (chunk) of PRIMVS sources for TESS cross-matching.
        
        Parameters:
        -----------
        batch_idx : int
            Index of the batch to process
        
        Returns:
        --------
        list
            List of match dictionaries
        """
        batch_start = batch_idx * self.batch_size
        batch_end = min((batch_idx + 1) * self.batch_size, self.total_sources)
        
        # Process this chunk of the PRIMVS catalog
        batch_results = self.process_primvs_chunk(batch_start, batch_end)
        
        logger.info(f"Processed batch {batch_idx}: {len(batch_results)} matches found")
        return batch_results

    def run_crossmatch(self):
        """Run the full cross-matching process."""
        # Get information about the PRIMVS file
        self.total_sources = self.get_primvs_file_info()
        if self.total_sources == 0:
            logger.error("Failed to get PRIMVS file information. Aborting.")
            return False
        
        start_time = time.time()
        num_batches = (self.total_sources + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Starting cross-match with {num_batches} batches")
        
        # Initialize intermediate results file to save results as we go
        interim_results_file = os.path.join(self.output_dir, "interim_results.csv")
        interim_header_written = False
        
        # Process batches with a progress bar
        with tqdm(total=num_batches, desc="Processing batches") as pbar:
            for batch_idx in range(num_batches):
                try:
                    # Process this batch
                    batch_results = self.process_batch(batch_idx)
                    
                    # Save batch results to the intermediate file
                    if batch_results:
                        batch_df = pd.DataFrame(batch_results)
                        
                        # Write header only once
                        batch_df.to_csv(
                            interim_results_file, 
                            mode='a', 
                            header=not interim_header_written,
                            index=False
                        )
                        
                        if not interim_header_written:
                            interim_header_written = True
                        
                        # Also keep in memory for statistics
                        self.results.extend(batch_results)
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                
                # Update progress bar
                pbar.update(1)
        
        # Load all results from the intermediate file
        if os.path.exists(interim_results_file) and os.path.getsize(interim_results_file) > 0:
            try:
                self.matches_df = pd.read_csv(interim_results_file)
                logger.info(f"Loaded {len(self.matches_df)} matches from interim results file")
            except Exception as e:
                logger.error(f"Error loading interim results: {str(e)}")
                # Fallback to in-memory results
                self.matches_df = pd.DataFrame(self.results)
        else:
            self.matches_df = pd.DataFrame(self.results)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Cross-match completed in {elapsed_time:.2f} seconds")
        logger.info(f"Found {len(self.matches_df)} total matches")
        
        # Calculate statistics
        if not self.matches_df.empty:
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
            
            # Clean up interim file
            try:
                os.remove(interim_results_file)
                logger.info(f"Removed interim results file")
            except:
                pass
        
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

    def analyze_matches(self):
        """Analyze cross-match results and generate quality metrics."""
        if not hasattr(self, 'matches_df') or self.matches_df.empty:
            logger.warning("No matches to analyze!")
            return
        
        # Create a directory for analysis plots
        analysis_dir = os.path.join(self.output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            # 1. Distribution of separations
            plt.figure(figsize=(10, 6))
            plt.hist(self.matches_df['separation_arcsec'], bins=50, alpha=0.7)
            plt.xlabel('Separation (arcsec)')
            plt.ylabel('Number of matches')
            plt.title('Distribution of PRIMVS-TESS match separations')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(analysis_dir, 'separation_histogram.png'), dpi=300)
            plt.close()
            
            # 2. Magnitude comparison if available
            if 'primvs_mag_avg' in self.matches_df.columns and 'tess_mag' in self.matches_df.columns:
                plt.figure(figsize=(10, 6))
                
                # Remove NaN values for plotting
                valid_data = self.matches_df.dropna(subset=['primvs_mag_avg', 'tess_mag'])
                
                # Create 2D histogram
                plt.hist2d(
                    valid_data['primvs_mag_avg'], 
                    valid_data['tess_mag'], 
                    bins=50, 
                    norm=LogNorm(),
                    cmap='viridis'
                )
                
                plt.colorbar(label='Number of matches')
                plt.xlabel('PRIMVS magnitude (mag_avg)')
                plt.ylabel('TESS magnitude (Tmag)')
                plt.title('Magnitude comparison between PRIMVS and TESS matches')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(analysis_dir, 'magnitude_comparison.png'), dpi=300)
                plt.close()
            
            # 3. Spatial distribution of matches
            plt.figure(figsize=(12, 8))
            plt.scatter(
                self.matches_df['primvs_ra'], 
                self.matches_df['primvs_dec'], 
                c=self.matches_df['separation_arcsec'],
                s=2,
                alpha=0.6,
                cmap='viridis'
            )
            plt.colorbar(label='Separation (arcsec)')
            plt.xlabel('RA (degrees)')
            plt.ylabel('Dec (degrees)')
            plt.title('Spatial distribution of PRIMVS-TESS matches')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(analysis_dir, 'spatial_distribution.png'), dpi=300)
            plt.close()
            
            logger.info(f"Analysis plots saved to {analysis_dir}")
            
        except Exception as e:
            logger.warning(f"Error generating analysis plots: {str(e)}")

    def query_tess_observations(self, limit=100):
        """
        Query TESS observations for a subset of matched sources.
        This is useful for testing if light curves are available.
        
        Parameters:
        -----------
        limit : int
            Maximum number of sources to query
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing observation information
        """
        if not hasattr(self, 'matches_df') or self.matches_df.empty:
            logger.warning("No matches to query observations for!")
            return None
            
        sample_tess_ids = self.matches_df['tess_id'].unique()[:limit]
        logger.info(f"Querying TESS observations for {len(sample_tess_ids)} sources")
        
        all_observations = []
        
        for tic_id in tqdm(sample_tess_ids, desc="Querying observations"):
            try:
                obs = Observations.query_criteria(
                    target_name=f"TIC {tic_id}",
                    obs_collection="TESS"
                )
                
                if len(obs) > 0:
                    for i in range(len(obs)):
                        obs_info = {
                            'tess_id': tic_id,
                            'obs_id': obs[i]['obsid'],
                            'target_name': obs[i]['target_name'],
                            'exptime': obs[i]['t_exptime'],
                            'dataproduct_type': obs[i]['dataproduct_type'],
                            's_ra': obs[i]['s_ra'],
                            's_dec': obs[i]['s_dec'],
                            't_min': obs[i]['t_min'],
                            't_max': obs[i]['t_max']
                        }
                        all_observations.append(obs_info)
            except Exception as e:
                logger.warning(f"Error querying observations for TIC {tic_id}: {str(e)}")
                continue
        
        if all_observations:
            obs_df = pd.DataFrame(all_observations)
            obs_csv_path = os.path.join(self.output_dir, 'tess_observations_sample.csv')
            obs_df.to_csv(obs_csv_path, index=False)
            logger.info(f"Saved observation sample to {obs_csv_path}")
            return obs_df
        else:
            logger.warning("No observations found in sample!")
            return None

def main():
    """Main function to run the cross-match."""
    # Configuration
    primvs_file = "../PRIMVS_P.fits"  # Change to your PRIMVS file path
    output_dir = "./primvs_tess_crossmatch"
    search_radius = 30  # arcseconds
    
    # Memory optimization parameters
    batch_size = 1000  # Process 1000 sources at a time to minimize memory usage
    
    # Initialize cross-match object
    crossmatch = CrossMatch(
        primvs_file=primvs_file,
        output_dir=output_dir,
        search_radius=search_radius,
        batch_size=batch_size
    )
    
    try:
        # Run cross-match
        if crossmatch.run_crossmatch():
            # Save results
            crossmatch.save_results()
            
            # Analyze results
            crossmatch.analyze_matches()
            
            # Check for available TESS observations (optional)
            # Uncomment the next line to query TESS observations for a sample of matches
            # crossmatch.query_tess_observations(limit=50)
            
            logger.info("Cross-match process completed successfully!")
        else:
            logger.error("Cross-match process failed!")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Saving partial results...")
        if hasattr(crossmatch, 'matches_df') and not crossmatch.matches_df.empty:
            crossmatch.save_results(filename="primvs_tess_crossmatch_partial")
            logger.info("Partial results saved.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        # Try to save partial results if available
        if hasattr(crossmatch, 'matches_df') and not crossmatch.matches_df.empty:
            crossmatch.save_results(filename="primvs_tess_crossmatch_error")
            logger.info("Results saved despite error.")

if __name__ == "__main__":
    main()