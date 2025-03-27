#!/usr/bin/env python3
"""
Optimized TESS Contamination Analysis using Gaia
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import concurrent.futures
import time
from functools import partial
import warnings
from astropy.utils.exceptions import AstropyWarning

# Suppress Astropy warnings
warnings.filterwarnings('ignore', category=AstropyWarning)

def batch_gaia_query(coords_batch, search_radius_arcsec=21.0, batch_size=100):
    """
    Perform a single Gaia query for a batch of coordinates
    
    Parameters:
    -----------
    coords_batch : list of tuples
        List of (ra, dec) tuples for targets
    search_radius_arcsec : float
        Search radius in arcseconds
    batch_size : int
        Maximum number of coordinates to query at once
        
    Returns:
    --------
    dict
        Dictionary mapping (ra, dec) to list of nearby Gaia stars
    """
    results = {}
    
    # Process in smaller batches to avoid query timeouts
    for i in range(0, len(coords_batch), batch_size):
        batch = coords_batch[i:i+batch_size]
        
        # Create ADQL query with multiple target positions
        adql_constraints = []
        for j, (ra, dec) in enumerate(batch):
            adql_constraints.append(
                f"1=CONTAINS(POINT('ICRS', ra, dec), "
                f"CIRCLE('ICRS', {ra}, {dec}, {search_radius_arcsec}/3600))"
            )
        
        adql_constraint = " OR ".join(adql_constraints)
        
        query = f"""
        SELECT 
            source_id, ra, dec, 
            phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_mag,
            phot_g_mean_flux, phot_g_mean_flux_error
        FROM gaiadr3.gaia_source
        WHERE {adql_constraint}
        """
        
        try:
            job = Gaia.launch_job_async(query)
            all_stars = job.get_results()
            
            # Create SkyCoord object for all stars
            if len(all_stars) > 0:
                all_star_coords = SkyCoord(ra=all_stars['ra'].data.data, 
                                        dec=all_stars['dec'].data.data, 
                                        unit='deg')
                
                # For each target, find associated stars
                for ra, dec in batch:
                    target_coord = SkyCoord(ra=ra, dec=dec, unit='deg')
                    separations = target_coord.separation(all_star_coords)
                    
                    # Filter stars within search radius
                    mask = separations.arcsec <= search_radius_arcsec
                    if np.any(mask):
                        results[(ra, dec)] = all_stars[mask]
                    else:
                        results[(ra, dec)] = []
            else:
                # No stars found, set empty result for all targets in batch
                for ra, dec in batch:
                    results[(ra, dec)] = []
                    
        except Exception as e:
            print(f"Error in Gaia query batch {i//batch_size}: {e}")
            # Set empty results for this batch
            for ra, dec in batch:
                results[(ra, dec)] = []
                
        # Add a small delay to avoid overloading the Gaia server
        time.sleep(1)
            
    return results

def process_target_contamination(target_data, nearby_stars):
    """
    Process contamination data for a single target
    
    Parameters:
    -----------
    target_data : dict
        Dictionary with target information
    nearby_stars : astropy.table.Table
        Table of nearby Gaia stars
        
    Returns:
    --------
    dict
        Dictionary with contamination results
    """
    target_sourceid = target_data['sourceid']
    target_ra = target_data['ra']
    target_dec = target_data['dec']
    
    # Create SkyCoord object for the target
    target_coords = SkyCoord(ra=target_ra, dec=target_dec, unit='deg')
    
    # Default values if no target is found
    target_g_mag = 15.0
    target_flux = 10**(-0.4 * target_g_mag)
    target_flux_error = target_flux * 0.01
    target_variability = 0.001
    gaia_source_id = None
    target_g_rp = 0.5
    
    # No stars found in pixel
    if len(nearby_stars) == 0:
        contaminants = []
    else:
        # Create coordinates without units
        star_ra = nearby_stars['ra'].data.data
        star_dec = nearby_stars['dec'].data.data
        
        # Convert to SkyCoord objects with explicit units
        all_coords = SkyCoord(ra=star_ra, dec=star_dec, unit='deg')
        
        # Calculate separations
        separations = target_coords.separation(all_coords)
        
        # Find the closest star to target position
        closest_idx = np.argmin(separations.arcsec)
        
        # If within 2 arcsec, likely our target
        if separations[closest_idx].arcsec < 2.0:
            # Set target properties from closest match
            closest_star = nearby_stars[closest_idx]
            gaia_source_id = int(closest_star['SOURCE_ID'])
            target_g_mag = float(closest_star['phot_g_mean_mag'])
            target_flux = float(closest_star['phot_g_mean_flux'])
            target_flux_error = float(closest_star['phot_g_mean_flux_error'])
            target_variability = target_flux_error / target_flux if target_flux > 0 else 0.01
            
            # Get G-RP color if available
            if 'phot_rp_mean_mag' in nearby_stars.colnames and not np.isnan(closest_star['phot_rp_mean_mag']):
                target_g_rp = float(closest_star['phot_g_mean_mag'] - closest_star['phot_rp_mean_mag'])
        
        # Find contaminating stars (all except target)
        contaminants = nearby_stars
        if gaia_source_id is not None:
            # Create mask for all stars except target
            mask = nearby_stars['SOURCE_ID'] != gaia_source_id
            contaminants = nearby_stars[mask]
    
    # Count contaminants
    num_contaminants = len(contaminants)
    
    # Process each contaminant
    contaminant_info = []
    total_noise_contribution = 0.0
    total_flux_contamination_ratio = 0.0
    
    # Skip if no contaminants
    if num_contaminants > 0:
        # Get TESS PSF sigma (used multiple times)
        psf_sigma = 42.0 / 2.355  # TESS PSF FWHM ~42 arcsec
        
        for i in range(num_contaminants):
            # Get basic contaminant info
            contam = contaminants[i]
            contam_id = int(contam['SOURCE_ID'])
            contam_ra = float(contam['ra'])
            contam_dec = float(contam['dec'])
            
            # Calculate separation
            contam_coords = SkyCoord(ra=contam_ra, dec=contam_dec, unit='deg')
            contam_separation = target_coords.separation(contam_coords).arcsec
            
            # Get photometric data
            contam_g_mag = float(contam['phot_g_mean_mag'])
            contam_flux = float(contam['phot_g_mean_flux'])
            contam_flux_error = float(contam['phot_g_mean_flux_error'])
            
            # Calculate variability (just use flux error)
            contam_variability = contam_flux_error / contam_flux if contam_flux > 0 else 0.01
            
            # Calculate flux ratio
            flux_ratio = contam_flux / target_flux if target_flux > 0 else 0.0
            
            # TESS PSF weighting (Gaussian approximation)
            distance_weight = np.exp(-(contam_separation**2) / (2 * psf_sigma**2))
            
            # Calculate contamination metrics
            weighted_flux_ratio = flux_ratio * distance_weight
            noise_contribution_ppm = weighted_flux_ratio * contam_variability * 1e6
            
            # Update totals
            total_noise_contribution += noise_contribution_ppm
            total_flux_contamination_ratio += weighted_flux_ratio
            
            # Get color if available
            contam_g_rp = 0.5
            if 'phot_rp_mean_mag' in contam.colnames and not np.isnan(contam['phot_rp_mean_mag']):
                contam_g_rp = float(contam['phot_g_mean_mag'] - contam['phot_rp_mean_mag'])
            
            # Store contaminant info
            contaminant_info.append({
                'contam_sourceid': contam_id,
                'contam_ra': contam_ra,
                'contam_dec': contam_dec,
                'separation_arcsec': contam_separation,
                'contam_g_mag': contam_g_mag,
                'contam_g_rp': contam_g_rp,
                'contam_variability': contam_variability,
                'contam_flux': contam_flux,
                'flux_ratio': flux_ratio,
                'distance_weight': distance_weight,
                'weighted_flux_ratio': weighted_flux_ratio,
                'noise_contribution_ppm': noise_contribution_ppm
            })
    
    # Store target results
    result = {
        'target_sourceid': target_sourceid,
        'gaia_source_id': gaia_source_id,
        'target_ra': target_ra,
        'target_dec': target_dec,
        'target_g_mag': target_g_mag,
        'target_g_rp': target_g_rp,
        'target_flux': target_flux,
        'target_variability': target_variability,
        'num_contaminants': num_contaminants,
        'total_noise_contribution_ppm': total_noise_contribution,
        'total_flux_contamination_ratio': total_flux_contamination_ratio,
        'contaminant_details': contaminant_info
    }
    
    return result

def analyze_tess_contamination_optimized(target_list_csv, output_file=None, search_radius_arcsec=21.0, 
                                       batch_size=100, max_workers=10):
    """
    Optimized TESS contamination analysis using Gaia data with batched queries and parallelization
    """
    # Load the target list CSV
    target_list = pd.read_csv(target_list_csv)
    print(f"Loaded {len(target_list)} targets from {target_list_csv}")
    
    # Extract coordinates for batch querying
    coords = [(float(row['ra']), float(row['dec'])) for _, row in target_list.iterrows()]
    
    # Query Gaia for all targets in batches
    print("Querying Gaia for all targets in batches...")
    all_stars = {}
    
    # Process batches of coordinates
    for i in tqdm(range(0, len(coords), batch_size)):
        coords_batch = coords[i:i+batch_size]
        batch_results = batch_gaia_query(coords_batch, search_radius_arcsec, batch_size=min(50, batch_size))
        all_stars.update(batch_results)
    
    # Process results with parallelization
    print("Processing contamination results in parallel...")
    results = []
    
    # Function to process a single target with pre-queried stars
    def process_target(row, all_stars_dict):
        target_ra = float(row['ra'])
        target_dec = float(row['dec'])
        stars = all_stars_dict.get((target_ra, target_dec), [])
        return process_target_contamination(row, stars)
    
    # Use partial function to pass the stars dictionary
    process_func = partial(process_target, all_stars_dict=all_stars)
    
    # Use ThreadPoolExecutor for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_func, row): idx 
                        for idx, row in target_list.iterrows()}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = future_to_idx[future]
                print(f"Error processing target {idx}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to file if requested
    if output_file:
        results_df.to_pickle(output_file)
        print(f"Results saved to {output_file}")
    
    return results_df

# The visualization functions can remain mostly the same
# Just adding some optimizations to make them faster

def visualize_contamination_optimized(results_df, output_folder='contamination_plots'):
    """
    Creates visualization plots for the contamination analysis with optimizations.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # For large datasets, sample data to speed up visualization
    sample_size = min(10000, len(results_df))
    if len(results_df) > sample_size:
        print(f"Sampling {sample_size} targets for visualization...")
        results_sample = results_df.sample(sample_size, random_state=42)
    else:
        results_sample = results_df
    
    # Plot 1: Histogram of number of contaminants per target
    plt.figure(figsize=(10, 6))
    plt.hist(results_sample['num_contaminants'], bins=20)
    plt.xlabel('Number of Contaminants')
    plt.ylabel('Frequency')
    plt.title('Distribution of Contaminant Counts per Target')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'contaminant_counts.png'), dpi=300)
    plt.close()
    
    # Plot 2: Histogram of total noise contributions
    plt.figure(figsize=(10, 6))
    valid_noise = results_sample['total_noise_contribution_ppm'].replace(0, np.nan).dropna()
    if len(valid_noise) > 0:
        plt.hist(valid_noise, bins=20)
        plt.xlabel('Total Noise Contribution (ppm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Noise Contributions from Contaminating Sources')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_folder, 'noise_contributions.png'), dpi=300)
    plt.close()
    
    # Plot 3: Scatter plot of noise contribution vs. target magnitude
    plt.figure(figsize=(10, 6))
    valid_data = results_sample.dropna(subset=['target_g_mag', 'total_noise_contribution_ppm'])
    valid_data = valid_data[valid_data['total_noise_contribution_ppm'] > 0]
    if len(valid_data) > 0:
        # Use hexbin for large datasets for better performance
        if len(valid_data) > 1000:
            plt.hexbin(valid_data['target_g_mag'], valid_data['total_noise_contribution_ppm'],
                     gridsize=50, cmap='viridis', bins='log')
            plt.colorbar(label='log10(count)')
        else:
            plt.scatter(valid_data['target_g_mag'], 
                       valid_data['total_noise_contribution_ppm'],
                       alpha=0.6)
        plt.xlabel('Target G Magnitude')
        plt.ylabel('Noise Contribution (ppm)')
        plt.title('Contamination Noise vs. Target Magnitude')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(output_folder, 'noise_vs_magnitude.png'), dpi=300)
    plt.close()
    
    # Plot 4: Spatial distribution of targets colored by noise contribution
    plt.figure(figsize=(12, 10))
    valid_data = results_sample.dropna(subset=['target_ra', 'target_dec', 'total_noise_contribution_ppm'])
    valid_data = valid_data[valid_data['total_noise_contribution_ppm'] > 0]
    if len(valid_data) > 0:
        # Use hexbin for large datasets
        if len(valid_data) > 1000:
            hb = plt.hexbin(valid_data['target_ra'], valid_data['target_dec'],
                          C=np.log10(valid_data['total_noise_contribution_ppm']),
                          reduce_C_function=np.median,
                          gridsize=50, cmap='viridis')
            cb = plt.colorbar(hb)
            cb.set_label('Log10(Median Noise Contribution in ppm)')
        else:
            noise_log = np.log10(valid_data['total_noise_contribution_ppm'])
            sc = plt.scatter(valid_data['target_ra'], valid_data['target_dec'], 
                           c=noise_log,
                           cmap='viridis', alpha=0.7, s=30)
            cbar = plt.colorbar(sc)
            cbar.set_label('Log10(Noise Contribution in ppm)')
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('Spatial Distribution of Targets Colored by Contamination Noise')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_folder, 'spatial_noise_contribution.png'), dpi=300)
    plt.close()

def save_contamination_report(results_df, output_file='contamination_report.csv'):
    """
    Saves a detailed report of the contamination analysis.
    """
    # Create a list to store flattened results
    flattened_results = []
    
    # Process each target
    for _, target in results_df.iterrows():
        target_data = {
            'target_sourceid': target['target_sourceid'],
            'gaia_source_id': target['gaia_source_id'],
            'target_ra': target['target_ra'],
            'target_dec': target['target_dec'],
            'target_g_mag': target['target_g_mag'],
            'target_g_rp': target['target_g_rp'],
            'target_variability': target['target_variability'],
            'num_contaminants': target['num_contaminants'],
            'total_noise_contribution_ppm': target['total_noise_contribution_ppm'],
            'total_flux_contamination_ratio': target['total_flux_contamination_ratio']
        }
        
        # If there are no contaminants, add a single row
        if target['num_contaminants'] == 0 or not isinstance(target['contaminant_details'], list):
            flattened_results.append(target_data)
        else:
            # Add a row for each contaminant
            for contam in target['contaminant_details']:
                row = target_data.copy()
                row.update({
                    'contam_sourceid': contam.get('contam_sourceid', np.nan),
                    'contam_ra': contam.get('contam_ra', np.nan),
                    'contam_dec': contam.get('contam_dec', np.nan),
                    'separation_arcsec': contam.get('separation_arcsec', np.nan),
                    'contam_g_mag': contam.get('contam_g_mag', np.nan),
                    'contam_g_rp': contam.get('contam_g_rp', np.nan),
                    'contam_variability': contam.get('contam_variability', np.nan),
                    'contam_flux': contam.get('contam_flux', np.nan),
                    'flux_ratio': contam.get('flux_ratio', np.nan),
                    'distance_weight': contam.get('distance_weight', np.nan),
                    'weighted_flux_ratio': contam.get('weighted_flux_ratio', np.nan),
                    'noise_contribution_ppm': contam.get('noise_contribution_ppm', np.nan)
                })
                flattened_results.append(row)
    
    # Convert to DataFrame and save
    report_df = pd.DataFrame(flattened_results)
    report_df.to_csv(output_file, index=False)
    print(f"Contamination report saved to {output_file}")
    
    # Also save a summary version with just the target information
    summary_df = results_df.drop(columns=['contaminant_details'])
    summary_df.to_csv(output_file.replace('.csv', '_summary.csv'), index=False)
    print(f"Summary report saved to {output_file.replace('.csv', '_summary.csv')}")
    
    return report_df

if __name__ == "__main__":
    # Parameters
    output_fp = '../PRIMVS/cv_results/contamination/'
    target_list_csv = 'targets.csv'
    output_file = output_fp + "gaia_contamination_results.pkl"
    report_file = output_fp + "gaia_contamination_report.csv"
    plots_folder = output_fp + "gaia_contamination_plots"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_fp, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Run optimized analysis using Gaia data
    print(f"Analyzing TESS contamination for targets in {target_list_csv}...")
    results = analyze_tess_contamination_optimized(
        target_list_csv, 
        output_file,
        batch_size=100,  # Adjust based on your data size
        max_workers=8    # Adjust based on your CPU cores
    )
    
    # Create visualizations
    print(f"Creating visualization plots in {plots_folder}...")
    visualize_contamination_optimized(results, plots_folder)
    
    # Save detailed report
    print(f"Saving detailed contamination report to {report_file}...")
    save_contamination_report(results, report_file)
    
    # Print total runtime
    elapsed_time = time.time() - start_time
    print(f"Done! Total runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")