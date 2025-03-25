#!/usr/bin/env python3
"""
Simple TESS Contamination Analysis using Gaia
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia

def analyze_tess_contamination(target_list_csv, output_file=None, search_radius_arcsec=21.0):
    """
    Simple TESS contamination analysis using Gaia data
    """
    # Load the target list CSV
    target_list = pd.read_csv(target_list_csv)
    print(f"Loaded {len(target_list)} targets from {target_list_csv}")
    
    # Create empty lists to store results
    results = []
    
    # Process each target
    for idx, row in tqdm(target_list.iterrows(), total=len(target_list)):
        target_sourceid = int(row['sourceid'])
        target_ra = float(row['ra'])
        target_dec = float(row['dec'])
        
        # Create SkyCoord object for the target
        target_coords = SkyCoord(ra=target_ra, dec=target_dec, unit='deg')
        
        # Query Gaia for stars in TESS pixel
        query = f"""
        SELECT 
            source_id, ra, dec, 
            phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_mag,
            phot_g_mean_flux, phot_g_mean_flux_error
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {target_ra}, {target_dec}, {search_radius_arcsec}/3600))
        """
        
        job = Gaia.launch_job_async(query)
        stars = job.get_results()
        
        # Default values if no target is found
        target_g_mag = 15.0
        target_flux = 10**(-0.4 * target_g_mag)
        target_flux_error = target_flux * 0.01
        target_variability = 0.001
        gaia_source_id = None
        target_g_rp = 0.5
        
        # No stars found in pixel
        if len(stars) == 0:
            print(f"No Gaia sources near target {target_sourceid}")
        else:
            # Create coordinates without units
            star_ra = stars['ra'].data.data
            star_dec = stars['dec'].data.data
            
            # Convert to SkyCoord objects with explicit units
            all_coords = SkyCoord(ra=star_ra, dec=star_dec, unit='deg')
            
            # Calculate separations
            separations = target_coords.separation(all_coords)
            
            # Find the closest star to target position
            closest_idx = np.argmin(separations.arcsec)
            
            # If within 2 arcsec, likely our target
            if separations[closest_idx].arcsec < 2.0:
                # Set target properties from closest match
                closest_star = stars[closest_idx]
                gaia_source_id = int(closest_star['SOURCE_ID'])
                target_g_mag = float(closest_star['phot_g_mean_mag'])
                target_flux = float(closest_star['phot_g_mean_flux'])
                target_flux_error = float(closest_star['phot_g_mean_flux_error'])
                target_variability = target_flux_error / target_flux if target_flux > 0 else 0.01
                
                # Get G-RP color if available
                if 'phot_rp_mean_mag' in stars.colnames and not np.isnan(closest_star['phot_rp_mean_mag']):
                    target_g_rp = float(closest_star['phot_g_mean_mag'] - closest_star['phot_rp_mean_mag'])
            
            # Find contaminating stars (all except target)
            contaminants = stars
            if gaia_source_id is not None:
                # Create mask for all stars except target
                mask = stars['SOURCE_ID'] != gaia_source_id
                contaminants = stars[mask]
        
        # Count contaminants
        num_contaminants = len(contaminants) if 'contaminants' in locals() else 0
        
        # Process each contaminant
        contaminant_info = []
        total_noise_contribution = 0.0
        total_flux_contamination_ratio = 0.0
        
        # Skip if no contaminants
        if num_contaminants > 0:
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
                psf_sigma = 42.0 / 2.355  # TESS PSF FWHM ~42 arcsec
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
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to file if requested
    if output_file:
        results_df.to_pickle(output_file)
        print(f"Results saved to {output_file}")
    
    return results_df

def visualize_contamination(results_df, output_folder='contamination_plots'):
    """
    Creates visualization plots for the contamination analysis.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot 1: Histogram of number of contaminants per target
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['num_contaminants'], bins=20)
    plt.xlabel('Number of Contaminants')
    plt.ylabel('Frequency')
    plt.title('Distribution of Contaminant Counts per Target')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'contaminant_counts.png'), dpi=300)
    plt.close()
    
    # Plot 2: Histogram of total noise contributions
    plt.figure(figsize=(10, 6))
    valid_noise = results_df['total_noise_contribution_ppm'].replace(0, np.nan).dropna()
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
    valid_data = results_df.dropna(subset=['target_g_mag', 'total_noise_contribution_ppm'])
    valid_data = valid_data[valid_data['total_noise_contribution_ppm'] > 0]
    if len(valid_data) > 0:
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
    valid_data = results_df.dropna(subset=['target_ra', 'target_dec', 'total_noise_contribution_ppm'])
    valid_data = valid_data[valid_data['total_noise_contribution_ppm'] > 0]
    if len(valid_data) > 0:
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
    output_fp = '/beegfs/car/njm/PRIMVS/cv_results/contamination/'
    target_list_csv = '/beegfs/car/njm/PRIMVS/cv_results/tess_crossmatch_results/tess_big_targets.csv'
    output_file = output_fp + "gaia_contamination_results.pkl"
    report_file = output_fp + "gaia_contamination_report.csv"
    plots_folder = output_fp + "gaia_contamination_plots"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_fp, exist_ok=True)
    
    # Run analysis using Gaia data
    print(f"Analyzing TESS contamination for targets in {target_list_csv}...")
    results = analyze_tess_contamination(target_list_csv, output_file)
    
    # Create visualizations
    print(f"Creating visualization plots in {plots_folder}...")
    visualize_contamination(results, plots_folder)
    
    # Save detailed report
    print(f"Saving detailed contamination report to {report_file}...")
    save_contamination_report(results, report_file)
    
    print("Done!")