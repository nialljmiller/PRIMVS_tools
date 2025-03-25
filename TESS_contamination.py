import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.table import Table

def analyze_tess_contamination_gaia(target_list_csv, output_file=None, search_radius_arcsec=21.0):
    """
    Analyzes potential photometric contamination for TESS observations using Gaia data.
    Always uses online Gaia query with minimal error handling.
    
    Parameters:
    -----------
    target_list_csv : str
        Path to CSV file containing target list with at least 'sourceid', 'ra', 'dec' columns
    output_file : str, optional
        Path to save the contamination analysis results
    search_radius_arcsec : float, optional
        Search radius in arcseconds (default: 21.0 arcsec, which is the TESS pixel scale)
        
    Returns:
    --------
    DataFrame containing contamination analysis results
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
        target_coords = SkyCoord(ra=target_ra*u.degree, dec=target_dec*u.degree)
        
        # First, query Gaia for the target star to get its magnitude
        target_query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_mag, 
               phot_g_mean_flux, phot_g_mean_flux_error, parallax
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {target_ra}, {target_dec}, 2/3600))
        """
        
        target_result = Gaia.launch_job_async(target_query)
        target_data = target_result.get_results()
        
        # If not found, use generic values
        if len(target_data) == 0:
            print(f"Target {target_sourceid} not found in Gaia, using generic values")
            target_g_mag = 15.0  # Generic magnitude
            target_flux = 10**(-0.4 * target_g_mag)
            target_flux_error = target_flux * 0.01  # Assume 1% error
            target_variability = 0.001  # Assume low variability
            gaia_source_id = None
            target_g_rp = 0.5  # Assuming a typical G-RP color
        else:
            # Use the closest match
            distances = target_coords.separation(SkyCoord(ra=target_data['ra']*u.degree, 
                                                        dec=target_data['dec']*u.degree))
            closest_idx = np.argmin(distances)
            
            target_g_mag = float(target_data[closest_idx]['phot_g_mean_mag'])
            target_flux = float(target_data[closest_idx]['phot_g_mean_flux'])
            target_flux_error = float(target_data[closest_idx]['phot_g_mean_flux_error'])
            target_variability = target_flux_error / target_flux if target_flux > 0 else 0.01
            gaia_source_id = int(target_data[closest_idx]['source_id'])
            
            # Calculate G-RP color if available (useful for stellar type estimation)
            if 'phot_rp_mean_mag' in target_data.colnames and not np.isnan(target_data[closest_idx]['phot_rp_mean_mag']):
                target_g_rp = float(target_data[closest_idx]['phot_g_mean_mag'] - target_data[closest_idx]['phot_rp_mean_mag'])
            else:
                target_g_rp = 0.5  # Default value
        
        # Next, query Gaia for all stars in the TESS pixel around the target
        contam_query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_mag,
               phot_g_mean_flux, phot_g_mean_flux_error, parallax, pmra, pmdec,
               phot_variable_flag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {target_ra}, {target_dec}, {search_radius_arcsec}/3600))
            AND source_id != {gaia_source_id if gaia_source_id else 0}
        """
        
        contam_result = Gaia.launch_job_async(contam_query)
        contam_data = contam_result.get_results()
        
        # Count the number of contaminating sources
        num_contaminants = len(contam_data)
        
        # Process each contaminant
        contaminant_info = []
        total_noise_contribution = 0.0
        total_flux_contamination_ratio = 0.0
        
        for i in range(num_contaminants):
            contam_row = contam_data[i]
            contam_sourceid = int(contam_row['source_id'])
            contam_ra = float(contam_row['ra'])
            contam_dec = float(contam_row['dec'])
            
            # Calculate separation
            contam_coords = SkyCoord(ra=contam_ra*u.degree, dec=contam_dec*u.degree)
            contam_separation = target_coords.separation(contam_coords).arcsec
            
            # Get magnitude and flux
            contam_g_mag = float(contam_row['phot_g_mean_mag'])
            contam_flux = float(contam_row['phot_g_mean_flux'])
            contam_flux_error = float(contam_row['phot_g_mean_flux_error'])
            
            # Estimate variability based on flux error and Gaia variability flag
            contam_var_flag = contam_row['phot_variable_flag']
            if contam_var_flag == 'VARIABLE':
                # Higher variability estimate for stars flagged as variable
                contam_variability = max(0.05, contam_flux_error / contam_flux if contam_flux > 0 else 0.05)
            else:
                # Base variability on flux error
                contam_variability = contam_flux_error / contam_flux if contam_flux > 0 else 0.01
            
            # Calculate flux ratio of contaminant to target
            flux_ratio = contam_flux / target_flux if target_flux > 0 else 0.0
            
            # Calculate the distance-based weighting (PSF approximation)
            # Assuming PSF follows approximately Gaussian profile
            # TESS PSF FWHM is approximately 2 pixels or about 42 arcsec
            psf_sigma = 42.0 / 2.355  # Convert FWHM to sigma
            distance_weight = np.exp(-(contam_separation**2) / (2 * psf_sigma**2))
            
            # Calculate weighted flux contribution
            weighted_flux_ratio = flux_ratio * distance_weight
            
            # Calculate noise contribution
            # Units: relative photometric noise contributed to target (in parts-per-million)
            noise_contribution_ppm = weighted_flux_ratio * contam_variability * 1e6
            
            # Add to total noise contribution
            total_noise_contribution += noise_contribution_ppm
            
            # Add to total flux contamination ratio
            total_flux_contamination_ratio += weighted_flux_ratio
            
            # Get color information if available
            if 'phot_rp_mean_mag' in contam_row.colnames and not np.isnan(contam_row['phot_rp_mean_mag']):
                contam_g_rp = float(contam_row['phot_g_mean_mag'] - contam_row['phot_rp_mean_mag'])
            else:
                contam_g_rp = 0.5  # Default value
            
            # Get proper motion information if available
            if 'pmra' in contam_row.colnames and not np.isnan(contam_row['pmra']):
                pmra = float(contam_row['pmra'])
                pmdec = float(contam_row['pmdec'])
                pm_total = np.sqrt(pmra**2 + pmdec**2)
            else:
                pmra = 0.0
                pmdec = 0.0
                pm_total = 0.0
            
            # Store contaminant information
            contaminant_info.append({
                'contam_sourceid': contam_sourceid,
                'contam_ra': contam_ra, 
                'contam_dec': contam_dec,
                'separation_arcsec': contam_separation,
                'contam_g_mag': contam_g_mag,
                'contam_g_rp': contam_g_rp,
                'contam_variability': contam_variability,
                'contam_flux': contam_flux,
                'contam_var_flag': contam_var_flag,
                'pmra': pmra,
                'pmdec': pmdec,
                'pm_total': pm_total,
                'flux_ratio': flux_ratio,
                'distance_weight': distance_weight,
                'weighted_flux_ratio': weighted_flux_ratio,
                'noise_contribution_ppm': noise_contribution_ppm
            })
        
        # Create a record for this target
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
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary of the DataFrame
    print(f"Created DataFrame with {len(results_df)} rows and columns: {results_df.columns.tolist()}")
    
    # Save results if output file is specified
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
    plt.hist(results_df['num_contaminants'].dropna(), bins=20)
    plt.xlabel('Number of Contaminants')
    plt.ylabel('Frequency')
    plt.title('Distribution of Contaminant Counts per Target')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'contaminant_counts.png'), dpi=300)
    plt.close()
    
    # Plot 2: Histogram of total noise contributions (in ppm)
    plt.figure(figsize=(10, 6))
    valid_scores = results_df['total_noise_contribution_ppm'].replace(0, np.nan).dropna()
    if len(valid_scores) > 0:
        plt.hist(valid_scores, bins=20)
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
    
    # Plot 5: Create detailed plots for top 5 most contaminated targets
    valid_targets = results_df.dropna(subset=['total_noise_contribution_ppm'])
    valid_targets = valid_targets[valid_targets['total_noise_contribution_ppm'] > 0]
    if len(valid_targets) > 0:
        top_targets = valid_targets.sort_values('total_noise_contribution_ppm', ascending=False).head(5)
        for _, target in top_targets.iterrows():
            contaminants = target['contaminant_details']
            if not contaminants or len(contaminants) == 0:
                continue
            contam_df = pd.DataFrame(contaminants)
            if len(contam_df) == 0:
                continue
            
            plt.figure(figsize=(10, 8))
            plt.scatter(0, 0, s=300, c='red', marker='*', label='Target')
            
            size_scale = 500 * contam_df['weighted_flux_ratio'] / contam_df['weighted_flux_ratio'].max()
            size_scale = size_scale.clip(20, 500)
            
            sc = plt.scatter(
                (contam_df['contam_ra'] - target['target_ra']) * np.cos(np.radians(target['target_dec'])) * 3600,
                (contam_df['contam_dec'] - target['target_dec']) * 3600,
                s=size_scale,
                c=contam_df['noise_contribution_ppm'],
                cmap='plasma',
                alpha=0.7
            )
            
            pixel_size = 21.0
            plt.plot([-pixel_size/2, pixel_size/2, pixel_size/2, -pixel_size/2, -pixel_size/2],
                    [-pixel_size/2, -pixel_size/2, pixel_size/2, pixel_size/2, -pixel_size/2],
                    'k--', alpha=0.5, label='TESS pixel')
            
            cbar = plt.colorbar(sc)
            cbar.set_label('Noise Contribution (ppm)')
            plt.xlabel('ΔRA (arcsec)')
            plt.ylabel('ΔDec (arcsec)')
            plt.title(f'Target {target["target_sourceid"]} - Total Noise: {target["total_noise_contribution_ppm"]:.1f} ppm')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.xlim(-pixel_size * 0.75, pixel_size * 0.75)
            plt.ylim(-pixel_size * 0.75, pixel_size * 0.75)
            plt.savefig(os.path.join(output_folder, f'target_{target["target_sourceid"]}_contamination.png'), dpi=300)
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
                    'contam_var_flag': contam.get('contam_var_flag', ''),
                    'pmra': contam.get('pmra', np.nan),
                    'pmdec': contam.get('pmdec', np.nan),
                    'pm_total': contam.get('pm_total', np.nan),
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
    
    # Run analysis (always using online query)
    print(f"Analyzing TESS contamination for targets in {target_list_csv}...")
    results = analyze_tess_contamination_gaia(target_list_csv, output_file)
    
    # Create visualizations
    print(f"Creating visualization plots in {plots_folder}...")
    visualize_contamination(results, plots_folder)
    
    # Save detailed report
    print(f"Saving detailed contamination report to {report_file}...")
    save_contamination_report(results, report_file)
    
    print("Done!")