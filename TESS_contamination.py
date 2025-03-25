import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import virac
from astropy.coordinates import SkyCoord
import astropy.units as u

def analyze_tess_contamination(target_list_csv, output_file=None, search_radius_arcsec=21.0):
    """
    Analyzes potential photometric contamination for TESS observations.
    
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
        
        try:
            # Get target light curve data
            target_lc = virac.run_sourceid(target_sourceid)
            target_mag = np.median(target_lc['hfad_mag'][target_lc['filter'] == 'Ks'])
            
            # Calculate variability for target (using standard deviation of magnitude)
            target_var = np.std(target_lc['hfad_mag'][target_lc['filter'] == 'Ks'])
            
            # Convert target magnitude to flux (arbitrary units)
            target_flux = 10**(-0.4 * target_mag)
            
            # Find nearby sources that could contaminate
            try:
                # Get the filepath corresponding to this position
                filepath = virac.coords_to_filename(target_ra, target_dec)
                lc_file = virac.open_lc_file(filepath)
                
                # Get all sources in the file
                source_ra = lc_file["sourceList/ra"][:]  
                source_dec = lc_file["sourceList/dec"][:]
                source_ids = lc_file["sourceList/sourceid"][:]
                
                # Create SkyCoord objects for all sources
                catalog_coords = SkyCoord(ra=source_ra*u.degree, dec=source_dec*u.degree)
                
                # Find separations
                seps = target_coords.separation(catalog_coords)
                
                # Find nearby sources (excluding the target itself)
                nearby_mask = (seps.arcsec <= search_radius_arcsec) & (source_ids != target_sourceid)
                nearby_indices = np.where(nearby_mask)[0]
                
                # Count the number of contaminating sources
                num_contaminants = len(nearby_indices)
                
                # Analyze each contaminant
                contaminant_info = []
                total_noise_contribution = 0.0
                total_flux_contamination_ratio = 0.0
                
                for idx in nearby_indices:
                    contam_sourceid = source_ids[idx]
                    contam_ra = source_ra[idx]
                    contam_dec = source_dec[idx]
                    contam_separation = seps[idx].arcsec
                    
                    try:
                        # Get contaminant light curve
                        contam_lc = virac.run_sourceid(int(contam_sourceid))
                        
                        # Get Ks-band measurements
                        contam_ks_data = contam_lc['hfad_mag'][contam_lc['filter'] == 'Ks']
                        
                        # Skip if no Ks-band data
                        if len(contam_ks_data) == 0:
                            continue
                            
                        contam_mag = np.median(contam_ks_data)
                        contam_var = np.std(contam_ks_data)
                        
                        # Convert contaminant magnitude to flux
                        contam_flux = 10**(-0.4 * contam_mag)
                        
                        # Calculate flux ratio of contaminant to target
                        flux_ratio = contam_flux / target_flux
                        
                        # Calculate the distance-based weighting (PSF approximation)
                        # Assuming PSF follows approximately Gaussian profile
                        # TESS PSF FWHM is approximately 2 pixels or about 42 arcsec
                        psf_sigma = 42.0 / 2.355  # Convert FWHM to sigma
                        distance_weight = np.exp(-(contam_separation**2) / (2 * psf_sigma**2))
                        
                        # Calculate weighted flux contribution
                        weighted_flux_ratio = flux_ratio * distance_weight
                        
                        # Calculate noise contribution based on:
                        # 1. Weighted flux ratio (how much of contaminant's flux affects the target)
                        # 2. Contaminant's variability (more variable stars contribute more noise)
                        # Units: relative photometric noise contributed to target (in parts-per-million)
                        noise_contribution_ppm = weighted_flux_ratio * contam_var * 1e6
                        
                        # Add to total noise contribution
                        total_noise_contribution += noise_contribution_ppm
                        
                        # Add to total flux contamination ratio
                        total_flux_contamination_ratio += weighted_flux_ratio
                        
                        # Store contaminant information
                        contaminant_info.append({
                            'contam_sourceid': contam_sourceid,
                            'contam_ra': contam_ra, 
                            'contam_dec': contam_dec,
                            'separation_arcsec': contam_separation,
                            'contam_mag': contam_mag,
                            'contam_variability': contam_var,
                            'contam_flux': contam_flux,
                            'flux_ratio': flux_ratio,
                            'distance_weight': distance_weight,
                            'weighted_flux_ratio': weighted_flux_ratio,
                            'noise_contribution_ppm': noise_contribution_ppm
                        })
                    except Exception as e:
                        print(f"Error processing contaminant {contam_sourceid}: {e}")
                
                # Create a record for this target
                result = {
                    'target_sourceid': target_sourceid,
                    'target_ra': target_ra,
                    'target_dec': target_dec,
                    'target_mag': target_mag,
                    'target_flux': target_flux,
                    'target_variability': target_var,
                    'num_contaminants': num_contaminants,
                    'total_noise_contribution_ppm': total_noise_contribution,
                    'total_flux_contamination_ratio': total_flux_contamination_ratio,
                    'contaminant_details': contaminant_info
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error finding nearby sources for target {target_sourceid}: {e}")
                results.append({
                    'target_sourceid': target_sourceid,
                    'target_ra': target_ra,
                    'target_dec': target_dec,
                    'target_mag': target_mag,
                    'target_flux': target_flux,
                    'target_variability': target_var,
                    'num_contaminants': 0,
                    'total_noise_contribution_ppm': 0.0,
                    'total_flux_contamination_ratio': 0.0,
                    'contaminant_details': []
                })
                
        except Exception as e:
            print(f"Error processing target {target_sourceid}: {e}")
            results.append({
                'target_sourceid': target_sourceid,
                'target_ra': target_ra,
                'target_dec': target_dec,
                'target_mag': np.nan,
                'target_flux': np.nan,
                'target_variability': np.nan,
                'num_contaminants': 0,
                'total_noise_contribution_ppm': 0.0,
                'total_flux_contamination_ratio': 0.0,
                'contaminant_details': []
            })
            continue
    
    # Convert results to DataFrame for easier analysis
    if not results:
        print("Warning: No results to convert to DataFrame")
        return pd.DataFrame()
        
    # Create a list of dictionaries with all required fields
    processed_results = []
    for r in results:
        try:
            processed_results.append({
                'target_sourceid': r['target_sourceid'],
                'target_ra': r['target_ra'], 
                'target_dec': r['target_dec'],
                'target_mag': r.get('target_mag', np.nan),
                'target_flux': r.get('target_flux', np.nan),
                'target_variability': r.get('target_variability', np.nan),
                'num_contaminants': r.get('num_contaminants', 0),
                'total_noise_contribution_ppm': r.get('total_noise_contribution_ppm', 0.0),
                'total_flux_contamination_ratio': r.get('total_flux_contamination_ratio', 0.0),
                'contaminant_details': r.get('contaminant_details', [])
            })
        except KeyError as e:
            print(f"Warning: Missing key in result dict: {e}")
            # Skip this result if it's missing required keys
            
    results_df = pd.DataFrame(processed_results)
    
    # Print summary of the DataFrame
    print(f"Created DataFrame with {len(results_df)} rows and columns: {results_df.columns.tolist()}")
    print(f"Data types: {results_df.dtypes}")
    print(f"Number of non-null values in each column:\n{results_df.count()}")
    
    # Save results if output file is specified
    if output_file:
        results_df.to_pickle(output_file)
        print(f"Results saved to {output_file}")
    
    return results_df

def visualize_contamination(results_df, output_folder='contamination_plots'):
    """
    Creates visualization plots for the contamination analysis.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from analyze_tess_contamination function
    output_folder : str
        Folder to save visualization plots
    """
    # Check if DataFrame is empty
    if results_df.empty:
        print("Warning: No data to visualize. Results DataFrame is empty.")
        return
        
    # Check for required columns
    required_columns = ['num_contaminants', 'total_noise_contribution_ppm', 'target_mag', 
                       'target_ra', 'target_dec', 'total_flux_contamination_ratio']
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot 1: Histogram of number of contaminants per target
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['num_contaminants'].dropna(), bins=20)
    plt.xlabel('Number of Contaminants')
    plt.ylabel('Frequency')
    plt.title('Distribution of Contaminant Counts per Target')
    plt.savefig(os.path.join(output_folder, 'contaminant_counts.png'))
    plt.close()
    
    # Plot 2: Histogram of total noise contributions (in ppm)
    plt.figure(figsize=(10, 6))
    # Filter out zeros and NaN values
    valid_scores = results_df['total_noise_contribution_ppm'].replace(0, np.nan).dropna()
    if len(valid_scores) > 0:
        plt.hist(valid_scores, bins=20)
        plt.xlabel('Total Noise Contribution (ppm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Noise Contributions from Contaminating Sources')
        plt.savefig(os.path.join(output_folder, 'noise_contributions.png'))
    else:
        print("Warning: No valid noise contribution values to plot")
    plt.close()
    
    # Plot 3: Histogram of total noise contributions (log scale)
    plt.figure(figsize=(10, 6))
    if len(valid_scores) > 0:
        log_scores = np.log10(valid_scores)
        plt.hist(log_scores, bins=20)
        plt.xlabel('Log10(Noise Contribution in ppm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Noise Contributions (Log Scale)')
        plt.savefig(os.path.join(output_folder, 'noise_contributions_log.png'))
    plt.close()
    
    # Plot 4: Scatter plot of noise contribution vs. target magnitude
    plt.figure(figsize=(10, 6))
    valid_data = results_df.dropna(subset=['target_mag', 'total_noise_contribution_ppm'])
    if len(valid_data) > 0:
        plt.scatter(valid_data['target_mag'], 
                    valid_data['total_noise_contribution_ppm'],
                    alpha=0.6)
        plt.xlabel('Target Magnitude (Ks)')
        plt.ylabel('Noise Contribution (ppm)')
        plt.title('Contamination Noise vs. Target Magnitude')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
    else:
        plt.text(0.5, 0.5, 'No valid data points to plot', 
                 horizontalalignment='center', verticalalignment='center')
    plt.savefig(os.path.join(output_folder, 'noise_vs_magnitude.png'))
    plt.close()
    
    # Plot 5: Spatial distribution of targets colored by noise contribution
    plt.figure(figsize=(12, 10))
    valid_data = results_df.dropna(subset=['target_ra', 'target_dec', 'total_noise_contribution_ppm'])
    if len(valid_data) > 0:
        # Use log scale for coloring since noise contributions can vary by orders of magnitude
        noise_log = np.log10(valid_data['total_noise_contribution_ppm'].replace(0, 1e-1))
        sc = plt.scatter(valid_data['target_ra'], valid_data['target_dec'], 
                         c=noise_log,
                         cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label('Log10(Noise Contribution in ppm)')
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('Spatial Distribution of Targets Colored by Contamination Noise')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No valid data points to plot', 
                 horizontalalignment='center', verticalalignment='center')
    plt.savefig(os.path.join(output_folder, 'spatial_noise_contribution.png'))
    plt.close()
    
    # Plot 6: Create detailed plots for top 10 most contaminated targets
    valid_targets = results_df.dropna(subset=['total_noise_contribution_ppm'])
    if len(valid_targets) > 0:
        top_targets = valid_targets.sort_values('total_noise_contribution_ppm', ascending=False).head(10)
        
        for i, (_, target) in enumerate(top_targets.iterrows()):
            contaminants = target['contaminant_details']
            if not contaminants or len(contaminants) == 0:
                continue
                
            # Create DataFrame from contaminant details
            contam_df = pd.DataFrame(contaminants)
            if len(contam_df) == 0:
                continue
                
            # Create a plot showing the target and its contaminants
            plt.figure(figsize=(10, 8))
            
            # Plot target
            plt.scatter(0, 0, s=300, c='red', marker='*', label='Target')
            
            # Plot contaminants
            try:
                # Size of points proportional to weighted flux ratio
                size_scale = 500 * contam_df['weighted_flux_ratio'] / contam_df['weighted_flux_ratio'].max()
                
                # Color based on noise contribution
                sc = plt.scatter(
                    (contam_df['contam_ra'] - target['target_ra']) * np.cos(np.radians(target['target_dec'])) * 3600,
                    (contam_df['contam_dec'] - target['target_dec']) * 3600,
                    s=size_scale,
                    c=contam_df['noise_contribution_ppm'],
                    cmap='plasma',
                    alpha=0.7,
                    norm=plt.Normalize(vmin=0, vmax=contam_df['noise_contribution_ppm'].max())
                )
                
                # Draw TESS pixel
                pixel_size = 21.0  # arcsec
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
                
                # Set limits slightly larger than TESS pixel
                plt.xlim(-pixel_size * 0.75, pixel_size * 0.75)
                plt.ylim(-pixel_size * 0.75, pixel_size * 0.75)
                
                plt.savefig(os.path.join(output_folder, f'target_{target["target_sourceid"]}_contamination.png'))
            except Exception as e:
                print(f"Error creating detailed plot for target {target['target_sourceid']}: {e}")
            plt.close()
    
    # Plot 7: Noise contribution vs. flux contamination ratio
    plt.figure(figsize=(10, 6))
    valid_data = results_df.dropna(subset=['total_noise_contribution_ppm', 'total_flux_contamination_ratio'])
    if len(valid_data) > 0:
        plt.scatter(valid_data['total_flux_contamination_ratio'] * 100, 
                    valid_data['total_noise_contribution_ppm'],
                    alpha=0.6)
        plt.xlabel('Total Flux Contamination (%)')
        plt.ylabel('Noise Contribution (ppm)')
        plt.title('Relationship Between Flux Contamination and Noise Contribution')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_folder, 'noise_vs_flux_contamination.png'))
    plt.close()

def save_contamination_report(results_df, output_file='contamination_report.csv'):
    """
    Saves a detailed report of the contamination analysis.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from analyze_tess_contamination function
    output_file : str
        Path to save the report CSV
    """
    # Create a list to store flattened results
    flattened_results = []
    
    # Process each target
    for _, target in results_df.iterrows():
        target_data = {
            'target_sourceid': target['target_sourceid'],
            'target_ra': target['target_ra'],
            'target_dec': target['target_dec'],
            'target_mag': target['target_mag'],
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
                    'contam_mag': contam.get('contam_mag', np.nan),
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
    
    return report_df

if __name__ == "__main__":
    # Hard-coded parameters
    output_fp = '/beegfs/car/njm/PRIMVS/cv_results/contamination/'
    target_list_csv = '/beegfs/car/njm/PRIMVS/cv_results/tess_crossmatch_results/tess_big_targets.csv'
    output_file = output_fp + "contamination_results.pkl"  # Path to save results
    report_file = output_fp + "contamination_report.csv"  # Path to save detailed report
    plots_folder = output_fp + "contamination_plots"  # Folder to save visualization plots
    search_radius = 21.0  # TESS pixel size in arcseconds
    
    # Run the analysis
    print(f"Analyzing TESS contamination for targets in {target_list_csv}...")
    results = analyze_tess_contamination(target_list_csv, output_file, search_radius)
    
    # Create visualizations
    print(f"Creating visualization plots in {plots_folder}...")
    visualize_contamination(results, plots_folder)
    
    # Save detailed report
    print(f"Saving detailed contamination report to {report_file}...")
    save_contamination_report(results, report_file)
    
    print("Done!")