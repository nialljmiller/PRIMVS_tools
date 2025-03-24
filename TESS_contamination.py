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
                total_contamination_score = 0.0
                
                for idx in nearby_indices:
                    contam_sourceid = source_ids[idx]
                    contam_ra = source_ra[idx]
                    contam_dec = source_dec[idx]
                    contam_separation = seps[idx].arcsec
                    
                    try:
                        # Get contaminant light curve
                        contam_lc = virac.run_sourceid(int(contam_sourceid))
                        contam_mag = np.median(contam_lc['hfad_mag'][contam_lc['filter'] == 'Ks'])
                        contam_var = np.std(contam_lc['hfad_mag'][contam_lc['filter'] == 'Ks'])
                        
                        # Calculate contamination score based on the criteria:
                        # a) Brightness: brighter stars have higher impact
                        # b) Variability: more variable stars have higher impact
                        # c) Proximity: closer stars have higher impact
                        
                        # Brightness impact (use flux ratio, not magnitude)
                        brightness_impact = 10**((target_mag - contam_mag)/2.5)
                        
                        # Variability impact (normalized by target variability)
                        variability_impact = contam_var / max(target_var, 0.001)
                        
                        # Proximity impact (inverse of separation, normalized to search radius)
                        proximity_impact = (search_radius_arcsec - contam_separation) / search_radius_arcsec
                        
                        # Combined contamination score (product of all factors)
                        contam_score = brightness_impact * variability_impact * proximity_impact
                        
                        # Add to total contamination score
                        total_contamination_score += contam_score
                        
                        # Store contaminant information
                        contaminant_info.append({
                            'contam_sourceid': contam_sourceid,
                            'contam_ra': contam_ra, 
                            'contam_dec': contam_dec,
                            'separation_arcsec': contam_separation,
                            'contam_mag': contam_mag,
                            'contam_variability': contam_var,
                            'brightness_impact': brightness_impact,
                            'variability_impact': variability_impact,
                            'proximity_impact': proximity_impact,
                            'contamination_score': contam_score
                        })
                    except Exception as e:
                        print(f"Error processing contaminant {contam_sourceid}: {e}")
                
                # Create a record for this target
                result = {
                    'target_sourceid': target_sourceid,
                    'target_ra': target_ra,
                    'target_dec': target_dec,
                    'target_mag': target_mag,
                    'target_variability': target_var,
                    'num_contaminants': num_contaminants,
                    'total_contamination_score': total_contamination_score,
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
                    'target_variability': target_var,
                    'num_contaminants': 0,
                    'total_contamination_score': 0.0,
                    'contaminant_details': []
                })
                
        except Exception as e:
            print(f"Error processing target {target_sourceid}: {e}")
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame([{
        'target_sourceid': r['target_sourceid'],
        'target_ra': r['target_ra'], 
        'target_dec': r['target_dec'],
        'target_mag': r.get('target_mag', np.nan),
        'target_variability': r.get('target_variability', np.nan),
        'num_contaminants': r['num_contaminants'],
        'total_contamination_score': r['total_contamination_score'],
        'contaminant_details': r['contaminant_details']
    } for r in results])
    
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
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot 1: Histogram of number of contaminants per target
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['num_contaminants'], bins=20)
    plt.xlabel('Number of Contaminants')
    plt.ylabel('Frequency')
    plt.title('Distribution of Contaminant Counts per Target')
    plt.savefig(os.path.join(output_folder, 'contaminant_counts.png'))
    plt.close()
    
    # Plot 2: Histogram of total contamination scores
    plt.figure(figsize=(10, 6))
    # Use log scale for better visualization of wide range of values
    log_scores = np.log10(results_df['total_contamination_score'] + 1e-10)
    plt.hist(log_scores, bins=20)
    plt.xlabel('Log10(Total Contamination Score)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Contamination Scores (Log Scale)')
    plt.savefig(os.path.join(output_folder, 'contamination_scores.png'))
    plt.close()
    
    # Plot 3: Scatter plot of contamination score vs. target magnitude
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['target_mag'], 
                np.log10(results_df['total_contamination_score'] + 1e-10),
                alpha=0.6)
    plt.xlabel('Target Magnitude (Ks)')
    plt.ylabel('Log10(Total Contamination Score)')
    plt.title('Contamination Score vs. Target Magnitude')
    plt.savefig(os.path.join(output_folder, 'score_vs_magnitude.png'))
    plt.close()
    
    # Plot 4: Spatial distribution of targets colored by contamination
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(results_df['target_ra'], results_df['target_dec'], 
                     c=np.log10(results_df['total_contamination_score'] + 1e-10),
                     cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Log10(Contamination Score)')
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Spatial Distribution of Targets Colored by Contamination Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'spatial_contamination.png'))
    plt.close()
    
    # Plot 5: Create detailed plots for top 10 most contaminated targets
    top_targets = results_df.sort_values('total_contamination_score', ascending=False).head(10)
    
    for i, (_, target) in enumerate(top_targets.iterrows()):
        contaminants = target['contaminant_details']
        if not contaminants:
            continue
            
        # Create DataFrame from contaminant details
        contam_df = pd.DataFrame(contaminants)
        
        # Create a plot showing the target and its contaminants
        plt.figure(figsize=(10, 8))
        
        # Plot target
        plt.scatter(0, 0, s=300, c='red', marker='*', label='Target')
        
        # Plot contaminants
        sc = plt.scatter(
            (contam_df['contam_ra'] - target['target_ra']) * np.cos(np.radians(target['target_dec'])) * 3600,
            (contam_df['contam_dec'] - target['target_dec']) * 3600,
            s=100 * contam_df['brightness_impact'] / max(contam_df['brightness_impact'].max(), 1),
            c=contam_df['variability_impact'],
            cmap='plasma',
            alpha=0.7
        )
        
        # Draw TESS pixel
        pixel_size = 21.0  # arcsec
        plt.plot([-pixel_size/2, pixel_size/2, pixel_size/2, -pixel_size/2, -pixel_size/2],
                [-pixel_size/2, -pixel_size/2, pixel_size/2, pixel_size/2, -pixel_size/2],
                'k--', alpha=0.5, label='TESS pixel')
        
        plt.colorbar(sc, label='Variability Impact')
        plt.xlabel('ΔRA (arcsec)')
        plt.ylabel('ΔDec (arcsec)')
        plt.title(f'Target {target["target_sourceid"]} with Contamination Score: {target["total_contamination_score"]:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Set limits slightly larger than TESS pixel
        plt.xlim(-pixel_size * 0.75, pixel_size * 0.75)
        plt.ylim(-pixel_size * 0.75, pixel_size * 0.75)
        
        plt.savefig(os.path.join(output_folder, f'target_{target["target_sourceid"]}_contamination.png'))
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
            'total_contamination_score': target['total_contamination_score']
        }
        
        # If there are no contaminants, add a single row
        if target['num_contaminants'] == 0:
            flattened_results.append(target_data)
        else:
            # Add a row for each contaminant
            for contam in target['contaminant_details']:
                row = target_data.copy()
                row.update({
                    'contam_sourceid': contam['contam_sourceid'],
                    'contam_ra': contam['contam_ra'],
                    'contam_dec': contam['contam_dec'],
                    'separation_arcsec': contam['separation_arcsec'],
                    'contam_mag': contam['contam_mag'],
                    'contam_variability': contam['contam_variability'],
                    'brightness_impact': contam['brightness_impact'],
                    'variability_impact': contam['variability_impact'],
                    'proximity_impact': contam['proximity_impact'],
                    'contamination_score': contam['contamination_score']
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