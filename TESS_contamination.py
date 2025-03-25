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



def visualize_contamination_enhanced(results_df, output_folder='contamination_plots'):
    """
    Creates enhanced visualization plots for the contamination analysis
    with more informative features.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Set a consistent style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Enhanced histogram of contaminant counts with cumulative distribution
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(111)
    
    # Create histogram with density=True for percentage
    n, bins, patches = ax1.hist(results_df['num_contaminants'], bins=30, 
                               alpha=0.7, color='steelblue', 
                               edgecolor='black', density=True,
                               label='Frequency')
    
    # Add cumulative distribution
    ax2 = ax1.twinx()
    counts, edges = np.histogram(results_df['num_contaminants'], bins=30)
    cum_counts = np.cumsum(counts) / len(results_df)
    ax2.plot(edges[:-1], cum_counts, 'r-', linewidth=2, label='Cumulative %')
    ax2.set_ylim(0, 1.05)
    
    # Add statistics to the plot
    mean_contams = results_df['num_contaminants'].mean()
    median_contams = results_df['num_contaminants'].median()
    max_contams = results_df['num_contaminants'].max()
    
    stats_text = f"Mean: {mean_contams:.1f}\nMedian: {median_contams:.1f}\nMax: {max_contams:.0f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # Add labels and title
    ax1.set_xlabel('Number of Contaminating Sources', fontsize=14)
    ax1.set_ylabel('Density (Proportion per bin)', fontsize=14)
    ax2.set_ylabel('Cumulative Proportion', fontsize=14)
    plt.title('Distribution of Contaminating Sources per Target', fontsize=16)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'contaminant_counts_enhanced.png'), dpi=300)
    plt.close()
    
    # Plot 2: Enhanced noise contribution distribution
    plt.figure(figsize=(12, 8))
    valid_noise = results_df['total_noise_contribution_ppm'].replace(0, np.nan).dropna()
    
    if len(valid_noise) > 0:
        # Create subplot grid
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Plot histogram with log scale on x-axis for main plot
        log_noise = np.log10(valid_noise)
        sns.histplot(log_noise, bins=30, kde=True, ax=ax1, color='darkblue')
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        colors = ['green', 'orange', 'red', 'purple']
        
        for p, c in zip(percentiles, colors):
            percentile_val = np.percentile(log_noise, p)
            ax1.axvline(x=percentile_val, color=c, linestyle='--', 
                       linewidth=2, label=f'{p}th percentile')
        
        # Add box plot below
        sns.boxplot(x=log_noise, ax=ax2, orient='h', color='steelblue')
        
        # Add summary statistics
        stats_text = (
            f"Mean: {10**(np.mean(log_noise)):.1f} ppm\n"
            f"Median: {10**(np.median(log_noise)):.1f} ppm\n"
            f"95th percentile: {10**(np.percentile(log_noise, 95)):.1f} ppm\n"
            f"Max: {valid_noise.max():.1f} ppm"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Format x-axis with original values
        def log_format(x, pos):
            return f'{10**x:.0f}'
        
        from matplotlib.ticker import FuncFormatter
        ax1.xaxis.set_major_formatter(FuncFormatter(log_format))
        ax2.xaxis.set_major_formatter(FuncFormatter(log_format))
        
        # Add labels
        ax1.set_xlabel('')
        ax1.set_ylabel('Frequency', fontsize=14)
        ax2.set_xlabel('Noise Contribution (ppm)', fontsize=14)
        ax2.set_ylabel('')
        
        ax1.set_title('Distribution of Noise Contributions from Contaminating Sources', fontsize=16)
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'noise_contributions_enhanced.png'), dpi=300)
    plt.close()
    
    # Plot 3: Enhanced scatter plot with hexbin density and individual points
    plt.figure(figsize=(14, 10))
    valid_data = results_df.dropna(subset=['target_g_mag', 'total_noise_contribution_ppm'])
    valid_data = valid_data[valid_data['total_noise_contribution_ppm'] > 0]
    
    if len(valid_data) > 0:
        # Create a grid of subplots
        gs = GridSpec(3, 3, width_ratios=[3, 1, 0.2], height_ratios=[1, 3, 0.2])
        
        # Main scatter plot
        ax_main = plt.subplot(gs[1, 0])
        
        # Create hexbin plot for density
        hexbin = ax_main.hexbin(valid_data['target_g_mag'], 
                               np.log10(valid_data['total_noise_contribution_ppm']),
                               gridsize=40, cmap='viridis', 
                               mincnt=1, bins='log')
        
        # Add a colorbar
        ax_cb = plt.subplot(gs[1, 2])
        cb = plt.colorbar(hexbin, cax=ax_cb)
        cb.set_label('log10(count)', rotation=270, labelpad=15)
        
        # Add histogram on top and right showing marginal distributions
        ax_top = plt.subplot(gs[0, 0])
        ax_right = plt.subplot(gs[1, 1])
        
        sns.histplot(valid_data['target_g_mag'], ax=ax_top, color='navy')
        ax_top.set_xlim(ax_main.get_xlim())
        ax_top.set_ylabel('Count')
        ax_top.set_xlabel('')
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['top'].set_visible(False)
        
        sns.histplot(y=np.log10(valid_data['total_noise_contribution_ppm']), ax=ax_right, color='navy')
        ax_right.set_ylim(ax_main.get_ylim())
        ax_right.set_xlabel('Count')
        ax_right.set_ylabel('')
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        
        # Add trendline
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_data['target_g_mag'],
                np.log10(valid_data['total_noise_contribution_ppm']))
            
            x = np.array(ax_main.get_xlim())
            y = slope * x + intercept
            ax_main.plot(x, y, 'r-', label=f'Fit: $\\log{{Noise}} = {slope:.2f}G {intercept:+.2f}$\n$R^2 = {r_value**2:.2f}$')
            ax_main.legend(loc='upper left')
        except:
            # Skip trendline if regression fails
            pass
        
        # Format y-axis with original values
        def log10_format(y, pos):
            return f'{10**y:.0f}'
        
        ax_main.yaxis.set_major_formatter(FuncFormatter(log10_format))
        
        # Add labels and title
        ax_main.set_xlabel('Target G Magnitude', fontsize=14)
        ax_main.set_ylabel('Noise Contribution (ppm)', fontsize=14)
        ax_main.set_title('Contamination Noise vs. Target Magnitude', fontsize=16)
        
        # Add grid
        ax_main.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.savefig(os.path.join(output_folder, 'noise_vs_magnitude_enhanced.png'), dpi=300)
    plt.close()
    
    # Plot 4: Enhanced spatial distribution with sky density map
    plt.figure(figsize=(14, 12))
    valid_data = results_df.dropna(subset=['target_ra', 'target_dec', 'total_noise_contribution_ppm'])
    valid_data = valid_data[valid_data['total_noise_contribution_ppm'] > 0]
    
    if len(valid_data) > 0:
        # Create subplots
        gs = GridSpec(1, 2, width_ratios=[20, 1])
        ax = plt.subplot(gs[0, 0])
        
        # Use a normalized log scale for colors
        noise_log = np.log10(valid_data['total_noise_contribution_ppm'])
        vmin, vmax = noise_log.min(), noise_log.max()
        
        # Create a custom color normalization
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Create hexbin density map
        hexbin = ax.hexbin(valid_data['target_ra'], valid_data['target_dec'], 
                          C=noise_log, 
                          reduce_C_function=np.median,
                          gridsize=50, cmap='viridis', 
                          norm=norm)
        
        # Add colorbar
        ax_cb = plt.subplot(gs[0, 1])
        cb = plt.colorbar(hexbin, cax=ax_cb)
        
        # Format colorbar with original values
        def log10_format(x, pos):
            return f'{10**x:.0f}'
        
        cb.ax.yaxis.set_major_formatter(FuncFormatter(log10_format))
        cb.set_label('Median Noise Contribution (ppm)', rotation=270, labelpad=15)
        
        # Add ecliptic coordinates (approximate)
        ra = np.linspace(0, 360, 360)
        dec = np.zeros_like(ra)  # Ecliptic plane (approximately)
        ax.plot(ra, dec, 'r--', alpha=0.5, label='Ecliptic plane (approx.)')
        
        # Add TESS observing regions (approximate)
        # TESS observes in 13 sectors per hemisphere
        for sector in range(13):
            center_ra = (sector * 360/13 + 180) % 360
            ax.axvline(x=center_ra, color='grey', alpha=0.3, linestyle=':')
        
        # Add labels
        ax.set_xlabel('RA (deg)', fontsize=14)
        ax.set_ylabel('Dec (deg)', fontsize=14)
        ax.set_title('Spatial Distribution of Target Contamination', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add density information
        count_per_area = len(valid_data) / ((ax.get_xlim()[1] - ax.get_xlim()[0]) * 
                                        (ax.get_ylim()[1] - ax.get_ylim()[0]))
        density_text = f"Target density: {count_per_area:.1f} targets/sq.deg"
        ax.text(0.02, 0.02, density_text, transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.savefig(os.path.join(output_folder, 'spatial_noise_contribution_enhanced.png'), dpi=300)
    plt.close()
    
    # Plot 5: New plot - Contamination vs stellar color (G-RP)
    plt.figure(figsize=(12, 8))
    valid_data = results_df.dropna(subset=['target_g_rp', 'total_noise_contribution_ppm'])
    valid_data = valid_data[valid_data['total_noise_contribution_ppm'] > 0]
    
    if len(valid_data) > 0:
        # Create hexbin color plot
        hexbin = plt.hexbin(valid_data['target_g_rp'], 
                          np.log10(valid_data['total_noise_contribution_ppm']),
                          gridsize=40, cmap='plasma', 
                          mincnt=1, bins='log')
        
        # Add colorbar
        cb = plt.colorbar(hexbin)
        cb.set_label('log10(count)', rotation=270, labelpad=15)
        
        # Add stellar type annotations
        stellar_types = [
            {'g_rp': -0.3, 'type': 'O/B stars', 'y': 1.0},
            {'g_rp': 0.0, 'type': 'A stars', 'y': 1.5},
            {'g_rp': 0.4, 'type': 'F stars', 'y': 2.0},
            {'g_rp': 0.7, 'type': 'G stars', 'y': 2.5},
            {'g_rp': 1.0, 'type': 'K stars', 'y': 3.0},
            {'g_rp': 1.5, 'type': 'M stars', 'y': 3.5}
        ]
        
        for star in stellar_types:
            plt.annotate(star['type'], 
                        xy=(star['g_rp'], star['y']), 
                        xytext=(star['g_rp'], star['y']),
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        ha='center')
            plt.axvline(x=star['g_rp'], color='black', linestyle=':', alpha=0.3)
        
        # Try to add trendline
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_data['target_g_rp'],
                np.log10(valid_data['total_noise_contribution_ppm']))
            
            x = np.array([valid_data['target_g_rp'].min(), valid_data['target_g_rp'].max()])
            y = slope * x + intercept
            plt.plot(x, y, 'r-', linewidth=2, 
                   label=f'Fit: $\\log{{Noise}} = {slope:.2f}(G-RP) {intercept:+.2f}$\n$R^2 = {r_value**2:.2f}$')
            plt.legend(loc='upper left')
        except:
            pass
        
        # Format y-axis with original values
        def log10_format(y, pos):
            return f'{10**y:.0f}'
        
        plt.gca().yaxis.set_major_formatter(FuncFormatter(log10_format))
        
        # Add labels
        plt.xlabel('G-RP Color (Gaia)', fontsize=14)
        plt.ylabel('Noise Contribution (ppm)', fontsize=14)
        plt.title('Contamination Noise vs. Stellar Color', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_folder, 'color_vs_noise_contribution.png'), dpi=300)
    plt.close()
    
    # Plot 6: New plot - Contamination ratio vs target magnitude
    plt.figure(figsize=(12, 8))
    valid_data = results_df.dropna(subset=['target_g_mag', 'total_flux_contamination_ratio'])
    valid_data = valid_data[valid_data['total_flux_contamination_ratio'] > 0]
    
    if len(valid_data) > 0:
        # Define a custom colormap based on contamination severity
        cmap = plt.cm.get_cmap('RdYlGn_r')
        
        # Create scatter plot with custom coloring
        sc = plt.scatter(valid_data['target_g_mag'], 
                       valid_data['total_flux_contamination_ratio'], 
                       c=valid_data['total_flux_contamination_ratio'],
                       cmap=cmap,
                       norm=colors.LogNorm(vmin=max(0.001, valid_data['total_flux_contamination_ratio'].min()), 
                                       vmax=max(1.0, valid_data['total_flux_contamination_ratio'].max())),
                       s=30, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Flux Contamination Ratio', rotation=270, labelpad=15)
        
        # Add contamination level thresholds
        thresholds = [0.01, 0.05, 0.1, 0.5]
        labels = ['1%', '5%', '10%', '50%']
        colors = ['green', 'yellowgreen', 'orange', 'red']
        
        for thresh, label, col in zip(thresholds, labels, colors):
            plt.axhline(y=thresh, color=col, linestyle='--', 
                      label=f'{label} contamination')
        
        # Add labels
        plt.xlabel('Target G Magnitude', fontsize=14)
        plt.ylabel('Flux Contamination Ratio', fontsize=14)
        plt.title('Flux Contamination Ratio vs. Target Magnitude', fontsize=16)
        plt.yscale('log')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(loc='upper left')
        
        # Add text about interpretation
        text = (
            "High contamination (>10%)\n"
            "can significantly impact\n"
            "transit depth measurements"
        )
        plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
               verticalalignment='top')
        
        plt.savefig(os.path.join(output_folder, 'contamination_ratio_vs_magnitude.png'), dpi=300)
    plt.close()
    
    # Plot 7: New plot - Multivariate analysis
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    plt.figure(figsize=(16, 10))
    
    # Get valid data for all needed fields
    needed_columns = ['target_g_mag', 'target_g_rp', 'num_contaminants', 
                     'total_noise_contribution_ppm', 'total_flux_contamination_ratio']
    valid_data = results_df.dropna(subset=needed_columns)
    valid_data = valid_data[(valid_data['total_noise_contribution_ppm'] > 0) & 
                           (valid_data['total_flux_contamination_ratio'] > 0)]
    
    if len(valid_data) > 0:
        # Create a grid for 4 subplot panels
        gs = GridSpec(2, 2, wspace=0.4, hspace=0.4)
        
        # Panel 1: Magnitude vs # of contaminants colored by noise
        ax1 = plt.subplot(gs[0, 0])
        sc1 = ax1.scatter(valid_data['target_g_mag'], 
                         valid_data['num_contaminants'],
                         c=np.log10(valid_data['total_noise_contribution_ppm']), 
                         cmap='viridis',
                         alpha=0.7)
        
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        cb1 = plt.colorbar(sc1, cax=cax1)
        cb1.set_label('log10(Noise ppm)')
        
        ax1.set_xlabel('Target G Magnitude')
        ax1.set_ylabel('Number of Contaminants')
        ax1.set_title('Magnitude vs. Contaminant Count')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Noise vs. contamination ratio colored by G-RP
        ax2 = plt.subplot(gs[0, 1])
        sc2 = ax2.scatter(np.log10(valid_data['total_noise_contribution_ppm']), 
                         valid_data['total_flux_contamination_ratio'],
                         c=valid_data['target_g_rp'], 
                         cmap='plasma',
                         alpha=0.7)
        
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        cb2 = plt.colorbar(sc2, cax=cax2)
        cb2.set_label('G-RP Color')
        
        # Format x-axis with original values
        def log10_format(x, pos):
            return f'{10**x:.0f}'
        
        ax2.xaxis.set_major_formatter(FuncFormatter(log10_format))
        ax2.set_xlabel('Noise Contribution (ppm)')
        ax2.set_ylabel('Flux Contamination Ratio')
        ax2.set_yscale('log')
        ax2.set_title('Noise vs. Contamination Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: G-RP vs. contaminant count colored by magnitude
        ax3 = plt.subplot(gs[1, 0])
        sc3 = ax3.scatter(valid_data['target_g_rp'], 
                         valid_data['num_contaminants'],
                         c=valid_data['target_g_mag'], 
                         cmap='coolwarm',
                         alpha=0.7)
        
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.1)
        cb3 = plt.colorbar(sc3, cax=cax3)
        cb3.set_label('G Magnitude')
        
        ax3.set_xlabel('G-RP Color')
        ax3.set_ylabel('Number of Contaminants')
        ax3.set_title('Stellar Color vs. Contaminant Count')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Stellar density - add stellar type axis
        ax4 = plt.subplot(gs[1, 1])
        
        # Create a KDE (Kernel Density Estimate) plot
        sns.kdeplot(data=valid_data, x='target_g_rp', y='target_g_mag',
                   fill=True, cmap='Reds', alpha=0.7, 
                   levels=5, ax=ax4)
        
        # Add contour lines for target_g_mag
        contour = ax4.contour(
            *np.meshgrid(
                np.linspace(valid_data['target_g_rp'].min(), valid_data['target_g_rp'].max(), 100),
                np.linspace(valid_data['target_g_mag'].min(), valid_data['target_g_mag'].max(), 100)
            ),
            valid_data['total_noise_contribution_ppm'].values.reshape(-1, 1) * np.ones((1, 100)),
            cmap='viridis', alpha=0.7
        )
        
        # Invert y-axis (brighter stars at top)
        ax4.invert_yaxis()
        
        # Add stellar type annotations on top x-axis
        stellar_types = [
            {'g_rp': -0.3, 'type': 'O/B'},
            {'g_rp': 0.0, 'type': 'A'},
            {'g_rp': 0.4, 'type': 'F'},
            {'g_rp': 0.7, 'type': 'G'},
            {'g_rp': 1.0, 'type': 'K'},
            {'g_rp': 1.5, 'type': 'M'}
        ]
        
        for star in stellar_types:
            if (star['g_rp'] >= ax4.get_xlim()[0] and 
                star['g_rp'] <= ax4.get_xlim()[1]):
                ax4.annotate(star['type'], 
                           xy=(star['g_rp'], ax4.get_ylim()[0]), 
                           xytext=(star['g_rp'], ax4.get_ylim()[0] - 0.5),
                           ha='center')
        
        ax4.set_xlabel('G-RP Color')
        ax4.set_ylabel('G Magnitude')
        ax4.set_title('Stellar Population Density')
        
        # Add overall title
        plt.suptitle('Multivariate Analysis of TESS Target Contamination', fontsize=18, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_folder, 'multivariate_analysis.png'), dpi=300)
    plt.close()
    
    print(f"Enhanced visualization plots saved to {output_folder}")




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
    
    print(f"Creating enhanced visualization plots in {plots_folder}...")
    visualize_contamination_enhanced(results, plots_folder)
    
    # Save detailed report
    print(f"Saving detailed contamination report to {report_file}...")
    save_contamination_report(results, report_file)
    
    print("Done!")