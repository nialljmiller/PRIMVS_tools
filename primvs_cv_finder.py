#!/usr/bin/env python3
"""
PRIMVS CV Candidate Finder

This script identifies potential cataclysmic variable (CV) candidates from the PRIMVS catalog
using multiple detection strategies to ensure high completeness:

1. Feature-based filtering using known CV characteristics
2. Supervised classification with XGBoost
3. Anomaly detection to identify unusual parameter combinations

The tool is designed to be generous in its candidate selection to maximize completeness.
"""

import os
import time
import warnings
import argparse
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import joblib
import sys


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.patches import Polygon
import matplotlib.patheffects as path_effects

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactic, ICRS
import astropy.units as u
from matplotlib.patches import Polygon
import matplotlib.patheffects as path_effects
from matplotlib.transforms import Affine2D
import matplotlib as mpl



class PrimvsCVFinder:
    """
    A comprehensive tool for identifying cataclysmic variable star candidates
    in the PRIMVS catalog.
    """
    
    def __init__(self, 
                 primvs_file, 
                 output_dir='./cv_candidates',
                 known_cv_file=None,
                 period_limit=5.0,  # CVs typically have periods under 1 day, being generous with 5
                 amplitude_limit=0.05,  # Very low threshold to be inclusive
                 fap_limit=0.5):      # False alarm probability threshold
        """
        Initialize the CV finder with input files and parameters.
        
        Parameters:
        -----------
        primvs_file : str
            Path to the PRIMVS FITS file
        output_dir : str
            Directory to save outputs
        known_cv_file : str, optional
            Path to a file containing known CVs for training
        period_limit : float
            Maximum period (in days) for CV candidates
        amplitude_limit : float
            Minimum amplitude (in mag) for CV candidates
        fap_limit : float
            Maximum false alarm probability for period detection
        """
        self.primvs_file = primvs_file
        self.output_dir = output_dir
        self.known_cv_file = known_cv_file
        self.period_limit = period_limit
        self.amplitude_limit = amplitude_limit
        self.fap_limit = fap_limit
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Features used for CV identification
        self.cv_features = [
            # Period-related features
            'true_period', 'log_period',
            
            # Amplitude and variability features
            'true_amplitude', 'percent_amp', 'stet_k', 
            'max_slope', 'MAD', 'mean_var', 'roms',
            
            # Statistical features
            'skew', 'kurt', 'Cody_M', 'eta', 'eta_e',
            'med_BRP', 'p_to_p_var', 'lag_auto',
            
            # Signal quality
            'best_fap',
            
            # Color features (when available)
            'Z-K', 'Y-K', 'J-K', 'H-K',
            
            # Position features (for spatial distribution analysis)
            #'l', 'b'
        ]
        
        # Initialize data containers
        self.primvs_data = None
        self.cv_candidates = None
        self.model = None

    
    def post_processing_plots(self, max_top_candidates=20):
        """
        Enhanced post-processing routine that consistently displays three categories in plots:
        1. All candidates
        2. Known CVs (if available)
        3. Best CVs (based on optimal confidence threshold)
        
        Parameters
        ----------
        max_top_candidates : int
            Number of top candidates to show in the summary listing (default: 20).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.table import Table
        import pandas as pd
        from sklearn.metrics import roc_curve, auc

        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates available for post-processing plots.")
            return

        df = self.cv_candidates
    
        df['period_hours'] = df['true_period'] * 24.0
        df['log_period'] = np.log10(df['period_hours'])    
        df['l_centered'] = np.where(df['l'] > 180, df['l'] - 360, df['l'])


        # ---------------------------
        # 1) Determine optimal threshold for "best" candidates
        # ---------------------------
        has_known_cvs = ('is_known_cv' in df.columns) and df['is_known_cv'].any()
        
        # Determine confidence column
        if 'confidence' in df.columns:
            conf_col = 'confidence'
        elif 'cv_prob' in df.columns:
            conf_col = 'cv_prob'
        else:
            df['temp_conf'] = 0.5
            conf_col = 'temp_conf'
        
        # Find optimal threshold if we have known CVs
        if has_known_cvs and conf_col in df.columns:
            print("Determining optimal threshold from ROC curve...")
            fpr, tpr, thresholds = roc_curve(df['is_known_cv'], df[conf_col])
            roc_auc = auc(fpr, tpr)
            
            # Find optimal threshold (maximize tpr - fpr)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal threshold from ROC curve: {optimal_threshold:.3f}")
        else:
            # Default to 0.8 if we don't have known CVs for comparison
            optimal_threshold = 0.8
            print(f"Using default threshold: {optimal_threshold:.3f}")
        
        # Flag best candidates
        df['is_best_candidate'] = df[conf_col] >= optimal_threshold
        best_count = df['is_best_candidate'].sum()
        print(f"Identified {best_count} best candidates (confidence >= {optimal_threshold:.3f})")
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ---------------------------
        # 2) Prepare or reuse PCA columns from embeddings
        # ---------------------------
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in df.columns]
        
        # Only do PCA if we actually have a decent chunk of embedding features
        if len(embedding_features) >= 3:
            # Check if pca_1 already exists
            if 'pca_1' not in df.columns or 'pca_2' not in df.columns or 'pca_3' not in df.columns:
                print("Computing PCA on contrastive embeddings...")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                emb_values = df[embedding_features].values
                emb_3d = pca.fit_transform(emb_values)
                df['pca_1'] = emb_3d[:, 0]
                df['pca_2'] = emb_3d[:, 1]
                df['pca_3'] = emb_3d[:, 2]
            else:
                print("Reusing existing pca_1, pca_2, pca_3 columns.")
        else:
            print("No or insufficient embedding features found. Skipping PCA-based plots.")
        
        # Check for two-stage classification
        has_two_stage = ('cv_prob_trad' in df.columns) and ('cv_prob_emb' in df.columns)

        # ---------------------------
        # 3) 3D PCA scatter with three categories
        # ---------------------------
        if 'pca_1' in df.columns and 'pca_2' in df.columns and 'pca_3' in df.columns:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 1. Plot all candidates
            all_candidates = df
            sc_all = ax.scatter(
                all_candidates['pca_1'],
                all_candidates['pca_2'],
                all_candidates['pca_3'],
                c='lightgray',
                alpha=0.3,
                s=10,
                label='All Candidates'
            )
            
            # 2. Plot best candidates that aren't known CVs
            best_candidates = df[(df['is_best_candidate']) & (~df['is_known_cv'] if has_known_cvs else True)]
            if len(best_candidates) > 0:
                sc_best = ax.scatter(
                    best_candidates['pca_1'],
                    best_candidates['pca_2'],
                    best_candidates['pca_3'],
                    c='blue',
                    alpha=0.7,
                    s=20,
                    label=f'Best Candidates (conf ≥ {optimal_threshold:.2f})'
                )
            
            # 3. Plot known CVs if available
            if has_known_cvs:
                known_cvs = df[df['is_known_cv']]
                sc_known = ax.scatter(
                    known_cvs['pca_1'],
                    known_cvs['pca_2'],
                    known_cvs['pca_3'],
                    c='red',
                    alpha=1.0,
                    s=40,
                    marker='*',
                    label='Known CVs'
                )
            
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_zlabel('PCA 3')
            ax.set_title('3D PCA of Embeddings')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'embeddings_3d_categories.png'), dpi=300)
            plt.close()

            # ---------------------------
            # 4) 2D PCA scatter with three categories
            # ---------------------------
            plt.figure(figsize=(10, 8))
            
            # 1. Plot all candidates
            plt.scatter(
                all_candidates['pca_1'],
                all_candidates['pca_2'],
                c='lightgray',
                alpha=0.3,
                s=10,
                label='All Candidates'
            )
            
            # 2. Plot best candidates that aren't known CVs
            if len(best_candidates) > 0:
                plt.scatter(
                    best_candidates['pca_1'],
                    best_candidates['pca_2'],
                    c='blue',
                    alpha=0.7,
                    s=20,
                    label=f'Best Candidates (conf ≥ {optimal_threshold:.2f})'
                )
            
            # 3. Plot known CVs if available
            if has_known_cvs:
                plt.scatter(
                    known_cvs['pca_1'],
                    known_cvs['pca_2'],
                    c='red',
                    alpha=1.0,
                    s=40,
                    marker='*',
                    label='Known CVs'
                )
            
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.title('2D PCA of Embeddings')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'embeddings_2d_categories.png'), dpi=300)
            plt.close()

        # ---------------------------
        # 5) Hexbin of cv_prob_trad vs cv_prob_emb (two-stage only)
        # ---------------------------
        if has_two_stage:
            plt.figure(figsize=(8, 6))
            # Hexbin for density of all candidates
            hb = plt.hexbin(
                df['cv_prob_trad'], df['cv_prob_emb'],
                gridsize=30,
                cmap='Greys',
                bins='log',
                alpha=0.7
            )
            plt.colorbar(hb, label='log10(count)')
            
            # Overlay best candidates
            if len(best_candidates) > 0:
                plt.scatter(
                    best_candidates['cv_prob_trad'],
                    best_candidates['cv_prob_emb'],
                    c='blue',
                    alpha=0.7,
                    s=20,
                    label=f'Best Candidates (conf ≥ {optimal_threshold:.2f})'
                )
            
            # Overlay known CVs
            if has_known_cvs:
                plt.scatter(
                    known_cvs['cv_prob_trad'],
                    known_cvs['cv_prob_emb'],
                    c='red',
                    alpha=1.0,
                    s=40,
                    marker='*',
                    label='Known CVs'
                )
            
            plt.plot([0,1], [0,1], 'k--', alpha=0.7, label='Perfect Agreement')
            plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            
            plt.xlabel('Traditional Probability')
            plt.ylabel('Embedding Probability')
            plt.title('Two-Stage Classification: Probabilities Comparison')
            plt.legend(loc='upper left')
            plt.axis('square')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'prob_trad_vs_emb_categories.png'), dpi=300)
            plt.close()

        # ---------------------------
        # 6) Bailey diagram with three categories
        # ---------------------------
        # Convert period to hours & take log

        plt.figure(figsize=(10, 6))
        
        # 1. Plot all candidates
        plt.scatter(
            all_candidates['log_period'],
            all_candidates['true_amplitude'],
            c='lightgray',
            alpha=0.3,
            s=10,
            label='All Candidates'
        )
        
        # 2. Plot best candidates that aren't known CVs
        if len(best_candidates) > 0:
            plt.scatter(
                best_candidates['log_period'],
                best_candidates['true_amplitude'],
                c='blue',
                alpha=0.7,
                s=20,
                label=f'Best Candidates (conf ≥ {optimal_threshold:.2f})'
            )
        
        # 3. Plot known CVs if available
        if has_known_cvs:
            plt.scatter(
                known_cvs['log_period'],
                known_cvs['true_amplitude'],
                c='red',
                alpha=1.0,
                s=40,
                marker='*',
                label='Known CVs'
            )
        
        # Mark the period gap
        plt.axvspan(np.log10(2), np.log10(3), alpha=0.2, color='gray', label='Period Gap (2-3h)')
        
        plt.xlabel('log₁₀(Period) [hours]')
        plt.ylabel('Amplitude [mag]')
        plt.title('Bailey Diagram')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'bailey_diagram_categories.png'), dpi=300)
        plt.close()

        # ---------------------------
        # 7) Galactic spatial plot with three categories
        # ---------------------------

        plt.figure(figsize=(12, 8))
        
        # 1. Plot all candidates
        plt.scatter(
            all_candidates['l_centered'],
            all_candidates['b'],
            c='lightgray',
            alpha=0.3,
            s=10,
            label='All Candidates'
        )
        
        # 2. Plot best candidates that aren't known CVs
        if len(best_candidates) > 0:
            plt.scatter(
                best_candidates['l_centered'],
                best_candidates['b'],
                c='blue',
                alpha=0.7,
                s=20,
                label=f'Best Candidates (conf ≥ {optimal_threshold:.2f})'
            )
        
        # 3. Plot known CVs if available
        if has_known_cvs:
            plt.scatter(
                known_cvs['l_centered'],
                known_cvs['b'],
                c='red',
                alpha=1.0,
                s=40,
                marker='*',
                label='Known CVs'
            )
        
        # TESS overlay
        if 'TESSCycle8Overlay' in globals():
            tess_overlay = TESSCycle8Overlay()
            tess_overlay.add_to_plot(plt.gca())
        
        plt.xlabel('Galactic Longitude (shifted) [deg]')
        plt.ylabel('Galactic Latitude [deg]')
        plt.title('Galactic Spatial Distribution with TESS Overlay')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'spatial_galactic_categories.png'), dpi=300)
        plt.close()

        # ---------------------------
        # 8) Short summary text + top candidates
        # ---------------------------
        summary_path = os.path.join(self.output_dir, 'cv_summary_categories.txt')
        with open(summary_path, 'w') as f:
            f.write("CV Candidate Summary with Categories\n")
            f.write("====================================\n\n")
            f.write(f"Total candidates: {len(df)}\n")
            f.write(f"Best candidates (conf ≥ {optimal_threshold:.3f}): {best_count}\n")
            if has_known_cvs:
                known_count = df['is_known_cv'].sum()
                f.write(f"Known CVs: {known_count}\n")
            f.write("\n")

            # Quick period stats for each category
            if 'true_period' in df.columns:
                f.write("Period (hrs) Stats by Category:\n")
                f.write("---------------------------------\n")
                
                # All candidates
                period_hours = df['period_hours']
                f.write("All candidates:\n")
                f.write(f"  Min: {period_hours.min():.2f}\n")
                f.write(f"  Median: {period_hours.median():.2f}\n")
                f.write(f"  Max: {period_hours.max():.2f}\n\n")
                
                # Best candidates
                if best_count > 0:
                    best_period = best_candidates['period_hours']
                    f.write("Best candidates:\n")
                    f.write(f"  Min: {best_period.min():.2f}\n")
                    f.write(f"  Median: {best_period.median():.2f}\n")
                    f.write(f"  Max: {best_period.max():.2f}\n\n")
                
                # Known CVs
                if has_known_cvs and known_count > 0:
                    known_period = known_cvs['period_hours']
                    f.write("Known CVs:\n")
                    f.write(f"  Min: {known_period.min():.2f}\n")
                    f.write(f"  Median: {known_period.median():.2f}\n")
                    f.write(f"  Max: {known_period.max():.2f}\n\n")
            
            # Top candidates
            sort_col = None
            for col in ['cv_prob', 'confidence', 'blended_score']:
                if col in df.columns:
                    sort_col = col
                    break
            if sort_col is None and 'best_fap' in df.columns:
                sort_col = 'best_fap'
            
            if sort_col is not None:
                ascending = (sort_col == 'best_fap')  # For FAP, lower is better
                top_candidates = df.sort_values(sort_col, ascending=ascending).head(max_top_candidates)
                f.write(f"Top {max_top_candidates} candidates sorted by '{sort_col}':\n")
                f.write("------------------------------------------------\n")
                id_col = 'sourceid' if 'sourceid' in df.columns else 'primvs_id'
                for i, row in top_candidates.iterrows():
                    pid = str(row.get(id_col, '???'))
                    if 'period_hours' in row and 'true_amplitude' in row:
                        per_hrs = row['period_hours']
                        amp = row['true_amplitude']
                    else:
                        per_hrs, amp = -1, -1
                    val = row.get(sort_col, -1)
                    is_known = "*" if row.get('is_known_cv', False) else " "
                    is_best = "+" if row.get('is_best_candidate', False) else " "
                    f.write(f"  {is_known}{is_best} {pid:15s}  Per={per_hrs:.2f}h  Amp={amp:.2f}  {sort_col}={val:.3f}\n")
                
                f.write("\nLegend: * = Known CV, + = Best Candidate\n")
            else:
                f.write("No recognized sort column found for top candidates.\n")
        
        print(f"Enhanced post-processing complete. Summary written to: {summary_path}")





    def analyze_period_gap_distribution(self):
        """Analyze CV candidate distribution around the period gap"""
        
        if 'true_period' not in self.cv_candidates.columns:
            return
        
        period_hours = self.cv_candidates['true_period'] * 24.0
        
        # Define period bins with focus on the gap
        period_bins = [0, 1, 1.5, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 4, 6, 12, 24]
        
        # Calculate distribution
        period_dist, _ = np.histogram(period_hours, bins=period_bins)
        
        # Plot with focus on period gap
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(period_dist)), period_dist, alpha=0.7)
        plt.xticks(range(len(period_dist)), 
                   [f"{period_bins[i]:.1f}-{period_bins[i+1]:.1f}" for i in range(len(period_dist))],
                   rotation=45)
        
        # Highlight the period gap
        gap_start_idx = period_bins.index(1.9)
        gap_end_idx = period_bins.index(3.1)
        for i in range(gap_start_idx, gap_end_idx):
            plt.bar(i, period_dist[i], color='red', alpha=0.7)
        
        plt.axvline(gap_start_idx - 0.5, color='r', linestyle='--', alpha=0.5)
        plt.axvline(gap_end_idx - 0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Period Range (hours)')
        plt.ylabel('Number of CV Candidates')
        plt.title('Period Distribution of CV Candidates with Gap Highlighted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'period_gap_analysis.png'), dpi=300)



    def analyze_galactic_regions(self):
        """
        Analyze CV period distribution across different Galactic regions (bulge and disk)
        with statistical comparison between each region and the overall distribution.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        import os
        
        print("Analyzing CV distribution across Galactic regions...")
        
        # Define regions based on Galactic position
        self.cv_candidates['region'] = 'Other'
        
        # 1. Central Bulge
        central_mask = (self.cv_candidates['l'] > -5) & (self.cv_candidates['l'] < 5) & \
                       (self.cv_candidates['b'] > -5) & (self.cv_candidates['b'] < 5)
        self.cv_candidates.loc[central_mask, 'region'] = 'Central Bulge'
        
        # 2. Inner Bulge
        inner_mask = (self.cv_candidates['l'] > -10) & (self.cv_candidates['l'] < 10) & \
                     (self.cv_candidates['b'] > -10) & (self.cv_candidates['b'] < 10) & \
                     ~central_mask
        self.cv_candidates.loc[inner_mask, 'region'] = 'Inner Bulge'
        
        # 3. Outer Bulge
        outer_mask = (self.cv_candidates['l'] > -20) & (self.cv_candidates['l'] < 20) & \
                     (self.cv_candidates['b'] > -15) & (self.cv_candidates['b'] < 15) & \
                     ~central_mask & ~inner_mask
        self.cv_candidates.loc[outer_mask, 'region'] = 'Outer Bulge'
        
        # 4. Inner Disk - Use ±1 degree latitude for disk regions
        inner_disk_mask = ((np.abs(self.cv_candidates['l']) >= 20) & (np.abs(self.cv_candidates['l']) < 60)) & \
                          (np.abs(self.cv_candidates['b']) < 1)
        self.cv_candidates.loc[inner_disk_mask, 'region'] = 'Inner Disk'
        
        # 5. Outer Disk
        outer_disk_mask = (np.abs(self.cv_candidates['l']) >= 60) & \
                          (np.abs(self.cv_candidates['b']) < 1)
        self.cv_candidates.loc[outer_disk_mask, 'region'] = 'Outer Disk'
        
        # 6. High Latitude
        high_lat_mask = (np.abs(self.cv_candidates['b']) >= 5)
        self.cv_candidates.loc[high_lat_mask, 'region'] = 'High Latitude'
        
        # Prepare figure for period distribution analysis
        fig, axs = plt.subplots(3, 2, figsize=(15, 18))
        axs = axs.flatten()
        
        # Calculate overall period distribution (in hours) for reference
        all_periods = self.cv_candidates['true_period'] * 24.0  # Convert to hours
        log_all_periods = np.log10(all_periods)
        valid_log_periods = log_all_periods[~np.isnan(log_all_periods) & ~np.isinf(log_all_periods)]
        
        # Define regions to analyze
        regions = ['Central Bulge', 'Inner Bulge', 'Outer Bulge', 
                   'Inner Disk', 'Outer Disk', 'High Latitude']
        
        # Color map for visual distinction
        colors = ['#E63946', '#F1C453', '#A8DADC', '#457B9D', '#1D3557', '#6A994E']
        
        region_stats = {}
        
        # Common bin edges for all histograms
        min_log_period = -1.5  # ~0.03 hours
        max_log_period = 2.0   # 100 hours
        common_bins = np.linspace(min_log_period, max_log_period, 30)
        
        # Analyze each region
        for i, region in enumerate(regions):
            region_mask = self.cv_candidates['region'] == region
            count = region_mask.sum()
            
            # Extract period data for this region
            period_hours = self.cv_candidates.loc[region_mask, 'true_period'] * 24.0
            log_period = np.log10(period_hours)
            valid_log_period = log_period[~np.isnan(log_period) & ~np.isinf(log_period)]
            
            # Handle the case where there are no valid periods
            if len(valid_log_period) == 0 or len(valid_log_periods) == 0:
                # Skip KS test and use placeholder values
                ks_stat, p_value = np.nan, np.nan
                median_period = np.nan
                # Display message about insufficient data
                axs[i].text(0.5, 0.5, f"Insufficient data for {region}\n(n={count})", 
                           ha='center', va='center', transform=axs[i].transAxes, fontsize=14)
                axs[i].set_title(f"{region}", fontsize=16)
                continue
            else:
                # Perform Kolmogorov-Smirnov test against the overall distribution
                try:
                    ks_stat, p_value = stats.ks_2samp(valid_log_period, valid_log_periods)
                    median_period = np.median(period_hours)
                except Exception as e:
                    print(f"Error performing KS test for {region}: {str(e)}")
                    print(f"  valid_log_period: {len(valid_log_period)} values")
                    print(f"  valid_log_periods: {len(valid_log_periods)} values")
                    ks_stat, p_value = np.nan, np.nan
                    median_period = np.median(period_hours) if len(period_hours) > 0 else np.nan
            
            region_stats[region] = {
                'count': count,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'median_period': median_period
            }
            
            # Create normalized histogram for this region
            try:
                hist_region, _ = np.histogram(valid_log_period, bins=common_bins, density=True)
                hist_region_scaled = hist_region * count  # Scale back to counts for readability
                
                # Plot normalized histogram
                width = common_bins[1] - common_bins[0]
                axs[i].bar(common_bins[:-1], hist_region_scaled, width=width, alpha=0.7, color=colors[i],
                          label=f'{region} (n={count})')
                
                # Plot the overall distribution as a step function for comparison
                hist_all, _ = np.histogram(valid_log_periods, bins=common_bins, density=True)
                total_count = len(valid_log_periods)
                scale_factor = count / total_count if total_count > 0 else 1  # Scale to make visual comparison easier
                axs[i].step(common_bins[:-1], hist_all * total_count * scale_factor, where='post', 
                           color='black', linestyle='--', linewidth=2, 
                           label='Overall Distribution (scaled)')
                
                # Highlight the period gap
                axs[i].axvspan(np.log10(2), np.log10(3), alpha=0.2, color='red', label='Period Gap (2-3h)')
                
                # Add statistical information - format p-value to avoid showing "0.0000"
                if np.isnan(p_value):
                    p_value_str = "p=N/A"
                elif p_value < 0.0001:
                    p_value_str = "p<0.0001"
                else:
                    p_value_str = f"p={p_value:.4f}"
                    
                if np.isnan(median_period):
                    median_str = "N/A"
                else:
                    median_str = f"{median_period:.2f}h"
                    
                stat_text = f"KS test: {p_value_str}\n"
                stat_text += f"Median period: {median_str}"
                axs[i].text(0.02, 0.95, stat_text, transform=axs[i].transAxes, 
                           va='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
                
                # Setup axis labels and title
                axs[i].set_xlabel('log₁₀(Period) [hours]', fontsize=12)
                axs[i].set_ylabel('Normalized Count', fontsize=12)
                axs[i].set_title(f"{region}", fontsize=16)
                axs[i].legend(loc='upper right', fontsize=10)
                axs[i].grid(True, alpha=0.3)
                
                # Set consistent x-axis range
                axs[i].set_xlim(min_log_period, max_log_period)
                
            except Exception as e:
                print(f"Error creating histogram for {region}: {str(e)}")
                axs[i].text(0.5, 0.5, f"Error creating plot for {region}\n{str(e)}", 
                           ha='center', va='center', transform=axs[i].transAxes, fontsize=12)
                axs[i].set_title(f"{region}", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'galactic_regions_period_analysis.png'), dpi=300)
        plt.close()
        
        # Create an improved spatial plot showing the distribution of CVs by region
        try:
            plt.figure(figsize=(12, 8))
            
            # Define the colormap for consistency with the histogram plots
            region_colors = {region: color for region, color in zip(regions, colors)}
            
            # Create density-based scatter (to handle the large number of points)
            # Get all longitudes and adjust them to be in -180 to +180 range for better visualization
            longitudes = self.cv_candidates['l'].values
            longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
            latitudes = self.cv_candidates['b'].values
            
            # Create hexbin plots for each region
            for region, color in region_colors.items():
                mask = self.cv_candidates['region'] == region
                if mask.sum() > 0:
                    try:
                        region_longs = self.cv_candidates.loc[mask, 'l'].values
                        region_longs = np.where(region_longs > 180, region_longs - 360, region_longs)
                        
                        # Use appropriate colormap for each region
                        region_cmap = plt.cm.Reds if region == 'Central Bulge' else \
                                     plt.cm.YlOrBr if region == 'Inner Bulge' else \
                                     plt.cm.GnBu if region == 'Outer Bulge' else \
                                     plt.cm.Blues if region == 'Inner Disk' else \
                                     plt.cm.Purples if region == 'Outer Disk' else \
                                     plt.cm.Greens
                        
                        hb = plt.hexbin(region_longs, 
                                       self.cv_candidates.loc[mask, 'b'].values,
                                       gridsize=50, cmap=region_cmap,
                                       alpha=0.8, mincnt=1)
                    except Exception as e:
                        print(f"Error plotting {region}: {str(e)}")
            
            # Add region boundary lines (more subtle)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
            
            # Add labels for regions with counts
            for region in regions:
                mask = self.cv_candidates['region'] == region
                count = mask.sum()
                if region == 'Central Bulge':
                    plt.annotate(f'Central Bulge\n(n={count})', xy=(0, 0), xytext=(0, 0), 
                                ha='center', va='center', fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.7))
                elif region == 'Inner Bulge':
                    plt.annotate(f'Inner Bulge\n(n={count})', xy=(-7, 7), xytext=(-7, 7), 
                                ha='center', va='center', fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.7))
                elif region == 'Outer Bulge':
                    plt.annotate(f'Outer Bulge\n(n={count})', xy=(-15, 12), xytext=(-15, 12), 
                                ha='center', va='center', fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.7))
                elif region == 'Inner Disk':
                    plt.annotate(f'Inner Disk\n(n={count})', xy=(-40, 0), xytext=(-40, 0), 
                                ha='center', va='center', fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.7))
                elif region == 'Outer Disk':
                    plt.annotate(f'Outer Disk\n(n={count})', xy=(-120, 0), xytext=(-120, 0), 
                                ha='center', va='center', fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.7))
                elif region == 'High Latitude':
                    plt.annotate(f'High Latitude\n(n={count})', xy=(-60, 10), xytext=(-60, 10), 
                                ha='center', va='center', fontsize=10, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.7))
            
            # Set axis limits to focus on the relevant area
            plt.xlim(-180, 180)
            plt.ylim(-15, 15)
            
            plt.xlabel('Galactic Longitude (l) [deg]', fontsize=14)
            plt.ylabel('Galactic Latitude (b) [deg]', fontsize=14)
            plt.title('Spatial Distribution of CV Candidates by Galactic Region', fontsize=16)
            plt.grid(True, alpha=0.3)
            
            # Create a summary table of regional statistics
            table_data = []
            for region in regions:
                if region in region_stats:
                    stats = region_stats[region]
                    p_value = stats['p_value']
                    
                    # Format p-value
                    if np.isnan(p_value):
                        p_value_str = "N/A"
                    elif p_value < 0.0001:
                        p_value_str = "p<0.0001"
                    else:
                        p_value_str = f"p={p_value:.4f}"
                    
                    # Format median period
                    median_period = stats['median_period']
                    if np.isnan(median_period):
                        median_str = "N/A"
                    else:
                        median_str = f"{median_period:.2f}h"
                    
                    table_data.append([
                        region, 
                        f"{stats['count']}",
                        p_value_str,
                        median_str
                    ])
            
            # Add table below the plot
            column_labels = ['Region', 'Count', 'KS Test', 'Median Period']
            table = plt.table(
                cellText=table_data,
                colLabels=column_labels,
                loc='bottom',
                bbox=[0.0, -0.35, 1.0, 0.25]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Adjust figure layout to make room for the table
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3)
            
            plt.savefig(os.path.join(self.output_dir, 'galactic_regions_spatial.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            print(f"Error creating spatial plot: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"Galactic region analysis complete. Results saved to {self.output_dir}.")
        
        # Return statistics for potential further analysis
        return region_stats


    def load_primvs_data(self):
        """Load data from the PRIMVS FITS file."""
        print(f"Loading PRIMVS data from {self.primvs_file}...")
        
        try:
            # Attempt to open with memory mapping for large files
            with fits.open(self.primvs_file, memmap=True) as hdul:
                # Extract data from the first extension (index 1)
                data = Table(hdul[1].data)
                
                # Convert to pandas DataFrame for easier manipulation
                self.primvs_data = data.to_pandas()
                
                # Check if expected columns are present
                required_cols = ['sourceid', 'true_period', 'true_amplitude', 'best_fap']
                missing_cols = [col for col in required_cols if col not in self.primvs_data.columns]
                
                if missing_cols:
                    missing_str = ', '.join(missing_cols)
                    print(f"Warning: Missing required columns: {missing_str}")
                    return False
                
                # Handle NaN values in critical columns
                for col in required_cols:
                    if self.primvs_data[col].isnull().any():
                        print(f"Handling NaN values in {col}")
                        if col in ['true_period', 'true_amplitude']:
                            # For these, we'll drop rows with NaN
                            self.primvs_data = self.primvs_data.dropna(subset=[col])
                        else:
                            # For others, we can fill with a default
                            self.primvs_data[col] = self.primvs_data[col].fillna(1.0)
                
                # Add log_period for better scaling in ML models
                self.primvs_data['log_period'] = np.log10(self.primvs_data['true_period'])
                
                # Rename color columns if present with standard format
                color_mappings = {
                    'z_med_mag-ks_med_mag': 'Z-K',
                    'y_med_mag-ks_med_mag': 'Y-K',
                    'j_med_mag-ks_med_mag': 'J-K',
                    'h_med_mag-ks_med_mag': 'H-K'
                }
                
                for old_name, new_name in color_mappings.items():
                    if old_name in self.primvs_data.columns:
                        self.primvs_data.rename(columns={old_name: new_name}, inplace=True)
                
                print(f"Loaded {len(self.primvs_data)} sources from PRIMVS catalog")
                return True
                
        except Exception as e:
            print(f"Error loading PRIMVS data: {str(e)}")
            return False
    

    def apply_initial_filters(self):
        """
        Apply initial loose filters to select potential CV candidates.
        These filters are intentionally permissive to maximize completeness.
        """
        if self.primvs_data is None or len(self.primvs_data) == 0:
            print("No PRIMVS data loaded. Call load_primvs_data() first.")
            return False
        
        print("Applying initial CV selection filters...")
        
        # Start with the full dataset
        filtered_data = self.primvs_data.copy()
        initial_count = len(filtered_data)
        
        # 1. Period filter: CVs typically have periods < 1 day, but we're generous with 5 days
        period_mask = filtered_data['true_period'] < self.period_limit
        period_count = period_mask.sum()
        
        # 2. Amplitude filter: CVs show significant variability
        amplitude_mask = filtered_data['true_amplitude'] > self.amplitude_limit
        amplitude_count = amplitude_mask.sum()
        
        # 3. FAP filter: Period should be reliable
        fap_mask = filtered_data['best_fap'] < self.fap_limit
        fap_count = fap_mask.sum()
        
        # Apply all filters
        combined_mask = period_mask & amplitude_mask & fap_mask
        
        filtered_data = filtered_data[combined_mask]
        
        # Report on filtering
        print(f"Initial count: {initial_count} sources")
        print(f"Period filter (< {self.period_limit} days): {period_count} sources passed")
        print(f"Amplitude filter (> {self.amplitude_limit} mag): {amplitude_count} sources passed")
        print(f"FAP filter (< {self.fap_limit}): {fap_count} sources passed")
        print(f"After all filters: {len(filtered_data)} candidate sources")
        
        self.filtered_data = filtered_data
        return True


    def load_known_cvs(self):
        """
        Load known CVs for classifier training with enhanced ID matching capability.
        """
        if self.known_cv_file is None:
            print("No known CV file provided. Skipping.")
            return None
        
        print(f"Loading known CVs from {self.known_cv_file}...")
        
        try:
            # Determine file type and load accordingly
            if self.known_cv_file.endswith('.fits'):
                with fits.open(self.known_cv_file) as hdul:
                    # Convert to pandas DataFrame
                    cv_data = Table(hdul[1].data).to_pandas()
            elif self.known_cv_file.endswith('.csv'):
                cv_data = pd.read_csv(self.known_cv_file)
            else:
                print(f"Unsupported file format: {self.known_cv_file}")
                return None
            
            print(f"Loaded {len(cv_data)} known CV sources")
            
            # Try to find ID column for matching
            id_columns = ['sourceid', 'primvs_id', 'id', 'source_id', 'ID']
            id_col = None
            
            for col in id_columns:
                if col in cv_data.columns:
                    id_col = col
                    print(f"Using '{id_col}' column for CV identification")
                    break
            
            if id_col is None:
                print("Could not find ID column in known CV file.")
                print("Available columns:", cv_data.columns.tolist())
                return None
            
            # Extract IDs as strings to ensure consistent matching
            known_ids = set(cv_data[id_col].astype(str))
            print(f"Extracted {len(known_ids)} unique CV IDs")
            
            return known_ids
                
        except Exception as e:
            print(f"Error loading known CVs: {str(e)}")
            import traceback
            traceback.print_exc()
            return None









        
    def select_candidates(self):
        """
        Process all data and append calculated CV classification scores and embeddings info,
        without filtering by confidence and without using PCA dimensionality reduction.
        All original data plus calculated values will be saved.
        """
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return False
        
        print("Processing data and calculating CV candidate scores...")

        # Keep all data, just add the probability as a score
        print("Adding classifier probabilities to all data")
        
        # Set candidate flag for all rows, no filtering
        self.filtered_data['is_cv_candidate'] = True
        
        # Save the classifier probability directly

        # Only select candidates with probability >= 0.5
        prob_threshold = 0.0
        high_prob_mask = self.filtered_data['cv_prob'] >= prob_threshold
        high_prob_count = high_prob_mask.sum()
        
        print(f"Found {high_prob_count} candidates with probability >= {prob_threshold}")
        
        # Filter to only high probability candidates to save processing time
        self.filtered_data['is_cv_candidate'] = high_prob_mask
        self.cv_candidates = self.filtered_data[high_prob_mask].copy()
        
        # Set confidence directly from classifier probability
        self.cv_candidates['confidence'] = self.cv_candidates['cv_prob']
                
        # Use all the filtered data instead of just high confidence candidates
        #self.cv_candidates = self.filtered_data.copy()
        
        # If embedding information is available, add embedding-based features too
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in self.cv_candidates.columns]
        
        if embedding_features and self.known_cv_file is not None and len(embedding_features) >= 10:
            print("Adding embedding proximity information without PCA...")
            
            # Load known CV IDs
            known_ids = self.load_known_cvs()
            
            if known_ids is not None and len(known_ids) > 0:
                # Determine ID column for matching
                id_columns = ['sourceid', 'primvs_id', 'source_id', 'id']
                id_col = None
                
                for col in id_columns:
                    if col in self.cv_candidates.columns:
                        id_col = col
                        break
                
                if id_col is not None:
                    # Flag known CVs in candidates
                    self.cv_candidates['is_known_cv'] = self.cv_candidates[id_col].astype(str).isin(known_ids)
                    
                    # Split known CVs and other data
                    known_cvs = self.cv_candidates[self.cv_candidates['is_known_cv']]
                    other_data = self.cv_candidates[~self.cv_candidates['is_known_cv']]
                    
                    if len(known_cvs) > 0 and len(other_data) > 0:
                        # Calculate embedding distances directly in 64-dimensional space
                        from scipy.spatial.distance import cdist
                        
                        known_embeddings = known_cvs[embedding_features].values
                        other_embeddings = other_data[embedding_features].values
                        
                        print(f"Calculating distances in {known_embeddings.shape[1]}-dimensional embedding space...")
                        
                        # Calculate minimum distance from each point to any known CV
                        distances = cdist(other_embeddings, known_embeddings, 'euclidean')
                        min_distances = np.min(distances, axis=1)
                        
                        # Add distance to nearest known CV
                        other_data['distance_to_nearest_cv'] = min_distances
                        
                        # Normalized distance (0-1 scale, 0 is closest)
                        max_dist = min_distances.max()
                        if max_dist > 0:
                            other_data['norm_distance'] = min_distances / max_dist
                        else:
                            other_data['norm_distance'] = 0
                        
                        # Compute embedding similarity score (inverse of normalized distance)
                        other_data['embedding_similarity'] = 1 - other_data['norm_distance']
                        
                        # Calculate blended score but preserve original probabilities
                        classifier_weight = 0.8
                        embedding_weight = 0.2
                        
                        if 'cv_confidence' in other_data.columns:
                            other_data['blended_score'] = (
                                classifier_weight * other_data['cv_confidence'] + 
                                embedding_weight * other_data['embedding_similarity']
                            )
                        else:
                            other_data['blended_score'] = other_data['embedding_similarity']
                        
                        # Update main candidates dataframe with these scores
                        self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'distance_to_nearest_cv'] = other_data['distance_to_nearest_cv'].values
                        self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'norm_distance'] = other_data['norm_distance'].values
                        self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'embedding_similarity'] = other_data['embedding_similarity'].values
                        
                        if 'blended_score' in other_data.columns:
                            self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'blended_score'] = other_data['blended_score'].values
                        
                        print(f"Added raw embedding proximity information to {len(other_data)} sources")
        
        print(f"Processed {len(self.cv_candidates)} sources with classification scores")
        return True


    def save_candidates(self):
        """Save the CV candidates to CSV and FITS files."""
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates to save.")
            return False
        
        print(f"Saving {len(self.cv_candidates)} CV candidates...")
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'cv_candidates.csv')
        self.cv_candidates.to_csv(csv_path, index=False)
        print(f"Saved candidates to CSV: {csv_path}")
        
        # Save to FITS
        fits_path = os.path.join(self.output_dir, 'cv_candidates.fits')
        table = Table.from_pandas(self.cv_candidates)
        table.write(fits_path, overwrite=True)
        print(f"Saved candidates to FITS: {fits_path}")

        return True




    def extract_features(self):
        """
        Extract and normalize features for CV detection, including enhanced CV-specific features
        and contrastive curves embeddings.
        """
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return None
        
        print("Extracting features for CV detection...")
        
        # Create a copy of the filtered data to add new features
        enhanced_data = self.filtered_data.copy()
        
        # Add period-related features that better capture CV behavior
        enhanced_data['period_hours'] = enhanced_data['true_period'] * 24.0  # Convert to hours
        enhanced_data['inverse_period'] = 1.0 / enhanced_data['true_period']  # Frequency
        enhanced_data['period_to_amp_ratio'] = enhanced_data['true_period'] / enhanced_data['true_amplitude']
        enhanced_data['log_period'] = np.log10(enhanced_data['true_period'])
        
        # Add CV-specific features
        
        # Period gap feature (2-3 hours gap is significant in CV evolution)
        period_hours = enhanced_data['true_period'] * 24.0
        enhanced_data['in_period_gap'] = ((period_hours >= 2) & (period_hours <= 3)).astype(int)
        
        # Flickering indicator (CVs show rapid, stochastic variability)
        if all(col in enhanced_data.columns for col in ['std_nxs', 'lag_auto']):
            enhanced_data['flickering_index'] = enhanced_data['std_nxs'] * (1 - enhanced_data['lag_auto'])
        
        # Outburst likelihood feature
        if all(col in enhanced_data.columns for col in ['skew', 'kurt']):
            enhanced_data['outburst_indicator'] = enhanced_data['skew'] * enhanced_data['kurt'] * enhanced_data['true_amplitude']
        
        # Add color-based features specifically for CVs if color data is available
        # CVs often show blue excess due to accretion disk
        color_cols = ['Z-K', 'Y-K', 'J-K', 'H-K']
        if all(col in enhanced_data.columns for col in color_cols):
            print("Adding color-based features...")
            enhanced_data['color_slope'] = (enhanced_data['Z-K'] - enhanced_data['J-K']) / 2.0  # Slope of SED
            enhanced_data['nir_excess'] = enhanced_data['H-K'] - 0.2 * enhanced_data['J-K']  # Near-IR excess parameter
        
        # Add variability shape features
        if 'Cody_M' in enhanced_data.columns and 'true_amplitude' in enhanced_data.columns:
            enhanced_data['asymmetry_amp'] = enhanced_data['Cody_M'] * enhanced_data['true_amplitude']
        
        # Update the filtered data with new features
        self.filtered_data = enhanced_data
        
        # Filter feature list to include only available columns
        # First, add our new features to the feature list
        new_features = [
            'period_hours', 'inverse_period', 'period_to_amp_ratio', 
            'in_period_gap', 'flickering_index', 'outburst_indicator',
            'color_slope', 'nir_excess', 'asymmetry_amp'
        ]
        extended_features = self.cv_features.copy()
        extended_features.extend([f for f in new_features if f in enhanced_data.columns])
        
        # Now get only available columns
        available_features = [f for f in extended_features if f in enhanced_data.columns]
        
        # Extract contrastive curves embeddings if available
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in enhanced_data.columns]
        
        if embedding_features:
            print(f"Found {len(embedding_features)} contrastive curves embedding features")
            available_features.extend(embedding_features)
        else:
            print("No contrastive curves embeddings found in the data")
        
        # Extract feature matrix
        X = enhanced_data[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in [np.float64, np.int64]:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
        
        # Scale features (but not embedding features which are already normalized)
        self.scaler = StandardScaler()
        cols_to_scale = [col for col in X.columns if col not in embedding_features]
        
        if cols_to_scale:
            X_scaled = X.copy()
            X_scaled[cols_to_scale] = self.scaler.fit_transform(X[cols_to_scale])
        else:
            X_scaled = X.copy()
        
        self.features = X
        self.scaled_features = X_scaled
        self.feature_names = available_features
        
        print(f"Extracted {len(available_features)} features for classification")
        for feat_category, feats in [
            ("Period-related", ['true_period', 'period_hours', 'inverse_period', 'log_period']),
            ("Amplitude-related", ['true_amplitude', 'period_to_amp_ratio']),
            ("Shape-related", ['Cody_M', 'skew', 'kurt', 'asymmetry_amp']),
            ("Color-related", ['color_slope', 'nir_excess']),
            ("CV-specific", ['in_period_gap', 'flickering_index', 'outburst_indicator'])
        ]:
            present_feats = [f for f in feats if f in available_features]
            if present_feats:
                print(f"  - {feat_category}: {', '.join(present_feats)}")
        
        return X_scaled




    def run_classifier(self):
        """
        Run the trained classifier on all filtered data to identify CV candidates.
        If no model is available, this will train one first.
        """
        if not hasattr(self, 'model') or self.model is None:
            print("No trained model available. Training now...")
            self.train_classifier()

        print("Running classifier on all filtered data...")
        
        # Get predictions and probabilities
        cv_probs = self.model.predict_proba(self.scaled_features)[:, 1]
        
        # Add to filtered data
        self.filtered_data['cv_prob'] = cv_probs
        
        # Plot probability distribution
        plt.figure(figsize=(10, 6))
        plt.hist(cv_probs, bins=50, alpha=0.7)
        plt.axvline(0.5, color='r', linestyle='--', label='Default threshold (0.5)')
        plt.axvline(0.8, color='g', linestyle='--', label='High confidence (0.8)')
        plt.xlabel('CV Probability')
        plt.ylabel('Number of Sources')
        plt.title('Distribution of CV Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cv_probability_distribution.png'), dpi=300)
        plt.close()
        
        return True




    def train_two_stage_classifier(self):
        """
        Streamlined two-stage classifier that combines traditional features and embeddings
        using weighted averaging - simple, robust, and effective.
        """
        import numpy as np
        import xgboost as xgb
        import time
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        import joblib
        import matplotlib.pyplot as plt
        import os
        
        print("\nTRAINING CV CLASSIFIER")
        start_time = time.time()
        
        # Setup data
        known_ids = self.load_known_cvs()
        candidate_ids = self.filtered_data['sourceid'].astype(str)
        y = candidate_ids.isin(known_ids).astype(int)
        print(f"Found {y.sum()} CVs in dataset of {len(y)} candidates")
        
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in self.scaled_features.columns if col in cc_embedding_cols]
        traditional_features = [col for col in self.scaled_features.columns if col not in embedding_features]
        
        X_trad = self.scaled_features[traditional_features].values
        X_emb = self.scaled_features[embedding_features].values
        
        # Train-test split
        X_indices = np.arange(len(X_trad))
        train_indices, val_indices = train_test_split(X_indices, test_size=0.25, random_state=42, stratify=y)
        
        X_trad_train, X_trad_val = X_trad[train_indices], X_trad[val_indices]
        y_train, y_val = y.iloc[train_indices].values, y.iloc[val_indices].values
        X_emb_train, X_emb_val = X_emb[train_indices], X_emb[val_indices]
        
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        # Train traditional model with best hyperparameters
        print("\nTraining traditional features model...")
        trad_model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=3, learning_rate=0.01, min_child_weight=1,
            subsample=0.7, colsample_bytree=0.7, gamma=0.1, objective='binary:logistic',
            scale_pos_weight=pos_weight, n_jobs=-1, random_state=42
        )
        trad_model.fit(X_trad_train, y_train)
        
        # Train embedding model with best hyperparameters
        print("\nTraining embedding features model...")
        emb_model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=7, learning_rate=0.05, min_child_weight=5,
            subsample=0.7, colsample_bytree=0.8, gamma=0.1, objective='binary:logistic',
            scale_pos_weight=pos_weight, n_jobs=-1, random_state=42
        )


        emb_model.fit(X_emb_train, y_train)
        
        # Get predictions on validation set
        trad_probs_val = trad_model.predict_proba(X_trad_val)[:, 1]
        emb_probs_val = emb_model.predict_proba(X_emb_val)[:, 1]
        
        # Evaluate base models
        trad_auc = roc_auc_score(y_val, trad_probs_val)
        emb_auc = roc_auc_score(y_val, emb_probs_val)
        print(f"\nTraditional model AUC: {trad_auc:.4f}")
        print(f"Embedding model AUC: {emb_auc:.4f}")
        
        # Find optimal weight for the ensemble
        best_weight = 0.5  # Default equal weighting
        if trad_auc > 0.5 and emb_auc > 0.5:  # Both models better than random
            # Try different weights to find the optimal blend
            weight_range = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
            best_auc = 0
            
            for weight in weight_range:
                ensemble_probs = weight * trad_probs_val + (1 - weight) * emb_probs_val
                ensemble_auc = roc_auc_score(y_val, ensemble_probs)
                
                if ensemble_auc > best_auc:
                    best_auc = ensemble_auc
                    best_weight = weight
            
            print(f"Ensemble AUC: {best_auc:.4f} (trad_weight={best_weight:.2f})")
        elif trad_auc > emb_auc:
            best_weight = 1.0  # Use only traditional model
            print(f"Using only traditional model (weight=1.0)")
        else:
            best_weight = 0.0  # Use only embedding model
            print(f"Using only embedding model (weight=0.0)")
        
        # Apply models to all data
        print("\nApplying models to all data...")
        trad_probs = trad_model.predict_proba(X_trad)[:, 1]

        # When applying models to all data for final predictions:
        trad_probs = trad_model.predict_proba(X_trad)[:, 1]
        # Use the same PCA model to transform the entire embedding dataset      
        emb_probs = emb_model.predict_proba(X_emb)[:, 1]

        # Now combine the probabilities as before:
        final_probs = best_weight * trad_probs + (1 - best_weight) * emb_probs
        
        # Show feature importance
        importance = trad_model.feature_importances_
        indices = np.argsort(importance)[::-1]
        print("\nTop traditional features:")
        for i in range(min(100, len(traditional_features))):
            print(f"  {i+1}. {traditional_features[indices[i]]}: {importance[indices[i]]:.4f}")
        
        # Store models and probabilities
        self.model_trad = trad_model
        self.model_emb = emb_model
        self.model = trad_model  # For compatibility
        
        self.filtered_data['cv_prob_trad'] = trad_probs
        self.filtered_data['cv_prob_emb'] = emb_probs
        self.filtered_data['cv_prob'] = final_probs
        
        # Print confidence levels
        sorted_probs = np.sort(final_probs)[::-1]
        if len(sorted_probs) >= 100:
            print(f"\nConfidence scores:")
            print(f"  Top candidate:    {sorted_probs[0]:.4f}")
            print(f"  10th candidate:   {sorted_probs[9]:.4f}")
            print(f"  50th candidate:   {sorted_probs[49]:.4f}")
            print(f"  100th candidate:  {sorted_probs[99]:.4f}")
            if len(sorted_probs) >= 200:
                print(f"  200th candidate:  {sorted_probs[199]:.4f}")
        
        # Save models
        os.makedirs(self.output_dir, exist_ok=True)
        joblib.dump(trad_model, os.path.join(self.output_dir, 'cv_classifier_traditional.joblib'))
        joblib.dump(emb_model, os.path.join(self.output_dir, 'cv_classifier_embedding.joblib'))
        
        # Create visualization of model predictions
        plt.figure(figsize=(10, 8))
        plt.scatter(trad_probs, emb_probs, c=final_probs, cmap='viridis', alpha=0.6, s=5)
        plt.colorbar(label='Final Probability')
        plt.xlabel('Traditional Model Probability')
        plt.ylabel('Embedding Model Probability')
        plt.grid(True, alpha=0.3)
        plt.title(f'Model Predictions (trad_weight={best_weight:.2f})')
        plt.savefig(os.path.join(self.output_dir, 'model_prediction_comparison.png'), dpi=300)
        plt.close()
        
        print(f"\nClassifier training completed in {time.time() - start_time:.2f} seconds")
        
        return True









    def plot_roc_curves(self):
        """
        Create ROC curves comparing the traditional, embedding, and ensemble models.
        
        This function evaluates all three models on the validation set and generates
        ROC curves with AUC scores to visualize their relative performance.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        from sklearn.model_selection import train_test_split
        import os
        
        print("Generating ROC curve comparison...")
        
        # Check if we have the models
        if not hasattr(self, 'model_trad') or self.model_trad is None:
            print("Error: Traditional model not available. Train the classifier first.")
            return False
            
        if not hasattr(self, 'model_emb') or self.model_emb is None:
            print("Error: Embedding model not available. Train the classifier first.")
            return False
        
        # Prepare the data
        known_ids = self.load_known_cvs()
        if known_ids is None or len(known_ids) == 0:
            print("Error: No known CVs available for evaluation.")
            return False
        
        # Determine ID column for matching
        id_columns = ['sourceid', 'primvs_id', 'source_id', 'id']
        id_col = None
        for col in id_columns:
            if col in self.filtered_data.columns:
                id_col = col
                break
        
        if id_col is None:
            print("Error: No suitable identifier column found in data.")
            return False
        
        # Create binary classification labels
        candidate_ids = self.filtered_data[id_col].astype(str)
        y = candidate_ids.isin(known_ids).astype(int)
        
        # Split features into traditional and embedding sets
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in self.scaled_features.columns]
        traditional_features = [col for col in self.scaled_features.columns if col not in embedding_features]
        
        # Extract feature sets
        X_trad = self.scaled_features[traditional_features].values
        X_emb = self.scaled_features[embedding_features].values if embedding_features else None
        
        # Split data for validation
        X_indices = np.arange(len(X_trad))
        train_indices, val_indices = train_test_split(
            X_indices, test_size=0.25, random_state=42, stratify=y
        )
        
        X_trad_val = X_trad[val_indices]
        y_val = y.iloc[val_indices].values
        
        # Get predictions from traditional model
        y_score_trad = self.model_trad.predict_proba(X_trad_val)[:, 1]
        
        # Get predictions from embedding model if available
        if X_emb is not None and hasattr(self, 'model_emb') and self.model_emb is not None:
            X_emb_val = X_emb[val_indices]
            
            # Apply PCA transformation if necessary
            if hasattr(self, 'pca') and self.pca is not None:
                X_emb_val_pca = self.pca.transform(X_emb_val)
                y_score_emb = self.model_emb.predict_proba(X_emb_val_pca)[:, 1]
            else:
                y_score_emb = self.model_emb.predict_proba(X_emb_val)[:, 1]
        else:
            y_score_emb = np.zeros_like(y_score_trad)
        
        # Get ensemble predictions
        meta_features_val = np.column_stack([y_score_trad, y_score_emb])
        y_score_ensemble = self.model_meta.predict_proba(meta_features_val)[:, 1] if hasattr(self, 'model_meta') else None
        
        # If we don't have a meta model, try to determine the ensemble weights from CV probabilities
        if y_score_ensemble is None and 'cv_prob_trad' in self.filtered_data and 'cv_prob_emb' in self.filtered_data:
            # Estimate weights by looking at the full dataset
            trad_probs = self.filtered_data['cv_prob_trad'].values
            emb_probs = self.filtered_data['cv_prob_emb'].values
            cv_probs = self.filtered_data['cv_prob'].values
            
            # Simple linear regression to estimate weights
            from sklearn.linear_model import LinearRegression
            X_weights = np.column_stack([trad_probs, emb_probs])
            reg = LinearRegression(fit_intercept=False).fit(X_weights, cv_probs)
            
            # Apply estimated weights to validation set
            trad_weight = reg.coef_[0]
            emb_weight = reg.coef_[1]
            y_score_ensemble = trad_weight * y_score_trad + emb_weight * y_score_emb
            
            print(f"Estimated ensemble weights: Traditional {trad_weight:.2f}, Embedding {emb_weight:.2f}")
        
        # If still no ensemble, use simple average
        if y_score_ensemble is None:
            y_score_ensemble = 0.5 * y_score_trad + 0.5 * y_score_emb
            print("Using simple average for ensemble model (0.5/0.5)")
        
        # Calculate ROC curve and AUC for each model
        fpr_trad, tpr_trad, _ = roc_curve(y_val, y_score_trad)
        roc_auc_trad = auc(fpr_trad, tpr_trad)
        
        fpr_emb, tpr_emb, _ = roc_curve(y_val, y_score_emb)
        roc_auc_emb = auc(fpr_emb, tpr_emb)
        
        fpr_ens, tpr_ens, _ = roc_curve(y_val, y_score_ensemble)
        roc_auc_ens = auc(fpr_ens, tpr_ens)
        
        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        
        # Plot Traditional model
        plt.plot(
            fpr_trad, tpr_trad,
            lw=2, label=f'Traditional Model (AUC = {roc_auc_trad:.3f})',
            color='blue'
        )
        
        # Plot Embedding model
        plt.plot(
            fpr_emb, tpr_emb,
            lw=2, label=f'Embedding Model (AUC = {roc_auc_emb:.3f})',
            color='green'
        )
        
        # Plot Ensemble model
        plt.plot(
            fpr_ens, tpr_ens,
            lw=2, label=f'Ensemble Model (AUC = {roc_auc_ens:.3f})',
            color='red'
        )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: CV Classification Models Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, 'roc_curves_comparison.png'), dpi=300)
        plt.close()
        
        print(f"ROC curve comparison saved to {os.path.join(self.output_dir, 'roc_curves_comparison.png')}")
        
        # Also create a zoomed-in view of the top-left corner
        plt.figure(figsize=(10, 8))
        
        # Plot Traditional model
        plt.plot(
            fpr_trad, tpr_trad,
            lw=2, label=f'Traditional Model (AUC = {roc_auc_trad:.3f})',
            color='blue'
        )
        
        # Plot Embedding model
        plt.plot(
            fpr_emb, tpr_emb,
            lw=2, label=f'Embedding Model (AUC = {roc_auc_emb:.3f})',
            color='green'
        )
        
        # Plot Ensemble model
        plt.plot(
            fpr_ens, tpr_ens,
            lw=2, label=f'Ensemble Model (AUC = {roc_auc_ens:.3f})',
            color='red'
        )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set plot properties for zoomed view
        plt.xlim([0.0, 0.3])
        plt.ylim([0.7, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: CV Models Comparison (Zoomed)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save the zoomed plot
        plt.savefig(os.path.join(self.output_dir, 'roc_curves_comparison_zoomed.png'), dpi=300)
        plt.close()
        
        print(f"Zoomed ROC curve comparison saved to {os.path.join(self.output_dir, 'roc_curves_comparison_zoomed.png')}")
        
        return True






    def run_pipeline(self):
        """Run the complete CV finder pipeline with two-stage classification."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("RUNNING PRIMVS CV FINDER PIPELINE WITH TWO-STAGE CLASSIFICATION")
        print("="*80 + "\n")
        
        # Step 1: Load PRIMVS data
        self.load_primvs_data()
        
        # Step 2: Apply initial filters
        self.apply_initial_filters()

        # Step 3: Extract features for classification
        self.extract_features()

        self.train_two_stage_classifier()

        self.select_candidates()
        
        self.save_candidates()
        
        self.post_processing_plots()
        self.analyze_period_gap_distribution()
        self.analyze_galactic_regions()
        self.plot_roc_curves()

        end_time = time.time()
        runtime = end_time - start_time
        
        print("\n" + "="*80)
        print(f"PIPELINE COMPLETED in {timedelta(seconds=int(runtime))}")
        print(f"Found {len(self.cv_candidates)} CV candidates")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")
        
        return True





def main():
    """Main function to run the CV finder."""
    # Fixed paths to match your description
    #primvs_file = '../PRIMVS/PRIMVS_CC_CV_cand.fits'
    primvs_file = '../PRIMVS/PRIMVS_CC.fits'    
    output_dir = "../PRIMVS/cv_results"
    known_cvs = "../PRIMVS/PRIMVS_CC_CV.fits"
    period_limit = 10.0  # Very generous upper limit (days)
    amplitude_limit = 0.003  # Very low amplitude threshold (mag)
    fap_limit = 1.0  # Permissive FAP threshold

    print(f"Starting CV finder with parameters:")
    print(f"  - PRIMVS file: {primvs_file}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Known CVs file: {known_cvs}")
    print(f"  - Period limit: {period_limit} days")
    print(f"  - Amplitude limit: {amplitude_limit} mag")
    print(f"  - FAP limit: {fap_limit}")
    
    # Create CV finder
    finder = PrimvsCVFinder(
        primvs_file=primvs_file,
        output_dir=output_dir,
        known_cv_file=known_cvs,
        period_limit=period_limit,
        amplitude_limit=amplitude_limit,
        fap_limit=fap_limit
    )
    
    # Run pipeline
    success = finder.run_pipeline()
    
    if success:
        print(f"CV finder completed successfully. Results in {output_dir}")
        return 0
    else:
        print("CV finder failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
















































