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
        if 'true_period' in df.columns and 'true_amplitude' in df.columns:
            # Convert period to hours & take log
            df['period_hours'] = df['true_period'] * 24.0
            df['log_period'] = np.log10(df['period_hours'])
            
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
        if 'l' in df.columns and 'b' in df.columns:
            # Because TESS overlay uses -180..+180, shift your data similarly
            df['l_centered'] = np.where(df['l'] > 180, df['l'] - 360, df['l'])
            
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


    def analyze_metallicity_effects(self):
        """Analyze CV distribution along bulge metallicity gradient"""
        
        if 'l' not in self.cv_candidates.columns or 'b' not in self.cv_candidates.columns:
            return
        
        # Define metallicity regions based on Galactic position (approximate)
        self.cv_candidates['region'] = 'Other'
        
        # Central bulge (metal-rich)
        central_mask = (self.cv_candidates['l'] > -5) & (self.cv_candidates['l'] < 5) & \
                       (self.cv_candidates['b'] > -5) & (self.cv_candidates['b'] < 5)
        self.cv_candidates.loc[central_mask, 'region'] = 'Central_Bulge'
        self.cv_candidates.loc[central_mask, 'metallicity'] = 0.3  # [Fe/H] ~ +0.3
        
        # Inner bulge
        inner_mask = (self.cv_candidates['l'] > -10) & (self.cv_candidates['l'] < 10) & \
                     (self.cv_candidates['b'] > -10) & (self.cv_candidates['b'] < 10) & \
                     ~central_mask
        self.cv_candidates.loc[inner_mask, 'region'] = 'Inner_Bulge'
        self.cv_candidates.loc[inner_mask, 'metallicity'] = 0.1  # [Fe/H] ~ +0.1
        
        # Outer bulge
        outer_mask = (self.cv_candidates['l'] > -20) & (self.cv_candidates['l'] < 20) & \
                     (self.cv_candidates['b'] > -15) & (self.cv_candidates['b'] < 15) & \
                     ~central_mask & ~inner_mask
        self.cv_candidates.loc[outer_mask, 'region'] = 'Outer_Bulge'
        self.cv_candidates.loc[outer_mask, 'metallicity'] = -0.1  # [Fe/H] ~ -0.1
        
        # Bulge-halo interface
        interface_mask = ~central_mask & ~inner_mask & ~outer_mask
        self.cv_candidates.loc[interface_mask, 'region'] = 'Bulge_Halo_Interface'
        self.cv_candidates.loc[interface_mask, 'metallicity'] = -0.4  # [Fe/H] ~ -0.4
        
        # Analyze period gap by metallicity
        plt.figure(figsize=(12, 8))
        
        for i, region in enumerate(['Central_Bulge', 'Inner_Bulge', 'Outer_Bulge', 'Bulge_Halo_Interface']):
            mask = self.cv_candidates['region'] == region
            if mask.sum() < 5:
                continue
                
            period_hours = self.cv_candidates.loc[mask, 'true_period'] * 24.0
            plt.subplot(2, 2, i+1)
            plt.hist(np.log10(period_hours), bins=20, alpha=0.7)
            plt.axvspan(np.log10(2), np.log10(3), alpha=0.2, color='red', label='Period Gap')
            plt.title(f"{region} ([Fe/H]={self.cv_candidates.loc[mask, 'metallicity'].iloc[0]})")
            plt.xlabel('log(Period [hours])')
            plt.ylabel('Count')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metallicity_period_analysis.png'), dpi=300)



















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
        prob_threshold = 0.7
        high_prob_mask = self.filtered_data['cv_prob'] >= prob_threshold
        high_prob_count = high_prob_mask.sum()
        
        print(f"Found {high_prob_count} candidates with probability >= {prob_threshold}")
        
        # Filter to only high probability candidates to save processing time
        self.filtered_data['is_cv_candidate'] = high_prob_mask
        self.cv_candidates = self.filtered_data[high_prob_mask].copy()
        
        # Set confidence directly from classifier probability
        self.cv_candidates['confidence'] = self.cv_candidates['cv_prob']
                
        # Use all the filtered data instead of just high confidence candidates
        self.cv_candidates = self.filtered_data.copy()
        
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
        Train a two-stage classifier combining traditional feature-based models with
        embedding-based models for optimal CV candidate selection.
        
        This approach maintains interpretability while leveraging the representational
        power of contrastive curve embeddings through an ensemble methodology.
        """
        import logging
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import classification_report
        from sklearn.decomposition import PCA
        import joblib
        import matplotlib.pyplot as plt
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # --------------------------------------------------------------------------------
        # 1) Check for features
        # --------------------------------------------------------------------------------
        if not hasattr(self, 'scaled_features') or len(self.scaled_features) == 0:
            logger.error("No features available. Call extract_features() first.")
            return False
        
        print("Training two-stage CV classifier...")

        # --------------------------------------------------------------------------------
        # 2) Load known CV IDs
        # --------------------------------------------------------------------------------
        known_ids = self.load_known_cvs()
        if known_ids is None or len(known_ids) == 0:
            logger.error("No known CVs available for training. Classifier aborted.")
            return False
        
        # --------------------------------------------------------------------------------
        # 3) Identify ID column and create binary labels
        # --------------------------------------------------------------------------------
        id_columns = ['sourceid', 'primvs_id', 'source_id', 'id']
        id_col = None
        for col in id_columns:
            if col in self.filtered_data.columns:
                id_col = col
                break
        if id_col is None:
            logger.error("No suitable identifier column found in candidate data.")
            return False
        
        print(f"Using identifier column '{id_col}' for matching")
        candidate_ids = self.filtered_data[id_col].astype(str)
        
        y = candidate_ids.isin(known_ids).astype(int)
        positive_count = y.sum()
        print(f"Found {positive_count} matches between known CVs and candidate sources")
        
        if positive_count < 10:
            logger.warning(f"Very few positive examples ({positive_count}). Classification may be unreliable.")
            if positive_count < 3:
                logger.error("Insufficient positive examples for training. Aborting classifier.")
                return False

        # --------------------------------------------------------------------------------
        # 4) Separate Traditional and Embedding features
        # --------------------------------------------------------------------------------
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in self.scaled_features.columns if col in cc_embedding_cols]
        traditional_features = [col for col in self.scaled_features.columns if col not in embedding_features]
        
        print(f"Traditional features: {len(traditional_features)}")
        print(f"Embedding features: {len(embedding_features)}")

        X_trad = self.scaled_features[traditional_features].values
        X_emb = self.scaled_features[embedding_features].values if embedding_features else None

        # --------------------------------------------------------------------------------
        # 5) Train-test split
        # --------------------------------------------------------------------------------
        X_indices = np.arange(len(X_trad))
        train_indices, val_indices = train_test_split(
            X_indices, test_size=0.25, random_state=42, stratify=y
        )
        X_trad_train, X_trad_val = X_trad[train_indices], X_trad[val_indices]
        y_train, y_val = y.iloc[train_indices].values, y.iloc[val_indices].values
        
        if X_emb is not None:
            X_emb_train, X_emb_val = X_emb[train_indices], X_emb[val_indices]

        # Handle class imbalance with XGBoost's built-in scale_pos_weight
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0

        # --------------------------------------------------------------------------------
        # 6) Stage 1: Traditional Model Tuning
        # --------------------------------------------------------------------------------
        print("Tuning traditional feature model...")
        param_grid_trad = {
            'n_estimators': [10],#, 100, 500],
            'max_depth': [5],#, 10, 100, 200],
            'learning_rate': [0.1],#, 0.05, 0.01, 0.001],
            'min_child_weight': [1],#, 3, 5],  # Helps with imbalanced data
            'gamma': [0],#, 0.1, 0.2],  # Minimum loss reduction
            'subsample': [0.8],#, 0.9, 1.0],  # Prevents overfitting
        }


        xgb_trad = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='auc'
        )
        grid_trad = GridSearchCV(
            xgb_trad,
            param_grid_trad,
            scoring='roc_auc',
            cv=4,
            verbose=1
        )
        grid_trad.fit(X_trad_train, y_train)
        best_trad = grid_trad.best_estimator_
        print(f"Best traditional model params: {grid_trad.best_params_}")

        # Evaluate on validation
        y_pred_trad = best_trad.predict(X_trad_val)
        prob_trad = best_trad.predict_proba(X_trad_val)[:, 1]
        print("\nTraditional Feature Model Performance:")
        print(classification_report(y_val, y_pred_trad))

        # Optional: feature importance
        trad_importance = pd.DataFrame({
            'Feature': traditional_features,
            'Importance': best_trad.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nTop 5 traditional features:")
        imp_sum = 0
        for i, (_, row) in enumerate(trad_importance.head(10).iterrows()):
            print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
            imp_sum = imp_sum + row['Importance']

        print('Top 10 features account for :', imp_sum)  



        # --------------------------------------------------------------------------------
        # 7) Stage 2: Embedding Model Tuning (with PCA)
        # --------------------------------------------------------------------------------
        if X_emb is not None and len(embedding_features) > 0:
            print("Reducing embedding dimensionality via PCA to capture ~90% variance...")
            pca_full = PCA().fit(X_emb_train)
            explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = min(np.argmax(explained_variance >= 0.9) + 1, len(explained_variance))
            print(f"Using {n_components} PCA components (explaining {explained_variance[n_components-1]:.2%} variance)")

            pca = PCA(n_components=n_components)
            X_emb_train_pca = X_emb_train#pca.fit_transform(X_emb_train)
            X_emb_val_pca = X_emb_val#pca.transform(X_emb_val)

            print("Tuning embedding feature model...")
            param_grid_emb = {
                'n_estimators': [10],#, 200, 500],
                'max_depth': [3],#, 10, 100, 200],
                'learning_rate': [0.1],#, 0.05, 0.01, 0.001],
            }
            xgb_emb = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=pos_weight,
                n_jobs=-1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='auc'
            )
            grid_emb = GridSearchCV(
                xgb_emb,
                param_grid_emb,
                scoring='roc_auc',
                cv=3,
                verbose=1
            )
            grid_emb.fit(X_emb_train_pca, y_train)
            best_emb = grid_emb.best_estimator_
            print(f"Best embedding model params: {grid_emb.best_params_}")

            y_pred_emb = best_emb.predict(X_emb_val_pca)
            prob_emb = best_emb.predict_proba(X_emb_val_pca)[:, 1]
            print("\nEmbedding Feature Model Performance:")
            print(classification_report(y_val, y_pred_emb))

        else:
            # If no embeddings, default to zeros
            pca = None
            best_emb = None
            prob_emb = np.zeros_like(prob_trad)

        # --------------------------------------------------------------------------------
        # 8) Stage 3: Meta-Model (Stacking)
        # --------------------------------------------------------------------------------
        print("Training meta-model to blend predictions...")

        # Build meta-features from training sets
        trad_probs_train = best_trad.predict_proba(X_trad_train)[:, 1]
        if best_emb is not None:
            X_emb_train_pca_full = X_emb_train#pca.transform(X_emb_train)
            emb_probs_train = best_emb.predict_proba(X_emb_train_pca_full)[:, 1]
        else:
            emb_probs_train = np.zeros_like(trad_probs_train)

        meta_features_train = np.column_stack([trad_probs_train, emb_probs_train])
        meta_features_val = np.column_stack([prob_trad, prob_emb])

        meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.1,
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=42,
            eval_metric='auc'
        )
        meta_model.fit(meta_features_train, y_train)

        # Evaluate meta-model
        y_pred_meta = meta_model.predict(meta_features_val)
        print("\nMeta-Model Performance:")
        print(classification_report(y_val, y_pred_meta))

        # Check how it's weighting the two base models
        blend_weights = meta_model.feature_importances_
        print(f"\nModel blend weights: Traditional {blend_weights[0]:.2f}, Embedding {blend_weights[1]:.2f}")

        # --------------------------------------------------------------------------------
        # 9) Apply to Full Dataset
        # --------------------------------------------------------------------------------
        print("Applying two-stage classifier to all data...")

        # Get probabilities from both base models on the entire dataset
        trad_probs = best_trad.predict_proba(X_trad)[:, 1]
        if best_emb is not None:
            X_emb_full_pca = X_emb#pca.transform(X_emb)
            emb_probs = best_emb.predict_proba(X_emb_full_pca)[:, 1]
        else:
            emb_probs = np.zeros_like(trad_probs)

        # Combine using the meta-model
        meta_features_full = np.column_stack([trad_probs, emb_probs])
        final_probs = meta_model.predict_proba(meta_features_full)[:, 1]

        # Store references
        self.model_trad = best_trad
        self.model_emb = best_emb
        self.model_meta = meta_model
        self.pca = pca

        # Add predictions to self.filtered_data
        self.filtered_data['cv_prob_trad'] = trad_probs
        self.filtered_data['cv_prob_emb'] = emb_probs
        self.filtered_data['cv_prob'] = final_probs

        # --------------------------------------------------------------------------------
        # 10) Save Models
        # --------------------------------------------------------------------------------
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        joblib.dump(best_trad, os.path.join(self.output_dir, 'cv_classifier_traditional.joblib'))
        if best_emb is not None:
            joblib.dump(best_emb, os.path.join(self.output_dir, 'cv_classifier_embedding.joblib'))
            #joblib.dump(pca, os.path.join(self.output_dir, 'embedding_pca.joblib'))
        joblib.dump(meta_model, os.path.join(self.output_dir, 'cv_classifier_meta.joblib'))

        # --------------------------------------------------------------------------------
        # 11) Plot Probability Distribution
        # --------------------------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        plt.hist(final_probs, bins=50, alpha=0.7)
        plt.axvline(0.5, color='r', linestyle='--', label='Default threshold (0.5)')
        plt.axvline(0.8, color='g', linestyle='--', label='High confidence (0.8)')
        plt.xlabel('CV Probability')
        plt.ylabel('Number of Sources')
        plt.title('Distribution of CV Probabilities from Two-Stage Classifier')
        plt.yscale('log')  # Log scale for better visibility
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cv_two_stage_probability_distribution.png'), dpi=300)
        plt.close()

        print("Two-stage classifier training complete.")
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
        self.analyze_metallicity_effects()

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
    primvs_file = '../PRIMVS/PRIMVS_CC_CV_cand.fits'
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
















































