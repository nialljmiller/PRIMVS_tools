#!/usr/bin/env python
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Catalogs, Observations
from matplotlib.patches import Polygon
from matplotlib import patheffects
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import umap
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# TESSCycle8Overlay class (for plotting footprints)
# -------------------------
class TESSCycle8Overlay:
    """TESS Cycle 8 camera footprint visualization for CV candidates"""
    
    def __init__(self):
        # TESS Year 8 camera positions (RA, Dec, Roll) in degrees
        self.camera_positions = {
            97: [(24.13, -9.32, 292.44), (34.60, -31.26, 296.14), (51.45, -51.83, 127.55), (90.00, -66.56, 161.25)],
            98: [(76.26, 4.75, 275.70), (78.71, -19.13, 276.02), (82.05, -42.96, 97.78), (90.00, -66.56, 104.42)],
            99: [(136.70, -10.03, 296.66), (149.01, -31.13, 301.08), (168.24, -50.40, 133.88), (206.58, -62.97, 166.47)],
            100: [(160.77, -19.40, 290.79), (171.88, -41.46, 296.53), (194.15, -61.36, 134.29), (251.44, -70.27, 187.40)],
            101: [(184.53, -29.77, 289.32), (197.12, -51.89, 297.73), (231.12, -70.22, 148.05), (303.95, -68.82, 217.35)],
            102: [(210.05, -39.39, 292.95), (228.76, -60.37, 307.55), (284.10, -72.46, 179.13), (340.57, -60.80, 231.85)],
            103: [(239.66, -46.44, 302.50), (269.50, -63.95, 327.47), (328.67, -66.46, 202.01), (5.45, -50.96, 234.00)],
            104: [(273.80, -48.76, 317.02), (310.89, -60.43, 347.75), (357.50, -56.84, 208.16), (26.00, -41.28, 230.08)],
            105: [(308.41, -45.00, 331.21), (343.49, -51.67, 357.83), (19.73, -46.57, 205.65), (45.61, -32.84, 222.47)],
            106: [(338.92, -36.33, 339.59), (9.24, -40.96, 358.81), (39.96, -37.24, 198.48), (65.56, -26.76, 212.26)],
            107: [(5.23, -25.55, 341.34), (31.87, -30.75, 354.10), (59.79, -30.20, 188.47), (85.93, -24.07, 200.58)]
        }
        self.galactic_positions = self._convert_to_galactic()
    
    def _convert_to_galactic(self):
        """Convert equatorial to galactic coordinates"""
        galactic_positions = {}
        for sector, cameras in self.camera_positions.items():
            galactic_positions[sector] = []
            for ra, dec, roll in cameras:
                coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                gal = coords.galactic
                l_deg = gal.l.degree
                if l_deg > 180:
                    l_deg -= 360
                galactic_positions[sector].append((l_deg, gal.b.degree, roll))
        return galactic_positions
    
    def add_to_plot(self, ax, focus_region=None, alpha=0.2):
        """
        Add TESS Cycle 8 camera footprints to a Galactic plot.
        """
        camera_colors = ['red', 'purple', 'blue', 'green']
        if focus_region == 'bulge':
            sectors = [103, 104, 105]
            camera_indices = [0, 1]
        elif focus_region == 'disk':
            sectors = [102, 103]
            camera_indices = [1, 2]
        else:
            sectors = list(self.galactic_positions.keys())
            camera_indices = [0, 1, 2, 3]
        
        for sector in sectors:
            for i in camera_indices:
                if i >= len(self.galactic_positions[sector]):
                    continue
                l, b, roll = self.galactic_positions[sector][i]
                self._add_camera_footprint(ax, l, b, roll, 
                                           color=camera_colors[i],
                                           alpha=alpha,
                                           label=f"Camera {i+1}" if f"Camera {i+1}" not in [child.get_label() for child in ax.get_children() if hasattr(child, 'get_label')] else "",
                                           sector=sector)
        # Add legend for cameras
        handles = []
        labels = []
        for i, color in enumerate(camera_colors):
            if i in camera_indices:
                handles.append(plt.Line2D([], [], color=color, marker='s', 
                                          linestyle='None', markersize=10, alpha=0.6))
                labels.append(f'Camera {i+1}')
        if handles and labels:
            ax.legend(handles, labels, loc='upper right', title="TESS Cycle 8")
        return ax
    
    def _add_camera_footprint(self, ax, l, b, roll, color='red', alpha=0.2, label="", sector=None, size=12):
        """Add a single camera footprint on a Galactic plot"""
        vertices_l = np.array([-size, size, size, -size, -size])
        vertices_b = np.array([-size, -size, size, size, -size])
        roll_rad = np.radians(roll-90)
        rotated_l = vertices_l * np.cos(roll_rad) - vertices_b * np.sin(roll_rad)
        rotated_b = vertices_l * np.sin(roll_rad) + vertices_b * np.cos(roll_rad)
        vertices_l = l + rotated_l
        vertices_b = b + rotated_b
        polygon = Polygon(np.column_stack([vertices_l, vertices_b]),
                          alpha=alpha, color=color, closed=True, label=label)
        ax.add_patch(polygon)
        if sector:
            text = ax.text(l, b, str(sector), fontsize=8, ha='center', va='center',
                           color='white', fontweight='bold')
            text.set_path_effects([
                patheffects.Stroke(linewidth=2, foreground='black'),
                patheffects.Normal()
            ])

# -------------------------
# Helper function to overlay TESS footprints in Equatorial coordinates
# -------------------------
def add_tess_overlay_equatorial(ax, alpha=0.2, size=12):
    camera_colors = ['red', 'purple', 'blue', 'green']
    from matplotlib.patches import Polygon
    tess = TESSCycle8Overlay()
    for sector, cameras in tess.camera_positions.items():
        for i, (ra, dec, roll) in enumerate(cameras):
            vertices_ra = np.array([-size, size, size, -size, -size])
            vertices_dec = np.array([-size, -size, size, size, -size])
            roll_rad = np.radians(roll-90)
            rotated_ra = vertices_ra * np.cos(roll_rad) - vertices_dec * np.sin(roll_rad)
            rotated_dec = vertices_ra * np.sin(roll_rad) + vertices_dec * np.cos(roll_rad)
            vertices_ra = ra + rotated_ra
            vertices_dec = dec + rotated_dec
            polygon = Polygon(np.column_stack([vertices_ra, vertices_dec]),
                              alpha=alpha, color=camera_colors[i % len(camera_colors)],
                              closed=True)
            ax.add_patch(polygon)
            ax.text(ra, dec, str(sector), fontsize=8, ha='center', va='center',
                    color='white', fontweight='bold',
                    path_effects=[patheffects.Stroke(linewidth=2, foreground='black'),
                                  patheffects.Normal()])
    handles = [plt.Line2D([], [], color=color, marker='s', linestyle='None', markersize=10, alpha=0.6) for color in camera_colors]
    labels = [f'Camera {i+1}' for i in range(len(camera_colors))]
    ax.legend(handles, labels, loc='upper right', title="TESS Cycle 8")
    return ax

# -------------------------
# PrimvsTessCrossMatch Pipeline Class
# -------------------------
class PrimvsTessCrossMatch:
    """
    Pipeline for cross-matching PRIMVS CV candidates with TESS data and generating a target list.
    Cycle 8–specific logic has been removed.
    """
    def __init__(self, cv_candidates_file, output_dir='./tess_crossmatch', 
                 search_radius=3.0, cv_prob_threshold=0.5, tess_mag_limit=16.0):
        self.cv_candidates_file = cv_candidates_file
        self.output_dir = output_dir
        self.search_radius = search_radius
        self.cv_prob_threshold = cv_prob_threshold
        self.tess_mag_limit = tess_mag_limit
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.cv_candidates = None
        self.filtered_candidates = None
        self.crossmatch_results = None
        self.target_list = None  # final 100 targets
    
    def load_cv_candidates(self):
        try:
            if self.cv_candidates_file.endswith('.fits'):
                candidates = Table.read(self.cv_candidates_file).to_pandas()
            elif self.cv_candidates_file.endswith('.csv'):
                candidates = pd.read_csv(self.cv_candidates_file)
            else:
                raise ValueError(f"Unsupported file format: {self.cv_candidates_file}")
            
            self.cv_candidates = candidates
            print(f"Loaded {len(candidates)} total CV candidates")
            
            if 'cv_prob' in candidates.columns:
                self.filtered_candidates = candidates[candidates['cv_prob'] >= self.cv_prob_threshold].copy()
                print(f"Filtered to {len(self.filtered_candidates)} candidates with cv_prob >= {self.cv_prob_threshold}")
            elif 'confidence' in candidates.columns:
                self.filtered_candidates = candidates[candidates['confidence'] >= self.cv_prob_threshold].copy()
                print(f"Filtered to {len(self.filtered_candidates)} candidates with confidence >= {self.cv_prob_threshold}")
            elif 'probability' in candidates.columns:
                self.filtered_candidates = candidates[candidates['probability'] >= self.cv_prob_threshold].copy()
                print(f"Filtered to {len(self.filtered_candidates)} candidates with probability >= {self.cv_prob_threshold}")
            else:
                print("Warning: No CV probability column found. Using all candidates.")
                self.filtered_candidates = candidates.copy()
            
            filtered_path = os.path.join(self.output_dir, 'filtered_candidates.csv')
            self.filtered_candidates.to_csv(filtered_path, index=False)
            print(f"Saved filtered candidates to: {filtered_path}")
            return True
        except Exception as e:
            print(f"Error loading CV candidates: {e}")
            return False
    
    def perform_crossmatch(self):
        if self.filtered_candidates is None:
            print("No filtered candidates available. Call load_cv_candidates() first.")
            return False
        
        print("Cross-matching with TESS Input Catalog...")
        if 'ra' in self.filtered_candidates.columns and 'dec' in self.filtered_candidates.columns:
            ra_col, dec_col = 'ra', 'dec'
        elif 'RAJ2000' in self.filtered_candidates.columns and 'DEJ2000' in self.filtered_candidates.columns:
            ra_col, dec_col = 'RAJ2000', 'DEJ2000'
        else:
            print("Error: Could not find RA/Dec columns in the candidates file")
            return False
        
        coords = SkyCoord(ra=self.filtered_candidates[ra_col].values * u.degree, 
                          dec=self.filtered_candidates[dec_col].values * u.degree)
        
        results = self.filtered_candidates.copy()
        results['in_tic'] = False
        results['tic_id'] = np.nan
        results['tic_tmag'] = np.nan
        results['separation_arcsec'] = np.nan
        results['has_tess_data'] = False
        results['tess_sectors'] = None
        results['not_in_tic_reason'] = None
        
        print(f"Cross-matching {len(coords)} filtered candidates...")
        for i, (idx, coord) in enumerate(tqdm(zip(results.index, coords), total=len(coords))):
            try:
                tic_result = Catalogs.query_region(coord, radius=self.search_radius * u.arcsec, catalog="TIC")
                if len(tic_result) > 0:
                    tic_result.sort('dstArcSec')
                    best_match = tic_result[0]
                    results.loc[idx, 'in_tic'] = True
                    results.loc[idx, 'tic_id'] = int(best_match['ID'])
                    results.loc[idx, 'tic_tmag'] = best_match['Tmag']
                    results.loc[idx, 'separation_arcsec'] = best_match['dstArcSec']
                    try:
                        tic_id = str(int(best_match['ID']))
                        obs = Observations.query_criteria(target_name=f"TIC {tic_id}", obs_collection='TESS')
                        if len(obs) > 0:
                            results.loc[idx, 'has_tess_data'] = True
                            sectors = set()
                            for o in obs:
                                if 's' in o['sequence_number']:
                                    sectors.add(int(o['sequence_number'].split('s')[1][:2]))
                            results.loc[idx, 'tess_sectors'] = ','.join(map(str, sorted(sectors)))
                    except Exception as obs_err:
                        print(f"  Warning: Error querying observations for TIC {best_match['ID']}: {obs_err}")
                else:
                    if 'mag_avg' in results.columns:
                        ks_mag = results.loc[idx, 'mag_avg']
                        est_tmag = ks_mag + 0.5
                        if est_tmag > self.tess_mag_limit:
                            results.loc[idx, 'not_in_tic_reason'] = 'below_limit'
                        else:
                            bright_sources = Catalogs.query_region(coord, radius=30 * u.arcsec, catalog="TIC")
                            bright_sources = bright_sources[bright_sources['Tmag'] < est_tmag - 1]
                            if len(bright_sources) > 0:
                                results.loc[idx, 'not_in_tic_reason'] = 'confused_with_brighter'
                            else:
                                results.loc[idx, 'not_in_tic_reason'] = 'unknown'
                    else:
                        results.loc[idx, 'not_in_tic_reason'] = 'unknown'
            except Exception as e:
                print(f"Error processing candidate {i}: {e}")
        
        self.crossmatch_results = results
        crossmatch_path = os.path.join(self.output_dir, 'tess_crossmatch_results.csv')
        results.to_csv(crossmatch_path, index=False)
        in_tic_count = results['in_tic'].sum()
        has_data_count = results['has_tess_data'].sum()
        below_limit_count = (results['not_in_tic_reason'] == 'below_limit').sum()
        confused_count = (results['not_in_tic_reason'] == 'confused_with_brighter').sum()
        print(f"Cross-match summary:")
        print(f"  - In TIC: {in_tic_count} ({in_tic_count/len(results)*100:.1f}%)")
        print(f"  - Has TESS data: {has_data_count} ({has_data_count/len(results)*100:.1f}%)")
        print(f"  - Not in TIC (below limit): {below_limit_count} ({below_limit_count/len(results)*100:.1f}%)")
        print(f"  - Not in TIC (confused): {confused_count} ({confused_count/len(results)*100:.1f}%)")
        print(f"Saved crossmatch results to: {crossmatch_path}")
        return True
    
    def download_tess_lightcurves(self):
        if self.crossmatch_results is None:
            print("No cross-match results available. Call perform_crossmatch() first.")
            return False
        data_dir = os.path.join(self.output_dir, 'tess_data')
        os.makedirs(data_dir, exist_ok=True)
        sources_with_data = self.crossmatch_results[self.crossmatch_results['has_tess_data']].copy()
        if len(sources_with_data) == 0:
            print("No sources with existing TESS data found.")
            return False
        print(f"Downloading TESS timeseries data for ALL {len(sources_with_data)} sources with data...")
        download_results = []
        for idx, row in sources_with_data.iterrows():
            tic_id = row['tic_id']
            print(f"Downloading data for TIC {tic_id} (index {idx})")
            target_dir = os.path.join(data_dir, f"TIC{tic_id}")
            os.makedirs(target_dir, exist_ok=True)
            if 'ra' in row and 'dec' in row:
                ra_val = row['ra']
                dec_val = row['dec']
            elif 'RAJ2000' in row and 'DEJ2000' in row:
                ra_val = row['RAJ2000']
                dec_val = row['DEJ2000']
            else:
                err_msg = f"Error: No valid RA/Dec columns for TIC {tic_id}"
                print(err_msg)
                download_results.append(err_msg)
                continue
            coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg)
            try:
                query = Observations.query_region(coord, radius=self.search_radius * u.arcsec, obs_collection='TESS')
                products = Observations.get_product_list(query)
            except Exception as e:
                err_msg = f"Error querying products for TIC {tic_id}: {e}"
                print(err_msg)
                download_results.append(err_msg)
                continue
            if len(products) > 0:
                try:
                    download_result = Observations.download_products(products, download_dir=target_dir, cache=True)
                    msg = f"Downloaded {len(download_result)} files for TIC {tic_id}"
                    print(msg)
                    download_results.append(msg)
                except Exception as e:
                    err_msg = f"Error downloading products for TIC {tic_id}: {e}"
                    print(err_msg)
                    download_results.append(err_msg)
            else:
                msg = f"No TESS timeseries products found for TIC {tic_id}"
                print(msg)
                download_results.append(msg)
        log_path = os.path.join(self.output_dir, 'tess_download_log.txt')
        with open(log_path, 'w') as f:
            for res in download_results:
                f.write(res + "\n")
        print(f"Download complete. Data saved to: {data_dir}")
        return True
    
    def generate_target_list(self):
        """
        Generate the final target list for the TESS GI proposal.
        The final list will be exactly 100 targets and must include all known CVs.
        For remaining slots, best candidates are selected based on a composite score:
            composite_score = cv_prob / tic_tmag
        """
        if self.crossmatch_results is None:
            print("No cross-match results available. Call perform_crossmatch() first.")
            return False
        print("Generating target list for TESS proposal...")
        # Select only objects with a TIC match.
        pool = self.crossmatch_results[self.crossmatch_results['in_tic'] == True].copy()
        if ('tic_tmag' in pool.columns) and ('cv_prob' in pool.columns):
            pool['composite_score'] = pool['cv_prob'] / pool['tic_tmag']
        else:
            pool['composite_score'] = 0.0
            print("Warning: cv_prob or tic_tmag not available; composite score set to 0.")
        # Ensure known CVs are flagged
        if 'is_known_cv' not in pool.columns:
            pool['is_known_cv'] = False
            print("Warning: is_known_cv column not found; assuming all are unknown.")
        known = pool[pool['is_known_cv'] == True].copy()
        others = pool[pool['is_known_cv'] == False].copy()
        others = others.sort_values('composite_score', ascending=False)
        num_needed = 100 - len(known)
        if num_needed < 0:
            # In the unlikely event that known CVs exceed 100, select top 100 known by composite score.
            final_targets = known.sort_values('composite_score', ascending=False).head(100)
        else:
            final_targets = pd.concat([known, others.head(num_needed)], ignore_index=True)
        # If final list is less than 100, warn the user.
        if len(final_targets) < 100:
            print(f"Warning: Final target list has only {len(final_targets)} targets.")
        else:
            final_targets = final_targets.head(100)
        self.target_list = final_targets.copy()
        target_list_path = os.path.join(self.output_dir, 'tess_targets.csv')
        self.target_list.to_csv(target_list_path, index=False)
        print(f"Generated target list with {len(self.target_list)} sources (100 targets expected).")
        print(f"Full target list saved to: {target_list_path}")
        # Create a simplified version for proposal submission.
        proposal_columns = ['priority', 'sourceid', 'tic_id', 'ra', 'dec', 'tic_tmag', 'cv_prob', 'composite_score']
        proposal_columns = [col for col in proposal_columns if col in self.target_list.columns]
        proposal_targets = self.target_list[proposal_columns].copy()
        proposal_targets['target'] = proposal_targets['tic_id'].apply(lambda x: f"TIC {x}")
        proposal_targets_path = os.path.join(self.output_dir, 'tess_proposal_targets.csv')
        proposal_targets.to_csv(proposal_targets_path, index=False)
        print(f"Proposal-formatted target list saved to: {proposal_targets_path}")
        return self.target_list
    
    def generate_summary_plots(self):
        if self.cv_candidates is None:
            print("No candidate data available. Run load_cv_candidates() first.")
            return False
        print("Generating summary plots...")
        plots_dir = os.path.join(self.output_dir, 'summary_plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12

        # Define three groups: All candidates, Known CVs, and Target List
        all_candidates = self.cv_candidates.copy()
        if 'is_known_cv' not in all_candidates.columns:
            all_candidates['is_known_cv'] = False
        known_candidates = all_candidates[all_candidates['is_known_cv'] == True]
        if self.target_list is None:
            print("Target list not generated; cannot plot target list group.")
            targets = pd.DataFrame()
        else:
            targets = self.target_list.copy()

        # Spatial Plot in Galactic Coordinates with TESS Overlay
        plt.figure(figsize=(12,10))
        hb = plt.hexbin(all_candidates['l'], all_candidates['b'], gridsize=75, cmap='Greys', bins='log')
        plt.colorbar(hb, label='log10(count)')
        plt.scatter(known_candidates['l'], known_candidates['b'], label='Known CVs', color='red', marker='*', s=80)
        if not targets.empty:
            plt.scatter(targets['l'], targets['b'], label='Target List', color='blue', s=30, alpha=0.8)
        plt.xlabel('Galactic Longitude (l)')
        plt.ylabel('Galactic Latitude (b)')
        plt.title('Spatial Distribution (Galactic) - All, Known CVs, Target List')
        ax_gal = plt.gca()
        tess_overlay = TESSCycle8Overlay()
        tess_overlay.add_to_plot(ax_gal, focus_region=None, alpha=0.2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'spatial_galactic_groups.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'spatial_galactic_groups.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # ROC Curve Plot using all candidates
        print("Generating ROC curve plot...")
        if 'true_label' not in all_candidates.columns:
            all_candidates['true_label'] = all_candidates.get('is_known_cv', False).astype(int)
        fpr, tpr, thresholds = roc_curve(all_candidates['true_label'], all_candidates['cv_prob'])
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], linestyle='--', color='grey')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal threshold = {optimal_threshold:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for CV Classifier')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=300)
        plt.close()

        # Bailey Diagram Plot: Period vs Amplitude
        print("Generating Bailey diagram...")
        plt.figure(figsize=(10,8))
        hb = plt.hexbin(all_candidates['true_period'], all_candidates['true_amplitude'], 
                        gridsize=50, cmap='Greys', bins='log')
        plt.colorbar(hb, label='log10(count)')
        plt.scatter(known_candidates['true_period'], known_candidates['true_amplitude'], 
                    label='Known CVs', color='red', marker='*', s=80)
        if not targets.empty:
            plt.scatter(targets['true_period'], targets['true_amplitude'], 
                        label='Target List', color='blue', s=30, alpha=0.8)
        plt.xlabel('True Period (days)')
        plt.ylabel('True Amplitude (mag)')
        plt.title('Bailey Diagram: Period vs Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "bailey_diagram_groups.png"), dpi=300)
        plt.close()

        # 2D PCA of Embedding Space (if embedding features exist)
        embedding_features = [str(i) for i in range(64)]
        available_features = [col for col in embedding_features if col in all_candidates.columns]
        if len(available_features) >= 3:
            print("Computing PCA on embeddings...")
            embeddings = all_candidates[available_features].values
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(embeddings)
            all_candidates['pca_1'] = pca_result[:, 0]
            all_candidates['pca_2'] = pca_result[:, 1]
            if not targets.empty:
                # Map PCA results for target list by matching TIC IDs
                target_pca = all_candidates[all_candidates['tic_id'].isin(targets['tic_id'])]
            else:
                target_pca = pd.DataFrame()
            plt.figure(figsize=(10,8))
            hb = plt.hexbin(all_candidates['pca_1'], all_candidates['pca_2'], 
                            gridsize=100, cmap='Greys', bins='log')
            plt.colorbar(hb, label='log10(count)')
            if not target_pca.empty:
                plt.scatter(target_pca['pca_1'], target_pca['pca_2'], label='Target List', alpha=0.7, color='blue', s=30)
            if not known_candidates.empty:
                known_pca = all_candidates[all_candidates['is_known_cv'] == True]
                plt.scatter(known_pca['pca_1'], known_pca['pca_2'], label='Known CVs', color='red', marker='*', s=80)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.title('2D PCA of Embedding Space')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "embedding_pca_2d_groups.png"), dpi=300)
            plt.close()
        else:
            print("Insufficient embedding features for PCA.")
        
        print(f"Generated publication-quality summary plots in {plots_dir}")
        return True



    def generate_summary_plots(self):
        if self.cv_candidates is None:
            print("No candidate data available. Run load_cv_candidates() first.")
            return False
        print("Generating summary plots...")
        plots_dir = os.path.join(self.output_dir, 'summary_plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12

        # Use crossmatch_results instead of cv_candidates to ensure 'tic_id' is present.
        all_candidates = self.crossmatch_results.copy()
        if 'is_known_cv' not in all_candidates.columns:
            all_candidates['is_known_cv'] = False
        known_candidates = all_candidates[all_candidates['is_known_cv'] == True]
        if self.target_list is None:
            print("Target list not generated; cannot plot target list group.")
            targets = pd.DataFrame()
        else:
            targets = self.target_list.copy()

        # Spatial Plot in Galactic Coordinates with TESS Overlay
        plt.figure(figsize=(12,10))
        hb = plt.hexbin(all_candidates['l'], all_candidates['b'], gridsize=75, cmap='Greys', bins='log')
        plt.colorbar(hb, label='log10(count)')
        plt.scatter(known_candidates['l'], known_candidates['b'], label='Known CVs', color='red', marker='*', s=80)
        if not targets.empty:
            plt.scatter(targets['l'], targets['b'], label='Target List', color='blue', s=30, alpha=0.8)
        plt.xlabel('Galactic Longitude (l)')
        plt.ylabel('Galactic Latitude (b)')
        plt.title('Spatial Distribution (Galactic) - All, Known CVs, Target List')
        ax_gal = plt.gca()
        tess_overlay = TESSCycle8Overlay()
        tess_overlay.add_to_plot(ax_gal, focus_region=None, alpha=0.2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'spatial_galactic_groups.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, 'spatial_galactic_groups.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # ROC Curve Plot using all candidates
        print("Generating ROC curve plot...")
        if 'true_label' not in all_candidates.columns:
            all_candidates['true_label'] = all_candidates.get('is_known_cv', False).astype(int)
        fpr, tpr, thresholds = roc_curve(all_candidates['true_label'], all_candidates['cv_prob'])
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], linestyle='--', color='grey')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal threshold = {optimal_threshold:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for CV Classifier')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=300)
        plt.close()

        # Bailey Diagram Plot: Period vs Amplitude
        print("Generating Bailey diagram...")
        plt.figure(figsize=(10,8))
        hb = plt.hexbin(all_candidates['true_period'], all_candidates['true_amplitude'], 
                        gridsize=50, cmap='Greys', bins='log')
        plt.colorbar(hb, label='log10(count)')
        plt.scatter(known_candidates['true_period'], known_candidates['true_amplitude'], 
                    label='Known CVs', color='red', marker='*', s=80)
        if not targets.empty:
            plt.scatter(targets['true_period'], targets['true_amplitude'], 
                        label='Target List', color='blue', s=30, alpha=0.8)
        plt.xlabel('True Period (days)')
        plt.ylabel('True Amplitude (mag)')
        plt.title('Bailey Diagram: Period vs Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "bailey_diagram_groups.png"), dpi=300)
        plt.close()

        # 2D PCA of Embedding Space (if embedding features exist)
        embedding_features = [str(i) for i in range(64)]
        available_features = [col for col in embedding_features if col in all_candidates.columns]
        if len(available_features) >= 3:
            print("Computing PCA on embeddings...")
            embeddings = all_candidates[available_features].values
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(embeddings)
            all_candidates['pca_1'] = pca_result[:, 0]
            all_candidates['pca_2'] = pca_result[:, 1]
            if not targets.empty:
                # Map PCA results for target list by matching TIC IDs.
                target_pca = all_candidates[all_candidates['tic_id'].isin(targets['tic_id'])]
            else:
                target_pca = pd.DataFrame()
            plt.figure(figsize=(10,8))
            hb = plt.hexbin(all_candidates['pca_1'], all_candidates['pca_2'], 
                            gridsize=100, cmap='Greys', bins='log')
            plt.colorbar(hb, label='log10(count)')
            if not target_pca.empty:
                plt.scatter(target_pca['pca_1'], target_pca['pca_2'], label='Target List', alpha=0.7, color='blue', s=30)
            if not known_candidates.empty:
                known_pca = all_candidates[all_candidates['is_known_cv'] == True]
                plt.scatter(known_pca['pca_1'], known_pca['pca_2'], label='Known CVs', color='red', marker='*', s=80)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.title('2D PCA of Embedding Space')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "embedding_pca_2d_groups.png"), dpi=300)
            plt.close()
        else:
            print("Insufficient embedding features for PCA.")
        
        print(f"Generated publication-quality summary plots in {plots_dir}")
        return True


        

    def report_tess_data_details(self):
        if self.crossmatch_results is None:
            print("No cross-match results available. Run perform_crossmatch() first.")
            return
        tic_results = self.crossmatch_results[self.crossmatch_results['in_tic'] == True]
        print("\n--- Detailed TESS Data Report for TIC-matched Candidates ---\n")
        for idx, row in tic_results.iterrows():
            tic_id = int(row['tic_id'])
            print(f"TIC ID: {tic_id}")
            print(f"  TIC Tmag: {row['tic_tmag']}")
            print(f"  Separation: {row['separation_arcsec']} arcsec")
            try:
                obs = Observations.query_criteria(target_name=f"TIC {tic_id}", obs_collection='TESS')
                print(f"  Number of TESS observations found: {len(obs)}")
                if len(obs) > 0:
                    print("  Observations:")
                    print(obs)
                else:
                    print("  No TESS lightcurve data available for this TIC.")
            except Exception as e:
                print(f"  Error querying TESS data: {e}")
            print("-" * 80)
    
    def run_pipeline(self):
        start_time = time.time()
        print("\n" + "="*80)
        print("RUNNING PRIMVS-TESS CROSS-MATCH PIPELINE")
        print("="*80 + "\n")
        self.load_cv_candidates()
        self.perform_crossmatch()
        self.download_tess_lightcurves()
        self.generate_target_list()
        self.generate_summary_plots()
        end_time = time.time()
        runtime = end_time - start_time
        print("\n" + "="*80)
        print(f"PIPELINE COMPLETED in {time.strftime('%H:%M:%S', time.gmtime(runtime))}")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")
        return True

# -------------------------
# Main Execution
# -------------------------
def main():
    cv_candidates_file = "../PRIMVS/cv_results/cv_candidates.fits"  # or CSV
    output_dir = "../PRIMVS/cv_results/tess_crossmatch_results"
    cv_prob_threshold = 0.999
    search_radius = 5.0  # arcseconds
    tess_mag_limit = 16.0
    print(f"Initializing PRIMVS-TESS cross-matching pipeline with parameters:")
    print(f"  - CV candidates file: {cv_candidates_file}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - CV probability threshold: {cv_prob_threshold}")
    print(f"  - Search radius: {search_radius} arcsec")
    print(f"  - TESS magnitude limit: {tess_mag_limit}")
    
    crossmatcher = PrimvsTessCrossMatch(
        cv_candidates_file=cv_candidates_file,
        output_dir=output_dir,
        search_radius=search_radius,
        cv_prob_threshold=cv_prob_threshold,
        tess_mag_limit=tess_mag_limit
    )
    
    crossmatcher.run_pipeline()
    print("Cross-matching pipeline completed successfully.")
    crossmatcher.report_tess_data_details()

if __name__ == "__main__":
    main()
