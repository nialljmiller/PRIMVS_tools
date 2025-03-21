import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Catalogs, Observations
from astropy.io import fits
import os
import warnings
import time
from tqdm import tqdm

class PrimvsTessCrossMatch:
    """
    A comprehensive framework for cross-matching PRIMVS CV candidates with TESS data,
    with probability-based pre-filtering.
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
        Generate the final target list for the TESS proposal.
        Only includes candidates that are in the TIC.
        """
        if self.crossmatch_results is None:
            print("No cross-match results available. Call perform_crossmatch() first.")
            return False
        
        print("Generating target list for TESS proposal...")
        
        target_mask = self.crossmatch_results['in_tic']
        target_list = self.crossmatch_results[target_mask].copy()
        
        target_list['priority'] = 0
        if 'cv_prob' in target_list.columns:
            target_list.loc[target_list['cv_prob'] >= 0.9, 'priority'] = 3
            target_list.loc[(target_list['cv_prob'] < 0.9) & (target_list['cv_prob'] >= 0.7), 'priority'] = 2
            target_list.loc[(target_list['cv_prob'] < 0.7) & (target_list['cv_prob'] >= 0.5), 'priority'] = 1
        
        if 'cv_prob' in target_list.columns:
            target_list = target_list.sort_values(['priority', 'cv_prob'], ascending=[False, False])
        else:
            target_list = target_list.sort_values('priority', ascending=False)
        
        target_list_path = os.path.join(self.output_dir, 'tess_targets.csv')
        target_list.to_csv(target_list_path, index=False)
        print(f"Generated target list with {len(target_list)} sources")
        print(f"Full target list saved to: {target_list_path}")
        return target_list

    def _generate_tess_proposal_format(self, targets):
        tess_format = pd.DataFrame()
        if 'tic_id' in targets.columns:
            tess_format['target'] = targets['tic_id'].apply(lambda x: f"TIC {x}" if pd.notnull(x) else "")
        else:
            tess_format['target'] = ""
        
        id_col = None
        for col in ['sourceid', 'primvs_id', 'source_id']:
            if col in targets.columns:
                id_col = col
                break
        if id_col:
            tess_format['alt_target'] = targets[id_col].astype(str)
        else:
            tess_format['alt_target'] = ""
        
        if 'ra' in targets.columns and 'dec' in targets.columns:
            tess_format['ra'] = targets['ra']
            tess_format['dec'] = targets['dec']
        
        if 'tic_tmag' in targets.columns:
            tess_format['tmag'] = targets['tic_tmag']
        elif 'mag_avg' in targets.columns:
            tess_format['tmag'] = targets['mag_avg'] + 0.5
        else:
            tess_format['tmag'] = ""
        
        if 'priority' in targets.columns:
            priority_max = targets['priority'].max()
            if priority_max > 0:
                tess_format['priority'] = np.minimum(5, np.maximum(1, np.round(targets['priority'] * 5 / priority_max))).astype(int)
            else:
                tess_format['priority'] = 3
        else:
            tess_format['priority'] = 3
        
        tess_format['duration'] = 2
        
        comments = []
        for _, row in targets.iterrows():
            comment_parts = []
            if 'cv_prob' in row and pd.notnull(row['cv_prob']):
                comment_parts.append(f"CVprob={row['cv_prob']:.2f}")
            comments.append(", ".join(comment_parts))
        tess_format['comments'] = comments
        
        return tess_format

    def generate_summary_plots(self):
        if self.crossmatch_results is None:
            print("No cross-match results available. Call perform_crossmatch() first.")
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
        
        if 'l' in self.crossmatch_results.columns and 'b' in self.crossmatch_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'cv_prob' in self.crossmatch_results.columns:
                scatter = ax.scatter(
                    self.crossmatch_results['l'], 
                    self.crossmatch_results['b'],
                    c=self.crossmatch_results['cv_prob'],
                    cmap='viridis',
                    alpha=0.7,
                    s=15,
                    edgecolor='none'
                )
                cbar = plt.colorbar(scatter)
                cbar.set_label('CV Probability')
            else:
                in_tic = self.crossmatch_results['in_tic']
                scatter = ax.scatter(
                    self.crossmatch_results['l'], 
                    self.crossmatch_results['b'],
                    c=in_tic.astype(int),
                    cmap='viridis',
                    alpha=0.7,
                    s=15,
                    edgecolor='none'
                )
                plt.colorbar(scatter, label='In TIC')
            ax.set_xlabel('Galactic Longitude (deg)')
            ax.set_ylabel('Galactic Latitude (deg)')
            ax.set_title('Distribution of CV Candidates in Galactic Coordinates')
            plt.savefig(os.path.join(plots_dir, 'galactic_distribution.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, 'galactic_distribution.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        if 'true_period' in self.crossmatch_results.columns and 'true_amplitude' in self.crossmatch_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            period_hours = self.crossmatch_results['true_period'] * 24.0
            log_period = np.log10(period_hours)
            if 'cv_prob' in self.crossmatch_results.columns:
                scatter = ax.scatter(
                    log_period,
                    self.crossmatch_results['true_amplitude'],
                    c=self.crossmatch_results['cv_prob'],
                    cmap='viridis',
                    alpha=0.7,
                    s=15,
                    edgecolor='none'
                )
                cbar = plt.colorbar(scatter)
                cbar.set_label('CV Probability')
            else:
                scatter = ax.scatter(
                    log_period,
                    self.crossmatch_results['true_amplitude'],
                    c=self.crossmatch_results['in_tic'].astype(int),
                    cmap='coolwarm',
                    alpha=0.7,
                    s=15,
                    edgecolor='none'
                )
                plt.colorbar(scatter, label='In TIC')
            ax.set_xlabel(r'$\log_{10}$(Period) [hours]')
            ax.set_ylabel('Amplitude [mag]')
            ax.set_title('Bailey Diagram of CV Candidates')
            ax.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'bailey_diagram.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, 'bailey_diagram.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        if 'tic_tmag' in self.crossmatch_results.columns or 'mag_avg' in self.crossmatch_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'tic_tmag' in self.crossmatch_results.columns:
                tmags = self.crossmatch_results.loc[self.crossmatch_results['in_tic'], 'tic_tmag']
                title = 'Distribution of TESS Magnitudes for CV Candidates'
                xlabel = 'TESS Magnitude (T)'
            else:
                tmags = self.crossmatch_results['mag_avg'] + 0.5
                title = 'Estimated TESS Magnitudes for CV Candidates'
                xlabel = 'Estimated TESS Magnitude'
            ax.hist(tmags, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
            ax.axvline(self.tess_mag_limit, color='red', linestyle='--', linewidth=2,
                       label=f'Standard limit ({self.tess_mag_limit:.1f})')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Number of Sources')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'tess_magnitude_distribution.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, 'tess_magnitude_distribution.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        print(f"Generated publication-quality summary plots in {plots_dir}")
        return True

    def report_tess_data_details(self):
        """
        For every candidate with a TIC match, query MAST for TESS observations and
        report all details found. This conclusively shows what TESS data (if any) exists.
        """
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

def main():
    cv_candidates_file = "../PRIMVS/cv_results/cv_candidates.fits"
    output_dir = "../PRIMVS/cv_results/tess_crossmatch_results"
    cv_prob_threshold = 0.984
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
    
    # Now report detailed TESS data for each TIC candidate.
    crossmatcher.report_tess_data_details()

if __name__ == "__main__":
    main()
