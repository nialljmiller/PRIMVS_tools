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
from concurrent.futures import ThreadPoolExecutor

class PrimvsTessCrossMatch:
    """
    A comprehensive framework for cross-matching PRIMVS CV candidates with TESS data
    and evaluating observability during TESS Cycle 8, with probability-based pre-filtering.
    """
    
    def __init__(self, cv_candidates_file, output_dir='./tess_crossmatch', 
                 search_radius=3.0, cv_prob_threshold=0.5, tess_mag_limit=16.0):
        """
        Initialize the cross-matcher with input files and parameters.
        
        Parameters:
        -----------
        cv_candidates_file : str
            Path to the PRIMVS CV candidates file (FITS or CSV)
        output_dir : str
            Directory to save outputs
        search_radius : float
            Search radius for cross-matching in arcseconds
        cv_prob_threshold : float
            Minimum CV probability threshold for candidate selection (0.0-1.0)
        tess_mag_limit : float
            Magnitude limit for TESS detectability
        """
        self.cv_candidates_file = cv_candidates_file
        self.output_dir = output_dir
        self.search_radius = search_radius
        self.cv_prob_threshold = cv_prob_threshold
        self.tess_mag_limit = tess_mag_limit
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data containers
        self.cv_candidates = None
        self.filtered_candidates = None
        self.crossmatch_results = None
    
    def load_cv_candidates(self):
        """
        Load CV candidates identified by the primvs_cv_finder and apply
        probability threshold filtering.
        """
        try:
            if self.cv_candidates_file.endswith('.fits'):
                candidates = Table.read(self.cv_candidates_file).to_pandas()
            elif self.cv_candidates_file.endswith('.csv'):
                candidates = pd.read_csv(self.cv_candidates_file)
            else:
                raise ValueError(f"Unsupported file format: {self.cv_candidates_file}")
            
            self.cv_candidates = candidates
            print(f"Loaded {len(candidates)} total CV candidates")
            
            # Apply probability-based filtering
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
            
            # Save the filtered candidates
            filtered_path = os.path.join(self.output_dir, 'filtered_candidates.csv')
            self.filtered_candidates.to_csv(filtered_path, index=False)
            print(f"Saved filtered candidates to: {filtered_path}")
            
            return True
        except Exception as e:
            print(f"Error loading CV candidates: {e}")
            return False
    
    def perform_crossmatch(self):
        """Cross-match filtered CV candidates with the TESS Input Catalog (TIC)."""
        if self.filtered_candidates is None:
            print("No filtered candidates available. Call load_cv_candidates() first.")
            return False
        
        print("Cross-matching with TESS Input Catalog...")
        
        # Prepare coordinates for cross-matching
        if 'ra' in self.filtered_candidates.columns and 'dec' in self.filtered_candidates.columns:
            ra_col, dec_col = 'ra', 'dec'
        elif 'RAJ2000' in self.filtered_candidates.columns and 'DEJ2000' in self.filtered_candidates.columns:
            ra_col, dec_col = 'RAJ2000', 'DEJ2000'
        else:
            print("Error: Could not find RA/Dec columns in the candidates file")
            return False
        
        # Create SkyCoord object for candidates
        coords = SkyCoord(ra=self.filtered_candidates[ra_col].values * u.degree, 
                         dec=self.filtered_candidates[dec_col].values * u.degree)
        
        # Initialize results dataframe
        results = self.filtered_candidates.copy()
        results['in_tic'] = False
        results['tic_id'] = np.nan
        results['tic_tmag'] = np.nan
        results['separation_arcsec'] = np.nan
        results['has_tess_data'] = False
        results['tess_sectors'] = None
        results['not_in_tic_reason'] = None
        results['cycle8_observable'] = False
        
        # Cross-match each candidate with progress tracking
        print(f"Cross-matching {len(coords)} filtered candidates...")
        for i, (idx, coord) in enumerate(tqdm(zip(results.index, coords), total=len(coords))):
            try:
                # Query TIC around the candidate position
                tic_result = Catalogs.query_region(coord, radius=self.search_radius * u.arcsec, catalog="TIC")
                
                if len(tic_result) > 0:
                    # Sort results by separation
                    tic_result.sort('dstArcSec')
                    
                    # Get the closest match
                    best_match = tic_result[0]
                    
                    # Update results
                    results.loc[idx, 'in_tic'] = True
                    results.loc[idx, 'tic_id'] = best_match['ID']
                    results.loc[idx, 'tic_tmag'] = best_match['Tmag']
                    results.loc[idx, 'separation_arcsec'] = best_match['dstArcSec']
                    
                    # Check if this TIC ID has TESS observations

                    try:
                        tic_id = str(int(tic_result[0]['ID']))
                        
                        # Initial TESS observation query
                        obs = Observations.query_criteria(target_name=f"TIC {tic_id}", obs_collection='TESS')


                        
                        if len(obs) > 0:
                            results.loc[idx, 'has_tess_data'] = True
                            # Extract observed sectors
                            sectors = set()
                            for o in obs:
                                if 's' in o['sequence_number']:
                                    sectors.add(int(o['sequence_number'].split('s')[1][:2]))
                            results.loc[idx, 'tess_sectors'] = ','.join(map(str, sorted(sectors)))
                    except Exception as obs_err:
                        print(f"  Warning: Error querying observations for TIC {best_match['ID']}: {obs_err}")
                        # Continue processing even if observation query fails
                else:
                    # Not in TIC - determine reason
                    # Estimate TESS magnitude from Ks
                    if 'mag_avg' in results.columns:  # Ks magnitude column from PRIMVS
                        ks_mag = results.loc[idx, 'mag_avg']
                        # Approximate conversion (refined from empirical color terms)
                        est_tmag = ks_mag + 0.5  # Simple adjustment based on typical color terms
                        
                        if est_tmag > self.tess_mag_limit:
                            results.loc[idx, 'not_in_tic_reason'] = 'below_limit'
                        else:
                            # Check for nearby bright sources that might cause confusion
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
        
        # Save crossmatch results
        crossmatch_path = os.path.join(self.output_dir, 'tess_crossmatch_results.csv')
        results.to_csv(crossmatch_path, index=False)
        
        # Summary statistics
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
        

    def perform_crossmatch(self):
        """Cross-match filtered CV candidates with the TESS Input Catalog (TIC)."""
        if self.filtered_candidates is None:
            print("No filtered candidates available. Call load_cv_candidates() first.")
            return False

        print("Cross-matching with TESS Input Catalog...")
        # Determine which RA/Dec columns to use
        if 'ra' in self.filtered_candidates.columns and 'dec' in self.filtered_candidates.columns:
            ra_col, dec_col = 'ra', 'dec'
        elif 'RAJ2000' in self.filtered_candidates.columns and 'DEJ2000' in self.filtered_candidates.columns:
            ra_col, dec_col = 'RAJ2000', 'DEJ2000'
        else:
            print("Error: Could not find RA/Dec columns in the candidates file")
            return False

        # Create SkyCoord objects for candidates
        coords = SkyCoord(ra=self.filtered_candidates[ra_col].values * u.deg, 
                          dec=self.filtered_candidates[dec_col].values * u.deg)

        # Copy candidates to results dataframe and initialize new columns
        results = self.filtered_candidates.copy()
        results['in_tic'] = False
        results['tic_id'] = np.nan
        results['tic_tmag'] = np.nan
        results['separation_arcsec'] = np.nan
        results['has_tess_data'] = False
        results['tess_sectors'] = None
        results['not_in_tic_reason'] = None
        results['cycle8_observable'] = False

        print(f"Cross-matching {len(coords)} filtered candidates...")
        for i, (idx, coord) in enumerate(tqdm(zip(results.index, coords), total=len(coords))):
            try:
                # Log the candidate coordinates for debugging
                print(f"Querying TIC for candidate {idx} at RA={coord.ra.deg:.4f}, Dec={coord.dec.deg:.4f}")
                tic_result = Catalogs.query_region(coord, radius=self.search_radius * u.arcsec, catalog="TIC")
                
                if len(tic_result) > 0:
                    tic_result.sort('dstArcSec')
                    best_match = tic_result[0]
                    results.loc[idx, 'in_tic'] = True
                    # Cast the TIC ID to int to prevent dtype issues
                    results.loc[idx, 'tic_id'] = int(best_match['ID'])
                    results.loc[idx, 'tic_tmag'] = best_match['Tmag']
                    results.loc[idx, 'separation_arcsec'] = best_match['dstArcSec']

                    # Query for TESS observations using the formatted target name
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
                    # No TIC match: try to determine why
                    if 'mag_avg' in results.columns:
                        ks_mag = results.loc[idx, 'mag_avg']
                        est_tmag = ks_mag + 0.5  # Rough conversion from Ks to TESS mag
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
        """
        Download ALL TESS timeseries data for matched sources.
        This version runs serially.
        """
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

            # Pull coordinates (supporting different column names)
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




    def evaluate_cycle8_observability(self):
        """
        Evaluate which sources not in TIC could be observed in Cycle 8.
        Focuses on sources below the detection limit that could benefit
        from longer exposures or different observing strategies.
        """
        if self.crossmatch_results is None:
            print("No cross-match results available. Call perform_crossmatch() first.")
            return False
        
        print("Evaluating Cycle 8 observability...")
        
        # Define Cycle 8 sectors covering the Galactic bulge
        cycle8_bulge_sectors = range(99, 106)  # Sectors 99-105
        
        # Estimate achievable depth in Cycle 8 with 2-minute cadence
        # This is a refined model based on TESS performance metrics
        cycle8_mag_limit = self.tess_mag_limit + 0.5  # 0.5 mag deeper with optimal strategy
        
        # Filter for sources not in TIC because they're below the limit
        below_limit_mask = self.crossmatch_results['not_in_tic_reason'] == 'below_limit'
        below_limit_sources = self.crossmatch_results[below_limit_mask].copy()
        
        # Count sources potentially observable in Cycle 8
        if 'mag_avg' in below_limit_sources.columns:
            # Refined Ks to TESS conversion based on empirical calibration
            est_tmag = below_limit_sources['mag_avg'] + 0.5
            observable_mask = est_tmag <= cycle8_mag_limit
            
            # Update main results
            self.crossmatch_results.loc[below_limit_sources[observable_mask].index, 'cycle8_observable'] = True
            
            observable_count = observable_mask.sum()
            print(f"Found {observable_count} sources below TIC limit but potentially observable in Cycle 8")
        else:
            print("Warning: Could not evaluate Cycle 8 observability due to missing magnitude data")
        
        # Save updated results with cycle8_observable flag
        updated_path = os.path.join(self.output_dir, 'tess_crossmatch_results_with_cycle8.csv')
        self.crossmatch_results.to_csv(updated_path, index=False)
        print(f"Saved updated results with Cycle 8 observability to: {updated_path}")
        
        return True
    
    def generate_target_list(self):
        """
        Generate the final target list for the TESS Cycle 8 proposal.
        This includes:
        1. CV candidates already in TIC
        2. Sources not in TIC but potentially observable in Cycle 8
        """
        if self.crossmatch_results is None:
            print("No cross-match results available. Call perform_crossmatch() first.")
            return False
        
        print("Generating target list for TESS Cycle 8 proposal...")
        
        # Select sources either in TIC or potentially observable in Cycle 8
        target_mask = (self.crossmatch_results['in_tic']) | (self.crossmatch_results['cycle8_observable'])
        target_list = self.crossmatch_results[target_mask].copy()
        
        # Add a priority column (higher for sources with known CV characteristics)
        target_list['priority'] = 0
        
        # Prioritize sources based on various factors
        # 1. Known CVs with strong CV characteristics get highest priority
        if 'cv_prob' in target_list.columns:
            target_list.loc[target_list['cv_prob'] >= 0.9, 'priority'] = 3
            target_list.loc[(target_list['cv_prob'] < 0.9) & (target_list['cv_prob'] >= 0.7), 'priority'] = 2
            target_list.loc[(target_list['cv_prob'] < 0.7) & (target_list['cv_prob'] >= 0.5), 'priority'] = 1
        
        # 2. Increase priority for sources in the period gap (2-3 hours)
        if 'true_period' in target_list.columns:
            period_hours = target_list['true_period'] * 24.0  # Convert days to hours
            period_gap_mask = (period_hours >= 2.0) & (period_hours <= 3.0)
            target_list.loc[period_gap_mask, 'priority'] += 1
        
        # 3. Sources with high amplitude get slight priority boost
        if 'true_amplitude' in target_list.columns:
            target_list.loc[target_list['true_amplitude'] > 1.0, 'priority'] += 1
        
        # Sort by priority (descending) and then by CV probability or other relevant metric
        if 'cv_prob' in target_list.columns:
            target_list = target_list.sort_values(['priority', 'cv_prob'], ascending=[False, False])
        else:
            target_list = target_list.sort_values('priority', ascending=False)
        
        # Add Galactic Bulge flag for sources in the region of interest
        if 'l' in target_list.columns and 'b' in target_list.columns:
            # Define approximate bulge region boundaries
            bulge_mask = (target_list['l'] > 350) | (target_list['l'] < 10)
            bulge_mask &= (target_list['b'] > -10) & (target_list['b'] < 10)
            target_list['in_bulge'] = bulge_mask
            
            # Increase priority for bulge sources to focus on metallicity gradient study
            target_list.loc[bulge_mask, 'priority'] += 1
        
        # Save the target list
        target_list_path = os.path.join(self.output_dir, 'tess_cycle8_targets.csv')
        target_list.to_csv(target_list_path, index=False)
        
        # Save a simplified version formatted for the proposal
        proposal_columns = ['priority', 'sourceid', 'tic_id', 'ra', 'dec', 'l', 'b', 'mag_avg', 
                           'true_period', 'true_amplitude', 'cv_prob', 'in_tic', 'has_tess_data', 
                           'tess_sectors', 'in_bulge']
        proposal_columns = [col for col in proposal_columns if col in target_list.columns]
        
        proposal_targets = target_list[proposal_columns].copy()
        
        # Add period in hours for easier reference
        if 'true_period' in proposal_targets.columns:
            proposal_targets['period_hours'] = proposal_targets['true_period'] * 24.0
        
        proposal_targets_path = os.path.join(self.output_dir, 'tess_cycle8_proposal_targets.csv')
        proposal_targets.to_csv(proposal_targets_path, index=False)
        
        # Generate target list in format compatible with TESS proposal submission
        tess_proposal_format = self._generate_tess_proposal_format(proposal_targets)
        tess_format_path = os.path.join(self.output_dir, 'tess_cycle8_proposal_targets_formatted.csv')
        tess_proposal_format.to_csv(tess_format_path, index=False)
        
        print(f"Generated target list with {len(target_list)} sources")
        print(f"Full target list saved to: {target_list_path}")
        print(f"Proposal-formatted target list saved to: {proposal_targets_path}")
        print(f"TESS submission format saved to: {tess_format_path}")
        
        return target_list
    
    def _generate_tess_proposal_format(self, targets):
        """
        Generate a target list in the format required for TESS Cycle 8 proposal submission.
        """
        # Create a new DataFrame with required TESS proposal columns
        tess_format = pd.DataFrame()
        
        # Extract TIC ID if available, otherwise use placeholder
        if 'tic_id' in targets.columns:
            tess_format['target'] = targets['tic_id'].apply(lambda x: f"TIC {x}" if pd.notnull(x) else "")
        else:
            tess_format['target'] = ""
        
        # Include PRIMVS ID as alternate target identifier
        id_col = None
        for col in ['sourceid', 'primvs_id', 'source_id']:
            if col in targets.columns:
                id_col = col
                break
        
        if id_col:
            tess_format['alt_target'] = targets[id_col].astype(str)
        else:
            tess_format['alt_target'] = ""
        
        # Add coordinates
        if 'ra' in targets.columns and 'dec' in targets.columns:
            tess_format['ra'] = targets['ra']
            tess_format['dec'] = targets['dec']
        
        # Add magnitude
        if 'tic_tmag' in targets.columns:
            tess_format['tmag'] = targets['tic_tmag']
        elif 'mag_avg' in targets.columns:
            # Approximate TESS magnitude from Ks
            tess_format['tmag'] = targets['mag_avg'] + 0.5
        else:
            tess_format['tmag'] = ""
        
        # Add priority (scaled to TESS 1-5 scale where 5 is highest)
        if 'priority' in targets.columns:
            # Scale to 1-5 range based on distribution of priorities
            priority_max = targets['priority'].max()
            if priority_max > 0:
                # Scale and round to integer
                tess_format['priority'] = np.minimum(5, np.maximum(1, 
                    np.round(targets['priority'] * 5 / priority_max))).astype(int)
            else:
                tess_format['priority'] = 3  # Default mid-priority
        else:
            tess_format['priority'] = 3  # Default mid-priority
        
        # Add observation duration (2 minutes for all)
        tess_format['duration'] = 2
        
        # Add comments with key scientific info
        comments = []
        for _, row in targets.iterrows():
            comment_parts = []
            if 'period_hours' in row and pd.notnull(row['period_hours']):
                comment_parts.append(f"P={row['period_hours']:.2f}h")
            if 'true_amplitude' in row and pd.notnull(row['true_amplitude']):
                comment_parts.append(f"Amp={row['true_amplitude']:.2f}mag")
            if 'cv_prob' in row and pd.notnull(row['cv_prob']):
                comment_parts.append(f"CVprob={row['cv_prob']:.2f}")
            if 'in_bulge' in row and row['in_bulge']:
                comment_parts.append("Bulge")
            if 'not_in_tic_reason' in row and pd.notnull(row['not_in_tic_reason']):
                comment_parts.append(f"NotInTIC:{row['not_in_tic_reason']}")
            
            comments.append(", ".join(comment_parts))
        
        tess_format['comments'] = comments
        
        return tess_format
    


    def run_pipeline(self):
        """Run the complete TESS cross-match pipeline."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("RUNNING PRIMVS-TESS CROSS-MATCH PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Load CV candidates and apply pre-filtering
        self.load_cv_candidates()
        
        # Step 2: Perform cross-match with TIC
        self.perform_crossmatch()
        
        # Step 3: Download ALL TESS light curves for matched sources
        self.download_tess_lightcurves()
        
        # Step 4: Evaluate Cycle 8 observability
        self.evaluate_cycle8_observability()
        
        # Step 5: Generate target list for proposal
        self.generate_target_list()
        
        # Step 6: Generate summary plots
        self.generate_summary_plots()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print("\n" + "="*80)
        print(f"PIPELINE COMPLETED in {time.strftime('%H:%M:%S', time.gmtime(runtime))}")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")
        
        return True

    def generate_summary_plots(self):
        """Generate publication-quality summary plots for the cross-match results."""
        if self.crossmatch_results is None:
            print("No cross-match results available. Call perform_crossmatch() first.")
            return False
        
        print("Generating summary plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'summary_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set plotting style for publication quality
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        
        # 1. CV candidates on the sky (Galactic coordinates)
        if 'l' in self.crossmatch_results.columns and 'b' in self.crossmatch_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create custom colormap for CV probability if available
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
                # Color by TIC status
                in_tic = self.crossmatch_results['in_tic']
                has_data = self.crossmatch_results['has_tess_data']
                cycle8 = self.crossmatch_results['cycle8_observable']
                
                # Create categories
                categories = np.zeros(len(self.crossmatch_results))
                categories[in_tic] = 1
                categories[has_data] = 2
                categories[cycle8] = 3
                
                scatter = ax.scatter(
                    self.crossmatch_results['l'], 
                    self.crossmatch_results['b'],
                    c=categories,
                    cmap='viridis',
                    alpha=0.7,
                    s=15,
                    edgecolor='none'
                )
                
                # Create custom legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0), label='Not in TIC', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0.33), label='In TIC', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0.66), label='Has TESS data', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(1.0), label='Cycle 8 target', markersize=8)
                ]
                ax.legend(handles=legend_elements, loc='upper right')
            
            # Add TESS Cycle 8 footprint overlay (simplified representation)
            # This is a simplified representation of TESS coverage
            cycle8_sectors = {
                'Sector 99': {'l_range': (-30, 30), 'b_range': (-30, 30)},
                'Sector 100': {'l_range': (-50, 10), 'b_range': (-30, 10)},
                'Sector 101': {'l_range': (-70, -10), 'b_range': (-20, 20)},
                'Sector 102': {'l_range': (-90, -30), 'b_range': (-30, 10)},
                'Sector 103': {'l_range': (-50, 10), 'b_range': (-10, 30)},
                'Sector 104': {'l_range': (-10, 50), 'b_range': (-20, 20)},
                'Sector 105': {'l_range': (10, 70), 'b_range': (-30, 10)}
            }
            
            for sector, coords in cycle8_sectors.items():
                l_min, l_max = coords['l_range']
                b_min, b_max = coords['b_range']
                
                # Create a rectangle patch
                rect = plt.Rectangle(
                    (l_min, b_min), 
                    l_max - l_min, 
                    b_max - b_min,
                    fill=False,
                    edgecolor='red',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.7,
                    label=sector if sector == 'Sector 99' else None
                )
                ax.add_patch(rect)
                
                # Add sector label
                ax.text(
                    (l_min + l_max) / 2, 
                    (b_min + b_max) / 2,
                    sector.replace('Sector ', 'S'),
                    color='red',
                    fontsize=10,
                    ha='center',
                    va='center'
                )
            
            ax.set_xlabel('Galactic Longitude (deg)')
            ax.set_ylabel('Galactic Latitude (deg)')
            ax.set_title('Distribution of CV Candidates in Galactic Coordinates')
            
            # Add rectangular patch indicating the bulge region
            bulge_patch = plt.Rectangle(
                (350, -10), 
                20, 
                20,
                fill=False,
                edgecolor='blue',
                linestyle='-',
                linewidth=2,
                alpha=0.7,
                label='Galactic Bulge'
            )
            ax.add_patch(bulge_patch)
            
            # Handle the longitude discontinuity for the bulge patch
            bulge_patch2 = plt.Rectangle(
                (0, -10), 
                10, 
                20,
                fill=False,
                edgecolor='blue',
                linestyle='-',
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(bulge_patch2)
            
            # Add legend for the bulge
            if 'Sector 99' not in cycle8_sectors:  # Only add if not already in legend
                from matplotlib.lines import Line2D
                bulge_legend = Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='Galactic Bulge')
                handles, labels = ax.get_legend_handles_labels()
                handles.append(bulge_legend)
                labels.append('Galactic Bulge')
                ax.legend(handles=handles, labels=labels, loc='upper right')
            
            plt.savefig(os.path.join(plots_dir, 'galactic_distribution.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, 'galactic_distribution.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        # 2. Bailey diagram (Period vs. Amplitude)
        if 'true_period' in self.crossmatch_results.columns and 'true_amplitude' in self.crossmatch_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get log period in hours
            period_hours = self.crossmatch_results['true_period'] * 24.0
            log_period = np.log10(period_hours)
            
            # Color by CV probability if available, otherwise by TIC status
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
                # Distinguish by TESS data availability
                scatter = ax.scatter(
                    log_period,
                    self.crossmatch_results['true_amplitude'],
                    c=self.crossmatch_results['has_tess_data'].astype(int),
                    cmap='coolwarm',
                    alpha=0.7,
                    s=15,
                    edgecolor='none'
                )
                plt.colorbar(scatter, label='Has TESS Data')
            
            # Mark the period gap
            ax.axvspan(np.log10(2), np.log10(3), alpha=0.2, color='red')
            
            # Add text annotation for the period gap
            ax.text(
                np.log10(2.5), 
                ax.get_ylim()[1] * 0.9,
                'Period Gap\n(2-3 hours)',
                color='darkred',
                fontsize=12,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="red")
            )
            
            # Add contours to highlight density distribution
            try:
                from scipy.stats import gaussian_kde
                
                # Create a 2D kernel density estimate
                xy = np.vstack([log_period, self.crossmatch_results['true_amplitude']])
                z = gaussian_kde(xy)(xy)
                
                # Sort points by density for better visualization
                idx = z.argsort()
                x_sorted = log_period[idx]
                y_sorted = self.crossmatch_results['true_amplitude'].values[idx]
                z_sorted = z[idx]
                
                # Add contour lines
                xgrid = np.linspace(log_period.min(), log_period.max(), 100)
                ygrid = np.linspace(self.crossmatch_results['true_amplitude'].min(), 
                                   self.crossmatch_results['true_amplitude'].max(), 100)
                X, Y = np.meshgrid(xgrid, ygrid)
                
                # Reshape data for contour plotting
                from scipy.interpolate import griddata
                Z = griddata((log_period, self.crossmatch_results['true_amplitude']), z, (X, Y), method='cubic')
                
                # Plot contours
                contour = ax.contour(X, Y, Z, levels=5, colors='black', alpha=0.5, linewidths=0.5)
            except:
                # Skip contours if scipy not available or other error
                pass
            
            ax.set_xlabel(r'$\log_{10}$(Period) [hours]')
            ax.set_ylabel('Amplitude [mag]')
            ax.set_title('Bailey Diagram of CV Candidates')
            ax.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(plots_dir, 'bailey_diagram.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, 'bailey_diagram.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        # 3. Histogram of TESS magnitudes with detection thresholds
        if 'tic_tmag' in self.crossmatch_results.columns or 'mag_avg' in self.crossmatch_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use TESS magnitudes if available, otherwise estimate from Ks
            if 'tic_tmag' in self.crossmatch_results.columns:
                tmags = self.crossmatch_results.loc[self.crossmatch_results['in_tic'], 'tic_tmag']
                title = 'Distribution of TESS Magnitudes for CV Candidates'
                xlabel = 'TESS Magnitude (T)'
            else:
                # Estimate TESS magnitude from Ks
                tmags = self.crossmatch_results['mag_avg'] + 0.5
                title = 'Estimated TESS Magnitudes for CV Candidates'
                xlabel = 'Estimated TESS Magnitude'
            
            # Plot histogram
            ax.hist(tmags, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
            
            # Add vertical lines for detection thresholds
            ax.axvline(self.tess_mag_limit, color='red', linestyle='--', linewidth=2,
                      label=f'Standard limit ({self.tess_mag_limit:.1f})')
            
            # Add estimated achievable depth in Cycle 8
            cycle8_mag_limit = self.tess_mag_limit + 0.5
            ax.axvline(cycle8_mag_limit, color='green', linestyle='--', linewidth=2,
                      label=f'Cycle 8 extended depth ({cycle8_mag_limit:.1f})')
            
            # Add annotations explaining the thresholds
            ax.annotate(
                'Standard\nLimit', 
                xy=(self.tess_mag_limit, ax.get_ylim()[1] * 0.8),
                xytext=(self.tess_mag_limit + 0.5, ax.get_ylim()[1] * 0.9),
                arrowprops=dict(arrowstyle='->'),
                fontsize=10
            )
            
            ax.annotate(
                'Cycle 8\nExtended Depth', 
                xy=(cycle8_mag_limit, ax.get_ylim()[1] * 0.6),
                xytext=(cycle8_mag_limit + 0.5, ax.get_ylim()[1] * 0.7),
                arrowprops=dict(arrowstyle='->'),
                fontsize=10
            )
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Number of Sources')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(plots_dir, 'tess_magnitude_distribution.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, 'tess_magnitude_distribution.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        # 4. Custom Venn diagram of candidate distribution
        try:
            from matplotlib_venn import venn3
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Count statistics for Venn diagram
            in_tic_count = int(self.crossmatch_results['in_tic'].sum())
            has_data_count = int(self.crossmatch_results['has_tess_data'].sum())
            cycle8_count = int(self.crossmatch_results['cycle8_observable'].sum())
            
            # Calculate intersections
            in_tic_only = in_tic_count - has_data_count
            has_data_only = 0  # This should always be 0 (can't have data without being in TIC)
            cycle8_only = int(sum(self.crossmatch_results['cycle8_observable'] & ~self.crossmatch_results['in_tic']))
            
            in_tic_and_cycle8 = int(sum(self.crossmatch_results['in_tic'] & self.crossmatch_results['cycle8_observable'] & ~self.crossmatch_results['has_tess_data']))
            has_data_and_cycle8 = int(sum(self.crossmatch_results['has_tess_data'] & self.crossmatch_results['cycle8_observable']))
            in_tic_and_has_data = has_data_count  # All sources with data are in TIC
            
            all_three = has_data_and_cycle8  # If it has data and is cycle8 observable, it's also in TIC
            
            # Create the Venn diagram
            venn = venn3(
                subsets=(in_tic_only, has_data_only, in_tic_and_has_data, 
                         cycle8_only, in_tic_and_cycle8, has_data_and_cycle8, 
                         all_three),
                set_labels=('In TIC', 'Has TESS Data', 'Cycle 8 Target')
            )
            
            # Set colors
            venn.get_patch_by_id('100').set_color('skyblue')
            venn.get_patch_by_id('010').set_color('lightgreen')
            venn.get_patch_by_id('001').set_color('salmon')
            
            # Add title
            plt.title('Distribution of CV Candidates in TESS Analysis')
            
            plt.savefig(os.path.join(plots_dir, 'candidate_distribution_venn.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, 'candidate_distribution_venn.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        except:
            # Skip Venn diagram if matplotlib_venn not available
            print("Warning: matplotlib_venn not available, skipping Venn diagram")
        
        print(f"Generated publication-quality summary plots in {plots_dir}")
        return True




def main():
    """
    Main function to execute the PRIMVS-TESS CV candidate cross-matching pipeline
    with probability pre-filtering and comprehensive light curve acquisition.
    """
    # Define input parameters
    cv_candidates_file = "../PRIMVS/cv_results/cv_candidates.fits"
    output_dir = "./tess_crossmatch_results"
    cv_prob_threshold = 0.982  # Only process candidates with CV probability ≥ 0.7
    search_radius = 5.0  # arcseconds
    tess_mag_limit = 16.0  # default TESS magnitude limit
    
    print(f"Initializing PRIMVS-TESS cross-matching pipeline with parameters:")
    print(f"  - CV candidates file: {cv_candidates_file}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - CV probability threshold: {cv_prob_threshold}")
    print(f"  - Search radius: {search_radius} arcsec")
    print(f"  - TESS magnitude limit: {tess_mag_limit}")
    
    # Initialize cross-matcher with specified parameters
    crossmatcher = PrimvsTessCrossMatch(
        cv_candidates_file=cv_candidates_file,
        output_dir=output_dir,
        search_radius=search_radius,
        cv_prob_threshold=cv_prob_threshold,
        tess_mag_limit=tess_mag_limit
    )
    
    # Execute the complete pipeline
    crossmatcher.run_pipeline()
    
    print("Cross-matching pipeline completed successfully.")

if __name__ == "__main__":
    main()







