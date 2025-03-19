import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import warnings
import multiprocessing
import concurrent.futures
from joblib import Parallel, delayed
import os

# Get number of available cores
N_JOBS = os.cpu_count()
print(f"Detected {N_JOBS} CPU cores for parallel processing")

warnings.filterwarnings('ignore')

class CVFinder:
    """
    A class to identify cataclysmic variable (CV) star candidates from 
    PRIMVS and TESS cross-matched data.
    
    This implementation uses multiple detection strategies:
    1. Feature-based filtering using known CV characteristics
    2. Anomaly detection to find outliers in parameter space
    3. Supervised classification using a gradient boosting model
    """
    
    def __init__(self, primvs_file, tess_match_file, output_dir='./output'):
        """
        Initialize the CV finder with input data files.
        
        Parameters:
        -----------
        primvs_file : str
            Path to the PRIMVS FITS file
        tess_match_file : str
            Path to the PRIMVS-TESS cross-match file
        output_dir : str
            Directory to save outputs
        """
        self.primvs_file = primvs_file
        self.tess_match_file = tess_match_file
        self.output_dir = output_dir
        
        # Initialize storage for data
        self.primvs_data = None
        self.tess_matches = None
        self.combined_data = None
        self.cv_candidates = None
        
        # CV detection parameters
        self.cv_features = [
            # PRIMVS time domain features
            'true_amplitude', 'true_period', 'log_period',
            'best_fap', 'stet_k', 'skew', 'kurt', 
            'Cody_M', 'eta', 'eta_e', 'med_BRP',
            'max_slope', 'MAD', 'mean_var', 'percent_amp',
            'roms', 'p_to_p_var', 'lag_auto',
            
            # Color features
            'Z-K', 'Y-K', 'J-K', 'H-K',
            
            # TESS features (to be added after loading)
            'tess_mag', 'separation_arcsec',
            
            # Spatial features
            'l', 'b'
        ]
        
    def load_data(self):
        """Load PRIMVS and TESS cross-matched data"""
        print("Loading PRIMVS data...")
        try:
            with fits.open(self.primvs_file, memmap=True) as hdul:
                self.primvs_data = Table(hdul[1].data).to_pandas()
                
            # Add log period for better scaling
            self.primvs_data['log_period'] = np.log10(self.primvs_data['true_period'])
            
            # Rename color columns for easier reference
            for col in self.primvs_data.columns:
                if col == 'z_med_mag-ks_med_mag':
                    self.primvs_data.rename(columns={col: 'Z-K'}, inplace=True)
                elif col == 'y_med_mag-ks_med_mag':
                    self.primvs_data.rename(columns={col: 'Y-K'}, inplace=True)
                elif col == 'j_med_mag-ks_med_mag':
                    self.primvs_data.rename(columns={col: 'J-K'}, inplace=True)
                elif col == 'h_med_mag-ks_med_mag':
                    self.primvs_data.rename(columns={col: 'H-K'}, inplace=True)
            
            print(f"Loaded {len(self.primvs_data)} PRIMVS sources")
        except Exception as e:
            print(f"Error loading PRIMVS data: {str(e)}")
            return False
            
        print("Loading TESS cross-match data...")
        try:
            # Determine file type
            if self.tess_match_file.endswith('.fits'):
                with fits.open(self.tess_match_file) as hdul:
                    self.tess_matches = Table(hdul[1].data).to_pandas()
            elif self.tess_match_file.endswith('.csv'):
                self.tess_matches = pd.read_csv(self.tess_match_file)
            else:
                raise ValueError(f"Unsupported file format: {self.tess_match_file}")
                
            print(f"Loaded {len(self.tess_matches)} TESS cross-matched sources")
        except Exception as e:
            print(f"Error loading TESS match data: {str(e)}")
            return False
            
        return True
    
    def merge_datasets(self):
        """Merge PRIMVS and TESS data for joint analysis"""
        if self.primvs_data is None or self.tess_matches is None:
            print("Data not loaded yet. Call load_data() first.")
            return False
            
        print("Merging PRIMVS and TESS datasets...")
        
        # Assuming 'sourceid' in PRIMVS matches 'primvs_id' in TESS matches
        # Adjust column names if they differ
        primvs_id_col = 'sourceid' if 'sourceid' in self.primvs_data.columns else 'primvs_id'
        tess_id_col = 'primvs_id' if 'primvs_id' in self.tess_matches.columns else 'primvs_id'
        
        # Merge datasets
        self.combined_data = pd.merge(
            self.tess_matches, 
            self.primvs_data,
            left_on=tess_id_col,
            right_on=primvs_id_col,
            how='inner'
        )
        
        print(f"Merged dataset contains {len(self.combined_data)} sources")
        
        # Check if we have the required CV features
        missing_features = [f for f in self.cv_features if f not in self.combined_data.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Remove missing features from our list
            self.cv_features = [f for f in self.cv_features if f not in missing_features]
            
        return True
    
    def apply_loose_filters(self):
        """
        Apply very loose preselection filters to prioritize completeness over purity.
        This creates a wide candidate pool that will be further refined.
        """
        if self.combined_data is None:
            print("No combined data available. Call merge_datasets() first.")
            return False
            
        print("Applying loose CV selection filters...")
        
        # Make a copy to avoid modifying the original data
        filtered_data = self.combined_data.copy()
        
        # Filter out rows with NaN values in critical columns
        critical_columns = ['true_period', 'true_amplitude']
        filtered_data = filtered_data.dropna(subset=critical_columns)
        
        # Extremely loose initial filters based on CV properties
        # These are deliberately permissive to avoid excluding potential candidates
        
        # 1. Period filter: CVs typically have periods < 1 day
        # We'll be very loose and include up to 5 days to catch unusual systems
        period_mask = filtered_data['true_period'] < 5.0
        
        # 2. Amplitude filter: CVs often show significant variability
        # But we'll set a very low threshold to be inclusive
        amplitude_mask = filtered_data['true_amplitude'] > 0.05
        
        # Apply the filters
        filtered_data = filtered_data[period_mask & amplitude_mask]
        
        print(f"After loose filtering: {len(filtered_data)} candidate sources")
        self.filtered_data = filtered_data
        
        return True
    
    def extract_cv_features(self):
        """Extract and normalize features for CV detection"""
        if self.filtered_data is None:
            print("No filtered data available. Call apply_loose_filters() first.")
            return None
            
        print("Extracting CV features...")
        
        # Select features present in our dataset
        available_features = [f for f in self.cv_features if f in self.filtered_data.columns]
        print(f"Using {len(available_features)} features for CV detection")
        
        # Extract feature matrix
        X = self.filtered_data[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in [np.float64, np.int64]:
                    # Fill numeric with median
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # Fill others with mode
                    X[col] = X[col].fillna(X[col].mode()[0])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled, X.index
    
    def detect_anomalies(self, X_scaled, indices):
        """
        Use Isolation Forest to identify anomalous sources that might be CVs
        CVs often occupy distinct regions of parameter space
        """
        print("Detecting anomalous sources with Isolation Forest...")
        
        # Initialize and fit Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Expect ~10% of sources to be unusual
            random_state=42,
            n_jobs=-1
        )
        
        # Fit and predict
        anomaly_scores = iso_forest.fit_predict(X_scaled)
        
        # Get anomaly scores (decision function)
        # More negative = more anomalous
        decision_scores = iso_forest.decision_function(X_scaled)
        
        # Select anomalies (where prediction is -1)
        anomaly_mask = anomaly_scores == -1
        anomaly_indices = indices[anomaly_mask]
        
        print(f"Identified {sum(anomaly_mask)} anomalous sources")
        
        # Add anomaly scores to filtered data
        self.filtered_data['anomaly_score'] = pd.Series(
            decision_scores, 
            index=indices
        )
        
        return anomaly_indices
    
    def calculate_cv_score_batch(self, batch_df):
        """
        Calculate CV score for a batch of data - enables parallel processing
        
        Parameters:
        -----------
        batch_df : DataFrame
            Batch of filtered data to process
            
        Returns:
        --------
        DataFrame
            Batch with CV scores added
        """
        # Make a copy to avoid modifying original
        batch = batch_df.copy()
        
        # 1. Period-based score: Higher for shorter periods
        # CVs typically have short periods (hours)
        batch['period_score'] = np.clip(1.0 - np.log10(batch['true_period']) / 2.0, 0, 1)
        
        # 2. Amplitude-based score: Higher for larger amplitudes
        # Normalize amplitude to 0-1 scale with sigmoid function
        batch['amplitude_score'] = 1 / (1 + np.exp(-2 * (batch['true_amplitude'] - 0.5)))
        
        # 3. Skewness score: CV light curves often show asymmetry
        # Normalize absolute skewness
        if 'skew' in batch.columns:
            batch['skew_score'] = np.clip(np.abs(batch['skew']) / 3.0, 0, 1)
        else:
            batch['skew_score'] = 0.5  # Default if not available
            
        # 4. Color score: CVs often have blue colors
        # This is a simplified approach - would need refinement for specific surveys
        if all(col in batch.columns for col in ['J-K', 'H-K']):
            # Blue objects have lower J-K, H-K values
            batch['color_score'] = 1 - np.clip((batch['J-K'] + batch['H-K']) / 2, 0, 1)
        else:
            batch['color_score'] = 0.5  # Default if colors not available
            
        # 5. Periodicity confidence: Higher for more reliable periods
        batch['fap_score'] = 1 - np.clip(batch['best_fap'], 0, 1)
        
        # Combine scores with different weights
        batch['cv_score'] = (
            0.3 * batch['period_score'] +
            0.3 * batch['amplitude_score'] +
            0.15 * batch['skew_score'] +
            0.15 * batch['color_score'] +
            0.1 * batch['fap_score']
        )
        
        return batch
    
    def apply_cv_knowledge(self):
        """
        Apply domain knowledge to identify CV candidates.
        This combines multiple heuristics based on CV properties.
        Uses parallel processing for large datasets.
        """
        if self.filtered_data is None:
            print("No filtered data available. Call apply_loose_filters() first.")
            return False
            
        print("Applying CV domain knowledge...")
        
        # Extract features for modeling
        X_scaled, indices = self.extract_cv_features()
        if X_scaled is None:
            return False
            
        # Find anomalous sources
        anomaly_indices = self.detect_anomalies(X_scaled, indices)
        
        # Determine optimal batch size based on data size and CPU cores
        rows_per_core = min(1000, max(100, len(self.filtered_data) // N_JOBS))
        n_batches = max(1, len(self.filtered_data) // rows_per_core)
        
        print(f"Processing CV scores in {n_batches} batches using {N_JOBS} cores...")
        
        # Split data into batches
        df_splits = np.array_split(self.filtered_data, n_batches)
        
        # Process batches in parallel
        processed_batches = Parallel(n_jobs=N_JOBS)(
            delayed(self.calculate_cv_score_batch)(batch) for batch in df_splits
        )
        
        # Combine results
        self.filtered_data = pd.concat(processed_batches, ignore_index=False)
        
        # Mark candidates based on combined score and anomaly detection
        # Very loose threshold for completeness
        self.filtered_data['is_cv_candidate'] = (
            (self.filtered_data['cv_score'] > 0.5) |  # Score-based
            (self.filtered_data.index.isin(anomaly_indices))  # Anomaly-based
        )
        
        # Sort candidates by CV score
        self.cv_candidates = self.filtered_data[self.filtered_data['is_cv_candidate']].sort_values(
            by='cv_score', ascending=False
        )
        
        print(f"Identified {len(self.cv_candidates)} CV candidates")
        return True
    
    def train_classifier(self, known_cv_file=None):
        """
        Train a classifier to identify CVs if known examples are available.
        
        Parameters:
        -----------
        known_cv_file : str, optional
            Path to file containing known CVs for training
        """
        if known_cv_file is None:
            print("No known CV file provided, skipping classifier training")
            return False
            
        print(f"Training CV classifier using known examples from {known_cv_file}")
        
        try:
            # Load known CVs
            if known_cv_file.endswith('.fits'):
                with fits.open(known_cv_file) as hdul:
                    known_cvs = Table(hdul[1].data).to_pandas()
            elif known_cv_file.endswith('.csv'):
                known_cvs = pd.read_csv(known_cv_file)
            else:
                raise ValueError(f"Unsupported file format: {known_cv_file}")
                
            # Extract features for modeling
            X_scaled, indices = self.extract_cv_features()
            if X_scaled is None:
                return False
                
            # Create labels based on known CVs
            # Assuming known_cvs has a column 'sourceid' or 'primvs_id'
            id_col = 'sourceid' if 'sourceid' in known_cvs.columns else 'primvs_id'
            known_ids = set(known_cvs[id_col])
            
            # Create labels
            y = self.filtered_data.index.isin(known_ids).astype(int)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train classifier
            classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42
            )
            
            classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = classifier.predict(X_test)
            print(classification_report(y_test, y_pred))
            
            # Apply to all data
            cv_probs = classifier.predict_proba(X_scaled)[:, 1]
            self.filtered_data['cv_prob'] = pd.Series(cv_probs, index=indices)
            
            # Update candidates
            threshold = 0.5  # Adjust as needed
            self.filtered_data['is_cv_candidate'] = (
                self.filtered_data['is_cv_candidate'] | 
                (self.filtered_data['cv_prob'] > threshold)
            )
            
            # Update candidate list
            self.cv_candidates = self.filtered_data[self.filtered_data['is_cv_candidate']].sort_values(
                by=['cv_prob', 'cv_score'], ascending=False
            )
            
            print(f"Updated candidate list now contains {len(self.cv_candidates)} sources")
            return True
            
        except Exception as e:
            print(f"Error in classifier training: {str(e)}")
            return False
    
    def plot_candidates(self, n_examples=10):
        """
        Generate plots to visualize the CV candidates
        
        Parameters:
        -----------
        n_examples : int
            Number of example candidates to plot
        """
        if self.cv_candidates is None or len(self.cv_candidates) == 0:
            print("No CV candidates to plot")
            return
            
        print("Generating plots for CV candidates...")
        
        # Plot 1: Period vs Amplitude (Bailey diagram)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            np.log10(self.filtered_data['true_period']), 
            self.filtered_data['true_amplitude'],
            alpha=0.3, s=5, color='gray', label='All sources'
        )
        plt.scatter(
            np.log10(self.cv_candidates['true_period']), 
            self.cv_candidates['true_amplitude'],
            alpha=0.7, s=10, color='red', label='CV candidates'
        )
        plt.xlabel('log10(Period) [days]')
        plt.ylabel('Amplitude [mag]')
        plt.title('Bailey Diagram of CV Candidates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cv_bailey_diagram.png", dpi=300)
        plt.close()
        
        # Plot 2: Spatial distribution
        if all(col in self.filtered_data.columns for col in ['l', 'b']):
            plt.figure(figsize=(12, 6))
            plt.scatter(
                self.filtered_data['l'], 
                self.filtered_data['b'],
                alpha=0.2, s=2, color='gray', label='All sources'
            )
            plt.scatter(
                self.cv_candidates['l'], 
                self.cv_candidates['b'],
                alpha=0.7, s=5, color='red', label='CV candidates'
            )
            plt.xlabel('Galactic Longitude (l)')
            plt.ylabel('Galactic Latitude (b)')
            plt.title('Spatial Distribution of CV Candidates')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/cv_spatial_distribution.png", dpi=300)
            plt.close()
        
        # Plot 3: CV Score Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.filtered_data['cv_score'], bins=50, alpha=0.5, label='All sources')
        plt.hist(self.cv_candidates['cv_score'], bins=50, alpha=0.5, label='CV candidates')
        plt.xlabel('CV Score')
        plt.ylabel('Number of Sources')
        plt.title('Distribution of CV Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cv_score_distribution.png", dpi=300)
        plt.close()
        
        # Plot 4: Example candidates (if we had light curves)
        # This would require accessing light curve data
        print("Light curve plotting not implemented - would require light curve data")
        
    def save_candidates(self):
        """Save the CV candidates to a file"""
        if self.cv_candidates is None or len(self.cv_candidates) == 0:
            print("No CV candidates to save")
            return
            
        # Save as CSV
        csv_path = f"{self.output_dir}/cv_candidates.csv"
        self.cv_candidates.to_csv(csv_path, index=False)
        print(f"Saved {len(self.cv_candidates)} CV candidates to {csv_path}")
        
        # Save as FITS for compatibility with astronomical tools
        fits_path = f"{self.output_dir}/cv_candidates.fits"
        t = Table.from_pandas(self.cv_candidates)
        t.write(fits_path, overwrite=True)
        print(f"Saved {len(self.cv_candidates)} CV candidates to {fits_path}")
        
        return True
        
    def run_pipeline(self, known_cv_file=None):
        """
        Run the complete CV finder pipeline
        
        Parameters:
        -----------
        known_cv_file : str, optional
            Path to file containing known CVs for training
        
        Returns:
        --------
        bool
            True if pipeline completed successfully
        """
        # 1. Load data
        if not self.load_data():
            return False
            
        # 2. Merge datasets
        if not self.merge_datasets():
            return False
            
        # 3. Apply initial filters
        if not self.apply_loose_filters():
            return False
            
        # 4. Apply CV knowledge to find candidates
        if not self.apply_cv_knowledge():
            return False
            
        # 5. Train classifier if known examples available
        if known_cv_file is not None:
            self.train_classifier(known_cv_file)
            
        # 6. Generate plots
        self.plot_candidates()
        
        # 7. Save candidates
        self.save_candidates()
        
        print("CV finder pipeline completed successfully!")
        return True


# Example usage (to be customized with your file paths)
if __name__ == "__main__":
    # File paths
    primvs_file = "/path/to/PRIMVS_P.fits"
    tess_match_file = "/path/to/primvs_tess_crossmatch.fits"
    output_dir = "./cv_results"
    
    # Optional: file with known CVs for training
    known_cv_file = None  # "/path/to/known_cvs.csv" if available
    
    # Create and run the CV finder
    finder = CVFinder(primvs_file, tess_match_file, output_dir)
    finder.run_pipeline(known_cv_file)