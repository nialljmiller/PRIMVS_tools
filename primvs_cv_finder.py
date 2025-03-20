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

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

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
            'l', 'b'
        ]
        
        # Initialize data containers
        self.primvs_data = None
        self.cv_candidates = None
        self.model = None
    
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
    
    def extract_features(self):
        """Extract and normalize features for CV detection."""
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return None
        
        print("Extracting features for CV detection...")
        
        # Filter feature list to include only available columns
        available_features = [f for f in self.cv_features if f in self.filtered_data.columns]
        
        if len(available_features) < 5:
            print(f"Warning: Only {len(available_features)} features available: {available_features}")
            print("This may affect classification accuracy.")
        else:
            print(f"Using {len(available_features)} features for classification")
        
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
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        self.features = X
        self.scaled_features = X_scaled
        self.feature_names = available_features
        
        return X_scaled
    
    def detect_anomalies(self):
        """Use Isolation Forest to identify anomalous sources that might be CVs."""
        if not hasattr(self, 'scaled_features') or len(self.scaled_features) == 0:
            print("No features available. Call extract_features() first.")
            return None
        
        print("Detecting anomalous sources with Isolation Forest...")
        
        # Initialize Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.05,  # Expect 5% of sources to be unusual
            random_state=42,
            n_jobs=-1  # Use all processors
        )
        
        # Fit and predict
        anomaly_scores = iso_forest.fit_predict(self.scaled_features)
        
        # Get anomaly scores (decision function)
        decision_scores = iso_forest.decision_function(self.scaled_features)
        
        # Add anomaly scores to filtered data
        self.filtered_data['anomaly_score'] = decision_scores
        
        # Mark anomalies (where prediction is -1)
        anomaly_mask = anomaly_scores == -1
        self.filtered_data['is_anomaly'] = anomaly_mask
        
        n_anomalies = anomaly_mask.sum()
        anomaly_percent = 100 * n_anomalies / len(self.filtered_data)
        print(f"Identified {n_anomalies} anomalous sources ({anomaly_percent:.2f}%)")
        
        return n_anomalies > 0
    
    def calculate_cv_score(self):
        """
        Calculate a CV score based on domain knowledge of CV characteristics.
        This combines multiple heuristics based on CV properties.
        """
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return False
        
        print("Calculating CV scores based on domain knowledge...")
        
        # Make a copy of the filtered data
        data = self.filtered_data.copy()
        
        # 1. Period-based score: Higher for shorter periods
        # CVs typically have periods in the hours range
        data['period_score'] = np.clip(1.0 - np.log10(data['true_period']) / 2.0, 0, 1)
        
        # 2. Amplitude-based score: Higher for larger amplitudes
        # Scale to 0-1 using a sigmoid function centered at 0.5 mag
        data['amplitude_score'] = 1 / (1 + np.exp(-2 * (data['true_amplitude'] - 0.5)))
        
        # 3. Skewness-based score: CV light curves often show asymmetry
        if 'skew' in data.columns:
            # Absolute skewness matters (positive or negative)
            data['skew_score'] = np.clip(np.abs(data['skew']) / 2.0, 0, 1)
        else:
            data['skew_score'] = 0.5  # Default if not available
        
        # 4. Color-based score: CVs often have colors compatible with hot components
        if all(col in data.columns for col in ['J-K', 'H-K']):
            # This is a simplified approximation - would need refinement
            # Most CVs have relatively blue colors
            data['color_score'] = 1 - np.clip((data['J-K'] + data['H-K']) / 2, 0, 1)
        else:
            data['color_score'] = 0.5  # Default if colors not available
        
        # 5. Periodicity quality: Higher for more reliable periods
        data['fap_score'] = 1 - np.clip(data['best_fap'], 0, 1)
        
        # Combine scores with different weights
        data['cv_score'] = (
            0.30 * data['period_score'] +
            0.30 * data['amplitude_score'] +
            0.15 * data['skew_score'] +
            0.15 * data['color_score'] +
            0.10 * data['fap_score']
        )
        
        # Update the filtered data
        self.filtered_data = data
        
        # Create a histogram of the scores to help set thresholds
        plt.figure(figsize=(10, 6))
        plt.hist(data['cv_score'], bins=50, alpha=0.7)
        plt.axvline(0.5, color='r', linestyle='--', label='Default threshold (0.5)')
        plt.axvline(0.7, color='g', linestyle='--', label='High confidence threshold (0.7)')
        plt.xlabel('CV Score')
        plt.ylabel('Number of Sources')
        plt.title('Distribution of CV Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cv_score_distribution.png'), dpi=300)
        plt.close()
        
        return True
    
    def load_known_cvs(self):
        """
        Load known CVs for training the classifier.
        This function can load from various formats with flexible matching.
        """
        if self.known_cv_file is None:
            print("No known CV file provided. Skipping.")
            return None
        
        print(f"Loading known CVs from {self.known_cv_file}...")
        
        try:
            # Determine file type and load accordingly
            if self.known_cv_file.endswith('.fits'):
                with fits.open(self.known_cv_file) as hdul:
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
                    break
            
            if id_col is None:
                print("Could not find ID column in known CV file.")
                return None
            
            # Extract IDs as a set for faster lookup
            known_ids = set(cv_data[id_col])
            print(f"Extracted {len(known_ids)} unique CV IDs")
            
            return known_ids
            
        except Exception as e:
            print(f"Error loading known CVs: {str(e)}")
            return None
    
    def train_classifier(self):
        """
        Train an XGBoost classifier to identify CVs.
        If known CVs are provided, they are used as positive examples.
        Otherwise, high-scoring candidates are used as positives.
        """
        if not hasattr(self, 'scaled_features') or len(self.scaled_features) == 0:
            print("No features available. Call extract_features() first.")
            return False
        
        print("Training CV classifier...")
        
        # Get known CVs if available
        known_ids = self.load_known_cvs()
        
        # Prepare training data
        X = self.scaled_features
        
        if known_ids is not None:
            # Create labels based on known CVs
            y = self.filtered_data.index.isin(known_ids).astype(int)
            print(f"Using {sum(y)} known CVs as positive examples")
            
            # Check if we have enough positive examples
            if sum(y) < 10:
                print("Warning: Very few positive examples. Classification may be unreliable.")
                if sum(y) < 3:
                    print("Error: Insufficient positive examples for training. Aborting classifier.")
                    return False
        else:
            # No known CVs, use high-scoring candidates as positives
            if 'cv_score' not in self.filtered_data.columns:
                print("No cv_score available. Call calculate_cv_score() first.")
                return False
                
            threshold = 0.7  # Use high-scoring candidates
            y = (self.filtered_data['cv_score'] > threshold).astype(int)
            print(f"Using {sum(y)} high-scoring candidates (score > {threshold}) as positive examples")
            
            if sum(y) < 10:
                print("Warning: Very few positive examples. Using anomaly detection instead.")
                if 'is_anomaly' in self.filtered_data.columns:
                    y = self.filtered_data['is_anomaly'].astype(int)
                    print(f"Using {sum(y)} anomalous sources as positive examples")
                else:
                    print("Error: No suitable training examples. Aborting classifier.")
                    return False
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Create class weights to handle imbalance
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
        
        # Train XGBoost classifier
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            objective='binary:logistic',
            scale_pos_weight=pos_weight,  # Handle class imbalance
            n_jobs=-1,  # Use all processors
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save feature importances
        feature_imp = pd.DataFrame(
            sorted(zip(model.feature_importances_, X.columns)),
            columns=['Importance', 'Feature']
        ).sort_values('Importance', ascending=False)
        
        print("\nTop 10 important features:")
        print(feature_imp.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp.head(20))
        plt.title('Feature Importance for CV Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
        # Save model
        model_path = os.path.join(self.output_dir, 'cv_classifier.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        self.model = model
        return True
    
    def run_classifier(self):
        """
        Run the trained classifier on all filtered data to identify CV candidates.
        If no model is available, this will train one first.
        """
        if not hasattr(self, 'model') or self.model is None:
            print("No trained model available. Training now...")
            if not self.train_classifier():
                print("Could not train classifier. Using simplified selection.")
                return self.select_candidates_simple()
        
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
    
    def select_candidates_simple(self):
        """
        Simplified CV candidate selection based on heuristics.
        Used as a fallback when ML classification is not possible.
        """
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return False
        
        print("Selecting CV candidates using simplified heuristics...")
        
        # Selection criteria
        period_hours = self.filtered_data['true_period'] * 24.0
        
        # 1. Strong period candidates: systems with periods in the 1-10 hour range
        period_mask = (period_hours >= 1.0) & (period_hours <= 10.0)
        
        # 2. High amplitude candidates
        amp_mask = self.filtered_data['true_amplitude'] > 0.2
        
        # 3. Reliable periods (low FAP)
        fap_mask = self.filtered_data['best_fap'] < 0.2
        
        # Combine criteria (any of these makes a candidate)
        candidate_mask = period_mask | amp_mask | fap_mask
        
        # Add selection flags
        self.filtered_data['is_period_candidate'] = period_mask
        self.filtered_data['is_amplitude_candidate'] = amp_mask
        self.filtered_data['is_reliability_candidate'] = fap_mask
        self.filtered_data['is_cv_candidate'] = candidate_mask
        
        # Select candidates
        self.cv_candidates = self.filtered_data[candidate_mask].copy()
        
        print(f"Selected {len(self.cv_candidates)} CV candidates:")
        print(f"  - Period-based (1-10 hours): {period_mask.sum()}")
        print(f"  - Amplitude-based (>0.2 mag): {amp_mask.sum()}")
        print(f"  - Reliability-based (FAP<0.2): {fap_mask.sum()}")
        
        return True
    
    def select_candidates(self):
        """
        Select final CV candidates using all available methods.
        This combines ML classification, anomaly detection, and heuristic scores.
        """
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return False
        
        print("Selecting final CV candidates...")
        
        # Define selection criteria based on available methods
        criteria = []
        
        # 1. ML-based selection
        if 'cv_prob' in self.filtered_data.columns:
            ml_mask = self.filtered_data['cv_prob'] > 0.5
            criteria.append(('ML-based (prob > 0.5)', ml_mask))
        
        # 2. Score-based selection
        if 'cv_score' in self.filtered_data.columns:
            score_mask = self.filtered_data['cv_score'] > 0.6
            criteria.append(('Score-based (score > 0.6)', score_mask))
        
        # 3. Anomaly-based selection
        if 'is_anomaly' in self.filtered_data.columns:
            anomaly_mask = self.filtered_data['is_anomaly']
            criteria.append(('Anomaly-based', anomaly_mask))
            
        # If no advanced criteria are available, use simplified selection
        if not criteria:
            print("No advanced selection criteria available. Using simplified selection.")
            return self.select_candidates_simple()
        
        # Combine all criteria (any match makes a candidate)
        combined_mask = np.zeros(len(self.filtered_data), dtype=bool)
        
        for name, mask in criteria:
            combined_mask = combined_mask | mask
            print(f"  - {name}: {mask.sum()} candidates")
        
        # Add candidate flag
        self.filtered_data['is_cv_candidate'] = combined_mask
        
        # Create final candidate set
        self.cv_candidates = self.filtered_data[combined_mask].copy()
        
        # Add confidence level based on how many criteria were met
        n_criteria = len(criteria)
        if n_criteria > 1:
            confidence = np.zeros(len(self.cv_candidates))
            
            for i, (name, mask) in enumerate(criteria):
                # Sources that match this criterion
                matches = self.cv_candidates.index.isin(self.filtered_data[mask].index)
                confidence[matches] += 1 / n_criteria
                
            self.cv_candidates['confidence'] = confidence
        else:
            # With only one criterion, use its score directly
            name, mask = criteria[0]
            if name.startswith('ML-based'):
                self.cv_candidates['confidence'] = self.cv_candidates['cv_prob']
            elif name.startswith('Score-based'):
                self.cv_candidates['confidence'] = self.cv_candidates['cv_score']
            else:
                # Default confidence
                self.cv_candidates['confidence'] = 0.7
        
        print(f"Selected {len(self.cv_candidates)} CV candidates")
        
        # Sort by confidence
        self.cv_candidates = self.cv_candidates.sort_values('confidence', ascending=False)
        
        return True
    
    def plot_candidates(self):
        """Generate plots to visualize the CV candidates."""
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates to plot.")
            return
        
        print("Generating plots for CV candidates...")
        
        # 1. Period-Amplitude diagram (Bailey diagram)
        plt.figure(figsize=(10, 6))
        
        # Plot all filtered sources as background
        if hasattr(self, 'filtered_data'):
            plt.scatter(
                np.log10(self.filtered_data['true_period'] * 24),  # Convert to hours
                self.filtered_data['true_amplitude'],
                alpha=0.2, s=5, color='gray', label='All filtered sources'
            )
        
        # Plot candidates
        plt.scatter(
            np.log10(self.cv_candidates['true_period'] * 24),  # Convert to hours
            self.cv_candidates['true_amplitude'],
            alpha=0.7, 
            s=10, 
            c=self.cv_candidates['confidence'] if 'confidence' in self.cv_candidates else 'red',
            cmap='viridis',
            label='CV candidates'
        )
        
        plt.colorbar(label='Confidence' if 'confidence' in self.cv_candidates else None)
        plt.xlabel('log₁₀(Period) [hours]')
        plt.ylabel('Amplitude [mag]')
        plt.title('Bailey Diagram of CV Candidates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cv_bailey_diagram.png'), dpi=300)
        plt.close()
        
        # 2. Spatial distribution
        if all(col in self.cv_candidates.columns for col in ['l', 'b']):
            plt.figure(figsize=(12, 6))
            
            # Plot all filtered sources as background
            if hasattr(self, 'filtered_data'):
                plt.scatter(
                    self.filtered_data['l'],
                    self.filtered_data['b'],
                    alpha=0.1, s=1, color='gray', label='All filtered sources'
                )
            
            # Plot candidates
            plt.scatter(
                self.cv_candidates['l'],
                self.cv_candidates['b'],
                alpha=0.7, 
                s=10, 
                c=self.cv_candidates['confidence'] if 'confidence' in self.cv_candidates else 'red',
                cmap='viridis',
                label='CV candidates'
            )
            
            plt.colorbar(label='Confidence' if 'confidence' in self.cv_candidates else None)
            plt.xlabel('Galactic Longitude (l)')
            plt.ylabel('Galactic Latitude (b)')
            plt.title('Spatial Distribution of CV Candidates')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'cv_spatial_distribution.png'), dpi=300)
            plt.close()
        
        # 3. Period distribution
        plt.figure(figsize=(10, 6))
        period_hours = self.cv_candidates['true_period'] * 24.0
        
        plt.hist(period_hours, bins=50, alpha=0.7)
        plt.axvline(2.0, color='r', linestyle='--', label='2 hours')
        plt.axvline(3.0, color='g', linestyle='--', label='3 hours (period gap lower bound)')
        plt.axvline(4.0, color='b', linestyle='--', label='4 hours (period gap upper bound)')
        
        plt.xlabel('Period (hours)')
        plt.ylabel('Number of Candidates')
        plt.title('Period Distribution of CV Candidates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cv_period_distribution.png'), dpi=300)
        plt.close()
        
        # 4. Period-FAP diagram (to identify potentially unreliable periods)
        plt.figure(figsize=(10, 6))
        
        # Scatter plot colored by confidence
        plt.scatter(
            period_hours,
            self.cv_candidates['best_fap'],
            alpha=0.7, 
            s=10, 
            c=self.cv_candidates['confidence'] if 'confidence' in self.cv_candidates else 'red',
            cmap='viridis'
        )
        
        plt.colorbar(label='Confidence' if 'confidence' in self.cv_candidates else None)
        plt.xlabel('Period (hours)')
        plt.ylabel('False Alarm Probability')
        plt.title('Period vs FAP for CV Candidates')
        plt.yscale('log')  # Log scale for FAP
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cv_period_fap.png'), dpi=300)
        plt.close()
        
        # 5. Color-color diagram (if colors available)
        color_cols = ['J-K', 'H-K']
        if all(col in self.cv_candidates.columns for col in color_cols):
            plt.figure(figsize=(10, 6))
            
            # Plot all filtered sources as background
            if hasattr(self, 'filtered_data'):
                plt.scatter(
                    self.filtered_data['J-K'],
                    self.filtered_data['H-K'],
                    alpha=0.1, s=1, color='gray', label='All filtered sources'
                )
            
            # Plot candidates
            plt.scatter(
                self.cv_candidates['J-K'],
                self.cv_candidates['H-K'],
                alpha=0.7, 
                s=10, 
                c=self.cv_candidates['confidence'] if 'confidence' in self.cv_candidates else 'red',
                cmap='viridis',
                label='CV candidates'
            )
            
            plt.colorbar(label='Confidence' if 'confidence' in self.cv_candidates else None)
            plt.xlabel('J-K Color')
            plt.ylabel('H-K Color')
            plt.title('Color-Color Diagram of CV Candidates')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'cv_color_diagram.png'), dpi=300)
            plt.close()
    
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
        
        # Generate summary statistics
        self.generate_summary()
        
        return True
    
    def generate_summary(self):
        """Generate a summary report of the CV candidates."""
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates for summary.")
            return
        
        print("Generating summary report...")
        
        # Calculate period statistics in hours
        period_hours = self.cv_candidates['true_period'] * 24.0
        
        # Period distribution by range
        p1 = (period_hours < 2).sum()
        p2 = ((period_hours >= 2) & (period_hours < 3)).sum()
        p3 = ((period_hours >= 3) & (period_hours < 4)).sum()
        p4 = ((period_hours >= 4) & (period_hours < 5)).sum()
        p5 = ((period_hours >= 5) & (period_hours < 10)).sum()
        p6 = (period_hours >= 10).sum()
        
        # Create summary file
        summary_path = os.path.join(self.output_dir, 'cv_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("PRIMVS CV Candidate Summary\n")
            f.write("==========================\n\n")
            
            f.write(f"Total candidates: {len(self.cv_candidates)}\n\n")
            
            # Period distribution
            f.write("Period Distribution:\n")
            f.write(f"  < 2 hours:  {p1} ({100*p1/len(self.cv_candidates):.1f}%)\n")
            f.write(f"  2-3 hours:  {p2} ({100*p2/len(self.cv_candidates):.1f}%)\n")
            f.write(f"  3-4 hours:  {p3} ({100*p3/len(self.cv_candidates):.1f}%)\n")
            f.write(f"  4-5 hours:  {p4} ({100*p4/len(self.cv_candidates):.1f}%)\n")
            f.write(f"  5-10 hours: {p5} ({100*p5/len(self.cv_candidates):.1f}%)\n")
            f.write(f"  > 10 hours: {p6} ({100*p6/len(self.cv_candidates):.1f}%)\n\n")
            
            # Amplitude statistics
            f.write("Amplitude Statistics:\n")
            f.write(f"  Minimum: {self.cv_candidates['true_amplitude'].min():.2f} mag\n")
            f.write(f"  Maximum: {self.cv_candidates['true_amplitude'].max():.2f} mag\n")
            f.write(f"  Median:  {self.cv_candidates['true_amplitude'].median():.2f} mag\n")
            f.write(f"  Mean:    {self.cv_candidates['true_amplitude'].mean():.2f} mag\n\n")
            
            # False alarm probability
            f.write("False Alarm Probability:\n")
            f.write(f"  Minimum: {self.cv_candidates['best_fap'].min():.4f}\n")
            f.write(f"  Maximum: {self.cv_candidates['best_fap'].max():.4f}\n")
            f.write(f"  Median:  {self.cv_candidates['best_fap'].median():.4f}\n")
            f.write(f"  Mean:    {self.cv_candidates['best_fap'].mean():.4f}\n\n")
            
            # Confidence if available
            if 'confidence' in self.cv_candidates.columns:
                f.write("Confidence Statistics:\n")
                f.write(f"  Minimum: {self.cv_candidates['confidence'].min():.2f}\n")
                f.write(f"  Maximum: {self.cv_candidates['confidence'].max():.2f}\n")
                f.write(f"  Median:  {self.cv_candidates['confidence'].median():.2f}\n")
                f.write(f"  Mean:    {self.cv_candidates['confidence'].mean():.2f}\n\n")
            
            # Top 20 candidates
            f.write("Top 20 CV Candidates:\n")
            
            # Sort by confidence if available, otherwise by FAP
            if 'confidence' in self.cv_candidates.columns:
                top_candidates = self.cv_candidates.sort_values('confidence', ascending=False).head(20)
                sort_col = 'confidence'
            else:
                top_candidates = self.cv_candidates.sort_values('best_fap').head(20)
                sort_col = 'best_fap'
                
            for i, (_, cand) in enumerate(top_candidates.iterrows()):
                source_id = cand['sourceid']
                period_hr = cand['true_period'] * 24.0
                amp = cand['true_amplitude']
                fap = cand['best_fap']
                conf = cand[sort_col]
                
                f.write(f"  {i+1}. Source ID: {source_id}, ")
                f.write(f"Period: {period_hr:.2f} hr, ")
                f.write(f"Amplitude: {amp:.2f} mag, ")
                f.write(f"FAP: {fap:.4f}, ")
                f.write(f"{sort_col.capitalize()}: {conf:.4f}\n")
                
        print(f"Summary saved to {summary_path}")
    
    def run_pipeline(self):
        """Run the complete CV finder pipeline."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("RUNNING PRIMVS CV FINDER PIPELINE")
        print("="*80 + "\n")
        
        # Step 1: Load PRIMVS data
        if not self.load_primvs_data():
            print("Failed to load PRIMVS data. Aborting.")
            return False
        
        # Step 2: Apply initial filters
        if not self.apply_initial_filters():
            print("Failed to apply initial filters. Aborting.")
            return False
        
        # Step 3: Extract features for classification
        if self.extract_features() is None:
            print("Failed to extract features. Aborting.")
            return False
        
        # Step 4: Calculate CV score
        if not self.calculate_cv_score():
            print("Failed to calculate CV scores. Continuing with limited features.")
        
        # Step 5: Detect anomalies
        if not self.detect_anomalies():
            print("Anomaly detection failed or found no anomalies. Continuing without anomaly features.")
        
        # Step 6: Train classifier if known CVs available
        if self.known_cv_file is not None:
            if not self.train_classifier():
                print("Classifier training failed. Continuing with heuristic selection.")
            else:
                # Step 7: Run classifier on all data
                if not self.run_classifier():
                    print("Classifier prediction failed. Continuing with heuristic selection.")
        
        # Step 8: Select final candidates
        if not self.select_candidates():
            print("Failed to select candidates. Aborting.")
            return False
        
        # Step 9: Plot candidates
        self.plot_candidates()
        
        # Step 10: Save candidates
        if not self.save_candidates():
            print("Failed to save candidates.")
            return False
        
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


    primvs_file = '../PRIMVS/PRIMVS_CV_cand.fits'
    output_dir = "../PRIMVS/"
    known_cvs = "../PRIMVS/PRIMVS_CV.fits:"
    period_limit = 10.0  # Very generous upper limit (days)
    amplitude_limit = 0.03  # Very low amplitude threshold (mag)
    fap_limit = 0.7  # Permissive FAP threshold

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
        print(f"CV finder completed successfully. Results in {args.output_dir}")
        return 0
    else:
        print("CV finder failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())    