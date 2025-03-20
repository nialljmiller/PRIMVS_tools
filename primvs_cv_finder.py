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
            #'l', 'b'
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
        """Extract and normalize features for CV detection, including contrastive curves embeddings."""
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return None
        
        print("Extracting features for CV detection...")
        
        # Filter feature list to include only available columns
        available_features = [f for f in self.cv_features if f in self.filtered_data.columns]
        
        # Extract contrastive curves embeddings if available
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in self.filtered_data.columns]
        
        if embedding_features:
            print(f"Found {len(embedding_features)} contrastive curves embedding features")
            available_features.extend(embedding_features)
        else:
            print("No contrastive curves embeddings found in the data")
        
        # Extract feature matrix
        X = self.filtered_data[available_features].copy()
        
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
        
        return X_scaled







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





    def train_two_stage_classifier(self):
        """
        Train a two-stage classifier that integrates traditional astronomical features with 
        contrastive curve embeddings for robust CV identification in time-domain survey data.
        
        This implementation utilizes the complete known CV dataset structure for comprehensive 
        training, properly handling feature alignment between training and candidate sets.
        """
        if not hasattr(self, 'scaled_features') or len(self.scaled_features) == 0:
            print("No features available. Call extract_features() first.")
            return False
        
        print("Training two-stage CV classifier...")
        
        # Load the complete known CV dataset
        if self.known_cv_file is None:
            print("Error: No known CV file provided. Classifier aborted.")
            return False
        
        try:
            # Load complete known CV dataset with appropriate format handling
            if self.known_cv_file.endswith('.fits'):
                with fits.open(self.known_cv_file) as hdul:
                    known_cv_data = Table(hdul[1].data).to_pandas()
            elif self.known_cv_file.endswith('.csv'):
                known_cv_data = pd.read_csv(self.known_cv_file)
            else:
                print(f"Unsupported file format: {self.known_cv_file}")
                return False
            
            print(f"Loaded {len(known_cv_data)} known CV records for training")
            
            # Identify feature categories
            cc_embedding_cols = [str(i) for i in range(64)]
            embedding_features = [col for col in cc_embedding_cols if col in self.scaled_features.columns]
            traditional_features = [col for col in self.scaled_features.columns if col not in embedding_features]
            
            print(f"Traditional features: {len(traditional_features)}")
            print(f"Embedding features: {len(embedding_features)}")
            
            # Prepare training and validation datasets
            
            # Ensure necessary columns exist in both datasets
            required_features = traditional_features + embedding_features
            known_cv_features = [f for f in required_features if f in known_cv_data.columns]
            
            # Extract common features between datasets
            candidate_features = self.scaled_features[known_cv_features].copy()
            known_cv_subset = known_cv_data[known_cv_features].copy()
            
            # Handle missing values in known CV dataset
            for col in known_cv_subset.columns:
                if known_cv_subset[col].isnull().any():
                    if known_cv_subset[col].dtype in [np.float64, np.int64]:
                        known_cv_subset[col] = known_cv_subset[col].fillna(known_cv_subset[col].median())
                    else:
                        known_cv_subset[col] = known_cv_subset[col].fillna(
                            known_cv_subset[col].mode()[0] if len(known_cv_subset[col].mode()) > 0 else 0
                        )
            
            # Scale known CV features using the same scaler as candidates
            for col in traditional_features:
                if col in known_cv_subset.columns:
                    # Reshape for scikit-learn compatibility
                    values = known_cv_subset[col].values.reshape(-1, 1)
                    scaled_values = self.scaler.transform(values)
                    known_cv_subset[col] = scaled_values.flatten()
            
            # Create positive examples (known CVs) and negative examples (candidates)
            X_positive = known_cv_subset.values
            y_positive = np.ones(len(X_positive))
            
            # For negative examples, use all filtered data (or a representative sample if too large)
            max_neg_samples = min(len(candidate_features), len(X_positive) * 5)  # Cap negative examples
            if len(candidate_features) > max_neg_samples:
                # Random sampling without replacement
                neg_indices = np.random.choice(len(candidate_features), max_neg_samples, replace=False)
                X_negative = candidate_features.values[neg_indices]
            else:
                X_negative = candidate_features.values
            y_negative = np.zeros(len(X_negative))
            
            # Combine datasets
            X_combined = np.vstack([X_positive, X_negative])
            y_combined = np.concatenate([y_positive, y_negative])
            
            print(f"Training dataset constructed: {len(y_positive)} positive examples, {len(y_negative)} negative examples")
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_combined, y_combined, test_size=0.25, random_state=42, stratify=y_combined
            )
            
            # Create class weights to account for potential imbalance
            pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train) if np.sum(y_train) > 0 else 1.0
            
            # Separate features by type
            trad_indices = [i for i, col in enumerate(known_cv_features) if col in traditional_features]
            emb_indices = [i for i, col in enumerate(known_cv_features) if col in embedding_features]
            
            # Extract feature subsets for two-stage classification
            X_trad_train = X_train[:, trad_indices] if trad_indices else None
            X_trad_val = X_val[:, trad_indices] if trad_indices else None
            
            X_emb_train = X_train[:, emb_indices] if emb_indices else None
            X_emb_val = X_val[:, emb_indices] if emb_indices else None
            
            # STAGE 1: Train model with traditional features
            print("\nStage 1: Training model with traditional features...")
            
            if X_trad_train is not None and X_trad_train.shape[1] > 0:
                model_trad = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    scale_pos_weight=pos_weight,
                    n_jobs=-1,
                    random_state=42
                )
                
                model_trad.fit(X_trad_train, y_train)
                
                # Evaluate traditional model
                y_pred_trad = model_trad.predict(X_trad_val)
                prob_trad = model_trad.predict_proba(X_trad_val)[:, 1]
                
                print("\nTraditional Feature Model Performance:")
                print(classification_report(y_val, y_pred_trad))
                
                # Get feature importance for traditional model
                trad_feature_names = [known_cv_features[i] for i in trad_indices]
                trad_importance = pd.DataFrame({
                    'Feature': trad_feature_names,
                    'Importance': model_trad.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print("\nTop traditional features by importance:")
                for i, (_, row) in enumerate(trad_importance.head(5).iterrows()):
                    print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
            else:
                print("Insufficient traditional features available. Skipping this stage.")
                model_trad = None
            
            # STAGE 2: Train model with embedding features if available
            if X_emb_train is not None and X_emb_train.shape[1] > 0:
                print("\nStage 2: Training model with embedding features...")
                
                # Apply PCA to reduce dimensionality of embeddings while preserving variance
                from sklearn.decomposition import PCA
                
                # Determine optimal number of components (explaining ~90% variance)
                pca = PCA().fit(X_emb_train)
                explained_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = min(np.argmax(explained_variance >= 0.9) + 1, len(explained_variance))
                print(f"Using {n_components} PCA components (explaining {explained_variance[n_components-1]:.2%} variance)")
                
                # Apply PCA transformation
                pca = PCA(n_components=n_components)
                X_emb_train_pca = pca.fit_transform(X_emb_train)
                X_emb_val_pca = pca.transform(X_emb_val)
                
                # Train embedding model
                model_emb = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    scale_pos_weight=pos_weight,
                    n_jobs=-1,
                    random_state=42
                )
                
                model_emb.fit(X_emb_train_pca, y_train)
                
                # Evaluate embedding model
                y_pred_emb = model_emb.predict(X_emb_val_pca)
                prob_emb = model_emb.predict_proba(X_emb_val_pca)[:, 1]
                
                print("\nEmbedding Feature Model Performance:")
                print(classification_report(y_val, y_pred_emb))
                
                # STAGE 3: Train meta-model (stacking) if both models available
                if model_trad is not None:
                    print("\nStage 3: Training meta-model to combine predictions...")
                    
                    # Create meta-features (predictions from base models)
                    meta_features_train = np.column_stack([
                        model_trad.predict_proba(X_trad_train)[:, 1],
                        model_emb.predict_proba(X_emb_train_pca)[:, 1]
                    ])
                    
                    meta_features_val = np.column_stack([
                        prob_trad,
                        prob_emb
                    ])
                    
                    # Train meta-model
                    meta_model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=2,
                        learning_rate=0.1,
                        objective='binary:logistic',
                        scale_pos_weight=pos_weight,
                        n_jobs=-1,
                        random_state=42
                    )
                    
                    meta_model.fit(meta_features_train, y_train)
                    
                    # Evaluate meta-model
                    y_pred_meta = meta_model.predict(meta_features_val)
                    
                    print("\nMeta-Model Performance:")
                    print(classification_report(y_val, y_pred_meta))
                    
                    # Calculate blend weights
                    blend_weights = meta_model.feature_importances_
                    print(f"\nModel blend weights: Traditional {blend_weights[0]:.2f}, Embedding {blend_weights[1]:.2f}")
                    
                    # STAGE 4: Apply ensemble model to all candidate data
                    print("\nApplying ensemble classifier to all candidate data...")
                    
                    # Extract features from all candidates in the same order as training
                    X_trad_full = self.scaled_features[[known_cv_features[i] for i in trad_indices]].values
                    X_emb_full = self.scaled_features[[known_cv_features[i] for i in emb_indices]].values
                    
                    # Apply PCA to full embedding dataset
                    X_emb_full_pca = pca.transform(X_emb_full)
                    
                    # Get predictions from both models
                    trad_probs = model_trad.predict_proba(X_trad_full)[:, 1]
                    emb_probs = model_emb.predict_proba(X_emb_full_pca)[:, 1]
                    
                    # Combine using meta-model
                    meta_features_full = np.column_stack([trad_probs, emb_probs])
                    final_probs = meta_model.predict_proba(meta_features_full)[:, 1]
                    
                    # Store models for later use
                    self.model_trad = model_trad
                    self.model_emb = model_emb
                    self.model_meta = meta_model
                    self.pca = pca
                    
                    # Add predictions to filtered data
                    self.filtered_data['cv_prob_trad'] = trad_probs
                    self.filtered_data['cv_prob_emb'] = emb_probs
                    self.filtered_data['cv_prob'] = final_probs
                    
                    # Store feature names for later interpretation
                    self.trad_feature_names = [known_cv_features[i] for i in trad_indices]
                    self.emb_feature_names = [known_cv_features[i] for i in emb_indices]
                    
                    # Save models
                    joblib.dump(model_trad, os.path.join(self.output_dir, 'cv_classifier_traditional.joblib'))
                    joblib.dump(model_emb, os.path.join(self.output_dir, 'cv_classifier_embedding.joblib'))
                    joblib.dump(meta_model, os.path.join(self.output_dir, 'cv_classifier_meta.joblib'))
                    joblib.dump(pca, os.path.join(self.output_dir, 'embedding_pca.joblib'))
                    
                    # Set the ensemble as the primary model
                    self.model = meta_model
                else:
                    # Only embedding model available
                    print("\nUsing only embedding-based model (no traditional features available)")
                    
                    # Get predictions for all data
                    X_emb_full = self.scaled_features[[known_cv_features[i] for i in emb_indices]].values
                    X_emb_full_pca = pca.transform(X_emb_full)
                    emb_probs = model_emb.predict_proba(X_emb_full_pca)[:, 1]
                    
                    # Add predictions to filtered data
                    self.filtered_data['cv_prob'] = emb_probs
                    
                    # Store feature names for later interpretation
                    self.emb_feature_names = [known_cv_features[i] for i in emb_indices]
                    
                    # Save model
                    joblib.dump(model_emb, os.path.join(self.output_dir, 'cv_classifier_embedding.joblib'))
                    joblib.dump(pca, os.path.join(self.output_dir, 'embedding_pca.joblib'))
                    
                    # Set as primary model
                    self.model = model_emb
                    self.pca = pca
            elif model_trad is not None:
                # Only traditional model available
                print("\nUsing only traditional-feature model (no embedding features available)")
                
                # Get predictions for all data
                X_trad_full = self.scaled_features[[known_cv_features[i] for i in trad_indices]].values
                trad_probs = model_trad.predict_proba(X_trad_full)[:, 1]
                
                # Add predictions to filtered data
                self.filtered_data['cv_prob'] = trad_probs
                
                # Store feature names for later interpretation
                self.trad_feature_names = [known_cv_features[i] for i in trad_indices]
                
                # Save model
                joblib.dump(model_trad, os.path.join(self.output_dir, 'cv_classifier_traditional.joblib'))
                
                # Set as primary model
                self.model = model_trad
            else:
                print("Error: Neither traditional nor embedding features could be processed. Classification failed.")
                return False
            
            # Generate visualization of probability distribution
            plt.figure(figsize=(10, 6))
            plt.hist(self.filtered_data['cv_prob'], bins=50, alpha=0.7)
            plt.axvline(0.5, color='r', linestyle='--', label='Default threshold (0.5)')
            plt.axvline(0.8, color='g', linestyle='--', label='High confidence (0.8)')
            plt.xlabel('CV Probability')
            plt.ylabel('Number of Sources')
            plt.title('Distribution of CV Probabilities')
            plt.yscale('log')  # Log scale for better visualization of distribution
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'cv_probability_distribution.png'), dpi=300)
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error in two-stage classifier training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False





    
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
    

    def select_candidates(self):
        """
        Select final CV candidates primarily using XGBoost classification results when available,
        with embedding proximity as a secondary factor to refine the selection.
        """
        if not hasattr(self, 'filtered_data') or len(self.filtered_data) == 0:
            print("No filtered data available. Call apply_initial_filters() first.")
            return False
        
        print("Selecting final CV candidates...")
        
        # Define selection criteria prioritizing XGBoost classification
        # Check if XGBoost classifier probabilities are available
        if 'cv_prob' in self.filtered_data.columns:
            print("Using XGBoost classifier probabilities for candidate selection")
            
            # Identify high confidence candidates
            high_confidence_threshold = 0.7
            high_confidence_mask = self.filtered_data['cv_prob'] >= high_confidence_threshold
            medium_confidence_threshold = 0.5
            medium_confidence_mask = (self.filtered_data['cv_prob'] >= medium_confidence_threshold) & (self.filtered_data['cv_prob'] < high_confidence_threshold)
            
            high_confidence_count = high_confidence_mask.sum()
            medium_confidence_count = medium_confidence_mask.sum()
            
            print(f"Found {high_confidence_count} high confidence candidates (prob >= {high_confidence_threshold})")
            print(f"Found {medium_confidence_count} medium confidence candidates ({medium_confidence_threshold} <= prob < {high_confidence_threshold})")
            
            # Create combined mask for all candidates
            candidate_mask = self.filtered_data['cv_prob'] >= medium_confidence_threshold
            
            # Set candidate flag
            self.filtered_data['is_cv_candidate'] = candidate_mask
            
            # Add confidence level classification
            self.filtered_data['confidence_level'] = 'low'
            self.filtered_data.loc[medium_confidence_mask, 'confidence_level'] = 'medium'
            self.filtered_data.loc[high_confidence_mask, 'confidence_level'] = 'high'
            
            # Set confidence directly from classifier probability
            self.filtered_data['confidence'] = self.filtered_data['cv_prob']
            
        # If XGBoost results not available, try anomaly detection
        elif 'is_anomaly' in self.filtered_data.columns:
            print("No classifier results available. Using anomaly detection results.")
            
            anomaly_mask = self.filtered_data['is_anomaly']
            anomaly_count = anomaly_mask.sum()
            
            print(f"Found {anomaly_count} candidates using anomaly detection")
            
            # Set candidate flag
            self.filtered_data['is_cv_candidate'] = anomaly_mask
            
            # Default medium confidence for anomaly-detected candidates
            self.filtered_data['confidence_level'] = 'low'
            self.filtered_data.loc[anomaly_mask, 'confidence_level'] = 'medium'
            self.filtered_data['confidence'] = 0.5  # Default medium confidence
            
        # If neither is available, use domain-knowledge score as fallback
        elif 'cv_score' in self.filtered_data.columns:
            print("Using domain-knowledge CV score for candidate selection")
            
            high_score_threshold = 0.7
            medium_score_threshold = 0.5
            
            high_score_mask = self.filtered_data['cv_score'] >= high_score_threshold
            medium_score_mask = (self.filtered_data['cv_score'] >= medium_score_threshold) & (self.filtered_data['cv_score'] < high_score_threshold)
            
            high_score_count = high_score_mask.sum()
            medium_score_count = medium_score_mask.sum()
            
            print(f"Found {high_score_count} high score candidates (score >= {high_score_threshold})")
            print(f"Found {medium_score_count} medium score candidates ({medium_score_threshold} <= score < {high_score_threshold})")
            
            # Create combined mask for all candidates
            candidate_mask = self.filtered_data['cv_score'] >= medium_score_threshold
            
            # Set candidate flag
            self.filtered_data['is_cv_candidate'] = candidate_mask
            
            # Add confidence level classification
            self.filtered_data['confidence_level'] = 'low'
            self.filtered_data.loc[medium_score_mask, 'confidence_level'] = 'medium'
            self.filtered_data.loc[high_score_mask, 'confidence_level'] = 'high'
            
            # Set confidence directly from cv_score
            self.filtered_data['confidence'] = self.filtered_data['cv_score']
            
        # If none of the above are available, use simplified method
        else:
            print("No advanced selection criteria available. Using simplified selection.")
            return self.select_candidates_simple()
        
        # Create final candidate set
        self.cv_candidates = self.filtered_data[self.filtered_data['is_cv_candidate']].copy()
        
        # If embedding information is available, use it to refine the rankings
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in self.cv_candidates.columns]
        
        if embedding_features and self.known_cv_file is not None and len(embedding_features) >= 10:
            print("Refining candidate rankings using embedding proximity to known CVs...")
            
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
                    
                    # Extract embeddings and reduce dimensionality
                    from sklearn.decomposition import PCA
                    
                    # Extract embeddings for candidates and known CVs
                    all_embeddings = self.cv_candidates[embedding_features].values
                    
                    # Apply PCA
                    pca = PCA(n_components=3)
                    all_embeddings_3d = pca.fit_transform(all_embeddings)
                    
                    # Add PCA dimensions
                    self.cv_candidates['pca_1'] = all_embeddings_3d[:, 0]
                    self.cv_candidates['pca_2'] = all_embeddings_3d[:, 1]
                    self.cv_candidates['pca_3'] = all_embeddings_3d[:, 2]
                    
                    # Split known CVs and candidates
                    known_cvs = self.cv_candidates[self.cv_candidates['is_known_cv']]
                    candidates = self.cv_candidates[~self.cv_candidates['is_known_cv']]
                    
                    if len(known_cvs) > 0 and len(candidates) > 0:
                        # Calculate embedding distances
                        from scipy.spatial.distance import cdist
                        
                        known_points = known_cvs[['pca_1', 'pca_2', 'pca_3']].values
                        candidate_points = candidates[['pca_1', 'pca_2', 'pca_3']].values
                        
                        # Calculate minimum distance from each candidate to any known CV
                        distances = cdist(candidate_points, known_points, 'euclidean')
                        min_distances = np.min(distances, axis=1)
                        
                        # Add distance to nearest known CV
                        candidates['distance_to_nearest_cv'] = min_distances
                        
                        # Normalized distance (0-1 scale, 0 is closest)
                        max_dist = min_distances.max()
                        if max_dist > 0:
                            candidates['norm_distance'] = min_distances / max_dist
                        else:
                            candidates['norm_distance'] = 0
                        
                        # Compute embedding similarity score (inverse of normalized distance)
                        candidates['embedding_similarity'] = 1 - candidates['norm_distance']
                        
                        # Blend classifier confidence with embedding similarity (weighted average)
                        # Weight classifier confidence more heavily
                        classifier_weight = 0.8
                        embedding_weight = 0.2
                        
                        if 'confidence' in candidates.columns:
                            candidates['blended_score'] = (
                                classifier_weight * candidates['confidence'] + 
                                embedding_weight * candidates['embedding_similarity']
                            )
                        else:
                            candidates['blended_score'] = candidates['embedding_similarity']
                        
                        # Update main candidates dataframe with these scores
                        self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'distance_to_nearest_cv'] = candidates['distance_to_nearest_cv'].values
                        self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'embedding_similarity'] = candidates['embedding_similarity'].values
                        
                        if 'blended_score' in candidates.columns:
                            self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'blended_score'] = candidates['blended_score'].values
                            
                            # Replace confidence with blended score for ranking, but keep original confidence
                            self.cv_candidates['original_confidence'] = self.cv_candidates['confidence']
                            self.cv_candidates['confidence'] = self.cv_candidates['blended_score']
                        
                        print(f"Enhanced candidate scoring with embedding proximity information")
        
        # Sort by confidence (which may now be the blended score)
        self.cv_candidates = self.cv_candidates.sort_values('confidence', ascending=False)
        
        print(f"Selected {len(self.cv_candidates)} CV candidates")
        return True




    def train_two_stage_classifier(self):
        """
        Train a two-stage classifier combining traditional feature-based models with
        embedding-based models for optimal CV candidate selection.
        
        This approach maintains interpretability while leveraging the representational
        power of contrastive curve embeddings through an ensemble methodology.
        """
        if not hasattr(self, 'scaled_features') or len(self.scaled_features) == 0:
            print("No features available. Call extract_features() first.")
            return False
        
        print("Training two-stage CV classifier...")
        
        # Retrieve known CV identifiers
        known_ids = self.load_known_cvs()
        
        if known_ids is None or len(known_ids) == 0:
            print("Error: No known CVs available for training. Classifier aborted.")
            return False
        
        # Determine candidate identifier column
        id_columns = ['sourceid', 'primvs_id', 'source_id', 'id']
        id_col = None
        
        for col in id_columns:
            if col in self.filtered_data.columns:
                id_col = col
                break
        
        if id_col is None:
            print("Error: No suitable identifier column found in candidate data.")
            return False
        
        print(f"Using identifier column '{id_col}' for matching")
        
        # Convert both known IDs and candidate IDs to strings for consistent matching
        candidate_ids = self.filtered_data[id_col].astype(str)
        
        # Create binary classification labels
        y = candidate_ids.isin(known_ids).astype(int)
        positive_count = y.sum()
        
        print(f"Found {positive_count} matches between known CVs and candidate sources")
        
        if positive_count < 10:
            print(f"Warning: Very few positive examples ({positive_count}). Classification may be unreliable.")
            if positive_count < 3:
                print("Error: Insufficient positive examples for training. Aborting classifier.")
                return False
        
        # Split features into traditional and embedding sets
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in self.scaled_features.columns]
        traditional_features = [col for col in self.scaled_features.columns if col not in embedding_features]
        
        print(f"Traditional features: {len(traditional_features)}")
        print(f"Embedding features: {len(embedding_features)}")
        
        # Extract feature sets as numpy arrays to avoid indexing issues
        X_trad = self.scaled_features[traditional_features].values
        X_emb = self.scaled_features[embedding_features].values if embedding_features else None
        
        # Split data for training and validation using indices
        # We'll use array indexing rather than DataFrame iloc to avoid the observed error
        X_indices = np.arange(len(X_trad))
        train_indices, val_indices = train_test_split(
            X_indices, test_size=0.25, random_state=42, stratify=y
        )
        
        X_trad_train, X_trad_val = X_trad[train_indices], X_trad[val_indices]
        y_train, y_val = y.iloc[train_indices].values, y.iloc[val_indices].values
        
        if X_emb is not None:
            X_emb_train, X_emb_val = X_emb[train_indices], X_emb[val_indices]
        
        # Create class weights to handle imbalance
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
        
        # STAGE 1: Train model with traditional features
        print("\nStage 1: Training model with traditional features...")
        
        model_trad = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=42
        )
        
        model_trad.fit(
            X_trad_train, y_train,
            #eval_set=[(X_trad_val, y_val)],
            #eval_metric='auc',
            #early_stopping_rounds=20,
            #verbose=False
        )
        
        # Evaluate traditional model
        y_pred_trad = model_trad.predict(X_trad_val)
        prob_trad = model_trad.predict_proba(X_trad_val)[:, 1]
        
        print("\nTraditional Feature Model Performance:")
        print(classification_report(y_val, y_pred_trad))
        
        # Get feature importance for traditional model
        trad_importance = pd.DataFrame({
            'Feature': traditional_features,
            'Importance': model_trad.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 5 traditional features:")
        for i, (_, row) in enumerate(trad_importance.head(5).iterrows()):
            print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        

        print("\nStage 2: Training model with embedding features...")
        
        # Apply PCA to reduce dimensionality of embeddings while preserving variance
        from sklearn.decomposition import PCA
        
        # Determine optimal number of components (explaining ~90% variance)
        pca = PCA().fit(X_emb_train)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = min(np.argmax(explained_variance >= 0.9) + 1, len(explained_variance))
        print(f"Using {n_components} PCA components (explaining {explained_variance[n_components-1]:.2%} variance)")
        
        # Apply PCA transformation
        pca = PCA(n_components=n_components)
        X_emb_train_pca = pca.fit_transform(X_emb_train)
        X_emb_val_pca = pca.transform(X_emb_val)
        
        # Train embedding model
        model_emb = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=42
        )
        
        model_emb.fit(
            X_emb_train_pca, y_train,
            #eval_set=[(X_emb_val_pca, y_val)],
            #eval_metric='auc',
            #early_stopping_rounds=20,
            #verbose=False
        )
        
        # Evaluate embedding model
        y_pred_emb = model_emb.predict(X_emb_val_pca)
        prob_emb = model_emb.predict_proba(X_emb_val_pca)[:, 1]
        
        print("\nEmbedding Feature Model Performance:")
        print(classification_report(y_val, y_pred_emb))
        
        # STAGE 3: Train meta-model (stacking)
        print("\nStage 3: Training meta-model to combine predictions...")
        
        # Create meta-features (predictions from base models)
        meta_features_train = np.column_stack([
            model_trad.predict_proba(X_trad_train)[:, 1],
            model_emb.predict_proba(X_emb_train_pca)[:, 1]
        ])
        
        meta_features_val = np.column_stack([
            prob_trad,
            prob_emb
        ])
        
        # Train meta-model
        meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.1,
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=42
        )
        
        meta_model.fit(
            meta_features_train, y_train,
            #eval_set=[(meta_features_val, y_val)],
            #eval_metric='auc',
            #early_stopping_rounds=10,
            #verbose=False
        )
        
        # Evaluate meta-model
        y_pred_meta = meta_model.predict(meta_features_val)
        
        print("\nMeta-Model Performance:")
        print(classification_report(y_val, y_pred_meta))
        
        # Calculate blend weights
        blend_weights = meta_model.feature_importances_
        print(f"\nModel blend weights: Traditional {blend_weights[0]:.2f}, Embedding {blend_weights[1]:.2f}")
        
        # STAGE 4: Apply to all data
        print("\nApplying two-stage classifier to all data...")
        
        # Prepare PCA for full dataset
        X_emb_full_pca = pca.transform(X_emb) if X_emb is not None else None
        
        # Get predictions from both models
        trad_probs = model_trad.predict_proba(X_trad)[:, 1]
        emb_probs = model_emb.predict_proba(X_emb_full_pca)[:, 1]
        
        # Combine using meta-model
        meta_features_full = np.column_stack([trad_probs, emb_probs])
        final_probs = meta_model.predict_proba(meta_features_full)[:, 1]
        
        # Store models for later use
        self.model_trad = model_trad
        self.model_emb = model_emb
        self.model_meta = meta_model
        self.pca = pca
        
        # Add predictions to filtered data
        self.filtered_data['cv_prob_trad'] = trad_probs
        self.filtered_data['cv_prob_emb'] = emb_probs
        self.filtered_data['cv_prob'] = final_probs
        
        # Save traditional feature names for later interpretation
        self.traditional_features = traditional_features
        
        # Save models
        joblib.dump(model_trad, os.path.join(self.output_dir, 'cv_classifier_traditional.joblib'))
        joblib.dump(model_emb, os.path.join(self.output_dir, 'cv_classifier_embedding.joblib'))
        joblib.dump(meta_model, os.path.join(self.output_dir, 'cv_classifier_meta.joblib'))
        joblib.dump(pca, os.path.join(self.output_dir, 'embedding_pca.joblib'))
        
        # Set the ensemble as the primary model
        self.model = meta_model
        

            
        # Plot probability distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.filtered_data['cv_prob'], bins=50, alpha=0.7)
        plt.axvline(0.5, color='r', linestyle='--', label='Default threshold (0.5)')
        plt.axvline(0.8, color='g', linestyle='--', label='High confidence (0.8)')
        plt.xlabel('CV Probability')
        plt.ylabel('Number of Sources')
        plt.title('Distribution of CV Probabilities from Two-Stage Classifier')
        plt.yscale('log')  # Set y-axis to log scale
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'cv_two_stage_probability_distribution.png'), dpi=300)
        plt.close()
        
        return True




    def run_pipeline(self):
        """Run the complete CV finder pipeline with two-stage classification."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("RUNNING PRIMVS CV FINDER PIPELINE WITH TWO-STAGE CLASSIFICATION")
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
        
        # Step 4: Calculate CV score using domain knowledge
        if not self.calculate_cv_score():
            print("Failed to calculate CV scores. Continuing with limited features.")
        
        # Step 5: Detect anomalies
        #if not self.detect_anomalies():
        #    print("Anomaly detection failed or found no anomalies. Continuing without anomaly features.")
        
        # Step 6: Train two-stage classifier if known CVs are available
        if self.known_cv_file is not None:
            if not self.train_two_stage_classifier():
                print("Two-stage classifier training failed. Falling back to heuristic selection.")
            else:
                # Step 7: Visualize two-stage classification results
                self.visualize_two_stage_classification()
        
        # Step 8: Select final candidates
        if not self.select_candidates():
            print("Failed to select candidates. Aborting.")
            return False
        
        # Step 9: Plot candidates
        self.plot_candidates()


        # Step 10: Create embedding-specific visualizations
        print("\nGenerating embedding-specific visualizations...")
        
        # Step 10a: Visualize embeddings in reduced dimensional space
        self.visualize_embeddings()
        
        # Step 10b: Compare candidates with known CVs in embedding space
        self.compare_candidates_with_known_cvs()
        
        # Step 10c: Comprehensive visualization of classification in embedding space
        self.visualize_classification_in_embedding_space()
        



        
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
            






    def visualize_embeddings(self):
        """Visualize the contrastive curves embeddings in relation to CV classification."""
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates to visualize.")
            return
        
        # Check if embeddings are available
        cc_embedding_cols = [str(i) for i in range(64)]
        embedding_features = [col for col in cc_embedding_cols if col in self.cv_candidates.columns]
        
        if len(embedding_features) < 10:
            print("Insufficient embedding features for visualization.")
            return
            
        print("Generating embedding visualizations...")
        
        # Extract embeddings
        embeddings = self.cv_candidates[embedding_features].values
        
        # Apply PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        
        # Add PCA dimensions to candidates dataframe
        self.cv_candidates['pca_1'] = embeddings_3d[:, 0]
        self.cv_candidates['pca_2'] = embeddings_3d[:, 1]
        self.cv_candidates['pca_3'] = embeddings_3d[:, 2]
        
        # Create visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by confidence if available
        if 'confidence' in self.cv_candidates.columns:
            sc = ax.scatter(
                self.cv_candidates['pca_1'],
                self.cv_candidates['pca_2'],
                self.cv_candidates['pca_3'],
                c=self.cv_candidates['confidence'],
                cmap='viridis',
                alpha=0.7,
                s=10
            )
            plt.colorbar(sc, label='Confidence')
        else:
            ax.scatter(
                self.cv_candidates['pca_1'],
                self.cv_candidates['pca_2'],
                self.cv_candidates['pca_3'],
                alpha=0.7,
                s=10
            )
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        plt.title('Contrastive Curves Embedding Space for CV Candidates')
        
        plt.savefig(os.path.join(self.output_dir, 'cv_embeddings_3d.png'), dpi=300)
        plt.close()
        
        # Create 2D visualization of first two components
        plt.figure(figsize=(10, 8))
        
        if 'confidence' in self.cv_candidates.columns:
            sc = plt.scatter(
                self.cv_candidates['pca_1'],
                self.cv_candidates['pca_2'],
                c=self.cv_candidates['confidence'],
                cmap='viridis',
                alpha=0.7,
                s=10
            )
            plt.colorbar(sc, label='Confidence')
        else:
            plt.scatter(
                self.cv_candidates['pca_1'],
                self.cv_candidates['pca_2'],
                alpha=0.7,
                s=10
            )
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('2D Projection of Contrastive Curves Embedding Space')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'cv_embeddings_2d.png'), dpi=300)
        plt.close()



    def compare_candidates_with_known_cvs(self):
        """
        Compare CV candidates with known CVs in the contrastive curves embedding space
        to evaluate classification performance and identify potential new CV members.
        
        This method generates comprehensive visualizations illustrating the spatial 
        relationships between known CVs and candidate sources in the embedding space.
        """
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates available for comparison.")
            return
        
        # Load complete known CV dataset rather than just IDs
        if self.known_cv_file is None:
            print("No known CV file provided for comparison.")
            return
        
        try:
            # Load complete known CV dataset
            if self.known_cv_file.endswith('.fits'):
                with fits.open(self.known_cv_file) as hdul:
                    known_cv_data = Table(hdul[1].data).to_pandas()
            elif self.known_cv_file.endswith('.csv'):
                known_cv_data = pd.read_csv(self.known_cv_file)
            else:
                print(f"Unsupported file format: {self.known_cv_file}")
                return
            
            print(f"Loaded {len(known_cv_data)} known CV records for comparison")
            
            # Identify the ID column for matching
            id_columns = ['sourceid', 'primvs_id', 'id', 'source_id', 'ID']
            id_col = None
            
            for col in id_columns:
                if col in known_cv_data.columns and col in self.cv_candidates.columns:
                    id_col = col
                    break
            
            if id_col is None:
                print("Warning: Could not identify common ID column between known CVs and candidates.")
                print("Available columns in known CVs:", known_cv_data.columns.tolist())
                print("Available columns in candidates:", self.cv_candidates.columns.tolist())
                return
            
            print(f"Using '{id_col}' as identifier column for matching")
            
            # Extract embedding features
            cc_embedding_cols = [str(i) for i in range(64)]
            embedding_features = [col for col in cc_embedding_cols if col in self.cv_candidates.columns]
            
            if len(embedding_features) < 3:
                print("Insufficient embedding features for dimensional reduction and visualization.")
                return
            
            # Mark known CVs in the candidate dataset
            cv_ids = set(known_cv_data[id_col].astype(str))
            self.cv_candidates['is_known_cv'] = self.cv_candidates[id_col].astype(str).isin(cv_ids)
            known_count = self.cv_candidates['is_known_cv'].sum()
            
            print(f"Identified {known_count} known CVs among the candidates")
            
            # Proceed only if we have known CVs in the candidates
            if known_count == 0:
                print("No known CVs found among the candidates. Visualization skipped.")
                return
            
            # Extract embeddings for dimension reduction
            print("Applying dimensional reduction to embedding space...")
            from sklearn.decomposition import PCA
            
            # Extract embeddings from candidates
            candidate_embeddings = self.cv_candidates[embedding_features].values
            
            # Apply PCA for dimensional reduction
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(candidate_embeddings)
            
            # Add reduced dimensions to candidate dataframe
            self.cv_candidates['pca_1'] = embeddings_3d[:, 0]
            self.cv_candidates['pca_2'] = embeddings_3d[:, 1]
            self.cv_candidates['pca_3'] = embeddings_3d[:, 2]
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA explained variance: {explained_variance[0]:.2%}, {explained_variance[1]:.2%}, {explained_variance[2]:.2%}")
            
            # Separate known CVs from candidates
            known_cvs = self.cv_candidates[self.cv_candidates['is_known_cv']]
            unknown_candidates = self.cv_candidates[~self.cv_candidates['is_known_cv']]
            
            # 1. Create 3D visualization of known CVs vs candidates
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot candidates with low opacity
            ax.scatter(
                unknown_candidates['pca_1'],
                unknown_candidates['pca_2'],
                unknown_candidates['pca_3'],
                color='blue',
                alpha=0.3,
                s=10,
                label='CV Candidates'
            )
            
            # Plot known CVs with high visibility
            ax.scatter(
                known_cvs['pca_1'],
                known_cvs['pca_2'],
                known_cvs['pca_3'],
                color='red',
                marker='*',
                s=50,
                alpha=1.0,
                label='Known CVs'
            )
            
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
            ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
            ax.set_title('Known CVs vs. Candidates in Embedding Space')
            plt.legend()
            
            plt.savefig(os.path.join(self.output_dir, 'known_vs_candidates_3d.png'), dpi=300)
            plt.close()
            
            # 2. Calculate distances to find closest candidates to known CVs
            from scipy.spatial.distance import cdist
            
            # Calculate pairwise distances between known CVs and candidates
            known_points = known_cvs[['pca_1', 'pca_2', 'pca_3']].values
            candidate_points = unknown_candidates[['pca_1', 'pca_2', 'pca_3']].values
            
            # Calculate distances
            distances = cdist(known_points, candidate_points, 'euclidean')
            
            # Initialize list to store nearest neighbors
            nearest_neighbors = []
            
            # Find k nearest neighbors for each known CV
            k_neighbors = min(5, len(candidate_points))
            
            for i, (idx, known_cv) in enumerate(known_cvs.iterrows()):
                # Get indices of k nearest candidates
                nearest_indices = np.argsort(distances[i])[:k_neighbors]
                
                # Record these neighbors
                for rank, neighbor_idx in enumerate(nearest_indices):
                    candidate_idx = unknown_candidates.iloc[neighbor_idx].name
                    candidate = unknown_candidates.iloc[neighbor_idx]
                    
                    nearest_neighbors.append({
                        'known_cv_id': known_cv[id_col],
                        'candidate_id': candidate[id_col],
                        'distance': distances[i, neighbor_idx],
                        'rank': rank + 1,
                        'confidence': candidate.get('confidence', candidate.get('cv_prob', 0.0)),
                        'period_hours': candidate['true_period'] * 24.0 if 'true_period' in candidate else 0.0,
                        'amplitude': candidate.get('true_amplitude', 0.0),
                        'pca_1': candidate['pca_1'],
                        'pca_2': candidate['pca_2'],
                        'pca_3': candidate['pca_3']
                    })
            
            # Create DataFrame of nearest neighbors
            nn_df = pd.DataFrame(nearest_neighbors)
            
            # Save to CSV
            nn_file = os.path.join(self.output_dir, 'nearest_neighbors_to_known_cvs.csv')
            nn_df.to_csv(nn_file, index=False)
            
            print(f"Saved {len(nn_df)} nearest neighbors to {nn_file}")
            
            # 3. Create 2D visualization with density contours
            plt.figure(figsize=(12, 10))
            
            # Create contour of density for all candidates
            from scipy.stats import gaussian_kde
            
            # Compute density estimate
            x = self.cv_candidates['pca_1']
            y = self.cv_candidates['pca_2']
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            
            # Create grid for contour plot
            x_grid = np.linspace(x.min(), x.max(), 100)
            y_grid = np.linspace(y.min(), y.max(), 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)
            
            # Plot density contours
            plt.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.6)
            
            # Plot candidates
            plt.scatter(
                unknown_candidates['pca_1'],
                unknown_candidates['pca_2'],
                color='blue',
                alpha=0.5,
                s=15,
                label='CV Candidates'
            )
            
            # Plot known CVs
            plt.scatter(
                known_cvs['pca_1'],
                known_cvs['pca_2'],
                color='red',
                marker='*',
                s=100,
                edgecolors='black',
                label='Known CVs'
            )
            
            # Highlight top nearest neighbors (closest to any known CV)
            top_neighbors = nn_df.sort_values('distance').head(10)
            if len(top_neighbors) > 0:
                # Get candidate IDs
                top_ids = top_neighbors['candidate_id'].astype(str).tolist()
                
                # Find these candidates
                top_candidates = unknown_candidates[unknown_candidates[id_col].astype(str).isin(top_ids)]
                
                # Plot them
                plt.scatter(
                    top_candidates['pca_1'],
                    top_candidates['pca_2'],
                    color='green',
                    marker='o',
                    s=80,
                    facecolors='none',
                    edgecolors='green',
                    linewidth=2,
                    label='Top CV Candidates'
                )
            
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
            plt.title('Known CVs and Candidates in Principal Component Space')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(self.output_dir, 'known_vs_candidates_2d.png'), dpi=300)
            plt.close()
            
            # 4. Bailey diagram comparing known CVs and closest candidates
            if 'true_period' in self.cv_candidates.columns and 'true_amplitude' in self.cv_candidates.columns:
                plt.figure(figsize=(12, 8))
                
                # Convert periods to hours and take log for better visualization
                known_cvs['log_period_hours'] = np.log10(known_cvs['true_period'] * 24.0)
                unknown_candidates['log_period_hours'] = np.log10(unknown_candidates['true_period'] * 24.0)
                
                # Plot candidates
                plt.scatter(
                    unknown_candidates['log_period_hours'],
                    unknown_candidates['true_amplitude'],
                    color='blue',
                    alpha=0.3,
                    s=10,
                    label='CV Candidates'
                )
                
                # Plot known CVs
                plt.scatter(
                    known_cvs['log_period_hours'],
                    known_cvs['true_amplitude'],
                    color='red',
                    marker='*',
                    s=100,
                    edgecolors='black',
                    label='Known CVs'
                )
                
                # Highlight top nearest neighbors
                if len(top_neighbors) > 0:
                    top_candidates['log_period_hours'] = np.log10(top_candidates['true_period'] * 24.0)
                    
                    plt.scatter(
                        top_candidates['log_period_hours'],
                        top_candidates['true_amplitude'],
                        color='green',
                        marker='o',
                        s=80,
                        facecolors='none',
                        edgecolors='green',
                        linewidth=2,
                        label='Top CV Candidates'
                    )
                
                plt.xlabel('log₁₀(Period) [hours]')
                plt.ylabel('Amplitude [mag]')
                plt.title('Bailey Diagram: Known CVs vs. Candidates')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.savefig(os.path.join(self.output_dir, 'bailey_known_vs_candidates.png'), dpi=300)
                plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error in CV comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


    def visualize_classification_in_embedding_space(self):
        """
        Generate comprehensive visualizations of classification results in embedding space
        to evaluate model efficacy and examine the alignment between traditional feature-based 
        and embedding-based classifications within the two-stage framework.
        
        This analysis provides insight into the semantic structure captured by contrastive 
        curve embeddings and their relationship to domain-specific classification criteria.
        """
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates available for visualization.")
            return False
        
        print("Generating embedding space classification visualizations...")
        
        try:
            # Extract embedding features
            cc_embedding_cols = [str(i) for i in range(64)]
            embedding_features = [col for col in cc_embedding_cols if col in self.cv_candidates.columns]
            
            if len(embedding_features) < 3:
                print("Insufficient embedding features for visualization. Minimum of 3 required.")
                return False
            
            # Determine classification confidence metric
            confidence_metrics = ['confidence', 'cv_prob', 'cv_prob_trad', 'cv_prob_emb', 'cv_score', 'probability']
            confidence_col = None
            for col in confidence_metrics:
                if col in self.cv_candidates.columns:
                    confidence_col = col
                    break
            
            if confidence_col is None:
                print("No classification confidence metric found in candidates.")
                confidence_col = 'confidence'  # Default placeholder
                self.cv_candidates[confidence_col] = 0.5  # Default value
            
            print(f"Using '{confidence_col}' as classification confidence metric")
            
            # Check if we have both traditional and embedding probabilities
            has_two_stage = all(col in self.cv_candidates.columns for col in ['cv_prob_trad', 'cv_prob_emb'])
            
            # Apply PCA to reduce dimensionality of embeddings
            from sklearn.decomposition import PCA
            
            # Extract embeddings
            embeddings = self.cv_candidates[embedding_features].values
            
            # Apply PCA
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(embeddings)
            
            # Add PCA dimensions to candidates dataframe
            self.cv_candidates['pca_1'] = embeddings_3d[:, 0]
            self.cv_candidates['pca_2'] = embeddings_3d[:, 1]
            self.cv_candidates['pca_3'] = embeddings_3d[:, 2]
            
            # Calculate explained variance for labels
            explained_variance = pca.explained_variance_ratio_
            
            # 1. Generate 3D scatter plot colored by classification confidence
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create scatter plot with color based on confidence
            sc = ax.scatter(
                self.cv_candidates['pca_1'], 
                self.cv_candidates['pca_2'], 
                self.cv_candidates['pca_3'],
                c=self.cv_candidates[confidence_col],
                cmap='viridis',
                alpha=0.7,
                s=20
            )
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label(f'{confidence_col.replace("_", " ").title()}')
            
            # Set axis labels with explained variance
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
            ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
            
            plt.title('CV Classification Confidence in Embedding Space')
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'classification_confidence_3d.png'), dpi=300)
            plt.close()
            
            # 2. Generate 2D scatter plot with high confidence candidates highlighted
            plt.figure(figsize=(12, 10))
            
            # Plot density contours for all candidates
            from scipy.stats import gaussian_kde
            
            # Compute KDE for density estimation
            x = self.cv_candidates['pca_1']
            y = self.cv_candidates['pca_2']
            xy = np.vstack([x, y])
            
            # Only compute KDE if we have sufficient points
            if len(x) > 10:
                try:
                    kde = gaussian_kde(xy)
                    
                    # Create grid
                    x_grid = np.linspace(x.min(), x.max(), 100)
                    y_grid = np.linspace(y.min(), y.max(), 100)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    
                    # Compute density and reshape
                    Z = kde(positions).reshape(X.shape)
                    
                    # Plot contours
                    plt.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.4)
                except Exception as e:
                    print(f"Warning: Could not compute density contours: {str(e)}")
            
            # Plot all candidates with low opacity
            plt.scatter(
                self.cv_candidates['pca_1'],
                self.cv_candidates['pca_2'],
                c='gray',
                alpha=0.3,
                s=10,
                label='All Candidates'
            )
            
            # Plot high confidence candidates
            threshold = 0.8
            high_conf = self.cv_candidates[self.cv_candidates[confidence_col] > threshold]
            
            if len(high_conf) > 0:
                plt.scatter(
                    high_conf['pca_1'],
                    high_conf['pca_2'],
                    c=high_conf[confidence_col],
                    cmap='viridis',
                    alpha=1.0,
                    s=40,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'High Confidence (>{threshold})'
                )
                
                plt.colorbar(label=f'{confidence_col.replace("_", " ").title()}')
            
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
            plt.title('High Confidence CV Candidates in Principal Component Space')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(self.output_dir, 'high_confidence_candidates_2d.png'), dpi=300)
            plt.close()
            
            # 3. If two-stage classification is available, visualize agreement between models
            if has_two_stage:
                print("Generating two-stage classification agreement visualizations...")
                
                # Calculate agreement between traditional and embedding models
                self.cv_candidates['model_agreement'] = 1.0 - np.abs(
                    self.cv_candidates['cv_prob_trad'] - self.cv_candidates['cv_prob_emb']
                )
                
                # Create 2D scatter plot colored by agreement
                plt.figure(figsize=(12, 10))
                
                # Plot all candidates colored by model agreement
                sc = plt.scatter(
                    self.cv_candidates['pca_1'],
                    self.cv_candidates['pca_2'],
                    c=self.cv_candidates['model_agreement'],
                    cmap='RdYlGn',  # Red to Yellow to Green
                    alpha=0.7,
                    s=30,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                plt.colorbar(sc, label='Model Agreement (Traditional vs. Embedding)')
                
                # Mark high agreement, high confidence candidates
                high_agreement = (self.cv_candidates['model_agreement'] > 0.9) & (self.cv_candidates['cv_prob'] > 0.8)
                high_agreement_candidates = self.cv_candidates[high_agreement]
                
                if len(high_agreement_candidates) > 0:
                    plt.scatter(
                        high_agreement_candidates['pca_1'],
                        high_agreement_candidates['pca_2'],
                        marker='*',
                        s=100,
                        color='white',
                        edgecolors='black',
                        linewidths=1.0,
                        label='High Agreement, High Confidence'
                    )
                
                plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
                plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
                plt.title('Two-Stage Classification Agreement in Embedding Space')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.savefig(os.path.join(self.output_dir, 'model_agreement_2d.png'), dpi=300)
                plt.close()


                # Create scatter plot of traditional vs. embedding probabilities with hexagonal binning
                plt.figure(figsize=(10, 8))

                # Use hexbin for density visualization
                hb = plt.hexbin(
                    self.cv_candidates['cv_prob_trad'].values,
                    self.cv_candidates['cv_prob_emb'].values,
                    gridsize=30,
                    cmap='viridis',
                    mincnt=1,
                    bins='log',  # Logarithmic binning for better dynamic range
                    alpha=0.8
                )

                plt.colorbar(hb, label='log10(N)')

                # Add diagonal line for perfect agreement
                plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Agreement')

                # Add decision boundaries
                plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
                plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

                # Annotate quadrants
                plt.text(0.25, 0.75, "Traditional: No\nEmbedding: Yes", 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
                plt.text(0.75, 0.75, "Both: Yes", 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
                plt.text(0.25, 0.25, "Both: No", 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
                plt.text(0.75, 0.25, "Traditional: Yes\nEmbedding: No", 
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

                plt.xlabel('Traditional Features Probability')
                plt.ylabel('Embedding Features Probability')
                plt.title('Comparison of CV Probabilities: Traditional vs. Embedding Models')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper left')
                plt.axis('square')
                plt.xlim(0, 1)
                plt.ylim(0, 1)

                plt.savefig(os.path.join(self.output_dir, 'traditional_vs_embedding_probabilities.png'), dpi=300)
                plt.close()


                try:
                    # Create histogram of ensemble probabilities colored by agreement
                    plt.figure(figsize=(10, 6))
                    
                    # Define color mapping based on agreement
                    agreement = self.cv_candidates['model_agreement']
                    colors = plt.cm.RdYlGn(agreement)  # Red to Yellow to Green
                    
                    # Sort by agreement for better visualization
                    sort_idx = agreement.argsort()
                    
                    # Plot histogram with color based on agreement
                    for i in sort_idx:
                        plt.bar(
                            self.cv_candidates['cv_prob'].iloc[i], 
                            1, 
                            width=0.02, 
                            color=colors[i],
                            alpha=0.7
                        )
                    
                    plt.axvline(0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
                    
                    # Create a custom colorbar
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
                    sm.set_array([])
                    plt.colorbar(sm, label='Model Agreement')
                    
                    plt.xlabel('Ensemble CV Probability')
                    plt.ylabel('Count')
                    plt.title('Distribution of Ensemble Probabilities by Model Agreement')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.savefig(os.path.join(self.output_dir, 'ensemble_probability_distribution.png'), dpi=300)
                    plt.close()
                except:
                    pass
            
            # 4. Try to create UMAP visualization if available
            try:
                import umap
                
                print("Creating UMAP visualization of embedding space...")
                
                # Apply UMAP for non-linear dimensionality reduction
                reducer = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    metric='euclidean',
                    random_state=42
                )
                
                embedding_umap = reducer.fit_transform(embeddings)
                
                # Add UMAP coordinates to DataFrame
                self.cv_candidates['umap_1'] = embedding_umap[:, 0]
                self.cv_candidates['umap_2'] = embedding_umap[:, 1]
                
                # Create scatter plot
                plt.figure(figsize=(12, 10))
                
                # Plot all candidates with color based on confidence
                sc = plt.scatter(
                    self.cv_candidates['umap_1'],
                    self.cv_candidates['umap_2'],
                    c=self.cv_candidates[confidence_col],
                    cmap='viridis',
                    alpha=0.7,
                    s=30
                )
                
                plt.colorbar(sc, label=f'{confidence_col.replace("_", " ").title()}')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')
                plt.title('UMAP Projection of Contrastive Curves Embedding Space')
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(self.output_dir, 'umap_visualization.png'), dpi=300)
                plt.close()
                
                # If two-stage classification is available, create UMAP colored by agreement
                if has_two_stage:
                    plt.figure(figsize=(12, 10))
                    
                    # Plot all candidates with color based on model agreement
                    sc = plt.scatter(
                        self.cv_candidates['umap_1'],
                        self.cv_candidates['umap_2'],
                        c=self.cv_candidates['model_agreement'],
                        cmap='RdYlGn',
                        alpha=0.7,
                        s=30
                    )
                    
                    plt.colorbar(sc, label='Model Agreement')
                    plt.xlabel('UMAP Dimension 1')
                    plt.ylabel('UMAP Dimension 2')
                    plt.title('Model Agreement in UMAP Space')
                    plt.grid(True, alpha=0.3)
                    
                    plt.savefig(os.path.join(self.output_dir, 'umap_model_agreement.png'), dpi=300)
                    plt.close()
                
                print("UMAP visualization completed successfully")
                
            except ImportError:
                print("UMAP package not available. Skipping UMAP visualization.")
            except Exception as e:
                print(f"Error in UMAP visualization: {str(e)}")
            
            # 5. Save the DataFrame with embedding coordinates for further analysis
            embedding_file = os.path.join(self.output_dir, 'cv_candidates_with_embeddings.csv')
            self.cv_candidates.to_csv(embedding_file, index=False)
            print(f"Saved candidates with embedding coordinates to {embedding_file}")
            
            return True
            
        except Exception as e:
            print(f"Error in classification visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False





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



    
    def visualize_candidate_distributions(self, max_top_candidates=1000):
        """
        Generate comprehensive visualizations of CV candidate distributions across multiple parameter spaces.
        This method plots all candidates, known CVs, and top candidates without individual annotations,
        facilitating analysis of population-level distributions.
        
        Parameters:
        -----------
        max_top_candidates : int
            Maximum number of top candidates to highlight (default: 1000)
        """
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates available for visualization.")
            return
        
        print(f"Generating comprehensive distribution visualizations of CV candidates...")
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, 'distributions')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Determine ID column
        id_col = 'sourceid' if 'sourceid' in self.cv_candidates.columns else 'primvs_id'
        
        # Determine sorting column for top candidates
        sort_col = None
        for col in ['cv_prob', 'blended_score', 'confidence']:
            if col in self.cv_candidates.columns:
                sort_col = col
                break
        
        if sort_col is None:
            # Fall back to using FAP (lower is better)
            sort_col = 'best_fap'
            top_candidates = self.cv_candidates.sort_values(sort_col, ascending=True).head(max_top_candidates)
        else:
            # Higher values are better for probability/confidence metrics
            top_candidates = self.cv_candidates.sort_values(sort_col, ascending=False).head(max_top_candidates)
        
        # Check if we have known CV information
        has_known_cvs = 'is_known_cv' in self.cv_candidates.columns and any(self.cv_candidates['is_known_cv'])
        if has_known_cvs:
            known_cvs = self.cv_candidates[self.cv_candidates['is_known_cv']]
            other_candidates = self.cv_candidates[~self.cv_candidates['is_known_cv']]
        else:
            other_candidates = self.cv_candidates
        
        print(f"Preparing to visualize distributions with:")
        print(f"  - {len(self.cv_candidates)} total candidates")
        print(f"  - {len(top_candidates)} top candidates (based on {sort_col})")
        if has_known_cvs:
            print(f"  - {len(known_cvs)} known CVs")
        
        # 1. Bailey diagram (Period-Amplitude space)
        plt.figure(figsize=(14, 10))
        
        # Plot all filtered sources if available (as background context)
        if hasattr(self, 'filtered_data'):
            period_hours_filtered = self.filtered_data['true_period'] * 24.0
            plt.scatter(
                np.log10(period_hours_filtered),
                self.filtered_data['true_amplitude'],
                alpha=0.05,
                s=1,
                color='lightgray',
                label='All filtered sources'
            )
        
        # Plot all candidates
        period_hours_all = other_candidates['true_period'] * 24.0
        plt.scatter(
            np.log10(period_hours_all),
            other_candidates['true_amplitude'],
            alpha=0.3,
            s=5,
            color='blue',
            label='All candidates'
        )
        
        # Plot top candidates
        period_hours_top = top_candidates['true_period'] * 24.0
        plt.scatter(
            np.log10(period_hours_top),
            top_candidates['true_amplitude'],
            alpha=0.6,
            s=15,
            color='red',
            label=f'Top {len(top_candidates)} candidates'
        )
        
        # Plot known CVs if available
        if has_known_cvs:
            period_hours_known = known_cvs['true_period'] * 24.0
            plt.scatter(
                np.log10(period_hours_known),
                known_cvs['true_amplitude'],
                alpha=0.8,
                s=30,
                color='green',
                marker='*',
                label='Known CVs'
            )
        
        # Add period gap shaded region
        plt.axvspan(np.log10(2), np.log10(3), alpha=0.1, color='gray', label='Period gap')
        
        # Add descriptive annotations for key regions
        plt.annotate(
            'Period Gap\n(2-3 hrs)',
            xy=(np.log10(2.5), 0.8),
            xytext=(np.log10(2.5), 0.8),
            ha='center',
            va='center',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        # Short-period CVs region
        plt.annotate(
            'Short-period CVs',
            xy=(np.log10(1.5), 0.3),
            xytext=(np.log10(1.5), 0.3),
            ha='center',
            va='center',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        # Long-period CVs region
        plt.annotate(
            'Long-period CVs',
            xy=(np.log10(5), 1.0),
            xytext=(np.log10(5), 1.0),
            ha='center',
            va='center',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.xlabel('log₁₀(Period) [hours]')
        plt.ylabel('Amplitude [mag]')
        plt.title('Bailey Diagram of CV Candidates')
        plt.grid(True, alpha=0.3)
        plt.xlim(-1, 2.5)  # Limit x-axis to reasonable range for CVs
        plt.ylim(0, 2)     # Limit y-axis to reasonable amplitude range
        plt.legend(loc='upper right')
        
        # Add count information in text box
        info_text = f"Total candidates: {len(self.cv_candidates)}\n"
        info_text += f"Top candidates: {len(top_candidates)}\n"
        if has_known_cvs:
            info_text += f"Known CVs: {len(known_cvs)}"
        
        plt.figtext(
            0.02, 0.02, info_text,
            ha='left',
            va='bottom',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.savefig(os.path.join(viz_dir, 'bailey_diagram_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Embedding space visualization (if available)
        embedding_features = []
        cc_embedding_cols = [str(i) for i in range(64)]
        for col in cc_embedding_cols:
            if col in self.cv_candidates.columns:
                embedding_features.append(col)
        
        if len(embedding_features) >= 10 and hasattr(self, 'cv_candidates'):
            # Apply PCA to reduce dimensionality for visualization
            from sklearn.decomposition import PCA
            
            print("Applying PCA to embedding features for visualization...")
            
            # Extract embeddings
            embeddings = self.cv_candidates[embedding_features].values
            
            # Apply PCA
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(embeddings)
            
            # Add PCA coordinates to candidates dataframe (temporary)
            temp_candidates = self.cv_candidates.copy()
            temp_candidates['pca_1'] = embeddings_3d[:, 0]
            temp_candidates['pca_2'] = embeddings_3d[:, 1]
            temp_candidates['pca_3'] = embeddings_3d[:, 2]
            
            # Split into categories for plotting
            if has_known_cvs:
                temp_known_cvs = temp_candidates[temp_candidates['is_known_cv']]
                temp_other = temp_candidates[~temp_candidates['is_known_cv']]
            else:
                temp_other = temp_candidates
            
            temp_top = temp_candidates.loc[top_candidates.index]
            
            # Explained variance for axis labels
            explained_variance = pca.explained_variance_ratio_
            
            # Create 2D PCA plot
            plt.figure(figsize=(14, 10))
            
            # Plot all candidates
            plt.scatter(
                temp_other['pca_1'],
                temp_other['pca_2'],
                alpha=0.3,
                s=5,
                color='blue',
                label='All candidates'
            )
            
            # Plot top candidates
            plt.scatter(
                temp_top['pca_1'],
                temp_top['pca_2'],
                alpha=0.6,
                s=15,
                color='red',
                label=f'Top {len(temp_top)} candidates'
            )
            
            # Plot known CVs if available
            if has_known_cvs:
                plt.scatter(
                    temp_known_cvs['pca_1'],
                    temp_known_cvs['pca_2'],
                    alpha=0.8,
                    s=30,
                    color='green',
                    marker='*',
                    label='Known CVs'
                )
            
            plt.xlabel(f'PCA Component 1 ({explained_variance[0]:.1%} variance)')
            plt.ylabel(f'PCA Component 2 ({explained_variance[1]:.1%} variance)')
            plt.title('Contrastive Curves Embedding Space Distribution (PCA Projection)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # Add count information
            plt.figtext(
                0.02, 0.02, info_text,
                ha='left',
                va='bottom',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
            
            plt.savefig(os.path.join(viz_dir, 'embedding_pca_distribution.png'), dpi=300)
            plt.close()
            
            # Create 3D PCA plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all candidates
            ax.scatter(
                temp_other['pca_1'],
                temp_other['pca_2'],
                temp_other['pca_3'],
                alpha=0.3,
                s=5,
                color='blue',
                label='All candidates'
            )
            
            # Plot top candidates
            ax.scatter(
                temp_top['pca_1'],
                temp_top['pca_2'],
                temp_top['pca_3'],
                alpha=0.6,
                s=15,
                color='red',
                label=f'Top {len(temp_top)} candidates'
            )
            
            # Plot known CVs if available
            if has_known_cvs:
                ax.scatter(
                    temp_known_cvs['pca_1'],
                    temp_known_cvs['pca_2'],
                    temp_known_cvs['pca_3'],
                    alpha=0.8,
                    s=30,
                    color='green',
                    marker='*',
                    label='Known CVs'
                )
            
            ax.set_xlabel(f'PCA 1 ({explained_variance[0]:.1%})')
            ax.set_ylabel(f'PCA 2 ({explained_variance[1]:.1%})')
            ax.set_zlabel(f'PCA 3 ({explained_variance[2]:.1%})')
            ax.set_title('3D Contrastive Curves Embedding Space Distribution')
            ax.legend(loc='upper right')
            
            plt.savefig(os.path.join(viz_dir, 'embedding_pca_3d_distribution.png'), dpi=300)
            plt.close()
            
            # Alternative: Create UMAP visualization if available
            try:
                import umap
                
                print("Creating UMAP visualization of embedding space...")
                
                # Apply UMAP
                reducer = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    random_state=42,
                    metric='euclidean'
                )
                
                # Transform the embeddings
                embeddings_umap = reducer.fit_transform(embeddings)
                
                # Add UMAP coordinates
                temp_candidates['umap_1'] = embeddings_umap[:, 0]
                temp_candidates['umap_2'] = embeddings_umap[:, 1]
                
                # Split into categories again (using updated dataframe)
                if has_known_cvs:
                    temp_known_cvs = temp_candidates[temp_candidates['is_known_cv']]
                    temp_other = temp_candidates[~temp_candidates['is_known_cv']]
                else:
                    temp_other = temp_candidates
                
                temp_top = temp_candidates.loc[top_candidates.index]
                
                # Create UMAP plot
                plt.figure(figsize=(14, 10))
                
                # Plot all candidates
                plt.scatter(
                    temp_other['umap_1'],
                    temp_other['umap_2'],
                    alpha=0.3,
                    s=5,
                    color='blue',
                    label='All candidates'
                )
                
                # Plot top candidates
                plt.scatter(
                    temp_top['umap_1'],
                    temp_top['umap_2'],
                    alpha=0.6,
                    s=15,
                    color='red',
                    label=f'Top {len(temp_top)} candidates'
                )
                
                # Plot known CVs if available
                if has_known_cvs:
                    plt.scatter(
                        temp_known_cvs['umap_1'],
                        temp_known_cvs['umap_2'],
                        alpha=0.8,
                        s=30,
                        color='green',
                        marker='*',
                        label='Known CVs'
                    )
                
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')
                plt.title('Contrastive Curves Embedding Space Distribution (UMAP Projection)')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper right')
                
                # Add count information
                plt.figtext(
                    0.02, 0.02, info_text,
                    ha='left',
                    va='bottom',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
                
                plt.savefig(os.path.join(viz_dir, 'embedding_umap_distribution.png'), dpi=300)
                plt.close()
                
            except ImportError:
                print("UMAP not available. Skipping UMAP visualization.")


        # 3. Spatial distribution (if galactic coordinates available)
        plt.figure(figsize=(14, 10))
        
        # Convert longitudes to a more intuitive representation for VVV survey
        # VVV primarily covers regions around l≈0° (bulge) and l≈295°-350° (disk)
        # Convert to centered coordinate system (-180° to +180°) for better visualization
        other_candidates_centered = other_candidates.copy()
        other_candidates_centered['l_centered'] = np.where(
            other_candidates['l'] > 180, 
            other_candidates['l'] - 360, 
            other_candidates['l']
        )
        
        top_candidates_centered = top_candidates.copy()
        top_candidates_centered['l_centered'] = np.where(
            top_candidates['l'] > 180, 
            top_candidates['l'] - 360, 
            top_candidates['l']
        )
        
        if has_known_cvs:
            known_cvs_centered = known_cvs.copy()
            known_cvs_centered['l_centered'] = np.where(
                known_cvs['l'] > 180, 
                known_cvs['l'] - 360, 
                known_cvs['l']
            )
        
        # Plot all candidates using centered coordinates
        plt.scatter(
            other_candidates_centered['l_centered'],
            other_candidates_centered['b'],
            alpha=0.3,
            s=5,
            color='blue',
            label='All candidates'
        )
        
        # Plot top candidates
        plt.scatter(
            top_candidates_centered['l_centered'],
            top_candidates_centered['b'],
            alpha=0.6,
            s=15,
            color='red',
            label=f'Top {len(top_candidates)} candidates'
        )
        
        # Plot known CVs if available
        if has_known_cvs:
            plt.scatter(
                known_cvs_centered['l_centered'],
                known_cvs_centered['b'],
                alpha=0.8,
                s=30,
                color='green',
                marker='*',
                label='Known CVs'
            )
        
        # Mark the Galactic center
        plt.scatter(
            0, 0,
            marker='+',
            s=100,
            color='black',
            label='Galactic Center'
        )
        
        # Add VVV survey boundary outlines (approximate)
        # Bulge region: -10° < l < +10° and -10° < b < +5°
        bulge_x = np.array([-10, -10, 10, 10, -10])
        bulge_y = np.array([-10, 5, 5, -10, -10])
        plt.plot(bulge_x, bulge_y, 'k--', alpha=0.5, linewidth=1)
        
        # Disk region: -65° < l < -10° and -2° < b < -10°
        disk_x = np.array([-65, -65, -10, -10, -65])
        disk_y = np.array([-2, -10, -10, -2, -2])
        plt.plot(disk_x, disk_y, 'k--', alpha=0.5, linewidth=1)
        
        # Add region labels
        plt.annotate(
            'VVV Bulge',
            xy=(0, -2.5),
            xytext=(0, -2.5),
            ha='center',
            va='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.annotate(
            'VVV Disk',
            xy=(-35, -5),
            xytext=(-35, -5),
            ha='center',
            va='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.xlabel('Galactic Longitude (l) [deg]')
        plt.ylabel('Galactic Latitude (b) [deg]')
        plt.title('Spatial Distribution of CV Candidates in Galactic Coordinates (VVV Survey)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Set axis limits to focus on the VVV survey regions
        # Find the extremes of the data with some padding
        l_values_centered = np.where(
            self.cv_candidates['l'] > 180,
            self.cv_candidates['l'] - 360,
            self.cv_candidates['l']
        )
        
        # Determine appropriate limits based on data distribution
        l_min = np.percentile(l_values_centered, 0.1) - 5
        l_max = np.percentile(l_values_centered, 99.9) + 5
        b_min = np.percentile(self.cv_candidates['b'], 0.1) - 1
        b_max = np.percentile(self.cv_candidates['b'], 99.9) + 1
        
        plt.xlim(l_min, l_max)
        plt.ylim(b_min, b_max)
        
        # Add count information
        plt.figtext(
            0.02, 0.02, info_text,
            ha='left',
            va='bottom',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.savefig(os.path.join(viz_dir, 'galactic_distribution_vvv.png'), dpi=300)
        plt.close()
        
        # Create heat map of candidate density in Galactic coordinates
        plt.figure(figsize=(14, 10))
        
        # Create 2D histogram using centered coordinates for better visualization
        l_bins = np.linspace(l_min, l_max, 100)
        b_bins = np.linspace(b_min, b_max, 50)
        
        H, xedges, yedges = np.histogram2d(
            l_values_centered,
            self.cv_candidates['b'],
            bins=[l_bins, b_bins]
        )
        
        # Use logarithmic color scale for better visualization
        from matplotlib.colors import LogNorm
        
        # Smooth the histogram for better visualization
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=1.0)
        
        # Plot heatmap
        plt.pcolormesh(
            xedges, yedges, H_smooth.T,
            norm=LogNorm(vmin=0.1, vmax=H_smooth.max()),
            cmap='inferno',
            alpha=0.7
        )
        
        plt.colorbar(label='Candidate Density (log scale)')
        
        # Add VVV survey boundary lines
        plt.plot(bulge_x, bulge_y, 'w--', alpha=0.8, linewidth=1.5)
        plt.plot(disk_x, disk_y, 'w--', alpha=0.8, linewidth=1.5)
        
        # Annotate VVV survey regions
        plt.annotate(
            'VVV Bulge',
            xy=(0, -2.5),
            xytext=(0, -2.5),
            ha='center',
            va='center',
            fontsize=10,
            color='white',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7)
        )
        
        plt.annotate(
            'VVV Disk',
            xy=(-35, -5),
            xytext=(-35, -5),
            ha='center',
            va='center',
            fontsize=10,
            color='white',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7)
        )
        
        # Plot known CVs on top if available
        if has_known_cvs:
            plt.scatter(
                known_cvs_centered['l_centered'],
                known_cvs_centered['b'],
                alpha=0.8,
                s=30,
                color='lime',
                marker='*',
                edgecolors='black',
                label='Known CVs'
            )
            plt.legend(loc='upper right')
        
        plt.xlabel('Galactic Longitude (l) [deg]')
        plt.ylabel('Galactic Latitude (b) [deg]')
        plt.title('Density Distribution of CV Candidates in VVV Survey Region')
        plt.grid(True, alpha=0.3)
        
        # Use same axis limits as the scatter plot
        plt.xlim(l_min, l_max)
        plt.ylim(b_min, b_max)
        
        plt.savefig(os.path.join(viz_dir, 'galactic_density_vvv.png'), dpi=300)
        plt.close()
        
        # Create separate zoomed plots for bulge and disk regions
        # 1. Bulge region
        plt.figure(figsize=(10, 8))
        
        # Plot all candidates in bulge region
        plt.scatter(
            other_candidates_centered['l_centered'],
            other_candidates_centered['b'],
            alpha=0.3,
            s=5,
            color='blue',
            label='All candidates'
        )
        
        # Plot top candidates
        plt.scatter(
            top_candidates_centered['l_centered'],
            top_candidates_centered['b'],
            alpha=0.6,
            s=15,
            color='red',
            label=f'Top {len(top_candidates)} candidates'
        )
        
        # Plot known CVs if available
        if has_known_cvs:
            plt.scatter(
                known_cvs_centered['l_centered'],
                known_cvs_centered['b'],
                alpha=0.8,
                s=30,
                color='green',
                marker='*',
                label='Known CVs'
            )
        
        # Mark the Galactic center
        plt.scatter(
            0, 0,
            marker='+',
            s=100,
            color='black',
            label='Galactic Center'
        )
        
        plt.xlabel('Galactic Longitude (l) [deg]')
        plt.ylabel('Galactic Latitude (b) [deg]')
        plt.title('CV Candidates in VVV Bulge Region')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Set limits to focus on bulge region
        plt.xlim(-12, 12)
        plt.ylim(-10, 5)
        
        plt.savefig(os.path.join(viz_dir, 'galactic_distribution_bulge.png'), dpi=300)
        plt.close()
        
        # 2. Disk region
        plt.figure(figsize=(12, 6))
        
        # Plot all candidates in disk region
        plt.scatter(
            other_candidates_centered['l_centered'],
            other_candidates_centered['b'],
            alpha=0.3,
            s=5,
            color='blue',
            label='All candidates'
        )
        
        # Plot top candidates
        plt.scatter(
            top_candidates_centered['l_centered'],
            top_candidates_centered['b'],
            alpha=0.6,
            s=15,
            color='red',
            label=f'Top {len(top_candidates)} candidates'
        )
        
        # Plot known CVs if available
        if has_known_cvs:
            plt.scatter(
                known_cvs_centered['l_centered'],
                known_cvs_centered['b'],
                alpha=0.8,
                s=30,
                color='green',
                marker='*',
                label='Known CVs'
            )
        
        plt.xlabel('Galactic Longitude (l) [deg]')
        plt.ylabel('Galactic Latitude (b) [deg]')
        plt.title('CV Candidates in VVV Disk Region')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Set limits to focus on disk region
        plt.xlim(-65, -10)
        plt.ylim(-10, -1)
        
        plt.savefig(os.path.join(viz_dir, 'galactic_distribution_disk.png'), dpi=300)
        plt.close()
        # 4. Classification confidence distribution (if available)
        if sort_col is not None and sort_col in self.cv_candidates.columns:
            plt.figure(figsize=(14, 10))
            
            # Create histogram of classification confidence/probability
            plt.hist(
                self.cv_candidates[sort_col],
                bins=50,
                alpha=0.7,
                color='blue',
                label='All candidates'
            )
            
            # Mark top candidates threshold
            if len(top_candidates) < len(self.cv_candidates):
                min_top_value = top_candidates[sort_col].min()
                plt.axvline(
                    min_top_value,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'Top {len(top_candidates)} threshold: {min_top_value:.4f}'
                )
            
            # Add key thresholds
            if sort_col != 'best_fap':
                plt.axvline(0.8, color='green', linestyle='-', alpha=0.7, label='High confidence threshold (0.8)')
                plt.axvline(0.5, color='orange', linestyle='-', alpha=0.7, label='Medium confidence threshold (0.5)')
            else:
                # For FAP, lower is better
                plt.axvline(0.1, color='green', linestyle='-', alpha=0.7, label='High confidence threshold (0.1)')
                plt.axvline(0.3, color='orange', linestyle='-', alpha=0.7, label='Medium confidence threshold (0.3)')
            
            plt.xlabel(sort_col.replace('_', ' ').title())
            plt.ylabel('Number of Candidates')
            plt.title(f'Distribution of {sort_col.replace("_", " ").title()} Values')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            # Add count information
            plt.figtext(
                0.02, 0.02, info_text,
                ha='left',
                va='bottom',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
            
            plt.savefig(os.path.join(viz_dir, f'{sort_col}_distribution.png'), dpi=300)
            plt.close()
            
            # Create scatter plot of classification confidence vs period
            plt.figure(figsize=(14, 10))
            
            # Plot all candidates
            period_hours_all = self.cv_candidates['true_period'] * 24.0
            plt.scatter(
                np.log10(period_hours_all),
                self.cv_candidates[sort_col],
                alpha=0.3,
                s=5,
                color='blue',
                label='All candidates'
            )
            
            # Plot known CVs if available
            if has_known_cvs:
                period_hours_known = known_cvs['true_period'] * 24.0
                plt.scatter(
                    np.log10(period_hours_known),
                    known_cvs[sort_col],
                    alpha=0.8,
                    s=30,
                    color='green',
                    marker='*',
                    label='Known CVs'
                )
            
            # Add period gap shaded region
            plt.axvspan(np.log10(2), np.log10(3), alpha=0.1, color='gray', label='Period gap')
            
            # Add thresholds
            if sort_col != 'best_fap':
                plt.axhline(0.8, color='green', linestyle='-', alpha=0.7, label='High confidence threshold (0.8)')
                plt.axhline(0.5, color='orange', linestyle='-', alpha=0.7, label='Medium confidence threshold (0.5)')
            else:
                # For FAP, lower is better
                plt.axhline(0.1, color='green', linestyle='-', alpha=0.7, label='High confidence threshold (0.1)')
                plt.axhline(0.3, color='orange', linestyle='-', alpha=0.7, label='Medium confidence threshold (0.3)')
            
            plt.xlabel('log₁₀(Period) [hours]')
            plt.ylabel(sort_col.replace('_', ' ').title())
            plt.title(f'{sort_col.replace("_", " ").title()} vs. Period')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            plt.savefig(os.path.join(viz_dir, f'{sort_col}_vs_period.png'), dpi=300)
            plt.close()
        
        print(f"Distribution visualizations saved to {viz_dir}")
        return


    def generate_summary(self):
        """Generate a comprehensive summary report of the CV candidates with classification details."""
        if not hasattr(self, 'cv_candidates') or len(self.cv_candidates) == 0:
            print("No CV candidates for summary.")
            return
        
        print("Generating comprehensive summary report...")
        
        # Calculate period statistics in hours
        period_hours = self.cv_candidates['true_period'] * 24.0
        
        # Period distribution by range (important for CV population studies)
        p1 = (period_hours < 2).sum()
        p2 = ((period_hours >= 2) & (period_hours < 3)).sum()  # Period gap lower bound
        p3 = ((period_hours >= 3) & (period_hours < 4)).sum()  # Period gap
        p4 = ((period_hours >= 4) & (period_hours < 5)).sum()  # Period gap upper bound
        p5 = ((period_hours >= 5) & (period_hours < 10)).sum()
        p6 = (period_hours >= 10).sum()
        
        # Determine sorting criteria for "top candidates"
        # Prioritize XGBoost classifier results if available
        if 'cv_prob' in self.cv_candidates.columns:
            primary_sort_col = 'cv_prob'
            sort_label = 'Classification Probability'
        elif 'blended_score' in self.cv_candidates.columns:
            primary_sort_col = 'blended_score'
            sort_label = 'Blended Score'
        elif 'confidence' in self.cv_candidates.columns:
            primary_sort_col = 'confidence'
            sort_label = 'Confidence'
        else:
            # Fallback to FAP (lower is better)
            primary_sort_col = 'best_fap'
            sort_label = 'FAP'
            # Reverse sort for FAP
            top_candidates = self.cv_candidates.sort_values(primary_sort_col, ascending=True).head(20)
        
        # Get top candidates (default: descending order - higher probability/confidence is better)
        if primary_sort_col != 'best_fap':
            top_candidates = self.cv_candidates.sort_values(primary_sort_col, ascending=False).head(20)
        
        # Create summary file
        summary_path = os.path.join(self.output_dir, 'cv_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("PRIMVS CV Candidate Summary\n")
            f.write("==========================\n\n")
            
            f.write(f"Total candidates: {len(self.cv_candidates)}\n\n")
            
            # Period distribution - critical for CV population studies
            f.write("Period Distribution:\n")
            f.write(f"  < 2 hours:  {p1} ({100*p1/len(self.cv_candidates):.1f}%) - Short period CVs\n")
            f.write(f"  2-3 hours:  {p2} ({100*p2/len(self.cv_candidates):.1f}%) - Period gap lower bound\n")
            f.write(f"  3-4 hours:  {p3} ({100*p3/len(self.cv_candidates):.1f}%) - Period gap\n")
            f.write(f"  4-5 hours:  {p4} ({100*p4/len(self.cv_candidates):.1f}%) - Period gap upper bound\n")
            f.write(f"  5-10 hours: {p5} ({100*p5/len(self.cv_candidates):.1f}%) - Long period CVs\n")
            f.write(f"  > 10 hours: {p6} ({100*p6/len(self.cv_candidates):.1f}%) - Possible misclassifications or unusual CVs\n\n")
            
            # Amplitude statistics - important for CV subtype classification
            f.write("Amplitude Statistics:\n")
            f.write(f"  Minimum: {self.cv_candidates['true_amplitude'].min():.2f} mag\n")
            f.write(f"  1st Quartile: {self.cv_candidates['true_amplitude'].quantile(0.25):.2f} mag\n")
            f.write(f"  Median:  {self.cv_candidates['true_amplitude'].median():.2f} mag\n")
            f.write(f"  3rd Quartile: {self.cv_candidates['true_amplitude'].quantile(0.75):.2f} mag\n")
            f.write(f"  Maximum: {self.cv_candidates['true_amplitude'].max():.2f} mag\n")
            f.write(f"  Mean:    {self.cv_candidates['true_amplitude'].mean():.2f} mag\n\n")
            
            # Classification statistics if available (from XGBoost)
            if 'cv_prob' in self.cv_candidates.columns:
                f.write("XGBoost Classification Statistics:\n")
                f.write(f"  Minimum: {self.cv_candidates['cv_prob'].min():.4f}\n")
                f.write(f"  1st Quartile: {self.cv_candidates['cv_prob'].quantile(0.25):.4f}\n")
                f.write(f"  Median:  {self.cv_candidates['cv_prob'].median():.4f}\n")
                f.write(f"  3rd Quartile: {self.cv_candidates['cv_prob'].quantile(0.75):.4f}\n")
                f.write(f"  Maximum: {self.cv_candidates['cv_prob'].max():.4f}\n")
                f.write(f"  Mean:    {self.cv_candidates['cv_prob'].mean():.4f}\n\n")
                
                # Count high confidence candidates
                high_conf = (self.cv_candidates['cv_prob'] >= 0.8).sum()
                med_conf = ((self.cv_candidates['cv_prob'] >= 0.6) & (self.cv_candidates['cv_prob'] < 0.8)).sum()
                low_conf = (self.cv_candidates['cv_prob'] < 0.6).sum()
                
                f.write(f"  High confidence (≥0.8): {high_conf} ({100*high_conf/len(self.cv_candidates):.1f}%)\n")
                f.write(f"  Medium confidence (0.6-0.8): {med_conf} ({100*med_conf/len(self.cv_candidates):.1f}%)\n")
                f.write(f"  Low confidence (<0.6): {low_conf} ({100*low_conf/len(self.cv_candidates):.1f}%)\n\n")
                
            # Embedding proximity statistics if available
            if 'embedding_similarity' in self.cv_candidates.columns:
                f.write("Embedding Similarity Statistics (proximity to known CVs):\n")
                f.write(f"  Minimum: {self.cv_candidates['embedding_similarity'].min():.4f}\n")
                f.write(f"  1st Quartile: {self.cv_candidates['embedding_similarity'].quantile(0.25):.4f}\n")
                f.write(f"  Median:  {self.cv_candidates['embedding_similarity'].median():.4f}\n")
                f.write(f"  3rd Quartile: {self.cv_candidates['embedding_similarity'].quantile(0.75):.4f}\n")
                f.write(f"  Maximum: {self.cv_candidates['embedding_similarity'].max():.4f}\n")
                f.write(f"  Mean:    {self.cv_candidates['embedding_similarity'].mean():.4f}\n\n")
            
            # False alarm probability
            f.write("Period False Alarm Probability:\n")
            f.write(f"  Minimum: {self.cv_candidates['best_fap'].min():.4f}\n")
            f.write(f"  1st Quartile: {self.cv_candidates['best_fap'].quantile(0.25):.4f}\n")
            f.write(f"  Median:  {self.cv_candidates['best_fap'].median():.4f}\n")
            f.write(f"  3rd Quartile: {self.cv_candidates['best_fap'].quantile(0.75):.4f}\n")
            f.write(f"  Maximum: {self.cv_candidates['best_fap'].max():.4f}\n")
            f.write(f"  Mean:    {self.cv_candidates['best_fap'].mean():.4f}\n\n")
            
            # Top 20 candidates
            f.write(f"Top 20 CV Candidates (sorted by {sort_label}):\n")
            f.write("=================================================\n")
            
            # Create a formatted table header
            header = f"{'Rank':4} {'Source ID':15} {'Period (hr)':10} {'Amp (mag)':10} {'FAP':8}"
            
            # Add XGBoost probability if available
            if 'cv_prob' in top_candidates.columns:
                header += f" {'XGBoost':8}"
                
            # Add embedding similarity if available
            if 'embedding_similarity' in top_candidates.columns:
                header += f" {'Embed Sim':10}"
                
            # Add blended score if available
            if 'blended_score' in top_candidates.columns:
                header += f" {'Blend Score':12}"
                
            f.write(f"{header}\n")
            f.write("-" * len(header) + "\n")
            
            # ID column to display
            id_col = 'sourceid' if 'sourceid' in top_candidates.columns else 'primvs_id'
                    
            for i, (_, cand) in enumerate(top_candidates.iterrows()):
                # Format source ID to be readable
                source_id = str(cand[id_col])
                if len(source_id) > 15:
                    source_id = source_id[:12] + "..."
                
                period_hr = cand['true_period'] * 24.0
                amp = cand['true_amplitude']
                fap = cand['best_fap']
                
                line = f"{i+1:4d} {source_id:15} {period_hr:10.2f} {amp:10.2f} {fap:8.4f}"
                
                # Add XGBoost probability if available
                if 'cv_prob' in cand:
                    line += f" {cand['cv_prob']:8.4f}"
                
                # Add embedding similarity if available
                if 'embedding_similarity' in cand:
                    line += f" {cand['embedding_similarity']:10.4f}"
                    
                # Add blended score if available
                if 'blended_score' in cand:
                    line += f" {cand['blended_score']:12.4f}"
                    
                f.write(f"{line}\n")
                
            f.write("\n")
            
            # Distribution of candidates in the period-amplitude (Bailey) diagram
            f.write("Bailey Diagram Distribution (log P vs. Amplitude):\n")
            f.write("===============================================\n")
            
            # Create 2D histogram bins for period (log scale) and amplitude
            log_period_bins = np.linspace(-1, 2, 7)  # log10 of period in hours
            amp_bins = np.linspace(0, 3, 7)  # amplitude in mag
            
            # Convert period to log scale
            log_period = np.log10(period_hours)
            
            # Create 2D histogram
            hist, _, _ = np.histogram2d(log_period, self.cv_candidates['true_amplitude'], 
                                       bins=[log_period_bins, amp_bins])
            
            # Print text-based representation of the 2D histogram
            f.write(f"{'':10}|")
            for i in range(len(log_period_bins)-1):
                bin_center = (log_period_bins[i] + log_period_bins[i+1]) / 2
                period_val = 10**bin_center
                f.write(f" {period_val:.1f}h ")
            f.write("\n")
            
            f.write("-" * 10 + "+" + "-" * (6 * (len(log_period_bins)-1)) + "\n")
            
            for i in range(len(amp_bins)-1, 0, -1):
                amp_val = (amp_bins[i-1] + amp_bins[i]) / 2
                f.write(f"Amp {amp_val:4.1f} |")
                
                for j in range(len(log_period_bins)-1):
                    count = int(hist[j, i-1])
                    if count == 0:
                        f.write("     .")
                    elif count < 10:
                        f.write(f"    {count}")
                    elif count < 100:
                        f.write(f"   {count}")
                    else:
                        f.write(f"  {count}")
                f.write("\n")
                
            f.write("\n")
            
            # Additional information if available
            if 'is_known_cv' in self.cv_candidates.columns:
                known_count = self.cv_candidates['is_known_cv'].sum()
                f.write(f"Known CVs included in candidates: {known_count}\n")
                
                if known_count > 0 and 'distance_to_nearest_cv' in self.cv_candidates.columns:
                    f.write("\nDistance statistics from candidates to nearest known CV:\n")
                    distances = self.cv_candidates.loc[~self.cv_candidates['is_known_cv'], 'distance_to_nearest_cv']
                    f.write(f"  Minimum: {distances.min():.4f}\n")
                    f.write(f"  1st Quartile: {distances.quantile(0.25):.4f}\n")
                    f.write(f"  Median:  {distances.median():.4f}\n")
                    f.write(f"  3rd Quartile: {distances.quantile(0.75):.4f}\n")
                    f.write(f"  Maximum: {distances.max():.4f}\n")
            
        print(f"Comprehensive summary saved to {summary_path}")
        
        # Also save a detailed CSV with the top candidates
        top_csv_path = os.path.join(self.output_dir, 'top_cv_candidates.csv')
        top_candidates.to_csv(top_csv_path, index=False)
        print(f"Top candidates saved to {top_csv_path}")
        
        # Generate visualizations of top candidates
        #self.visualize_top_candidates(top_candidates)
        self.visualize_candidate_distributions()




    def visualize_two_stage_classification(self):
        """
        Generate visualizations that illustrate the performance and comparative contributions
        of traditional feature-based classification versus embedding-based classification
        in the two-stage ensemble model.
        """
        if not hasattr(self, 'filtered_data') or 'cv_prob' not in self.filtered_data.columns:
            print("No classification results available for visualization.")
            return
        
        # Check if we have results from both stages
        has_two_stage = all(col in self.filtered_data.columns for col in ['cv_prob_trad', 'cv_prob_emb'])
        
        if not has_two_stage:
            print("Single-stage classification detected. Skipping two-stage visualization.")
            return
        
        print("Generating two-stage classification visualizations...")
        

        # Probability comparison scatter plot
        plt.figure(figsize=(10, 8))

        # Use hexbin for density visualization
        hb = plt.hexbin(
            self.filtered_data['cv_prob_trad'], 
            self.filtered_data['cv_prob_emb'], 
            gridsize=50, cmap='viridis', bins='log'
        )

        # Add identity line
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Identity Line')

        # Add decision boundaries
        plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        # Annotate quadrants
        plt.text(0.25, 0.75, "Traditional: No\nEmbedding: Yes", 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.75, 0.75, "Both: Yes", 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.25, 0.25, "Both: No", 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.text(0.75, 0.25, "Traditional: Yes\nEmbedding: No", 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

        plt.xlabel('Traditional Features CV Probability')
        plt.ylabel('Embedding Features CV Probability')
        plt.title('Comparison of CV Probabilities: Traditional vs. Embedding Features')
        plt.colorbar(hb, label='Log Count')
        plt.grid(True, alpha=0.3)
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.savefig(os.path.join(self.output_dir, 'two_stage_probability_comparison.png'), dpi=300)
        plt.close()


        # Calculate agreement score (difference between probabilities)
        agreement = 1 - np.abs(self.filtered_data['cv_prob_trad'] - self.filtered_data['cv_prob_emb'])
        

        # 4. Bailey diagram with model agreement
        if all(col in self.filtered_data.columns for col in ['true_period', 'true_amplitude']):
            plt.figure(figsize=(10, 8))
            
            # Prepare data
            period_hours = self.filtered_data['true_period'] * 24.0
            log_period = np.log10(period_hours)
            amplitude = self.filtered_data['true_amplitude']
            
            # Plot scatter colored by model agreement
            sc = plt.scatter(
                log_period, 
                amplitude,
                c=agreement,
                cmap='coolwarm',
                alpha=0.7,
                s=8,
                edgecolor='none'
            )
            
            plt.colorbar(sc, label='Model Agreement')
            
            # Add period gap reference lines
            plt.axvline(np.log10(2), color='gray', linestyle='--', alpha=0.5, label='2 hours')
            plt.axvline(np.log10(3), color='gray', linestyle='--', alpha=0.5, label='3 hours')
            
            plt.xlabel('Log₁₀(Period) [hours]')
            plt.ylabel('Amplitude [mag]')
            plt.title('Bailey Diagram of CV Candidates with Model Agreement')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(self.output_dir, 'two_stage_bailey_diagram.png'), dpi=300)
            plt.close()
            
        print("Two-stage classification visualizations completed.")














def main():
    """Main function to run the CV finder."""
    # Fixed paths to match your description
    primvs_file = '../PRIMVS/PRIMVS_CC_CV_cand.fits'
    output_dir = "../PRIMVS/cv_results"
    known_cvs = "../PRIMVS/PRIMVS_CC_CV.fits"
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
        print(f"CV finder completed successfully. Results in {output_dir}")
        return 0
    else:
        print("CV finder failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    sys.exit(main())    