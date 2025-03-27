import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from astropy.table import Table
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Column
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

filename = '/beegfs/car/njm/OUTPUT/PRIMVS.fits'
# Load FITS file
data = Table.read(fits.open(filename, memmap=True)[1], format='fits')

# Load FITS file
spicydata = Table.read('/beegfs/car/njm/OUTPUT/PRIMVS_spicyclass1.fits', format='fits')

data_sourceids = np.array(data['sourceid'])
spicydata_sourceids = np.array(spicydata['sourceid'])
common_sourceids = np.intersect1d(data_sourceids, spicydata_sourceids)
subset_data_mask = np.isin(data_sourceids, common_sourceids)



selected_feature_names = ['l','b','parallax','pmra','pmdec','Z-K','Y-K','J-K','H-K',
			'mag_avg','Cody_M','Cody_Q','stet_k','eta','eta_e','med_BRP',
			'range_cum_sum','max_slope','MAD','mean_var','percent_amp',
			'true_amplitude','roms','p_to_p_var','lag_auto','AD',
			'std_nxs','weight_mean','weight_std','weight_skew','weight_kurt',
			'mean','std','skew','kurt','time_range','true_period']

feature_weights = [0.5,0.5,0.3,0.4,0.4,0.9,0.9,0.9,0.9,0.7,0.6,0.4,0.4,0.4,0.5,0.5,0.4,0.7,0.6,0.8,0.7,0.6,0.5,0.5,0.7,0.5,0.4,0.4,0.6,0.6,0.6,0.6,0.4,0.4,0.2,0.9]
#feature_weights = [0.5, 0.5, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.6, 0.9]





selected_features = data[selected_feature_names]

# Convert selected features to Pandas DataFrame
df = selected_features.to_pandas()

#Convert missing values to 0
df.fillna(0, inplace=True)

# Convert DataFrame to NumPy array
features = df.values

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features to PyTorch tensors
features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
weights_tensor = torch.tensor(feature_weights, dtype=torch.float32)

# Multiply the normalized features by the weights
data_features_tensor = features_tensor * weights_tensor


# Step 1: Prepare the training data
train_features = data_features_tensor[subset_data_mask]



# One-Class SVM
model_svm = OneClassSVM(gamma='auto')
model_svm.fit(train_features)
predictions_svm = model_svm.predict(features_tensor)
confidence_svm = model_svm.decision_function(features_tensor).flatten()  # Confidence scores
data.add_column(Column(name='SVM_predicted_class', data=predictions_svm))
data.add_column(Column(name='SVM_confidence', data=confidence_svm))

# Local Outlier Factor
model_lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
model_lof.fit(train_features)
predictions_lof = model_lof.predict(data_features_tensor)
confidence_lof = model_lof.negative_outlier_factor_  # Confidence scores (more negative means more of an outlier)
data.add_column(Column(name='LOF_predicted_class', data=predictions_lof))
data.add_column(Column(name='LOF_confidence', data=confidence_lof))

# Isolation Forest
model_if = IsolationForest(contamination=0.1)
model_if.fit(train_features)
predictions_if = model_if.predict(features_tensor)
confidence_if = model_if.score_samples(features_tensor)  # Anomaly scores (lower scores are more anomalous)
data.add_column(Column(name='IF_predicted_class', data=predictions_if))
data.add_column(Column(name='IF_confidence', data=-confidence_if))  # Negate to align with convention (higher is more confident)

# Gaussian Mixture Model
model_gmm = GaussianMixture(n_components=2)
model_gmm.fit(train_features)
log_likelihoods_gmm = model_gmm.score_samples(features_tensor)  # Log-likelihood as a confidence measure
data.add_column(Column(name='GMM_predicted_class', data=np.sign(log_likelihoods_gmm)))  # Sign of log-likelihood for class prediction
data.add_column(Column(name='GMM_confidence', data=log_likelihoods_gmm))

# Write the data table to a new FITS file with confidence measures included
data.write('/beegfs/car/njm/PRIMVS/autoencoder/forest_class1_with_confidence.fits', format='fits', overwrite=True)


