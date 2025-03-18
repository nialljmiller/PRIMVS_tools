import pandas as pd
from astropy.io import fits
import xgboost as xgb
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from FITS file
def read_fits_data(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        df = pd.DataFrame(data)
    return df

def read_fits_data(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        df = Table(data).to_pandas()  # Convert to a pandas DataFrame
        for column in df.columns:
            if df[column].dtype.byteorder == '>':  # Big-endian
                df[column] = df[column].byteswap().newbyteorder()
    return df

# Read your training and test data
#train_fits_file = 'PRIMVS_P_GAIA'
train_fits_file = 'PRIMVS_P_SIMBAD'
test_fits_file = 'PRIMVS_P'
output_fits_file = 'PRIMVS_P_CLASS'

train = False
if train:

    train_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + train_fits_file + '.fits')
    test_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + test_fits_file + '.fits')

    # Select features and target variable from training data
    # Replace 'feature_columns' with your list of feature column names
    # and 'target_column' with your target column name
    feature_columns = ['l', 'b','Cody_M', 
                    'stet_k', 'eta', 'eta_e',
                    'z_med_mag-ks_med_mag', 'z_mean_mag-ks_mean_mag', 'z_mad_mag-ks_mad_mag', 
                    'y_med_mag-ks_med_mag', 'y_mean_mag-ks_mean_mag', 'y_mad_mag-ks_mad_mag', 
                    'j_med_mag-ks_med_mag', 'j_mean_mag-ks_mean_mag', 'j_mad_mag-ks_mad_mag', 
                    'h_med_mag-ks_med_mag', 'h_mean_mag-ks_mean_mag', 'h_mad_mag-ks_mad_mag',
                    'med_BRP', 'range_cum_sum', 'max_slope', 
                    'MAD', 'mean_var', 'percent_amp',
                    'true_amplitude', 'roms', 'p_to_p_var',
                    'lag_auto', 'AD', 'std_nxs',
                    'weight_mean', 'weight_std', 'weight_skew',
                    'weight_kurt', 'mean', 'std', 
                    'skew', 'kurt', 'time_range', 'true_period']


    target_column = 'main_type'

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    # For numerical columns
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        median_value = train_df[col].median()
        train_df[col].fillna(median_value, inplace=True)
        test_df[col].fillna(median_value, inplace=True)

    # For categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = train_df[col].mode()[0]
        train_df[col].fillna(mode_value, inplace=True)
        test_df[col].fillna(mode_value, inplace=True)

    # 2. Convert String Labels to Numeric

    # Provided array of valid main types
    valid_main_types = ['EB*','EllipVar', 'RRLyr','LP*_Candidate', 'AGB*_Candidate','YSO_Candidate', 'PulsV*WVir','Star', 'V*','PulsV*delSct', 'OH/IR','deltaCep', 'RGB*','YSO', 'IR','Em*', 'Mira','LPV*', 'Pec*','EB*_Candidate', 'PulsV*RVTau','RRLyr_Candidate', 'TTau*','AGB*', 'Mi*_Candidate','PulsV*WVir_Candidate', '**','HB*', 'outflow?_Candidate','Cepheid', 'HMXB','EmObj', 'Nova','Nova_Candidate', 'Orion_V*','PulsV*', 'CataclyV*','Ae*_Candidate', 'BlueStraggler','C*_Candidate', 'RSCVn','HV*', 'HMXB_Candidate','SB*', 'Outflow','WR*', 'CV*_Candidate','Irregular_V*', 'BYDra','RotV*', 'C*','postAGB*_Candidate', 'Eruptive*','WD*_Candidate', 'Cepheid_Candidate','Pulsar', 'V*?_Candidate','TTau*_Candidate', 'Be*','OpCl']

    train_df['main_type'] = train_df['main_type'].str.strip().str.replace('?', '').str.replace('_Candidate', '')
    train_df = train_df[train_df['main_type'].isin(valid_main_types)]
    # Clean 'main_type' column

    # Filter rows where cleaned 'main_type' is in the provided array after cleaning

    # Re-map labels to integers after consolidation
    unique_labels = train_df['main_type'].unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    train_df['label_encoded'] = train_df['main_type'].map(label_to_int)

    # Now y_train should be updated to reflect this
    y_train = train_df['label_encoded']

    # You might want to update num_classes if it's used later for model training
    num_classes = len(unique_labels)

    # Prepare the DMatrix
    X_train = train_df[feature_columns]
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # XGBoost parameters
    # You may want to tune these parameters
    params = {
        'max_depth': 3,
        'eta': 0.3,
        'objective': 'multi:softprob',  # for multi-class classification
        'num_class': num_classes,  # specify the number of classes
        'eval_metric': 'mlogloss',  # evaluation metric
    }


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Example model
model = XGBClassifier(objective='multi:softprob', num_class=num_classes)

# Define parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'eta': [0.1, 0.01, 0.001],
    'subsample': [0.8, 0.9, 1.0]
}

# Grid search
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_log_loss', verbose=2)
grid_search.fit(X_train, y_train)

# Use the best parameters
best_params = grid_search.best_params_












    # Train the model
    num_round = 100  # number of boosting rounds
    bst = xgb.train(params, dtrain, num_round)

    # Prepare test data
    X_test = test_df[feature_columns]
    dtest = xgb.DMatrix(X_test)

    # Make predictions
    preds = bst.predict(dtest)
    # preds will contain the probability of each class for each instance

    # Convert probabilities to class labels if necessary
    # For example, you could take the argmax of the predictions
    pred_labels = preds.argmax(axis=1)

    # Assume preds is obtained from the bst.predict(dtest) as before

    # Convert probabilities to most likely class labels and their probabilities
    most_likely_classes = np.argmax(preds, axis=1)  # Most likely class
    max_probabilities = np.max(preds, axis=1)  # Probability of most likely class

    # Map numeric labels back to original string labels
    int_to_label = {v: k for k, v in label_to_int.items()}
    most_likely_labels = [int_to_label[cls] for cls in most_likely_classes]

    # Add predictions to test_df
    test_df['most_likely_class'] = most_likely_labels
    test_df['probability'] = max_probabilities


    class_mapping = {
	    # Pulsating Variables
	    'RRLyr': 'Pulsating Variables', 'Cepheid': 'Pulsating Variables', 
	    'deltaCep': 'Pulsating Variables', 'PulsV*delSct': 'Pulsating Variables',
	    'PulsV*WVir': 'Pulsating Variables', 
	    'PulsV*RVTau': 'Pulsating Variables',

	    'RotV*': 'Rotating Variables',
	    'BYDra': 'Rotating Variables',

	    # Binaries
	    'EB*': 'Binaries', 'SB*': 'Binaries', 
	    '**': 'Binaries',
	    'CataclyV*': 'Binaries',
	    'HMXB': 'Binaries',

	    # Young Stellar Objects (YSOs) and Protostars
	    'Orion_V*': 'YSOs', 
	    'YSO': 'YSOs', 'TTau*': 'YSOs', 'Outflow': 'YSOs',

	    # Evolved Stars
	    'Mira': 'Evolved Stars', 
	    'AGB*': 'Evolved Stars', 'RGB*': 'Evolved Stars', 
	    'OH/IR': 'Evolved Stars', 'C*': 'Evolved Stars', 
	    'HB*': 'Evolved Stars', 'WR*': 'Evolved Stars',

	    # Emission Line Stars
	    'Em*': 'Emission Line Stars', 'Be*': 'Emission Line Stars', 
	    'Nova': 'Emission Line Stars',

	    # Pulsars and Neutron Stars
	    'Pulsar': 'Neutron Stars',
    }

    # Apply the mapping to create a new 'broad_class' column
    test_df['broad_class'] = test_df['most_likely_class'].apply(lambda x: class_mapping.get(x, 'Miscellaneous'))

    # The DataFrame now contains a 'broad_class' column with the broader categories

    # Select columns to write to FITS (you may adjust this as needed)
    columns_to_write = test_df.columns  # This includes all columns; adjust as needed

    # Convert DataFrame to Astropy Table, then write to FITS
    table_to_write = Table.from_pandas(test_df[columns_to_write])
    table_to_write.write('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits', overwrite=True)
else:
    test_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits')





# Filter the DataFrame
df_filtered = test_df[test_df['probability'] > 0.8]
df_filtered = df_filtered[df_filtered['broad_class'] != 'Miscellaneous']

# Apply log transformation
df_filtered['log_true_period'] = np.log10(df_filtered['true_period'])
df_filtered['log_true_amplitude'] = np.log10(df_filtered['true_amplitude'])

# Obtain the unique classes and their corresponding codes after filtering
unique_classes = df_filtered['broad_class'].unique()
class_codes = pd.Categorical(df_filtered['broad_class'], categories=unique_classes).codes

# Create a colormap that maps class codes to colors
cmap = plt.cm.get_cmap('tab10', len(unique_classes))  # Adjust 'tab10' as needed

# Plot setup
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['log_true_period'], df_filtered['true_amplitude'], 
            c=class_codes,  # Use the class codes for color coding
            alpha=0.3,  # Semi-transparency
            s=5,  # Point size
            cmap=cmap)  # Use the created colormap

# Set custom x and y limits
plt.xlim([-2, 3])
plt.ylim([0.1, 5])

# Create legend for the broad classes
handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_, 
            markerfacecolor=cmap(i), markersize=8, alpha=1) 
            for i, class_ in enumerate(unique_classes)]

plt.legend(handles=handles, title='Broad Class',)# loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Save the plot
plt.savefig('/beegfs/car/njm/PRIMVS/'+str(output_fits_file)+'.jpg', dpi=300, bbox_inches='tight')
plt.clf()

# Make sure to run this code in your local Python environment.



