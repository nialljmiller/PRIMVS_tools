import pandas as pd
from astropy.io import fits
import xgboost as xgb
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

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
train_fits_file = 'PRIMVS_P_ASASSN'
test_fits_file = 'PRIMVS_P'
output_fits_file = 'PRIMVS_P_CLASS'

train = True
if train:

    train_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + train_fits_file + '.fits')
    test_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + test_fits_file + '.fits')
    # Select features and target variable from training data
    # Replace 'feature_columns' with your list of feature column names
    # and 'target_column' with your target column name
    train_df.rename(columns={'l_2': 'l'}, inplace=True)
    train_df.rename(columns={'b_2': 'b'}, inplace=True)
    train_df['l'] = ((train_df['l'] + 180) % 360) - 180
    test_df['l'] = ((test_df['l'] + 180) % 360) - 180    
    
    feature_columns = ['l', 'b','Cody_M', 
                    'stet_k', 'eta', 'eta_e',
                    'z_med_mag-ks_med_mag',
                    'y_med_mag-ks_med_mag',
                    'j_med_mag-ks_med_mag',
                    'h_med_mag-ks_med_mag',
                    'med_BRP', 'range_cum_sum', 'max_slope', 
                    'MAD', 'mean_var', 'percent_amp',
                    'true_amplitude', 'roms', 'p_to_p_var',
                    'lag_auto', 'AD', 'std_nxs',
                    'weight_mean', 'weight_std', 'weight_skew',
                    'weight_kurt', 'mean', 'std', 
                    'skew', 'kurt', 'time_range', 'true_period']


    class_mapping = {
	    # Pulsating Variables
	    'RRLy': 'Pulsating Variables', 'DCEPS': 'Pulsating Variables', 
	    'DCEP': 'Pulsating Variables', 'CWB': 'Pulsating Variables',
	    'CWA': 'Pulsating Variables', 
	    'RVA': 'Pulsating Variables',
	    'DSCT': 'Pulsating Variables',
	    'HADS': 'Pulsating Variables',
	    'GCAS': 'Pulsating Variables',
	    'RRAB': 'Pulsating Variables',
	    'RRC': 'Pulsating Variables',
	    'RRD': 'Pulsating Variables',

	    'ROT': 'Rotating Variables',
	    'BYDra': 'Rotating Variables',

	    # Binaries
	    'EB': 'Binaries', 'EW': 'Binaries', 
	    'EA': 'Binaries',
	    # Young Stellar Objects (YSOs) and Protostars
	    'YSO': 'YSOs',

	    # Evolved Stars
	    'M': 'Evolved Stars', 


	    # Misc
	    'L': 'Misc', 
	    'VAR': 'Misc', 
	    'SR': 'Misc', 


    }

    # Apply the mapping to create a new 'broad_class' column
    train_df['broad_class'] = train_df['variable_type'].apply(lambda x: class_mapping.get(x, 'Miscellaneous'))

    target_column = 'variable_type'

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    print(X_train)
    # For numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    # Check and fill missing values for numerical columns in train_df
    for col in numerical_cols:
        if col in train_df.columns:
            median_value = train_df[col].median()
            train_df[col].fillna(median_value, inplace=True)

    # Assuming the error occurs when you try to fill missing values in test_df
    # Check and fill missing values for numerical columns in test_df
    for col in numerical_cols:
        if col in test_df.columns:
            # Use median from train_df to fill missing values in test_df to avoid data leakage
            median_value = train_df[col].median()
            test_df[col].fillna(median_value, inplace=True)


    # For categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = train_df[col].mode()[0]
        if col in train_df.columns:
            train_df[col].fillna(mode_value, inplace=True)
        if col in test_df.columns:
            test_df[col].fillna(mode_value, inplace=True)

  
    train_df[target_column] = train_df[target_column].str.strip().str.replace('?', '').str.replace('_Candidate', '').str.replace(':', '')

    # Re-map labels to integers after consolidation
    unique_labels = train_df[target_column].unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    train_df['label_encoded'] = train_df[target_column].map(label_to_int)

    # Now y_train should be updated to reflect this
    y_train = train_df['label_encoded']

    # You might want to update num_classes if it's used later for model training
    num_classes = len(unique_labels)

    # Prepare the DMatrix
    X_train = train_df[feature_columns]
    dtrain = xgb.DMatrix(X_train, label=y_train)


    # Split your original training data into a new training set and a validation set
    X_train, X_val, y_train, y_val = train_test_split(train_df[feature_columns], train_df['label_encoded'], test_size=0.1, random_state=42)
    
    # Assuming 'X_train' and 'X_val' are your training and validation features
    scaler = MinMaxScaler()

    # Fit on training data
    scaler.fit(X_train[numerical_cols])

    # Transform both training and validation data
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
        
    
    print(X_train)
    # Prepare the DMatrix for the new training and validation sets
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
#[7073]  train-mlogloss:0.54103  val-mlogloss:1.07035
#[8739]  train-mlogloss:0.73107  val-mlogloss:1.05515

    # XGBoost parameters
    # You may want to tune these parameters
    params = {
        'max_depth': 4,
        'eta': 0.001,
        'objective': 'multi:softprob',  # for multi-class classification
        'num_class': num_classes,  # specify the number of classes
        'eval_metric': 'mlogloss',  # evaluation metric
        'min_child_weight': 1,  # Minimum sum of instance weight (hessian) needed in a child
        'gamma': 0,  # Minimum loss reduction required to make a further partition on a leaf node of the tree
        'subsample': 1,  # Subsample ratio of the training instances
        'colsample_bytree': 1,  # Subsample ratio of columns when constructing each tree
        'colsample_bylevel': 1,  # Subsample ratio of columns for each split, in each level
        'colsample_bynode': 1,  # Subsample ratio of columns for each split
        'lambda': 1,  # L2 regularization term on weights
        'alpha': 0,  # L1 regularization term on weights
        'tree_method': 'auto',  # The tree construction algorithm used in XGBoost
        'scale_pos_weight': 1,  # Balancing of positive and negative weights
        'grow_policy': 'lossguide',  # Strategy to grow trees
        'max_leaves': 0,  # Maximum number of nodes to be added. 0 means no limit
        'max_bin': 256,  # Max number of bins for histogram construction
        'predictor': 'auto',  # Type of predictor algorithm to use
        'sampling_method': 'uniform',  # Sampling method for training data
    }


    print(params)
    # Train the model
    num_round = 100000  # number of boosting rounds
    bst = xgb.train(params, dtrain, num_round)
    eval_set = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(params, dtrain, num_round, evals=eval_set, early_stopping_rounds=10)

    # Prepare test data
    X_test = test_df[feature_columns]
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    dtest = xgb.DMatrix(X_test)

    # Make predictions
    preds = bst.predict(dtest)
    # preds will contain the probability of each class for each instance
    print(preds)
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
    test_df['variable_type'] = most_likely_labels
    test_df['probability'] = max_probabilities


    # The DataFrame now contains a 'broad_class' column with the broader categories

    # Select columns to write to FITS (you may adjust this as needed)
    columns_to_write = test_df.columns  # This includes all columns; adjust as needed

    # Convert DataFrame to Astropy Table, then write to FITS
    table_to_write = Table.from_pandas(test_df[columns_to_write])
    table_to_write.write('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits', overwrite=True)
else:
    test_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits')





# Filter the DataFrame
test_df['broad_class'] = test_df['variable_type'].apply(lambda x: class_mapping.get(x, 'Miscellaneous'))
df_filtered = test_df[test_df['probability'] > 0.9]
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



