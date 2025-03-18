import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits
from astropy.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm





# Applying feature weights
def apply_feature_weights(df, feature_columns, weights):
    for feature in feature_columns:
        if feature in weights:
            df[feature] *= weights[feature]
    return df


feature_weights = {
		'Cody_M': 0.5,
		'stet_k': 0.5,
		'eta': 0.5,
		'eta_e': 0.5,
		'z_med_mag-ks_med_mag': 2.5,
		'y_med_mag-ks_med_mag': 2.5,
		'j_med_mag-ks_med_mag': 2.5,
		'h_med_mag-ks_med_mag': 2.5,
		'med_BRP': 0.5,
		'range_cum_sum': 0.1,
		'max_slope': 0.1,
		'MAD': 0.4,
		'mean_var': 0.4,
		'percent_amp': 0.2,
		'true_amplitude': 2.,
		'roms': 0.1,
		'p_to_p_var': 0.1,
		'lag_auto': 0.1,
		'AD': 0.5,
		'std_nxs': 0.1,
		'weight_mean': 0.1,
		'weight_std': 0.5,
		'weight_skew': 1.,
		'weight_kurt': 0.5,
		'mean': 0.1,
		'std': 0.5,
		'skew': 2.0,
		'kurt': 1.0,
		'true_period': 5.}
	
feature_columns = ['Cody_M', 
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
            'skew', 'kurt', 'true_period']


target_column = 'best_class_name'


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
        for column in df.columns:fnorm
            if df[column].dtype.byteorder == '>':  # Big-endian
                df[column] = df[column].byteswap().newbyteorder()
    return df

# Read your training and test data
train_fits_file = 'PRIMVS_P_GAIA'
test_fits_file = 'PRIMVS_P'
output_fits_file = 'PRIMVS_P_CLASS_GAIA'

train = False
if train:

    train_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + train_fits_file + '.fits')
    test_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + test_fits_file + '.fits')

    train_df.rename(columns={'l_2': 'l'}, inplace=True)
    train_df.rename(columns={'b_2': 'b'}, inplace=True)
    train_df['l'] = ((train_df['l'] + 180) % 360) - 180
    test_df['l'] = ((test_df['l'] + 180) % 360) - 180    
    
    train_df = train_df[train_df['best_class_score'] > 0.5]
    train_df = train_df[train_df['ls_bal_fap'] < 0.000000001]

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
        
        # Apply feature weights
    X_train = apply_feature_weights(X_train, feature_columns, feature_weights)
    X_val = apply_feature_weights(X_val, feature_columns, feature_weights)

    print(X_train)
    # Prepare the DMatrix for the new training and validation sets
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    # You may want to tune these parameters
    params = {
        'max_depth': 4,
        'eta': 0.001,
        'objective': 'multi:softprob',  # for multi-class classification
        'num_class': num_classes,  # specify the number of classes
        'eval_metric': 'mlogloss',  # evaluation metric
        'min_child_weight': 1,  # Minimum sum of instance weight (hessian) needed in a child
        'gamma': 0,  # Minimum loss reducti-on required to make a further partition on a leaf node of the tree
        'subsample': 0.8,  # Subsample ratio of the training instances
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
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
        'nthread': 64        
    }


    print("Starting model training...")

    # Define the number of boosting rounds
    num_round = 1000000

    # Define evaluation set
    eval_set = [(dtrain, 'train'), (dval, 'val')]

    # Start model training with early stopping
    bst = xgb.train(params, dtrain, num_round, evals=eval_set, early_stopping_rounds=10)

    print("Model training completed.")
    print(f"Best iteration: {bst.best_iteration}, Best score: {bst.best_score}")
    print("Preparing test data...")


    # Selecting the features from the test dataset
    X_test = test_df[feature_columns]
    # Transforming the numerical columns with the predefined scaler
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    X_test = apply_feature_weights(X_test, feature_columns, feature_weights)

    # Creating a DMatrix for the test dataset
    dtest = xgb.DMatrix(X_test)

    print("Test data preparation completed.")

    # Make predictions
    preds = bst.predict(dtest)
    # preds will contain the probability of each class for each instance
    # For example, you could take the argmax of the predictions
    pred_labels = preds.argmax(axis=1)

    # Convert probabilities to most likely class labels and their probabilities
    most_likely_classes = np.argmax(preds, axis=1)  # Most likely class
    max_probabilities = np.max(preds, axis=1)  # Probability of most likely class

    # Map numeric labels back to original string labels
    int_to_label = {v: k for k, v in label_to_int.items()}
    most_likely_labels = [int_to_label[cls] for cls in most_likely_classes]

    # Add predictions to test_df
    test_df['variable_type'] = most_likely_labels
    test_df['probability'] = max_probabilities

    columns_to_write = test_df.columns  # This includes all columns; adjust as needed

    # Convert DataFrame to Astropy Table, then write to FITS
    table_to_write = Table.from_pandas(test_df[columns_to_write])
    table_to_write.write('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits', overwrite=True)
    # Save the model to a file
    bst.save_model('/beegfs/car/njm/PRIMVS/dtree/gaia/XGBoost_gaia.model')
 

else:
    
         
    test_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits')

    test_df['l'] = ((test_df['l'] + 180) % 360) - 180    
    
    train_df = train_df[train_df['best_class_score'] > 0.5]
    train_df = train_df[train_df['ls_bal_fap'] < 0.000000001]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    print(X_train)
    # For numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
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



    # Load the model from the file
    bst = xgb.Booster()  # Initialize an empty model
    bst.load_model('/beegfs/car/njm/PRIMVS/dtree/gaia/XGBoost_gaia.model')
    
    

    train_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + train_fits_file + '.fits')
    train_df.rename(columns={'l_2': 'l'}, inplace=True)
    train_df.rename(columns={'b_2': 'b'}, inplace=True)

    
    train_df['l'] = ((train_df['l'] + 180) % 360) - 180
    
    train_df = train_df[train_df['best_class_score'] > 0.9]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if col in train_df.columns:
            median_value = train_df[col].median()
            train_df[col].fillna(median_value, inplace=True)

    # For categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = train_df[col].mode()[0]
        if col in train_df.columns:
            train_df[col].fillna(mode_value, inplace=True)
  
    train_df[target_column] = train_df[target_column].str.strip().str.replace('?', '').str.replace('_Candidate', '').str.replace(':', '')


    unique_labels = train_df[target_column].unique()
    
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {v: k for k, v in label_to_int.items()}
       
    train_df['label_encoded'] = train_df[target_column].map(label_to_int)
    y_train = train_df['label_encoded']
    num_classes = len(unique_labels)
    X_train = train_df[feature_columns]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    scaler = MinMaxScaler()
    scaler.fit(X_train[numerical_cols])
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_train = apply_feature_weights(X_train, feature_columns, feature_weights)
    dtrain = xgb.DMatrix(X_train, label=y_train)    
    
    





df_filtered = test_df[test_df['probability'] > 0.9]
df_filtered['log_true_period'] = np.log10(df_filtered['true_period'])
df_filtered['log_true_amplitude'] = np.log10(df_filtered['true_amplitude'])

df_filtered.rename(columns={'z_med_mag-ks_med_mag': 'Z-K'}, inplace=True)
df_filtered.rename(columns={'y_med_mag-ks_med_mag': 'Y-K'}, inplace=True)
df_filtered.rename(columns={'j_med_mag-ks_med_mag': 'J-K'}, inplace=True)
df_filtered.rename(columns={'h_med_mag-ks_med_mag': 'H-K'}, inplace=True)
df_filtered.rename(columns={'true_period': 'Period'}, inplace=True)
df_filtered.rename(columns={'true_amplitude': 'Amplitude'}, inplace=True)
df_filtered.rename(columns={'log_true_period': 'log10(Period)'}, inplace=True)
df_filtered.rename(columns={'log_true_amplitude': 'log10(Amplitude)'}, inplace=True)


df_filtered[target_column] = train_df[target_column].str.strip().str.replace('?', '').str.replace('_Candidate', '').str.replace(':', '')


# Your existing filter
df_filtered = df_filtered.loc[
    (df_filtered['Z-K'] != 0) & 
    (df_filtered['J-K'] != 0) & 
    (df_filtered['H-K'] != 0) & 
    (df_filtered['Y-K'] != 0)
]


# Compute mode for each column and remove rows with these mode values
for column in ['Z-K', 'J-K', 'H-K', 'Y-K']:
    mode_value = df_filtered[column].mode()[0]
    df_filtered = df_filtered.loc[df_filtered[column] != mode_value]

label_mapping = {
    'WD': 'White Dwarf',
    'ECL': 'Eclipsing Binary',
    'S': 'Short-timescale',
    'RS': 'RS Canum V',
    'ELL': 'Ellipsoidal',    
    'RR': 'RR Lyrae',        
    'CEP': 'Cepheid',        
    'SOLAR_LIKE': 'Solar-like',        
    'DSCT|GDOR|SXPHE': 'Delta Scuti',        
    'LPV': 'Long-period Variable',        
    'YSO': 'YSO',        
    'CV': 'Cataclysmic Variable',        
    'MICROLENSING': 'Microlensing',        
    'BE|GCAS|SDOR|WR': 'B-type',        
}

#label_mapping = {
#    'WD': 'White Dwarf',
#    'ECL': 'Eclipsing Binary',
#    'S': 'Short-timescale',
#    'RS': 'Eclipsing Binary',
#    'ELL': 'Eclipsing Binary',    
#    'RR': 'RR Lyrae',        
#    'CEP': 'Cepheid',        
#    'SOLAR_LIKE': 'Solar-like',        
#    'DSCT|GDOR|SXPHE': 'Delta Scuti',        
#    'LPV': 'Long-period Variable',        
#    'YSO': 'YSO',        
#    'CV': 'Cataclysmic Variable',        
#    'MICROLENSING': 'Microlensing',        
#    'BE|GCAS|SDOR|WR': 'B-type',        
#}

# Overwrite actual label names in the DataFrame
df_filtered['variable_type'] = df_filtered['variable_type'].map(label_mapping)

# Define the classes you want to keep
classes_to_keep = ['B-type', 'Cataclysmic Variable', 'YSO', 'Delta Scuti', 'Cepheid',  'RR Lyrae', 'White Dwarf', 'Eclipsing Binary', 'Solar-like']

# Filter the DataFrame to keep only the rows with the desired classes
df_filtered = df_filtered[df_filtered['variable_type'].isin(classes_to_keep)]

# Determine the top 5 most populated classes
top_5_classes = df_filtered['variable_type'].value_counts().head(5).index

# Filter the DataFrame to include only those top 5 classes
#df_filtered = df_filtered[df_filtered['variable_type'].isin(top_5_classes)]
df_filtered = df_filtered[df_filtered['Amplitude']<5]
df500 = df_filtered[df_filtered['Period']<500].copy()
df100 = df_filtered[df_filtered['Period']<100].copy()

# Obtain the unique classes and their corresponding codes after filtering
unique_classes = df_filtered['variable_type'].unique()
class_codes = pd.Categorical(df_filtered['variable_type'], categories=unique_classes).codes



#############################################################################################################################################################################################################
# Create a colormap that maps class codes to colors
cmap = plt.cm.get_cmap('tab10', len(unique_classes))  # Adjust 'tab10' as needed

train_preds_prob = bst.predict(dtrain)
train_preds = np.argmax(train_preds_prob, axis=1)  # Convert probabilities to predicted class labels

original_labels = [int_to_label[i] for i in np.unique(y_train)]
print(np.unique(y_train), np.unique(train_preds))
# Generate the confusion matrix with the filtered and mapped labels
cm = confusion_matrix(y_train, train_preds, normalize='true')
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=original_labels)
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='vertical', values_format='.3f', colorbar=False)
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_confusion_norm.jpg', dpi=300, bbox_inches='tight')
plt.clf()


#############################################################################################################################################################################################################

# Generate the confusion matrix with the filtered and mapped labels
cm = confusion_matrix(y_train, train_preds, normalize='true')
norm = LogNorm(vmin=cm.min(), vmax=cm.max())
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=original_labels)
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='vertical', values_format='.3f', colorbar=False)
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_confusion_log_norm.jpg', dpi=300, bbox_inches='tight')
plt.clf()

#############################################################################################################################################################################################################

cm = confusion_matrix(y_train, train_preds)
norm = LogNorm(vmin=cm.min(), vmax=cm.max())
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=original_labels)
disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='vertical', values_format='d', colorbar=False, norm=norm)
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_confusion_log.jpg', dpi=300, bbox_inches='tight')
plt.clf()

#############################################################################################################################################################################################################
# Plot setup
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['log10(Period)'], df_filtered['Amplitude'], 
            c=class_codes,  # Use the class codes for color coding
            alpha=0.4,  # Semi-transparency
            s=1,  # Point size
            cmap=cmap)  # Use the created colormap

# Set custom x and y limits
plt.xlim([-2, 3])
plt.ylim([0.1, 5])

# Create legend for the broad classes
handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_, 
            markerfacecolor=cmap(i), markersize=10, alpha=1) 
            for i, class_ in enumerate(unique_classes)]

plt.legend(handles=handles)# loc='upper left', bbox_to_anchor=(1, 1))
#plt.tight_layout()

plt.xlabel('Period')
plt.ylabel('Amplitude')
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/log10PAcut'+str(output_fits_file)+'.jpg', dpi=300, bbox_inches='tight')
plt.clf()






#############################################################################################################################################################################################################



# Plot setup
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['Period'], df_filtered['Amplitude'], 
            c=class_codes,  # Use the class codes for color coding
            alpha=0.4,  # Semi-transparency
            s=1,  # Point size
            cmap=cmap)  # Use the created colormap


# Create legend for the broad classes
handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_, 
            markerfacecolor=cmap(i), markersize=10, alpha=1) 
            for i, class_ in enumerate(unique_classes)]

plt.legend(handles=handles)# loc='upper left', bbox_to_anchor=(1, 1))
#plt.tight_layout()
plt.xlim([0,max(df_filtered['Period'])])
plt.ylim([0,max(df_filtered['Amplitude'])])

plt.xlabel('Period')
plt.ylabel('Amplitude')
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/PA'+str(output_fits_file)+'.jpg', dpi=300, bbox_inches='tight')
plt.clf()


#################################################################################################################################################################################################

# Plot setup
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['Period'], df_filtered['Amplitude'], 
            c=class_codes,  # Use the class codes for color coding
            alpha=0.4,  # Semi-transparency
            s=1,  # Point size
            cmap=cmap)  # Use the created colormap


# Create legend for the broad classes
handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_, 
            markerfacecolor=cmap(i), markersize=10, alpha=1) 
            for i, class_ in enumerate(unique_classes)]

plt.legend(handles=handles)# loc='upper left', bbox_to_anchor=(1, 1))
#plt.tight_layout()
plt.xlim([0,100])
plt.ylim([0,max(df_filtered['Amplitude'])])

plt.xlabel('Period')
plt.ylabel('Amplitude')
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/PA100'+str(output_fits_file)+'.jpg', dpi=300, bbox_inches='tight')
plt.clf()



#############################################################################################################################################################################################################


# Plotting the bar chart for class frequencies
f, ax = plt.subplots(figsize=(10,6))
ax.set_yscale("log")
df_filtered['variable_type'].value_counts().plot(kind='bar', color='orange', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_class_frequencies.jpg', dpi=300, bbox_inches='tight')
plt.clf()
#############################################################################################################################################################################################################

f, ax = plt.subplots(figsize=(10,6))
ax.set_yscale("log")
df_filtered['best_class_name'].value_counts().plot(kind='bar', color='orange', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_trainclass_frequencies.jpg', dpi=300, bbox_inches='tight')
plt.clf()


#############################################################################################################################################################################################################


# Plotting the boxplot of log10(Period) for each class
f, ax = plt.subplots(figsize=(12,8))
sns.boxplot(x='variable_type', y='log10(Period)', data=df_filtered)
plt.xlabel('Class')
plt.ylabel('Log True Period')
plt.title('Boxplot of Log True Period by Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_log10Period_boxplot_by_class.jpg', dpi=300, bbox_inches='tight')
plt.clf()


#############################################################################################################################################################################################################


# Plotting the violin plot of log10(Period) for each class
# Plotting the boxplot of log10(Period) for each class
f, ax = plt.subplots(figsize=(12,8))
sns.violinplot(x='variable_type', y='log10(Period)', data=df_filtered)
plt.xlabel('Class')
plt.ylabel('Log True Period')
plt.title('Violin Plot of Log True Period by Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_log10Period_violin.jpg', dpi=300, bbox_inches='tight')
plt.clf()
#############################################################################################################################################################################################################

f, ax = plt.subplots(figsize=(12,8))
sns.violinplot(x='variable_type', y='Period', data=df100)
plt.xlabel('Class')
plt.ylabel('Period')
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim([0,100])
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'_100Period_violin.jpg', dpi=300, bbox_inches='tight')
plt.clf()
#############################################################################################################################################################################################################

# Assuming 'variable_type' is the column with class labels
classes = df_filtered['variable_type'].value_counts().index.tolist()  # Get unique classes sorted by population size
class_to_color = {cls: idx for idx, cls in enumerate(classes)}

# Map each class label to a color index
colors = df_filtered['variable_type'].map(class_to_color)

# Create a colormap
cmap = plt.cm.get_cmap('tab20', len(classes))  # Adjust 'tab10' to your preference

# Scatter plot of galactic longitude and latitude, color-coded by class
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['l'], df_filtered['b'], c=colors, alpha=0.5, s=0.01, cmap=cmap)
plt.xlabel('Galactic Longitude (l)')
plt.ylabel('Galactic Latitude (b)')
plt.xlim([min(df_filtered['l']), max(df_filtered['l'])])
plt.ylim([min(df_filtered['b']), max(df_filtered['b'])])

# Create a legend
handles = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=cmap(idx), markersize=3, alpha=0.5) 
           for idx, cls in enumerate(classes)]
plt.legend(handles, classes, markerscale=5)
plt.gca().invert_xaxis()  # Get the current axes and invert the x-axis
plt.tight_layout()
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'galactic.jpg', dpi=300, bbox_inches='tight')
plt.clf()


#############################################################################################################################################################################################################

# Assuming 'variable_type' is the column with class labels
classes = df_filtered['variable_type'].value_counts().index.tolist()  # Get unique classes sorted by population size
class_to_color = {cls: idx for idx, cls in enumerate(classes)}

# Map each class label to a color index
colors = df_filtered['variable_type'].map(class_to_color)

# Create a colormap
cmap = plt.cm.get_cmap('tab10', len(classes))  # Adjust 'tab10' to your preference

# Scatter plot of galactic longitude and latitude, color-coded by class
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['H-K'], df_filtered['Y-K'], c=colors, alpha=0.5, s=0.01, cmap=cmap, marker = 'x')
plt.xlabel('H-K')
plt.ylabel('Y-K')
plt.xlim([-2.5, 2.5])
plt.ylim([-0.5,7.5])
# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(idx), markersize=2, alpha=1) 
           for idx, cls in enumerate(classes)]
plt.legend(handles, classes, markerscale=5)#, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'HYK_colorcoded_scatter_plot.jpg', dpi=900, bbox_inches='tight')
plt.clf()




#############################################################################################################################################################################################################

# Assuming 'variable_type' is the column with class labels
classes = df_filtered['variable_type'].value_counts().index.tolist()  # Get unique classes sorted by population size
class_to_color = {cls: idx for idx, cls in enumerate(classes)}


# Map each class label to a color index
colors = df_filtered['variable_type'].map(class_to_color)

# Create a colormap
cmap = plt.cm.get_cmap('tab10', len(classes))  # Adjust 'tab10' to your preference

# Scatter plot of galactic longitude and latitude, color-coded by class
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['Z-K'], df_filtered['J-K'], c=colors, alpha=0.5, s=0.01, cmap=cmap)
plt.xlabel('Z-K')
plt.ylabel('J-K')
plt.xlim([0, 10])
plt.ylim([-0.5, 5])
# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(idx), markersize=2, alpha=0.7) 
           for idx, cls in enumerate(classes)]
plt.legend(handles, classes, markerscale=5)

plt.tight_layout()
plt.savefig('/beegfs/car/njm/PRIMVS/dtree/gaia/'+str(output_fits_file)+'ZJK_colorcoded_scatter_plot.jpg', dpi=300, bbox_inches='tight')
plt.clf()





