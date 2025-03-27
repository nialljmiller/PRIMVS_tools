import pandas as pd
from astropy.io import fits
import xgboost as xgb
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


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
    inference_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + test_fits_file + '.fits')

    # Select features and target variable from training data
    # Replace 'feature_columns' with your list of feature column names
    # and 'target_column' with your target column name
    train_df.rename(columns={'l_2': 'l'}, inplace=True)
    train_df.rename(columns={'b_2': 'b'}, inplace=True)
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




    target_column = 'broad_class'

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
        if col in inference_df.columns:
            # Use median from train_df to fill missing values in test_df to avoid data leakage
            median_value = train_df[col].median()
            inference_df[col].fillna(median_value, inplace=True)


    # For categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = train_df[col].mode()[0]
        if col in train_df.columns:
            train_df[col].fillna(mode_value, inplace=True)
        if col in inference_df.columns:
            inference_df[col].fillna(mode_value, inplace=True)

  
    train_df[target_column] = train_df[target_column].str.strip().str.replace('?', '').str.replace('_Candidate', '').str.replace(':', '')

    # Re-map labels to integers after consolidation
    unique_labels = train_df[target_column].unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    train_df['label_encoded'] = train_df[target_column].map(label_to_int)
    
       
       
       
    # Assuming 'train_df' is your DataFrame and 'feature_columns' are your feature names
    X = train_df[feature_columns].values  # Extracting feature values
    y = train_df['label_encoded'].values  # Target values

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Converting to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Creating TensorDatasets and DataLoaders for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)




    # You might want to update num_classes if it's used later for model training
    num_classes = len(unique_labels)


    # Define the Neural Network Model
    class Net(nn.Module):
        def __init__(self, num_features, num_classes):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(num_features, 100)  # First hidden layer
            self.relu = nn.ReLU()  # Activation function
            self.fc2 = nn.Linear(100, 50)  # Second hidden layer
            self.fc3 = nn.Linear(50, num_classes)  # Output layer
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Instantiate the model, loss function, and optimizer
    model = Net(num_features=len(feature_columns), num_classes=len(train_df['label_encoded'].unique()))
    criterion = nn.CrossEntropyLoss()  # Add weight parameter for class imbalance if necessary
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100  # Adjust based on your needs
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluation code can be added here to assess the model

    # Ensure the model is in evaluation mode
    model.eval()

    # Collect all predictions and their probabilities
    all_preds = []
    all_probs = []

    with torch.no_grad():  # No need to track gradients
        for inputs in test_loader:  # Assuming your test_loader doesn't have labels
            outputs = model(inputs[0])  # Get model predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities
            all_probs.extend(probs.numpy())  # Store probabilities
            preds = torch.argmax(probs, dim=1)  # Convert probabilities to predicted class
            all_preds.extend(preds.numpy())  # Store predictions

    # Convert to numpy arrays for easier processing
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)


    # Map numeric labels to original string labels
    most_likely_labels = [int_to_label[cls] for cls in all_preds]

    # Extract the probability of the most likely class
    max_probabilities = np.max(all_probs, axis=1)


    inference_df['most_likely_class'] = most_likely_labels
    inference_df['probability'] = max_probabilities



    # The DataFrame now contains a 'broad_class' column with the broader categories

    # Select columns to write to FITS (you may adjust this as needed)
    columns_to_write = inference_df.columns  # This includes all columns; adjust as needed

    # Convert DataFrame to Astropy Table, then write to FITS
    table_to_write = Table.from_pandas(inference_df[columns_to_write])
    table_to_write.write('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits', overwrite=True)
else:
    inference_df = read_fits_data('/beegfs/car/njm/OUTPUT/' + output_fits_file + '.fits')





# Filter the DataFrame
df_filtered = inference_df[inference_df['probability'] > 0.8]
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



