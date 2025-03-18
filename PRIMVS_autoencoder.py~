import os
# Exclude CUDA device 0
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'  # Adjust this based on your available devices
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from astropy.table import Table
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from sklearn.utils import resample
from sklearn.neighbors import KernelDensity


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



def train_model(num_epochs, patience=20):
    train_losses = []
    valid_losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # Training phase
        autoencoder.train()
        for batch_features, _ in train_loader:
            batch_features = batch_features.to(DEVICE)  # Ensure data is on GPU
            optimizer.zero_grad()
            outputs = autoencoder(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        autoencoder.eval()
        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(DEVICE)  # Ensure data is on GPU            
                outputs = autoencoder(batch_features)
                loss = criterion(outputs, batch_features)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        scheduler.step(avg_valid_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, T-V Loss: {avg_train_loss-avg_valid_loss:.4f}")

        # Early stopping logic
        if avg_valid_loss < best_loss*1.01:
            best_loss = min([avg_valid_loss, best_loss])
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/beegfs/car/njm/PRIMVS/autoencoder/Loss.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


# Load a saved model
def load_model(model_path, autoencoder):
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()
    return autoencoder



fits_file = 'PRIMVS_P'
output_fp = '/beegfs/car/njm/PRIMVS/summary_plots/'
fits_file_path = '/beegfs/car/njm/OUTPUT/' + fits_file + '.fits'
sampled_file_path = output_fp + fits_file + '_sampled.fits'

filename = fits_file_path
outputfp = '/beegfs/car/njm/PRIMVS/autoencoder/'
# Load FITS file
data = Table.read(fits.open(filename, memmap=True)[1], format='fits')

# Define the list of feature names you want to use


selected_feature_names = ['l', 'b','Cody_M', 
                        'stet_k', 'eta', 
                        'z_med_mag-ks_med_mag',
                        'y_med_mag-ks_med_mag',
                        'j_med_mag-ks_med_mag',
                        'h_med_mag-ks_med_mag',
                        'med_BRP', 'range_cum_sum', 'max_slope', 
                        'MAD', 'mean_var', 
                        'true_amplitude', 'roms', 'p_to_p_var',
                        'lag_auto', 'AD', 'std_nxs',
                        'weight_mean', 'weight_std', 'weight_skew',
                        'weight_kurt', 'mean', 'std', 
                        'skew', 'kurt', 'true_period']
            

feature_weights = [1.,1.,0.5,1.,1.,1.5,1.5,1.5,1.5,1.,0.5,0.5,0.5,0.1,1.5,0.2,0.2,0.8,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0,1.0,2.]
selected_features = data[selected_feature_names]

# Convert selected features to Pandas DataFrame
df = selected_features.to_pandas()
df.fillna(0, inplace=True)
df['l'] = ((df['l'] + 180) % 360) - 180

features = df.values

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
weights_tensor = torch.tensor(feature_weights, dtype=torch.float32)
features_tensor = features_tensor * weights_tensor

X_train, X_test = train_test_split(features_tensor, test_size=0.2, random_state=42)
input_dim = X_train.size(1)

autoencoder = Autoencoder(input_dim, latent_dim=int(input_dim/2)).to(DEVICE)


criterion = nn.MSELoss().to(DEVICE)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

if torch.cuda.is_available():
    autoencoder.cuda()  # Move model to the default GPU before using DataParallel
    criterion.cuda()  # Also move your loss function to the GPU

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    autoencoder = nn.DataParallel(autoencoder)


train_dataset = TensorDataset(X_train, X_train)
train_loader = DataLoader(train_dataset, batch_size=4096*8, shuffle=True)
test_dataset = TensorDataset(X_test, X_test)
test_loader = DataLoader(test_dataset, batch_size=4096*8, shuffle=True)


Train_model = False
if Train_model == True:
    # Train the model
    num_epochs = 1000000
    train_model(num_epochs)
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')

else:
    # Example usage to load a saved model
    model_path = 'autoencoder.pth'
    autoencoder = load_model(model_path, autoencoder)


# Obtain the latent space representation
with torch.no_grad():
    # Assuming your model is on cuda:0 and you're using DataParallel
    # First, move features_tensor to the same device as your model
    features_tensor_gpu = features_tensor.to(DEVICE)

    # Now, encode the data
    encoded_data = autoencoder.module.encoder(features_tensor_gpu).detach().cpu().numpy()





def plot_3dgif(outputfp, name, x, y, z, x_lims, y_lims, z_lims, explained_var_ratios=None):
    # Setting visualization parameters
    alpha = 0.5
    s = 0.1  # Adjusted marker size for better visibility
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, cmap='viridis', alpha=alpha, s=s, c=density)

    # Setting ticks without numbers
    ax.set_xticks(np.linspace(*x_lims, num=5), minor=False)  # Adjust num for density of ticks
    ax.set_yticks(np.linspace(*y_lims, num=5), minor=False)
    ax.set_zticks(np.linspace(*z_lims, num=5), minor=False)

    ax.tick_params(axis='x', which='both', length=6, width=1, labelbottom=False)
    ax.tick_params(axis='y', which='both', length=6, width=1, labelleft=False)
    ax.tick_params(axis='z', which='both', length=6, width=1, labelleft=False)

    if explained_var_ratios is not None:
        ax.set_xlabel(f'{explained_var_ratios[0]*100:.2f}%', labelpad=2)
        ax.set_ylabel(f'{explained_var_ratios[1]*100:.2f}%', labelpad=2)
        ax.set_zlabel(f'{explained_var_ratios[2]*100:.2f}%', labelpad=2)

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_zlim(z_lims)

    def update(frame):
        ax.view_init(elev=20*(abs(frame-180)/180), azim=frame)
        return scatter

    animation = FuncAnimation(fig, update, frames=range(0, 360, 1), interval=25)
    animation.save(outputfp + name + '.gif', writer='imagemagick', dpi=300)

    plt.clf()
    plt.close()


def plot_3d(outputfp, name, x, y, z, x_lims, y_lims, z_lims, explained_var_ratios=None):
    alpha = 0.5
    s = 0.1  # Adjusted marker size for better visibility

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, cmap='viridis', alpha=alpha, s=s, c=density)

    # Setting ticks without numbers
    ax.set_xticks(np.linspace(*x_lims, num=5), minor=False)  # Adjust num for density of ticks
    ax.set_yticks(np.linspace(*y_lims, num=5), minor=False)
    ax.set_zticks(np.linspace(*z_lims, num=5), minor=False)

    ax.tick_params(axis='x', which='both', length=6, width=1, labelbottom=False)
    ax.tick_params(axis='y', which='both', length=6, width=1, labelleft=False)
    ax.tick_params(axis='z', which='both', length=6, width=1, labelleft=False)

    if explained_var_ratios is not None:
        ax.set_xlabel(f'{explained_var_ratios[0]*100:.2f}%', labelpad=-2)
        ax.set_ylabel(f'{explained_var_ratios[1]*100:.2f}%', labelpad=-2)
        ax.set_zlabel(f'{explained_var_ratios[2]*100:.2f}%', labelpad=-2)

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_zlim(z_lims)

    plt.savefig(outputfp + name + '.jpg', dpi=300, bbox_inches='tight')
    plt.clf()





# Initial subsampling
plotty = False
if plotty:
    subsample_size_initial = 100000  # Adjust based on your system's capabilities
    print("Subsampling data to", subsample_size_initial, "samples.")
    encoded_data = resample(encoded_data, n_samples=subsample_size_initial, random_state=42)

print("Applying PCA for dimensionality reduction.")
pca = PCA(n_components=3)
latent_pca = pca.fit_transform(encoded_data)
explained_var_ratios = pca.explained_variance_ratio_

x = latent_pca[:, 0]
y = latent_pca[:, 1]
z = latent_pca[:, 2]

if plotty:

    print("Calculating axis limits based on percentiles.")
    x_lims = np.percentile(latent_pca[:, 0], [0.001, 99.999])
    y_lims = np.percentile(latent_pca[:, 1], [0.001, 99.999])
    z_lims = np.percentile(latent_pca[:, 2], [0.001, 99.999])

    print("Calculating point densities.")
    xyz = np.vstack([x, y, z])
    density = gaussian_kde(xyz)(xyz)
    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]

    name = 'PCA'
    print("Plotting 3D scatter with density-based coloring and preparing animation.")
    plot_3d(outputfp, name, x, y, z, x_lims, y_lims, z_lims, explained_var_ratios = explained_var_ratios)
    print("Animation for density-based 3D scatter plot saved.")
    plot_3dgif(outputfp, name, x, y, z, x_lims, y_lims, z_lims, explained_var_ratios = explained_var_ratios)

else:

    data['PCA1'] = x
    data['PCA2'] = y
    data['PCA3'] = z






print("Applying UMAP for dimensionality reduction.")
latent_umap = UMAP(n_components=3, random_state=420).fit_transform(encoded_data) 
x = latent_umap[:, 0]
y = latent_umap[:, 1]
z = latent_umap[:, 2]

if plotty:
    print("Calculating axis limits based on percentiles.")
    x_lims = np.percentile(latent_umap[:, 0], [0.001, 99.999])
    y_lims = np.percentile(latent_umap[:, 1], [0.001, 99.999])
    z_lims = np.percentile(latent_umap[:, 2], [0.001, 99.999])

    print("Calculating point densities.")
    xyz = np.vstack([x, y, z])
    density = gaussian_kde(xyz)(xyz)
    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]

    print("Plotting 3D scatter with density-based coloring and preparing animation.")
    name = 'UMAP'
    plot_3d(outputfp, name, x, y, z, x_lims, y_lims, z_lims)
    print("Animation for density-based 3D scatter plot saved.")
    plot_3dgif(outputfp, name, x, y, z, x_lims, y_lims, z_lims)

else:

    data['UMAP1'] = x
    data['UMAP2'] = y
    data['UMAP3'] = z

    data.write('/beegfs/car/njm/OUTPUT/PRIMVS_P_auto.fits', overwrite=True)
