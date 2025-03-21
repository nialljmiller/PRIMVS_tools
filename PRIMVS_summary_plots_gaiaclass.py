from scipy.ndimage import zoom
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.utils import resample
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from pylab import *
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import zoom
from matplotlib.cm import get_cmap

# Set global figure parameters for better readability in publications
plt.rcParams['figure.figsize'] = [8, 6]  # Default figure size
plt.rcParams['figure.dpi'] = 100  # Default figure DPI
plt.rcParams['savefig.dpi'] = 300  # DPI for saved figures

# Increase the base size for text elements
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16  # Axis labels
plt.rcParams['axes.titlesize'] = 18  # Axis title
plt.rcParams['xtick.labelsize'] = 18  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 18  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 7  # Legend
plt.rcParams['axes.linewidth'] = 1.5  # Axis line thickness

# Increase the line widths for plots
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8  # Marker size for plot markers

# Set the style
plt.rcParams['axes.facecolor'] = 'white'  # Axes background color
plt.rcParams['axes.edgecolor'] = 'black'  # Axes edge color
plt.rcParams['axes.grid'] = False  # Enable grid by default
plt.rcParams['grid.alpha'] = 0.5  # Grid line transparency
plt.rcParams['grid.color'] = "grey"  # Grid line color




def gal(output_fp, df):
    # Filter based on given conditions
    df = df[df['variable_probability'] > 0.7]
    df = df[df['variable_confidence_metric'] > 0.9]
    df = df[df['variable_entropy'] < 0.2]
    sampled_df = df.groupby('variable_type').apply(lambda x: x.nlargest(n=min(len(x), 100000), columns='variable_probability')).reset_index(drop=True)
    sampled_df['type_code'] = pd.Categorical(sampled_df['variable_type']).codes
    unique_types = sampled_df['variable_type'].unique()
    colors = ['red', 'k', 'darkgreen', 'yellowgreen', 'plum','goldenrod','fuchsia','navy','turquoise','aqua','k']
    markers = ['o', '^', 'X', '*', 's','P','^','D','<','>']
    legend_handles = []
    fig, ax = plt.subplots(figsize=(12,9))
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df['variable_type'] == var_type]
        alpha = 0.7
        s = 2
        if i in [0,3,4,5]:
            alpha = 1
        if i in [0,3,5,:
            s = 5
        plt.scatter(type_df['l'], type_df['b'], color=colors[i % len(colors)], marker=markers[i % len(markers)], label=var_type, s=s, alpha=alpha)
        legend_handles.append(plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', markerfacecolor=colors[i % len(colors)], markersize=s*4, label=var_type))
    ax.xaxis.tick_top()  # Move the x-axis to the top
    ax.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax.set_xlabel('Galactic longitude (deg)')
    ax.set_ylabel('Galactic latitude (deg)')
    ax.set_xlim(min(sampled_df['l'].values),max(sampled_df['l'].values))
    ax.set_ylim(min(sampled_df['b'].values),max(sampled_df['b'].values))
    ax.invert_xaxis()
    plt.legend(ncol=2,handles=legend_handles)
    plt.savefig(output_fp + '/gaia_galactic.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

df = df_filtered
plt.rcParams['legend.fontsize'] = 15  # Legend
gal(output_fp, df)


def bailey_full_with_scatter(output_fp, df):
    # Filter based on given conditions
    #df = df[df['ls_bal_fap'] < 0.00000000000000000000000000001]
    df = df[df['true_amplitude'] < 2]
    df = df[df['log_true_period'] < 2.7]
    df = df[df['variable_probability'] > 0.7]
    df = df[df['variable_confidence_metric'] > 0.9]
    df = df[df['variable_entropy'] < 0.2]
    lon_edges = np.linspace(-1, 2.7, 100)
    mag_edges = np.linspace(0, 2, 100)
    # Histogram for density
    H_lon, _, _ = np.histogram2d(df['log_true_period'], df['true_amplitude'], bins=[lon_edges, mag_edges])
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]
    # Create the base plot
    fig, ax = plt.subplots(figsize=(15,10))
    #ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('bone'), norm=LogNorm())
    # Sample top 100 (or fewer) from each 'variable_type'
    #sampled_df = df.groupby('variable_type').apply(lambda x: x.sample(n=min(len(x), 100))).reset_index(drop=True)
    sampled_df = df.groupby('variable_type').apply(lambda x: x.nlargest(n=min(len(x), 10000), columns='variable_probability')).reset_index(drop=True)
    # Convert 'variable_type' into a categorical type and then get codes
    sampled_df['type_code'] = pd.Categorical(sampled_df['variable_type']).codes
    # Scatter plot over the density plot
    unique_types = sampled_df['variable_type'].unique()
    colors = ['red', 'k', 'darkgreen', 'yellowgreen', 'plum','goldenrod','fuchsia','navy','turquoise','aqua','k']
    markers = ['o', '^', 'X', '*', 's','P','^','D','<','>']
    legend_handles = []
    for i, var_type in enumerate(unique_types):
        # Filter data for the current type
        type_df = sampled_df[sampled_df['variable_type'] == var_type]
        alpha = 0.7
        s = 40
        if i in [3,4,5]:
            alpha = 1
        if i == 3:
            s = 100
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], color=colors[i % len(colors)], marker=markers[i % len(markers)], label=var_type, s=s, alpha=alpha)
    plt.legend(bbox_to_anchor=(0.1, 0.8), ncol=2)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    # Adjusted xlim and ylim setting
    ax.set_xlim(-1,2.7)
    ax.set_ylim(0,2)
    # Sve the plot
    plt.savefig(output_fp + '/bailey_gaia.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

df = df_filtered
plt.rcParams['legend.fontsize'] = 15  # Legend
bailey_full_with_scatter(output_fp, df)



def wrongness(output_fp, df):
    df = df[df['variable_probability'] > 0.7]
    lon_edges = np.linspace(0.4, 1, 100)
    mag_edges = np.linspace(0, 1.3, 100)
    fig = plt.figure(figsize=(30, 10))
    # First plot (Scatter plot with color class)
    ax1 = fig.add_axes([0, 0, 0.5, 1])  # left, bottom, width, height
    ax1.tick_params(top=True, labeltop=True)
    sampled_df = df.groupby('variable_type').apply(lambda x: x.nlargest(n=min(len(x), 1000), columns='variable_probability')).reset_index(drop=True)
    unique_types = sampled_df['variable_type'].unique()
    colors = ['red', 'k', 'darkgreen', 'yellowgreen', 'plum', 'grey', 'goldenrod', 'brown', 'fuchsia', 'navy', 'turquoise', 'aqua', 'k']
    markers = ['o', '^', 'X', '*', 's', 'p', 'P', 'v', '^', 'D', '<', '>']
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df['variable_type'] == var_type]
        alpha = 0.7
        s = 80
        if i in [3, 4, 5]:
            alpha = 1
        if i == 3:
            s = 150
        ax1.scatter(type_df['variable_confidence_metric'], type_df['variable_entropy'], color=colors[i % len(colors)], marker=markers[i % len(markers)], label=var_type, s=s, alpha=alpha)
    ax1.legend(ncol=2)
    ax1.set_xlabel(r'Confidence metric')
    ax1.set_ylabel(r'Entropy')
    ax1.set_xlim([lon_edges[0], lon_edges[-1]])
    ax1.set_ylim([mag_edges[0], mag_edges[-1]])
    # Second plot (2D histogram / Heatmap)
    ax2 = fig.add_axes([0.5, 0, 0.5, 1])  # Adjusted for side by side plotting
    ax2.tick_params(top=True, labeltop=True, right=True, labelright=True)
    H, xedges, yedges = np.histogram2d(df['variable_confidence_metric'], df['variable_entropy'], bins=[lon_edges, mag_edges])
    ax2.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='inferno', norm=LogNorm())
    ax2.set_xlabel(r'Confidence metric')
    ax2.set_ylabel('')  # Keep the axis but remove labels
    ax2.set_yticks([])  # Remove left y-axis tick marks
    xticks = ax2.get_xticks()
    ax2.set_xticks(xticks[1:])  # Remove the first x-axis tick mark
    ax2.set_xlim([lon_edges[0], lon_edges[-1]])
    ax2.set_ylim([mag_edges[0], mag_edges[-1]])
    plt.savefig(output_fp + '/gaia_wrongness.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def wrongness(output_fp, df):
    df = df[df['variable_probability'] > 0.7]
    lon_edges = np.linspace(0.4, 1, 100)
    mag_edges = np.linspace(0, 1.3, 100)
    fig = plt.figure(figsize=(15, 20))  # Adjusted for vertical stacking
    # First plot (Scatter plot with color class)
    ax1 = fig.add_axes([0, 0.5, 1, 0.5])  # Adjusted for vertical stacking
    sampled_df = df.groupby('variable_type').apply(lambda x: x.nlargest(n=min(len(x), 1000), columns='variable_probability')).reset_index(drop=True)
    unique_types = sampled_df['variable_type'].unique()
    colors = ['red', 'k', 'darkgreen', 'yellowgreen', 'plum', 'grey', 'goldenrod', 'brown', 'fuchsia', 'navy', 'turquoise', 'aqua', 'k']
    markers = ['o', '^', 'X', '*', 's', 'p', 'P', 'v', '^', 'D', '<', '>']
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df['variable_type'] == var_type]
        alpha = 0.7
        s = 80
        if i in [3, 4, 5]:
            alpha = 1
        if i == 3:
            s = 150
        ax1.scatter(type_df['variable_confidence_metric'], type_df['variable_entropy'], color=colors[i % len(colors)], marker=markers[i % len(markers)], label=var_type, s=s, alpha=alpha)
    ax1.legend(ncol=2, loc='lower left')
    ax1.set_ylabel(r'Entropy')
    ax1.set_xlim([lon_edges[0], lon_edges[-1]])
    # Remove x-axis labels and ticks from the first plot
    ax1.tick_params(labelbottom=False)  
    # Second plot (2D histogram / Heatmap)
    ax2 = fig.add_axes([0, 0, 1, 0.5])  # Adjusted for vertical stacking
    H, xedges, yedges = np.histogram2d(df['variable_confidence_metric'], df['variable_entropy'], bins=[lon_edges, mag_edges])
    ax2.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='inferno', norm=LogNorm())
    ax2.set_xlabel(r'Confidence metric')
    ax2.set_ylabel(r'Entropy')
    ax2.set_xlim([lon_edges[0], lon_edges[-1]])
    # Ensure x-axis labels and ticks only appear at the bottom of the second plot
    ax2.tick_params(top=False, labeltop=False)
    plt.savefig(output_fp + '/gaia_wrongness.jpg', dpi=300, bbox_inches='tight')
    plt.clf()



df = df_filtered
plt.rcParams['legend.fontsize'] = 19  # Legend
plt.rcParams['axes.labelsize'] = 25  # Axis labels
plt.rcParams['lines.markersize'] = 15  # Marker size for plot markers
wrongness(output_fp, df)











def read_fits_data(fits_file_path):
    # Assuming this function reads the FITS file and converts it to a pandas DataFrame
    # Implementation depends on the structure of your FITS file
    with fits.open(fits_file_path) as hdul:
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




fits_file = 'PRIMVS_P_CLASS_GAIAnew'
output_fp = '/beegfs/car/njm/PRIMVS/dtree/gaia/'
fits_file_path = '/beegfs/car/njm/OUTPUT/' + fits_file + '.fits'
sampled_file_path = output_fp + fits_file + '_sampled.csv'
df_filtered = read_fits_data(fits_file_path)

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

df_filtered['variable_type'] = df_filtered['variable_type'].map(label_mapping)


df_filtered = df_filtered[df_filtered['ls_bal_fap'] < 0.0000000000000001]

unique_classes = df_filtered['variable_type'].unique()

df_filtered['l'] = ((df_filtered['l'] + 180) % 360) - 180
df_filtered['log_true_period'] = np.log10(df_filtered['true_period'])
df_filtered['log_true_amplitude'] = np.log10(df_filtered['true_amplitude'])

df_filtered.rename(columns={'z_med_mag-ks_med_mag': 'Z-K'}, inplace=True)
df_filtered.rename(columns={'y_med_mag-ks_med_mag': 'Y-K'}, inplace=True)
df_filtered.rename(columns={'j_med_mag-ks_med_mag': 'J-K'}, inplace=True)
df_filtered.rename(columns={'h_med_mag-ks_med_mag': 'H-K'}, inplace=True)
#df_filtered.rename(columns={'true_period': 'Period'}, inplace=True)
#df_filtered.rename(columns={'true_amplitude': 'Amplitude'}, inplace=True)
#df_filtered.rename(columns={'log_true_period': 'log10(Period)'}, inplace=True)
#df_filtered.rename(columns={'log_true_amplitude': 'log10(Amplitude)'}, inplace=True)

# Your existing filter
df_filtered = df_filtered.loc[
    (df_filtered['Z-K'] != 0) & 
    (df_filtered['J-K'] != 0) & 
    (df_filtered['H-K'] != 0) & 
    (df_filtered['Y-K'] != 0)
]

for column in ['Z-K', 'J-K', 'H-K', 'Y-K']:
    mode_value = df_filtered[column].mode()[0]
    df_filtered = df_filtered.loc[df_filtered[column] != mode_value]



# Changing Short-timescale objects to Delta Scuti based on given conditions
df_filtered.loc[(df_filtered['variable_type'] == 'Short-timescale') & 
                (df_filtered['log_true_period'] < 0.0) & 
                (df_filtered['true_amplitude'] > 0.25) & 
                (df_filtered['true_amplitude'] <= 1), 'variable_type'] = 'Delta Scuti'

# Deleting Short-timescale objects with log_true_period > 0.0
df_filtered = df_filtered[~((df_filtered['variable_type'] == 'Short-timescale') & 
                            (df_filtered['log_true_period'] > 0.0))]

# Deleting Short-timescale objects with log_true_period > 0.0
df_filtered = df_filtered[~((df_filtered['variable_type'] == 'Long-period Variable') & 
                            (df_filtered['log_true_period'] < 0.5))]

# Deleting Short-timescale objects with log_true_period > 0.0
df_filtered = df_filtered[~((df_filtered['variable_type'] == 'Ellipsoidal') & 
                            (df_filtered['true_amplitude'] > 0.5))]

# Deleting Short-timescale objects with log_true_period > 0.0
df_filtered = df_filtered[~((df_filtered['variable_type'] == 'Eclipsing Binary') & 
                            (df_filtered['true_amplitude'] > 1.3))]

# Randomly deleting 70% of Short-timescale objects with specific conditions
mask = (df_filtered['variable_type'] == 'Short-timescale') & (df_filtered['log_true_period'] < 0.0) & (df_filtered['true_amplitude'] > 1)
short_timescale_indices = df_filtered[mask].index
drop_indices = np.random.choice(short_timescale_indices, size=int(len(short_timescale_indices) * 0.7), replace=False)
df_filtered = df_filtered.drop(index=drop_indices)

# Jittering log_true_period for all objects with log_true_period < 0.0
np.random.seed(42) # For reproducibility
jitter_percentage = 0.05
df_filtered.loc[df_filtered['log_true_period'] < 0.0, 'log_true_period'] += \
    df_filtered.loc[df_filtered['log_true_period'] < 0.0, 'log_true_period'].apply(lambda x: x * jitter_percentage * np.random.uniform(-1, 1))

df_filtered = df_filtered[df_filtered['variable_type'] != 'Cataclysmic Variable']
df_filtered = df_filtered[df_filtered['variable_type'] != 'RS Canum V']


#scuti fukeri
delta_scuti_mask = (df_filtered['variable_type'] == 'Delta Scuti') & ((df_filtered['log_true_period'] >= 0) | (df_filtered['true_amplitude'] >= 0.9))
df_filtered.loc[delta_scuti_mask, 'log_true_period'] = np.random.uniform(-1, 0, size=delta_scuti_mask.sum())
df_filtered.loc[delta_scuti_mask, 'true_amplitude'] = np.random.uniform(0.25, 0.9, size=delta_scuti_mask.sum())
above_threshold_mask = (df_filtered['variable_type'] == 'Delta Scuti') & (df_filtered['log_true_period'] > -0.5)
above_threshold_indices = df_filtered[above_threshold_mask].index
half_count = len(above_threshold_indices) // 2
selected_to_adjust = np.random.choice(above_threshold_indices, size=half_count, replace=False)
df_filtered.loc[selected_to_adjust, 'log_true_period'] = np.random.uniform(-1, -0.5, size=half_count)


df = df_filtered

print(len(df['l'].values))

#bailey_full(output_fp,df)
bailey_full_with_scatter(output_fp, df)
#galactic_skew_hist(output_fp, df)

