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
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels
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
    df = df[df['probability'] > 0.9]
    sampled_df = df.groupby('variable_type').apply(lambda x: x.nlargest(n=min(len(x), 1000), columns='probability')).reset_index(drop=True)
    sampled_df['type_code'] = pd.Categorical(sampled_df['variable_type']).codes
    unique_types = sampled_df['variable_type'].unique()
    colors = ['darkgreen','navy','plum','k','red','yellowgreen','turquoise']
    markers = ['P','D', 's', 'X', 'P', '*','>']
    legend_handles = []
    fig, ax = plt.subplots(figsize=(12,9))
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df['variable_type'] == var_type]
        plt.scatter(type_df['l'], type_df['b'], color=colors[i % len(colors)], marker=markers[i % len(markers)], label=var_type, s=15, alpha=0.5)
        if i == 5:
            legend_handles.append(plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', markerfacecolor=colors[i % len(colors)], markersize=17, label=var_type))
        else:
            legend_handles.append(plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', markerfacecolor=colors[i % len(colors)], markersize=10, label=var_type))
    ax.xaxis.tick_top()  # Move the x-axis to the top
    ax.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax.set_xlabel('Galactic longitude (deg)')
    ax.set_ylabel('Galactic latitude (deg)')
    ax.set_xlim(min(sampled_df['l'].values),max(sampled_df['l'].values))
    ax.set_ylim(min(sampled_df['b'].values),max(sampled_df['b'].values))
    ax.invert_xaxis()
    plt.legend(ncol=2,handles=legend_handles)
    plt.savefig(output_fp + '/simbad_galactic.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

plt.rcParams['legend.fontsize'] = 15  # Legend
df = df_filtered
gal(output_fp, df)


def bailey_full_with_scatter(output_fp, df):
    # Filter based on given conditions
    #df = df[df['ls_bal_fap'] < 0.00000000000000000000000000001]
    df = df[df['true_amplitude'] < 2]
    df = df[df['log_true_period'] < 2.7]
    df = df[df['probability'] > 0.9]
    lon_edges = np.linspace(-1, 2.7, 100)
    mag_edges = np.linspace(0, 2, 100)
    # Histogram for density
    H_lon, _, _ = np.histogram2d(df['log_true_period'], df['true_amplitude'], bins=[lon_edges, mag_edges])
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]
    # Create the base plot
    fig, ax = plt.subplots(figsize=(7,5))
    #ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('bone'), norm=LogNorm())
    # Sample top 100 (or fewer) from each 'variable_type'
    #sampled_df = df.groupby('variable_type').apply(lambda x: x.sample(n=min(len(x), 100))).reset_index(drop=True)
    sampled_df = df.groupby('variable_type').apply(lambda x: x.nlargest(n=min(len(x), 1000), columns='probability')).reset_index(drop=True)
    # Convert 'variable_type' into a categorical type and then get codes
    sampled_df['type_code'] = pd.Categorical(sampled_df['variable_type']).codes
    # Scatter plot over the density plot
    unique_types = sampled_df['variable_type'].unique()
    colors = ['darkgreen','navy','plum','k','red','yellowgreen','turquoise']
    markers = ['P','D', 's', 'X', 'P', '*','>']
    legend_handles = []
    for i, var_type in enumerate(unique_types):
        # Filter data for the current type
        type_df = sampled_df[sampled_df['variable_type'] == var_type]
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], 
                    color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], 
                    label=var_type, 
                    s=2, alpha=0.6)
                # Create a custom legend handle for this type
        if i == 5:
            legend_handles.append(plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', markerfacecolor=colors[i % len(colors)], markersize=9, label=var_type))
        else:
            legend_handles.append(plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', markerfacecolor=colors[i % len(colors)], markersize=7, label=var_type))
    plt.legend(handles=legend_handles, bbox_to_anchor=(0.3, 0.8), loc='lower left', ncol=2)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    # Adjusted xlim and ylim setting
    ax.set_xlim(-1,2.7)
    ax.set_ylim(0,2)
    # Sve the plot
    plt.savefig(output_fp + '/bailey_simbad.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


plt.rcParams['legend.fontsize'] = 7  # Legend
df = df_filtered
bailey_full_with_scatter(output_fp, df)



def bailey_full(output_fp, df):
    # Compute KDE on downsampled data
    df = df[df['ls_bal_fap'] < 0.00000000000000000000000000001]
    df = df[df['true_amplitude'] < 2]
    df = df[df['probability'] > 0.999]

    lon_edges = np.linspace(-1,2.7, 100)
    mag_edges = np.linspace(0,2, 100)
    H_lon, _, _ = np.histogram2d(df['log_true_period'], df['true_amplitude'], bins=[lon_edges, mag_edges])
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]


    fig, ax = plt.subplots()
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('bone'))
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('bone'), norm=LogNorm())

    # Interpolate H_lon for smoother contours
    H_lon_zoom = zoom(H_lon, 200)  # Increase the number by which you want to interpolate

    # Update the extent to account for the zoom
    lon_extent_zoom = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]

    # Define new levels for the interpolated data
    log_min_zoom = np.log10(np.percentile(H_lon_zoom[H_lon_zoom > 0], 20))
    log_max_zoom = np.log10(H_lon_zoom.max())
    levels_log_zoom = np.logspace(log_min_zoom, log_max_zoom, num=2)

    # Create contour with smoothed data
    ax.contour(H_lon_zoom.T, levels=levels_log_zoom, extent=lon_extent_zoom, colors='white', linewidths=0.5, norm=LogNorm())

    # Adjusted xlim and ylim setting
    ax.set_xlim(-1,2.7)
    ax.set_ylim(0,2)

    # Set labels (assuming period is in days and amplitude in magnitudes)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    plt.savefig(output_fp + '/bailey_full.jpg', dpi=300, bbox_inches='tight')
    plt.clf()





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




fits_file = 'PRIMVS_P_CLASS_SIMBAD'
output_fp = '/beegfs/car/njm/PRIMVS/dtree/simbad/'
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

#df_filtered['variable_type'] = df_filtered['variable_type'].map(label_mapping)


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

df = df_filtered

print(len(df['l'].values))

#bailey_full(output_fp,df)
bailey_full_with_scatter(output_fp, df)
#galactic_skew_hist(output_fp, df)

