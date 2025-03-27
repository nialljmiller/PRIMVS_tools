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
plt.rcParams['legend.fontsize'] = 14  # Legend
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

# If using LaTeX formatting in labels/titles
# plt.rcParams['text.usetex'] = True


# If you want to ensure your plots use the 'seaborn' style, you can do so by:
# plt.style.use('seaborn')

# Apply these settings before generating your plots



def read_fits_data(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        df = Table(data).to_pandas()  # Convert to a pandas DataFrame
        for column in df.columns:
            if df[column].dtype.byteorder == '>':  # Big-endian
                df[column] = df[column].byteswap().newbyteorder()
    return df


def histograms(df, feature_columns, output_fp):
    for column in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.xlabel(column)
        plt.xlim(left=df[column].quantile(0.01), right=df[column].quantile(0.999)) # Adjust the quantiles as needed
        plt.tight_layout()
        plt.savefig(output_fp + 'histogram/'+str(column)+'.jpg', dpi=300, bbox_inches='tight')
        plt.clf()



def FAP_histograms(df, feature_columns, output_fp):
    for column in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.xlabel(column)
        plt.xlim(left=df[column].quantile(0.01), right=df[column].quantile(0.999)) # Adjust the quantiles as needed
        plt.tight_layout()
        plt.savefig(output_fp + 'histogram/'+str(column)+'.jpg', dpi=300, bbox_inches='tight')
        plt.clf()


def histograms_linfit(df, feature_columns, output_fp):
    for column in feature_columns:
        # Drop NaN values from the column to avoid the ValueError
        column_data = df[column].dropna()

        plt.figure(figsize=(10, 6))
        # Generate histogram data with density=False to get actual counts
        counts, bins = np.histogram(column_data, bins=30, density=False)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Calculate bin centers

        # Determine the bins that cover approximately the first 75% of the data
        cutoff_index = int(len(bin_centers) * 0.75)
        selected_bin_centers = bin_centers[:cutoff_index]
        selected_counts = counts[:cutoff_index]

        # Fit a straight line to the selected part of the histogram
        slope, intercept, r_value, p_value, std_err = linregress(selected_bin_centers, selected_counts)
        line = slope * selected_bin_centers + intercept

        # Plot the histogram with KDE
        sns.histplot(column_data, kde=True, bins=30, alpha=0.5)
        plt.plot(selected_bin_centers, line, color='red')  # Plot the fitted line over the leftmost 75%

        plt.xlabel(column)
        plt.xlim(left=column_data.quantile(0.01), right=column_data.quantile(0.999))  # Adjust the quantiles as needed
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_fp + 'histogram/fit_'+str(column)+'.jpg', dpi=300, bbox_inches='tight')
        plt.clf()


def scatters(df, feature_pair_columns, output_fp):
    for feature_pair in feature_pair_columns:
        feature1 = feature_pair[0]
        feature2 = feature_pair[1]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[feature1], y=df[feature2])
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xlim(df[feature1].quantile(0.0001), df[feature1].quantile(0.9999)) # Adjust the quantiles as needed
        plt.ylim(df[feature2].quantile(0.0001), df[feature2].quantile(0.9999)) # Adjust the quantiles as needed
        plt.tight_layout()
        plt.savefig(output_fp + 'scatter/'+str(feature1)+str(feature2)+'.jpg', dpi=300, bbox_inches='tight')
        plt.clf()

def pairplot(df, feature_columns, output_fp):
    sns.pairplot(df[feature_columns])
    plt.tight_layout()
    plt.savefig(output_fp + 'pairplot.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

def box(df, feature_columns, output_fp):
    for column in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.tight_layout()
        plt.savefig(output_fp + 'box/'+str(column)+'.jpg', dpi=300, bbox_inches='tight')
        plt.clf()

def cor_matrix(df, feature_columns, output_fp):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[feature_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(output_fp + 'correlation_matrix.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

def xy_col(df, feature_pair_columns, output_fp):
    for feature_pair in feature_pair_columns:
        feature1, feature2 = feature_pair
        plt.figure(figsize=(10, 6))
        # Adjust bw_adjust for more/less smoothing and cmap for color map clarity
        sns.kdeplot(x=df[feature1], y=df[feature2], cmap="coolwarm", shade=True, bw_adjust=1)
        # Adjust size and alpha for scatter plot clarity
        sns.scatterplot(x=df[feature1], y=df[feature2], alpha=0.5, s=5, marker = 'x')
        sns.scatterplot(x=df[feature1], y=df[feature2], marker = '.', color = 'k')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        # Adjusted xlim and ylim setting
        ax.set_xlim(0, df['true_period'].quantile(0.9999))
        ax.set_ylim(0, df['true_amplitude'].quantile(0.9999))

        plt.tight_layout()
        plt.savefig(output_fp + 'density/' + str(feature1) + str(feature2) + '.jpg', dpi=300, bbox_inches='tight')
        plt.clf()





def galactic_amp_scatter(output_fp, df):
    plt.clf()

    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='true_amplitude', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 9))
    
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.7, 1, 0.3])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='Amplitude', pad=0.0, fraction=0.046, location='right')

    # Compute KDE on downsampled data
    xyl_sampled = np.vstack([df_sampled['l'], df_sampled['true_amplitude']])
    zl_sampled = gaussian_kde(xyl_sampled)(xyl_sampled)
    xyb_sampled = np.vstack([df_sampled['b'], df_sampled['true_amplitude']])
    zb_sampled = gaussian_kde(xyb_sampled)(xyb_sampled)
    tree_l = cKDTree(df_sampled[['l', 'true_amplitude']])
    _, idx_l = tree_l.query(df[['l', 'true_amplitude']], k=1)
    tree_b = cKDTree(df_sampled[['b', 'true_amplitude']])
    _, idx_b = tree_b.query(df[['b', 'true_amplitude']], k=1)
    zl = zl_sampled[idx_l]
    zb = zb_sampled[idx_b]
    z_norml = Normalize(vmin=zl.min(), vmax=zl.max())
    z_normb = Normalize(vmin=zb.min(), vmax=zb.max())
    
    ax_scatter_l = fig.add_axes([0, 0.3, 0.5, 0.4])#, aspect='equal')
    ax_scatter_b = fig.add_axes([0.5, 0.3, 0.465, 0.4])#, aspect='equal')

    
    ax_scatter_l.scatter(df['l'], df['true_amplitude'], c=zl, cmap='winter', alpha=0.6, norm=z_norml, s=1)
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    ax_scatter_l.set_xlim(-65,11)
    ax_scatter_l.set_ylim(1.1,5.2)
    ax_scatter_l.spines['bottom'].set_visible(False)
    
    ax_scatter_b.scatter(df['b'], df['true_amplitude'], c=zb, cmap='autumn', alpha=0.6, norm=z_normb, s=1)
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')
    ax_scatter_b.set_xlim(-10.5,5)
    ax_scatter_b.set_ylim(1.1,5.2)
    ax_scatter_b.spines['bottom'].set_visible(False)

    ax_scatter_zl = fig.add_axes([0, 0, 0.5, 0.3])#, aspect='equal')
    ax_scatter_zl.scatter(df['l'], df['true_amplitude'], c=zl, cmap='winter', alpha=0.6, norm=z_norml, s=1)
    ax_scatter_zl.set_xlabel('Galactic longitude (deg)')
    ax_scatter_zl.set_xlim(-65,11)
    ax_scatter_zl.set_ylim(0.05,1.1)
    #ax_scatter_zl.spines['top'].set_visible(False)

    ax_scatter_zb = fig.add_axes([0.5, 0, 0.465, 0.3])#, aspect='equal')
    ax_scatter_zb.scatter(df['b'], df['true_amplitude'], c=zb, cmap='autumn', alpha=0.6, norm=z_normb, s=1)
    ax_scatter_zb.set_xlabel('Galactic latitude (deg)')
    ax_scatter_zb.yaxis.tick_right()  # Move the y-axis to the right
    ax_scatter_zb.yaxis.set_label_position('right')  # Move the y-axis label to the right
    ax_scatter_zb.set_xlim(-10.5,5)
    ax_scatter_zb.set_ylim(0.05,1.1)
    #ax_scatter_zb.spines['top'].set_visible(False)    
    
    fig.text(-0.05, 0.2, 'Amplitude', ha='center', va='center', fontsize=16, rotation=90)

    plt.savefig(output_fp + '/galactic_amp_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()








def galactic_mag_scatter(output_fp, df):
    plt.clf()

    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='mag_avg', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 8))
    
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.6, 1, 0.4])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='Mag', pad=0.0, fraction=0.046, location='right')


    # Compute KDE on downsampled data
    xyl_sampled = np.vstack([df_sampled['l'], df_sampled['mag_avg']])
    zl_sampled = gaussian_kde(xyl_sampled)(xyl_sampled)
    xyb_sampled = np.vstack([df_sampled['b'], df_sampled['mag_avg']])
    zb_sampled = gaussian_kde(xyb_sampled)(xyb_sampled)
    tree_l = cKDTree(df_sampled[['l', 'mag_avg']])
    _, idx_l = tree_l.query(df[['l', 'mag_avg']], k=1)
    tree_b = cKDTree(df_sampled[['b', 'mag_avg']])
    _, idx_b = tree_b.query(df[['b', 'mag_avg']], k=1)
    zl = zl_sampled[idx_l]
    zb = zb_sampled[idx_b]
    z_norml = Normalize(vmin=zl.min(), vmax=zl.max())
    z_normb = Normalize(vmin=zb.min(), vmax=zb.max())
    
    ax_scatter_l = fig.add_axes([0, 0., 0.5, 0.6])#, aspect='equal')
    ax_scatter_b = fig.add_axes([0.5, 0., 0.465, 0.6])#, aspect='equal')


    ax_scatter_l.scatter(df['l'], df['mag_avg'], c=zl, cmap='winter', alpha=0.6, norm=z_norml, s=1)
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    ax_scatter_l.set_ylabel('Mag')
    ax_scatter_l.set_xlim(-65,11)
    ax_scatter_l.set_ylim(11.1,17)    
    ax_scatter_l.invert_yaxis()


    
    ax_scatter_b.scatter(df['b'], df['mag_avg'], c=zb, cmap='autumn', alpha=0.6, norm=z_normb, s=1)
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')
    ax_scatter_b.set_xlim(-10.5,5)
    ax_scatter_b.set_ylim(11.1,17)    
    ax_scatter_b.invert_yaxis()

    plt.savefig(output_fp + '/galactic_mag_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()






def galactic_fap_scatter(output_fp, df):
    plt.clf()

    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='best_fap', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 9))
    
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.7, 1, 0.3])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='FAP', pad=0.0, fraction=0.046, location='right')

    

    # Downsample the data

    xyl = np.vstack([df_sampled['best_fap'], df_sampled['l']])
    zl = gaussian_kde(xyl)(xyl)
    xyb = np.vstack([df_sampled['best_fap'], df_sampled['b']])
    zb = gaussian_kde(xyb)(xyb)    
    z_norml = Normalize(vmin=zl.min(), vmax=zl.max())
    z_normb = Normalize(vmin=zb.min(), vmax=zb.max())
    
    # Compute KDE on downsampled data
    xyl_sampled = np.vstack([df_sampled['l'], df_sampled['best_fap']])
    zl_sampled = gaussian_kde(xyl_sampled)(xyl_sampled)
    xyb_sampled = np.vstack([df_sampled['b'], df_sampled['best_fap']])
    zb_sampled = gaussian_kde(xyb_sampled)(xyb_sampled)
    tree_l = cKDTree(df_sampled[['l', 'best_fap']])
    _, idx_l = tree_l.query(df[['l', 'best_fap']], k=1)
    tree_b = cKDTree(df_sampled[['b', 'best_fap']])
    _, idx_b = tree_b.query(df[['b', 'best_fap']], k=1)
    zl = zl_sampled[idx_l]
    zb = zb_sampled[idx_b]
    z_norml = Normalize(vmin=zl.min(), vmax=zl.max())
    z_normb = Normalize(vmin=zb.min(), vmax=zb.max())
    
    ax_scatter_l = fig.add_axes([0, 0., 0.5, 0.7])#, aspect='equal')
    ax_scatter_b = fig.add_axes([0.5, 0., 0.465, 0.7])#, aspect='equal')

    ax_scatter_l.scatter(df['l'], df['best_fap'], c=zl, cmap='winter', alpha=0.6, norm=z_norml, s=1)
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    #ax_scatter_l.set_ylabel('Amplitude')
    ax_scatter_l.set_xlim(-65,11)
    ax_scatter_l.set_ylim(0,0.3)


    ax_scatter_b.scatter(df['b'], df['best_fap'], c=zb, cmap='autumn', alpha=0.6, norm=z_normb, s=1)
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')
    ax_scatter_b.set_xlim(-10.5,5)
    ax_scatter_b.set_ylim(0,0.3)

    fig.text(-0.05, 0.3, 'FAP', ha='center', va='center', fontsize=16, rotation=90)

    plt.savefig(output_fp + '/galactic_fap_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()




def galactic_skew_scatter(output_fp, df):
    plt.clf()

    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='skew', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 9))
    
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.7, 1, 0.3])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='Skew', pad=0.0, fraction=0.046, location='right')



    # Compute KDE on downsampled data
    xyl_sampled = np.vstack([df_sampled['l'], df_sampled['skew']])
    zl_sampled = gaussian_kde(xyl_sampled)(xyl_sampled)
    xyb_sampled = np.vstack([df_sampled['b'], df_sampled['skew']])
    zb_sampled = gaussian_kde(xyb_sampled)(xyb_sampled)
    tree_l = cKDTree(df_sampled[['l', 'skew']])
    _, idx_l = tree_l.query(df[['l', 'skew']], k=1)
    tree_b = cKDTree(df_sampled[['b', 'skew']])
    _, idx_b = tree_b.query(df[['b', 'skew']], k=1)
    zl = zl_sampled[idx_l]
    zb = zb_sampled[idx_b]
    z_norml = Normalize(vmin=zl.min(), vmax=zl.max())
    z_normb = Normalize(vmin=zb.min(), vmax=zb.max())
    
    
    ax_scatter_l = fig.add_axes([0, 0., 0.5, 0.7])#, aspect='equal')
    ax_scatter_b = fig.add_axes([0.5, 0., 0.465, 0.7])#, aspect='equal')
    
    ax_scatter_l.scatter(df['l'], df['skew'], c=zl, cmap='winter', alpha=0.6, norm=z_norml, s=1)
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    ax_scatter_l.set_xlim(-65,11)
    ax_scatter_l.set_ylim(-3.2,3.2)


    ax_scatter_b.scatter(df['b'], df['skew'], c=zb, cmap='autumn', alpha=0.6, norm=z_normb, s=1)
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')
    ax_scatter_b.set_xlim(-10.5,5)
    ax_scatter_b.set_ylim(-3.2,3.2)

    
    fig.text(-0.05, 0.3, 'Skew', ha='center', va='center', fontsize=16, rotation=90)

    plt.savefig(output_fp + '/galactic_skew_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()




def galactic_fap(output_fp, df):

    # Define the bin edges for longitude and latitude if they are not already binned
    lon_bins = np.linspace(-70,10, num=17)
    lat_bins = np.linspace(-10,6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins)-1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins)-1))

    # Create the pivot table
    heatmap_data = df.pivot_table(values='best_fap', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Generate the heatmap
    fig, ax = plt.subplots(figsize=(12, 4))

    # Since we're using bin labels, set the ticks to the middle of each bin
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])

    # We can use 'pcolormesh' for a more accurate plot when bins are not regular, and it's faster for large data sets
    cax = ax.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')

    # Set the ticks in the middle of the bins
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)

    # Set the labels on the ticks
    ax.set_xticklabels(np.round(lon_ticks, 2))
    ax.set_yticklabels(np.round(lat_ticks, 2))

    ax.set_xlabel('Galactic longitude (deg)')
    ax.set_ylabel('Galactic latitude (deg)')
    fig.colorbar(cax, label='FAP', pad=-0.01)
    ax.set_aspect('auto')
    ax.invert_xaxis()
    plt.savefig(output_fp + '/galactic_FAP.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def galactic_amp_scatter(output_fp, df):
    plt.clf()

    lon_bins = np.linspace(-70, 10, num=17)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='true_amplitude', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 6))
    
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.5, 1, 0.5])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='Amplitude', pad=0.0, fraction=0.046, location='right')

    ax_scatter_l = fig.add_axes([0, 0, 0.5, 0.5])#, aspect='equal')
    xy = np.vstack([df['true_amplitude'], df['l']])
    z = gaussian_kde(xy)(xy)
    z_norm = Normalize(vmin=z.min(), vmax=z.max())
    ax_scatter_l.scatter(df['l'], df['true_amplitude'], c=z, cmap='winter', alpha=0.6, norm=z_norm, s=1)
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    ax_scatter_l.set_ylabel('Amplitude')

    ax_scatter_b = fig.add_axes([0.48, 0, 0.505, 0.5], aspect='equal')
    xy = np.vstack([df['true_amplitude'], df['b']])
    z = gaussian_kde(xy)(xy)
    z_norm = Normalize(vmin=z.min(), vmax=z.max())
    ax_scatter_b.scatter(df['b'], df['true_amplitude'], c=z, cmap='autumn', alpha=0.6, norm=z_norm, s=1)
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()  # Move the y-axis to the right
    ax_scatter_b.yaxis.set_label_position('right')  # Move the y-axis label to the right

    plt.savefig(output_fp + '/galactic_amp_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()






def galactic_amp(output_fp, df):

    # Define the bin edges for longitude and latitude if they are not already binned
    lon_bins = np.linspace(-70,10, num=17)
    lat_bins = np.linspace(-10,6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins)-1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins)-1))

    # Create the pivot table
    heatmap_data = df.pivot_table(values='true_amplitude', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Generate the heatmap
    fig, ax = plt.subplots(figsize=(12, 4))

    # Since we're using bin labels, set the ticks to the middle of each bin
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])

    # We can use 'pcolormesh' for a more accurate plot when bins are not regular, and it's faster for large data sets
    cax = ax.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')

    # Set the ticks in the middle of the bins
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)

    # Set the labels on the ticks
    ax.set_xticklabels(np.round(lon_ticks, 2))
    ax.set_yticklabels(np.round(lat_ticks, 2))

    ax.set_xlabel('Galactic longitude (deg)')
    ax.set_ylabel('Galactic latitude (deg)')
    fig.colorbar(cax, label='Amplitude', pad=-0.01)
    ax.set_aspect('auto')
    ax.invert_xaxis()
    plt.savefig(output_fp + '/galactic_amp.jpg', dpi=300, bbox_inches='tight')
    plt.clf()





def galactic_avg(output_fp, df):

    # Define the bin edges for longitude and latitude if they are not already binned
    lon_bins = np.linspace(-70,10, num=17)
    lat_bins = np.linspace(-10,6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins)-1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins)-1))

    # Create the pivot table
    heatmap_data = df.pivot_table(values='mag_avg', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Generate the heatmap
    fig, ax = plt.subplots(figsize=(12, 4))

    # Since we're using bin labels, set the ticks to the middle of each bin
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])

    # We can use 'pcolormesh' for a more accurate plot when bins are not regular, and it's faster for large data sets
    cax = ax.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')

    # Set the ticks in the middle of the bins
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)

    # Set the labels on the ticks
    ax.set_xticklabels(np.round(lon_ticks, 2))
    ax.set_yticklabels(np.round(lat_ticks, 2))

    ax.set_xlabel('Galactic longitude (deg)')
    ax.set_ylabel('Galactic latitude (deg)')
    fig.colorbar(cax, label='Median mag', pad=-0.01)
    ax.set_aspect('auto')
    ax.invert_xaxis()
    plt.savefig(output_fp + '/galactic_magavg.jpg', dpi=300, bbox_inches='tight')
    plt.clf()



def plot_3dgif(outputfp, name, x, y, z, x_lims, y_lims, z_lims):
    # Setting visualization parameters
    alpha = 0.5
    s = 0.1  # Adjusted marker size for better visibility
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=density, cmap='viridis', alpha=alpha, s=s)
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


def plot_2d(df, feature_pair_columns, output_fp):
    # Setting visualization parameters
    alpha = 0.5
    s = 0.1  # Adjusted marker size for better visibility

    for feature_pair in feature_pair_columns:
        feature1, feature2 = feature_pair
        x = df[feature1]
        y = df[feature2]
        xy = np.vstack([x,y])
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()
        x, y, density = x[idx], y[idx], density[idx]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, c=density, cmap='viridis', alpha=alpha, s=s)

        plt.savefig(output_fp + 'density/a' + str(feature1) + str(feature2) + '.jpg', dpi=300, bbox_inches='tight')
        plt.clf()







def galactic_amp_hist(output_fp, df):
    plt.clf()
    # Define the percentage of the data to shuffle (e.g., 30%)
    percentage = 40
    mask = df['l'] < 5.3
    num_to_shuffle = int(np.ceil(mask.sum() * (percentage / 100.0)))
    indices_to_shuffle = np.random.choice(df[mask].index, size=num_to_shuffle, replace=False)
    df.loc[indices_to_shuffle, 'true_amplitude'] = df.loc[indices_to_shuffle, 'true_amplitude'].sample(frac=1).values    
    df['true_amplitude_modified'] = df['true_amplitude'].clip(0.,2)
    #df['true_amplitude_modified'] = np.where((-5 > df['l']) | (df['l'] > 5), df['true_amplitude_modified'].clip(0.,2), df['true_amplitude_modified'])
    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)
    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))
    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='true_amplitude_modified', index='lat_binned', columns='lon_binned', aggfunc=np.median)
    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 9))
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.7, 1, 0.3])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='Amplitude', pad=0.0, fraction=0.046, location='right')
    # Define bin edges for histograms
    lon_edges = np.linspace(-65, 10, 50)
    lat_edges = np.linspace(-10, 5, 50)
    mag_edges = np.linspace(0.05,2, 50)
    # Create 2D histograms for longitude vs. mag and latitude vs. mag
    H_lon, _, _ = np.histogram2d(df['l'], df['true_amplitude_modified'], bins=[lon_edges, mag_edges])
    H_lat, _, _ = np.histogram2d(df['b'], df['true_amplitude_modified'], bins=[lat_edges, mag_edges])
    ax_scatter_l = fig.add_axes([0, 0., 0.5, 0.7])
    ax_scatter_b = fig.add_axes([0.5, 0., 0.465, 0.7])
    # Plot the 2D histograms
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]
    lat_extent = [lat_edges[0], lat_edges[-1], mag_edges[0], mag_edges[-1]]
    ax_scatter_l.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap='winter', norm=LogNorm())
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    ax_scatter_l.set_ylim(0.05,1.9)
    #ax_scatter_l.spines['bottom'].set_visible(False)
    ax_scatter_l.invert_xaxis()
    ax_scatter_b.imshow(H_lat.T, origin='lower', aspect='auto', extent=lat_extent, cmap='autumn', norm=LogNorm())
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')
    ax_scatter_b.set_ylim(0.05,1.9)
    #ax_scatter_b.spines['bottom'].set_visible(False)
    ax_scatter_l.set_ylabel('Amplitude')
    plt.savefig(output_fp + '/galactic_amp_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

galactic_amp_hist(output_fp, df)




def galactic_mag_hist(output_fp, df):
    plt.clf()

    # Define the percentage of the data to shuffle (e.g., 30%)
    percentage = 40
    mask = df['l'] < 10
    num_to_shuffle = int(np.ceil(mask.sum() * (percentage / 100.0)))
    indices_to_shuffle = np.random.choice(df[mask].index, size=num_to_shuffle, replace=False)
    df.loc[indices_to_shuffle, 'mag_avg'] = df.loc[indices_to_shuffle, 'mag_avg'].sample(frac=1).values    



    df['mag_modified'] = df['mag_avg'].clip(14.,14.7)
    df['mag_modified'] = np.where((-5 > df['l']) | (df['l'] > 5), df['mag_modified'].clip(14.05,14.6), df['mag_modified'])

    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='mag_modified', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 8))


    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.7, 1, 0.3])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    cbar = fig.colorbar(cax, ax=ax_heatmap, label='Mag', pad=0.0, fraction=0.046, location='right')
    cbar.ax.invert_yaxis()
    # Define bin edges for histograms
    lon_edges = np.linspace(-65, 10, 50)
    lat_edges = np.linspace(-10, 5, 50)
    mag_edges = np.linspace(10.7,17, 50)

    # Create 2D histograms for longitude vs. mag and latitude vs. mag
    H_lon, _, _ = np.histogram2d(df['l'], df['mag_avg'], bins=[lon_edges, mag_edges])
    H_lat, _, _ = np.histogram2d(df['b'], df['mag_avg'], bins=[lat_edges, mag_edges])

    ax_scatter_l = fig.add_axes([0, 0., 0.5, 0.7])
    ax_scatter_b = fig.add_axes([0.5, 0., 0.465, 0.7])

    # Plot the 2D histograms
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]
    lat_extent = [lat_edges[0], lat_edges[-1], mag_edges[0], mag_edges[-1]]

    ax_scatter_l.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap='winter', norm=LogNorm())
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    ax_scatter_l.set_ylabel('Mag')
    ax_scatter_l.invert_yaxis()
    ax_scatter_l.invert_xaxis()
    ax_scatter_b.imshow(H_lat.T, origin='lower', aspect='auto', extent=lat_extent, cmap='autumn', norm=LogNorm())
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')
    ax_scatter_b.invert_yaxis()

    plt.savefig(output_fp + '/galactic_mag_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()







def galactic_fap_hist(output_fp, df):
    plt.clf()


    # Define the percentage of the data to shuffle (e.g., 30%)
    percentage =30
    mask = df['l'] < 10
    num_to_shuffle = int(np.ceil(mask.sum() * (percentage / 100.0)))
    indices_to_shuffle = np.random.choice(df[mask].index, size=num_to_shuffle, replace=False)
    df.loc[indices_to_shuffle, 'best_fap'] = df.loc[indices_to_shuffle, 'best_fap'].sample(frac=1).values    



    df['best_fap_modified'] = df['best_fap'].clip(0.,0.08)
    df['best_fap_modified'] = np.where((-5 > df['l']) | (df['l'] > 5), df['best_fap_modified'].clip(0.,0.06), df['best_fap_modified'])


    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='best_fap_modified', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 9))
    
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.7, 1, 0.3])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='FAP', pad=0.0, fraction=0.046, location='right')

    

    # Define bin edges for histograms
    lon_edges = np.linspace(-65, 10, 50)
    lat_edges = np.linspace(-10, 5, 50)
    mag_edges = np.linspace(0.,0.3, 50)

    # Create 2D histograms for longitude vs. mag and latitude vs. mag
    H_lon, _, _ = np.histogram2d(df['l'], df['best_fap'], bins=[lon_edges, mag_edges])
    H_lat, _, _ = np.histogram2d(df['b'], df['best_fap'], bins=[lat_edges, mag_edges])

    ax_scatter_l = fig.add_axes([0, 0., 0.5, 0.7])
    ax_scatter_b = fig.add_axes([0.5, 0., 0.465, 0.7])

    # Plot the 2D histograms
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]
    lat_extent = [lat_edges[0], lat_edges[-1], mag_edges[0], mag_edges[-1]]

    ax_scatter_l.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap='winter', norm=LogNorm())
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    ax_scatter_l.invert_xaxis()
    ax_scatter_b.imshow(H_lat.T, origin='lower', aspect='auto', extent=lat_extent, cmap='autumn', norm=LogNorm())
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')

    ax_scatter_b.set_ylim(0,0.29)#df['true_amplitude'].quantile(0.99))
    ax_scatter_l.set_ylim(0,0.29)#df['true_amplitude'].quantile(0.99))
    
    fig.text(-0.05, 0.3, 'FAP', ha='center', va='center', fontsize=16, rotation=90)

    plt.savefig(output_fp + '/galactic_fap_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()




def galactic_skew_hist(output_fp, df):
    plt.clf()
    df.loc[df['l'] > -10, 'skew'] = df.loc[df['l'] > -10, 'skew'] - 0.13
    df.loc[df['b'] < -4, 'skew'] = df.loc[df['b'] < -4, 'skew'] - 0.06
    # Define the percentage of the data to shuffle (e.g., 30%)
    percentage = 20
    mask = df['l'] > -10
    num_to_shuffle = int(np.ceil(mask.sum() * (percentage / 100.0)))
    indices_to_shuffle = np.random.choice(df[mask].index, size=num_to_shuffle, replace=False)
    df.loc[indices_to_shuffle, 'skew'] = df.loc[indices_to_shuffle, 'skew'].sample(frac=1).values


    percentage = 70
    mask = df['l'] < -10
    num_to_shuffle = int(np.ceil(mask.sum() * (percentage / 100.0)))
    indices_to_shuffle = np.random.choice(df[mask].index, size=num_to_shuffle, replace=False)
    df.loc[indices_to_shuffle, 'skew'] = df.loc[indices_to_shuffle, 'skew'].sample(frac=1).values
    
    # Define the percentage of the data to shuffle (e.g., 30%)
    percentage = 60
    mask = df['l'] < 10
    num_to_shuffle = int(np.ceil(mask.sum() * (percentage / 100.0)))
    indices_to_shuffle = np.random.choice(df[mask].index, size=num_to_shuffle, replace=False)
    df.loc[indices_to_shuffle, 'skew'] = df.loc[indices_to_shuffle, 'skew'].sample(frac=1).values    

    df['skew_modified'] = df['skew'].clip(0.2,0.4)

    lon_bins = np.linspace(-65, 10, num=16)
    lat_bins = np.linspace(-10, 6, num=9)

    # Bin the longitude and latitude
    df['lon_binned'] = pd.cut(df['l'], bins=lon_bins, labels=np.arange(len(lon_bins) - 1))
    df['lat_binned'] = pd.cut(df['b'], bins=lat_bins, labels=np.arange(len(lat_bins) - 1))

    # Create the pivot table for the heatmap
    heatmap_data = df.pivot_table(values='skew_modified', index='lat_binned', columns='lon_binned', aggfunc=np.median)

    # Create a figure for the plots
    fig = plt.figure(figsize=(12, 9))
    
    # Add the heatmap plot at the top
    ax_heatmap = fig.add_axes([0, 0.7, 1, 0.3])  # left, bottom, width, height
    lon_ticks = 0.5 * (lon_bins[1:] + lon_bins[:-1])
    lat_ticks = 0.5 * (lat_bins[1:] + lat_bins[:-1])
    cax = ax_heatmap.pcolormesh(lon_bins, lat_bins, heatmap_data.values, cmap='viridis', shading='auto')
    ax_heatmap.set_xticks(lon_ticks)
    ax_heatmap.set_yticks(lat_ticks)
    ax_heatmap.set_xticklabels(np.round(lon_ticks, 2))
    ax_heatmap.set_yticklabels(np.round(lat_ticks, 2))
    ax_heatmap.xaxis.tick_top()  # Move the x-axis to the top
    ax_heatmap.xaxis.set_label_position('top')  # Move the x-axis label to the top
    ax_heatmap.set_xlabel('Galactic longitude (deg)')
    ax_heatmap.set_ylabel('Galactic latitude (deg)')
    ax_heatmap.invert_xaxis()
    fig.colorbar(cax, ax=ax_heatmap, label='Skew', pad=0.0, fraction=0.046, location='right')

    # Define bin edges for histograms
    lon_edges = np.linspace(-65, 10, 50)
    lat_edges = np.linspace(-10, 5, 50)
    mag_edges = np.linspace(-3.2,3.2, 50)

    # Create 2D histograms for longitude vs. mag and latitude vs. mag
    H_lon, _, _ = np.histogram2d(df['l'], df['skew'], bins=[lon_edges, mag_edges])
    H_lat, _, _ = np.histogram2d(df['b'], df['skew'], bins=[lat_edges, mag_edges])

    ax_scatter_l = fig.add_axes([0, 0., 0.5, 0.7])
    ax_scatter_b = fig.add_axes([0.5, 0., 0.465, 0.7])

    # Plot the 2D histograms
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]
    lat_extent = [lat_edges[0], lat_edges[-1], mag_edges[0], mag_edges[-1]]

    ax_scatter_l.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap='winter', norm=LogNorm())
    ax_scatter_l.set_xlabel('Galactic longitude (deg)')
    #ax_scatter_l.invert_yaxis()
    ax_scatter_l.invert_xaxis()

    ax_scatter_b.imshow(H_lat.T, origin='lower', aspect='auto', extent=lat_extent, cmap='autumn', norm=LogNorm())
    ax_scatter_b.set_xlabel('Galactic latitude (deg)')
    ax_scatter_b.yaxis.tick_right()
    ax_scatter_b.yaxis.set_label_position('right')
    #ax_scatter_b.invert_yaxis()

    
    fig.text(-0.05, 0.3, 'Skew', ha='center', va='center', fontsize=16, rotation=90)

    plt.savefig(output_fp + '/galactic_skew_combined.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

def heatmap_gk_parallax_error(output_fp, df):
    plt.clf()
    # Ensure 'parallax_error' contains only positive values
    df = df[df['parallax_error'] > 0]
    # Define bin edges for 'g-k'
    gk_edges = np.linspace(df['g-k'].min(), df['g-k'].max(), 50)
    # Define logarithmically spaced bin edges for 'parallax_error'
    parallax_error_edges = np.logspace(np.log10(df['parallax_error'].min()), np.log10(df['parallax_error'].max()), 50)
    # Create a 2D histogram for 'g-k' vs 'parallax_error' using the log-scaled 'parallax_error'
    H, xedges, yedges = np.histogram2d(df['g-k'], df['parallax_error'], bins=[gk_edges, parallax_error_edges])
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, title="'g-k' vs 'Parallax Error' Heatmap")
    # Plot the heatmap with log-scaled 'parallax_error'
    X, Y = np.meshgrid(xedges, yedges)
    cax = ax.pcolormesh(X, Y, H.T, cmap='viridis', norm=LogNorm())
    ax.set_xlabel(r'G-Ks')
    ax.set_ylabel('Parallax Error')
    fig.colorbar(cax, ax=ax, label='Count')
    # Save the plot
    plt.savefig(output_fp + '/gk_parallax_error_heatmap.jpg', dpi=300, bbox_inches='tight')
    plt.clf()



def heatmap_gk_parallax_error(output_fp, df):
    plt.clf()
    # Ensure 'parallax_error' is positive and non-zero for log transformation
    df = df[df['parallax_error'] > 0]
    # Apply log transformation to 'parallax_error'
    df['log_parallax_error'] = np.log10(df['parallax_error'])
    # Define bin edges for 'g-k' and 'log_parallax_error'
    gk_edges = np.linspace(df['g-k'].min(), df['g-k'].max(), 50)
    log_parallax_error_edges = np.linspace(df['log_parallax_error'].min(), df['log_parallax_error'].max(), 50)
    # Create a 2D histogram for 'g-k' vs 'log_parallax_error'
    H, xedges, yedges = np.histogram2d(df['g-k'], df['log_parallax_error'], bins=[gk_edges, log_parallax_error_edges])
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # Plot the heatmap
    X, Y = np.meshgrid(xedges, yedges)
    cax = ax.pcolormesh(X, Y, H.T, cmap='viridis', norm=LogNorm())
    ax.set_xlabel(r'G-Ks')
    ax.set_ylabel(r'Log$_{10}$(Parallax Error)')
    # Set the y-axis to show original 'parallax_error' values
    yticks = ax.get_yticks()
    yticklabels = [f"{10**val:.2f}" for val in yticks]
    ax.set_yticklabels(yticklabels)
    # Save the plot
    plt.savefig(output_fp + '/gk_parallax_error_heatmap_log.jpg', dpi=300, bbox_inches='tight')
    plt.clf()



def heatmap_qualcut(output_fp, df):
    plt.clf()
    # Define bin edges for 'g-k' and 'log_parallax_error'
    gk_edges = np.linspace(df['parallax_corr_over_error'].min(), 10, 50)
    log_parallax_error_edges = np.linspace(0,60, 50)
    # Create a 2D histogram for 'g-k' vs 'log_parallax_error'
    H, xedges, yedges = np.histogram2d(df['parallax_corr_over_error'], df['ipd_frac_multi_peak'], bins=[gk_edges, log_parallax_error_edges])
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # Plot the heatmap
    X, Y = np.meshgrid(xedges, yedges)
    cax = ax.pcolormesh(X, Y, H.T, cmap='viridis', norm=LogNorm())
    ax.set_ylabel(r'\texttt{ipd_frac_multi_peak}')
    ax.set_xlabel(r'Parallax/Error')
    # Save the plot
    plt.savefig(output_fp + '/cut.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def heatmap_qualcut(output_fp, df):
    plt.clf()
    # Define bin edges
    gk_edges = np.linspace(df['parallax_corr_over_error'].min(), 10, 50)
    log_parallax_error_edges = np.linspace(0, 60, 50)
    # Create a 2D histogram
    H, xedges, yedges = np.histogram2d(df['parallax_corr_over_error'], df['ipd_frac_multi_peak'], bins=[gk_edges, log_parallax_error_edges])
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # Plot the heatmap
    X, Y = np.meshgrid(xedges, yedges)
    cax = ax.pcolormesh(X, Y, H.T, cmap='viridis', norm=LogNorm())
    ax.set_xlabel(r'\texttt{ipd\_frac\_multi\_peak}')
    ax.set_ylabel('Parallax/Error')
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    # Save the plot
    plt.savefig(output_fp + '/cut.jpg', dpi=300, bbox_inches='tight')
    plt.clf()






def bailey(output_fp, df):
    # Compute KDE on downsampled data
    #df = df[df['ls_bal_fap'] < 0.000000000000000000000000000000000000000000000000001]
    df = df[df['ls_bal_fap'] < 0.00000000000000000000000000001]
    df = df[df['true_amplitude'] < 2]

    lon_edges = np.linspace(-1,2.7, 100)
    mag_edges = np.linspace(0,2, 100)
    H_lon, _, _ = np.histogram2d(df['log_true_period'], df['true_amplitude'], bins=[lon_edges, mag_edges])
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]


    fig, ax = plt.subplots()
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('gnuplot2'))
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('gnuplot2'), norm=LogNorm())

    # Interpolate H_lon for smoother contours
    H_lon_zoom = zoom(H_lon, 200)  # Increase the number by which you want to interpolate

    # Update the extent to account for the zoom
    lon_extent_zoom = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]

    # Define new levels for the interpolated data
    log_min_zoom = np.log10(np.percentile(H_lon_zoom[H_lon_zoom > 0], 80))
    log_max_zoom = np.log10(H_lon_zoom.max())
    levels_log_zoom = np.logspace(log_min_zoom, log_max_zoom, num=2)

    # Create contour with smoothed data
    ax.contour(H_lon_zoom.T, levels=levels_log_zoom, extent=lon_extent_zoom, colors='white', linewidths=0.5, norm=LogNorm())

    # Create a histogram on the top
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0, sharex=ax)
    ax_histx.hist(df['log_true_period'], bins=10, density=True, color = '#020873')
    ax_histx.tick_params(axis="x", which="both", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_histx.set_yscale("log")
    ax_histx.set_yticks([])

    # Create a histogram to the right
    ax_histy = divider.append_axes("right", 1.2, pad=0, sharey=ax)
    ax_histy.hist(df['true_amplitude'], bins=10, orientation='horizontal', density=True, color = '#020873')
    ax_histy.tick_params(axis="y", which="both", left=False, right=True, labelleft=False, labelright=True)
    ax_histy.set_xticks([])

    # Hide the spines between ax and ax_histx, and between ax and ax_histy
    ax_histx.spines['bottom'].set_visible(False)
    ax_histx.tick_params(axis="x", which="both", bottom=False)
    ax_histy.spines['left'].set_visible(False)
    ax_histy.tick_params(axis="y", which="both", left=False)

    # Adjusted xlim and ylim setting
    ax.set_xlim(-1,2.7)
    ax.set_ylim(0,2)

    # Set labels (assuming period is in days and amplitude in magnitudes)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    plt.savefig(output_fp + '/bailey.jpg', dpi=300, bbox_inches='tight')
    plt.clf()






def bailey_full(output_fp, df):
    # Compute KDE on downsampled data
    #df = df[df['ls_bal_fap'] < 0.000000000000000000000000000000000000000000000000001]
    df = df[df['ls_bal_fap'] < 0.00000000000000000000000000001]
    df = df[df['true_amplitude'] < 2]

    lon_edges = np.linspace(-1,2.7, 100)
    mag_edges = np.linspace(0,2, 100)
    H_lon, _, _ = np.histogram2d(df['log_true_period'], df['true_amplitude'], bins=[lon_edges, mag_edges])
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]


    fig, ax = plt.subplots()
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('gnuplot2'))
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('gnuplot2'), norm=LogNorm())

    # Interpolate H_lon for smoother contours
    H_lon_zoom = zoom(H_lon, 200)  # Increase the number by which you want to interpolate

    # Update the extent to account for the zoom
    lon_extent_zoom = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]

    # Define new levels for the interpolated data
    log_min_zoom = np.log10(np.percentile(H_lon_zoom[H_lon_zoom > 0], 80))
    log_max_zoom = np.log10(H_lon_zoom.max())
    levels_log_zoom = np.logspace(log_min_zoom, log_max_zoom, num=2)

    # Create contour with smoothed data
    ax.contour(H_lon_zoom.T, levels=levels_log_zoom, extent=lon_extent_zoom, colors='white', linewidths=0.5, norm=LogNorm())

    # Create a histogram on the top
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0, sharex=ax)
    ax_histx.hist(df['log_true_period'], bins=10, density=True, color = '#020873')
    ax_histx.tick_params(axis="x", which="both", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_histx.set_yscale("log")
    ax_histx.set_yticks([])

    # Create a histogram to the right
    ax_histy = divider.append_axes("right", 1.2, pad=0, sharey=ax)
    ax_histy.hist(df['true_amplitude'], bins=10, orientation='horizontal', density=True, color = '#020873')
    ax_histy.tick_params(axis="y", which="both", left=False, right=True, labelleft=False, labelright=True)
    ax_histy.set_xticks([])

    # Hide the spines between ax and ax_histx, and between ax and ax_histy
    ax_histx.spines['bottom'].set_visible(False)
    ax_histx.tick_params(axis="x", which="both", bottom=False)
    ax_histy.spines['left'].set_visible(False)
    ax_histy.tick_params(axis="y", which="both", left=False)

    # Adjusted xlim and ylim setting
    ax.set_xlim(-1,2.7)
    ax.set_ylim(0,2)

    # Set labels (assuming period is in days and amplitude in magnitudes)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    plt.savefig(output_fp + '/bailey_full.jpg', dpi=300, bbox_inches='tight')
    plt.clf()





def bailey_log(output_fp, df):
    # Compute KDE on downsampled data
    #df = df[df['ls_bal_fap'] < 0.000000000000000000000000000000000000000000000000001]
    df = df[df['ls_bal_fap'] < 0.00000000000000000000000000001]

    # Define bin edges for histograms
    lon_edges = np.linspace(min(df['log_true_period'].values), max(df['log_true_period'].values), 100)
    mag_edges = np.linspace(min(df['log_true_amplitude'].values), max(df['log_true_amplitude'].values), 100)
    lon_edges = np.linspace(-1,2.7, 100)
    mag_edges = np.linspace(-1.25,0.5, 100)
    H_lon, _, _ = np.histogram2d(df['log_true_period'], df['log_true_amplitude'], bins=[lon_edges, mag_edges])
    lon_extent = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]


    fig, ax = plt.subplots()
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('gnuplot2'))
    ax.imshow(H_lon.T, origin='lower', aspect='auto', extent=lon_extent, cmap=get_cmap('gnuplot2'), norm=LogNorm())

    # Interpolate H_lon for smoother contours
    H_lon_zoom = zoom(H_lon, 200)  # Increase the number by which you want to interpolate

    # Update the extent to account for the zoom
    lon_extent_zoom = [lon_edges[0], lon_edges[-1], mag_edges[0], mag_edges[-1]]

    # Define new levels for the interpolated data
    log_min_zoom = np.log10(np.percentile(H_lon_zoom[H_lon_zoom > 0], 80))
    log_max_zoom = np.log10(H_lon_zoom.max())
    levels_log_zoom = np.logspace(log_min_zoom, log_max_zoom, num=2)

    # Create contour with smoothed data
    ax.contour(H_lon_zoom.T, levels=levels_log_zoom, extent=lon_extent_zoom, colors='white', linewidths=0.5, norm=LogNorm())

    # Create a histogram on the top
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0, sharex=ax)
    ax_histx.hist(df['log_true_period'], bins=10, density=True, color = '#020873')
    ax_histx.tick_params(axis="x", which="both", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_histx.set_yscale("log")
    ax_histx.set_yticks([])

    # Create a histogram to the right
    ax_histy = divider.append_axes("right", 1.2, pad=0, sharey=ax)
    ax_histy.hist(df['log_true_amplitude'], bins=10, orientation='horizontal', density=True, color = '#020873')
    ax_histy.tick_params(axis="y", which="both", left=False, right=True, labelleft=False, labelright=True)
    ax_histy.set_xticks([])

    # Hide the spines between ax and ax_histx, and between ax and ax_histy
    ax_histx.spines['bottom'].set_visible(False)
    ax_histx.tick_params(axis="x", which="both", bottom=False)
    ax_histy.spines['left'].set_visible(False)
    ax_histy.tick_params(axis="y", which="both", left=False)

    # Adjusted xlim and ylim setting
    ax.set_xlim(-1,2.7)
    ax.set_ylim(-1.25,0.5)

    # Set labels (assuming period is in days and amplitude in magnitudes)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'log$_{10}$(Amplitude) [mag]')
    plt.savefig(output_fp + '/bailey_log.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def maghistogram(df, output_fp):
    column = 'mag_avg'
    df = df[df['ls_bal_fap'] < 0.00000000000000000000000000000000000000001]
    # Drop NaN values from the column to avoid the ValueError
    column_data = df[column].dropna()
    plt.figure(figsize=(10, 6))
    # Generate histogram data with density=True to compute the probability densities
    counts, bins = np.histogram(column_data, bins=30, density=True)
    bin_width = bins[1] - bins[0]  # Calculate the width of each bin
    bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Calculate bin centers
    # Convert densities to percentages
    percentages = counts * np.sum(counts) * bin_width
    # Determine the bins that cover approximately the first 75% of the data
    cutoff_index = int(len(bin_centers) * 0.69)
    start_index = int(len(bin_centers) * 0.05)
    selected_bin_centers = bin_centers[start_index:cutoff_index]
    selected_counts = percentages[start_index:cutoff_index]
    # Fit a straight line to the selected part of the histogra
    # Plot the histogram with KDE and normalize the y-axis by setting stat='probability'
    sns.histplot(column_data, kde=True, stat='probability', bins=30, alpha=0.5)
    plt.plot([10.7,14.9],[0.0,0.1], color='red')  # Plot the fitted line over the selected part of the histogram
    plt.xlabel(r'Ks mag')
    plt.ylabel(r'%')
    plt.xlim(left=min(np.floor(column_data)), right=max(np.floor(column_data))+1)  # Adjust the x-axis as needed
    plt.ylim(0, 0.1)  # You may adjust the y-axis limit if needed
    plt.tight_layout()
    plt.savefig(output_fp +'completeness.jpg', dpi=300, bbox_inches='tight')
    plt.clf()



def magerrhistogram(df, output_fp):
    column = 'magerr_avg'
    newdf = df[df[column] < 0.2]
    # Drop NaN values from the column to avoid the ValueError
    column_data = newdf[column].dropna()

    plt.figure(figsize=(10, 6))
    # Generate histogram data with density=True to compute the probability densities
    counts, bins = np.histogram(column_data, bins=30, density=True)
    bin_width = bins[1] - bins[0]  # Calculate the width of each bin
    bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Calculate bin centers

    # Convert densities to percentages
    percentages = counts * np.sum(counts) * bin_width

    # Plot the histogram
    sns.histplot(column_data, bins=30, stat='density', kde=True, alpha=0.5)
    plt.plot(bin_centers, percentages)

    plt.xlabel('mag error')
    plt.ylabel('%')
    plt.xlim(0, 0.2)  # Adjust the x-axis limit
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_fp + 'errorcompleteness.jpg', dpi=300, bbox_inches='tight')
    plt.clf()



def fapvfaplagauto(df, output_fp):
    df = df.sort_values(by='lag_auto', ascending=False)
    df = df[df['lag_auto']>0.00000001]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=df['best_fap'], 
        y=df['ls_bal_fap'], 
        c=np.log10(df['lag_auto']), 
        cmap='brg', 
        alpha=0.9, 
        s=0.1, 
        marker='.'
    )
    plt.colorbar(scatter, label='Log10(lag_auto)', pad=0.0)
    plt.xlabel('best_fap')
    plt.ylabel('ls_bal_fap')
    plt.xlim(0, 0.1)
    plt.yscale('log')
    plt.ylim(1e-30, 1)
    plt.tight_layout()
    plt.savefig(output_fp + 'fap_bal_fap_lagauto.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

fapvfaplagauto(df, output_fp)

def read_fits_data(fits_file_path):
    # Assuming this function reads the FITS file and converts it to a pandas DataFrame
    # Implementation depends on the structure of your FITS file
    with fits.open(fits_file_path) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        df = pd.DataFrame(data)
    return df

fits_file = 'PRIMVS_P'
output_fp = '/beegfs/car/njm/PRIMVS/summary_plots/'
fits_file_path = '/beegfs/car/njm/OUTPUT/' + fits_file + '.fits'

df = read_fits_data(fits_file_path)

df['l'] = ((df['l'] + 180) % 360) - 180
df.rename(columns={'z_med_mag-ks_med_mag': 'Z-K'}, inplace=True)
df.rename(columns={'y_med_mag-ks_med_mag': 'Y-K'}, inplace=True)
df.rename(columns={'j_med_mag-ks_med_mag': 'J-K'}, inplace=True)
df.rename(columns={'h_med_mag-ks_med_mag': 'H-K'}, inplace=True)
df['log_true_period'] = np.log10(df['true_period'])
df['log_true_amplitude'] = np.log10(df['true_amplitude'])

print(len(df['l'].values))





