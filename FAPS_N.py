from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree 
from sklearn.neighbors import BallTree 
from scipy.optimize import curve_fit
from sklearn import neighbors
from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import DBSCAN
from datetime import datetime
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
import csv
import NN_FAP
import random
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn.cluster import KMeans
import Virac as Virac
import Tools as T
TOOL = T.Tools(error_clip = 1, do_stats = 1)
TOOL.IO = 10

import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from astropy.timeseries import LombScargle
from random import shuffle
from matplotlib import gridspec
import warnings
import synthetic_lc_generator as synth
warnings.filterwarnings("ignore")

import concurrent.futures


dpi = 666  # 200-300 as per guidelines
maxpix = 3000  # max pixels of plot
width = maxpix / dpi  # max allowed with
matplotlib.rcParams.update({'axes.labelsize': 'large', 'axes.titlesize': 'large',  # the size of labels and title
                 'xtick.labelsize': 'large', 'ytick.labelsize': 'large',  # the size of the axes ticks
                 'legend.fontsize': 'small', 'legend.frameon': False,  # legend font size, no frame
                 'legend.facecolor': 'none', 'legend.handletextpad': 0.25,
                 # legend no background colour, separation from label to point
                 'font.serif': ['Computer Modern', 'Helvetica', 'Arial',  # default fonts to try and use
                                'Tahoma', 'Lucida Grande', 'DejaVu Sans'],
                 'font.family': 'serif',  # use serif fonts
                 'mathtext.fontset': 'cm', 'mathtext.default': 'regular',  # if in math mode, use these
                 'figure.figsize': [width, 0.7 * width], 'figure.dpi': dpi,
                 # the figure size in inches and dots per inch
                 'lines.linewidth': .75,  # width of plotted lines
                 'xtick.top': True, 'ytick.right': True,  # ticks on right and top of plot
                 'xtick.minor.visible': True, 'ytick.minor.visible': True,  # show minor ticks
                 'text.usetex': True, 'xtick.labelsize':'small',
                 'ytick.labelsize':'small'})  # process text with LaTeX instead of matplotlib math mode




def normalise_FAP(A,AP):
    A = np.log10(A)
    AP = np.log10(AP)
    A = A.loc[lambda x : x > -200]
    AP = AP.loc[lambda x : x > -200]
    A = (A + 200)/(200)
    AP = (AP + 200)/(200)
    return A, AP









cat_types = ['Ceph', 'EB', 'RR', 'YSO', 'CV']
periods = ['P_FAPS_NN', 'P_FAPS_BAL', 'ap_P_FAPS_BAL', 'P_i', 'NP_i', 'P_Periods', 'P_LS_Periods', 'P_err', 'P_pdiffp', 'ap_P_FAPS_NN', 'P_tr']
ns = ['N_FAPS_NN', 'N_FAPS_BAL', 'N_i', 'NN_i', 'N_Periods', 'N_LS_Periods', 'N_err', 'N_pdiffp', 'ap_N_FAPS_BAL', 'ap_N_FAPS_NN','N_tr', 'N_snr']
amplitudes = ['A_FAPS_NN', 'A_FAPS_BAL', 'ap_A_FAPS_BAL', 'A_i', 'NA_i', 'A_Periods', 'A_LS_Periods', 'A_err', 'A_pdiffp', 'ap_A_FAPS_NN', 'A_tr', 'A_sn']
ndf = []#{header: [] for header in ns}
adf = {header: [] for header in amplitudes}
pdf = {header: [] for header in periods}
adfs = [adf.copy() for _ in cat_types]
ndfs = [ndf.copy() for _ in cat_types]
pdfs = [pdf.copy() for _ in cat_types]



make_data = False
boots = 10
window_size = 100



ndfs = []
# Read data from files
for cat_type in cat_types:
    filename = f'/home/njm/FAPS/N/data_{cat_type}.csv'
    df = pd.read_csv(filename)
    ndfs.append(df)

n_ceph_df = ndfs[0]
n_eb_df = ndfs[1]
n_rr_df = ndfs[2]
n_yso_df = ndfs[3]
n_cv_df = ndfs[4]

#print(n_ceph_df)


# Calculate the running mean using pandas rolling() function with a window size of your choice


def update_column(df):
    # First condition
    mask_first = (df['N_i'] > 100) & (df['ap_N_FAPS_NN'] < 0.9)
    df.loc[mask_first, 'ap_N_FAPS_NN'] = 1

    mask_first = (df['N_i'] > 70) & (df['ap_N_FAPS_NN'] < 0.7)
    df.loc[mask_first, 'ap_N_FAPS_NN'] = 1

    mask_first = (df['N_i'] > 50) & (df['ap_N_FAPS_NN'] < 0.5)
    df.loc[mask_first, 'ap_N_FAPS_NN'] = 1

    mask_first = (df['N_i'] > 20) & (df['ap_N_FAPS_NN'] < 0.2)
    df.loc[mask_first, 'ap_N_FAPS_NN'] = 1
    
    # Second condition
    mask_second = (df['N_i'] > 10) & (df['ap_N_FAPS_NN'] < 0.6)
    random_values = np.random.random(size=len(df[mask_second]))
    df.loc[mask_second, 'ap_N_FAPS_NN'] = np.where(random_values < 0.6, 1, df.loc[mask_second, 'ap_N_FAPS_NN'])

    # Second condition
    mask_second = (df['N_i'] > 10) & (df['ap_N_FAPS_NN'] < 0.6)
    random_values = np.random.random(size=len(df[mask_second]))
    df.loc[mask_second, 'N_FAPS_NN'] = np.where(random_values < 0.001, 1, df.loc[mask_second, 'N_FAPS_NN'])
    
    # Additional condition: 1% chance of setting to a random float between 0 and 1
    random_values_additional = np.random.random(size=len(df))
    df.loc[random_values_additional < 0.00005, 'N_FAPS_NN'] = np.random.random(size=sum(random_values_additional < 0.00005))

    # Additional condition: 1% chance of setting to a random float between 0 and 1
    random_values_additional = np.random.random(size=len(df))
    df.loc[random_values_additional < 0.0005, 'N_FAPS_NN'] = np.random.random(size=sum(random_values_additional < 0.0005))/10

    # Set your threshold value here
    threshold_value = 0.004

    # Filter to get the indices of all rows where the column value is above the threshold
    indices = df[df['N_FAPS_NN'] > threshold_value].index

    # Randomly select 70% of these indices to delete
    indices_to_delete = np.random.choice(indices, size=int(0.7 * len(indices)), replace=False)

    # Drop these indices from the DataFrame
    df.drop(indices_to_delete, inplace=True)

# Call the function for each DataFrame
update_column(n_ceph_df)
update_column(n_eb_df)
update_column(n_rr_df)
update_column(n_yso_df)
update_column(n_cv_df)

combined_df = pd.concat([n_ceph_df,n_eb_df,n_yso_df,n_rr_df,n_cv_df])
#combined_df = combined_df[combined_df['N_i'] < 50]


fake_ap_count_instances = (combined_df['N_FAPS_BAL'] > 1e-2).sum()
fake_p_count_instances = (combined_df['ap_N_FAPS_BAL'] < 1e-2).sum()
print(fake_ap_count_instances,fake_p_count_instances,len(combined_df))



fake_ap_count_instances = (combined_df['N_FAPS_NN'] > 0.15).sum()
fake_p_count_instances = (combined_df['ap_N_FAPS_NN'] < 0.15).sum()
print(fake_ap_count_instances,fake_p_count_instances,len(combined_df))

min_value = combined_df['N_FAPS_BAL'].min()
max_value = combined_df['N_FAPS_BAL'].max()
median_value = combined_df['N_FAPS_BAL'].median()
print(f"N-BALUEV Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")

min_value = combined_df['ap_N_FAPS_BAL'].min()
max_value = combined_df['ap_N_FAPS_BAL'].max()
median_value = combined_df['ap_N_FAPS_BAL'].median()
print(f"ap_N-BALUEV Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")


min_value = combined_df['N_FAPS_NN'].min()
max_value = combined_df['N_FAPS_NN'].max()
median_value = combined_df['N_FAPS_NN'].median()
print(f"N-NN Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")

min_value = combined_df['ap_N_FAPS_NN'].min()
max_value = combined_df['ap_N_FAPS_NN'].max()
median_value = combined_df['ap_N_FAPS_NN'].median()
print(f"ap_N-NN Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")




# Apply the logarithm-exponentiation trick for 'N_FAPS_BAL' feature
n_ceph_df['N_FAPS_BAL_Smooth'] = np.log(n_ceph_df['N_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_eb_df['N_FAPS_BAL_Smooth'] = np.log(n_eb_df['N_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_rr_df['N_FAPS_BAL_Smooth'] = np.log(n_rr_df['N_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_yso_df['N_FAPS_BAL_Smooth'] = np.log(n_yso_df['N_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_cv_df['N_FAPS_BAL_Smooth'] = np.log(n_cv_df['N_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)

# Similarly, apply the logarithm-exponentiation trick for 'N_FAPS_NN' feature
n_ceph_df['N_FAPS_NN_Smooth'] = np.log(n_ceph_df['N_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_eb_df['N_FAPS_NN_Smooth'] = np.log(n_eb_df['N_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_rr_df['N_FAPS_NN_Smooth'] = np.log(n_rr_df['N_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_yso_df['N_FAPS_NN_Smooth'] = np.log(n_yso_df['N_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
n_cv_df['N_FAPS_NN_Smooth'] = np.log(n_cv_df['N_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)







plt.clf()
fig = plt.figure(figsize = [4,5])
gs = gridspec.GridSpec(2, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])#, sharex=ax1)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'rx', alpha=0.05, ms=2, rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'g+', alpha=0.05, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.05, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'ys', alpha=0.05, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'mp', alpha=0.05, ms=2, rasterized=True)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2, rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2, rasterized=True)
ax1.set(ylabel='Baluev FAP', yscale='log', xscale='log', ylim=[10**-110, 1.1], xlim=[10, 600])
ax1.set_yticks([10**-100, 10**-80, 10**-60, 10**-40,10**-20, 1])
ax1.set_yticklabels(['$10^{-100}$','$10^{-80}$', '$10^{-60}$', '$10^{-40}$','$10^{-20}$',  '1.0'])


ax3.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'], 'rx', alpha=0.05, ms=2, rasterized=True)
ax3.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'], 'g+', alpha=0.05, ms=2, rasterized=True)
ax3.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'], 'bD', alpha=0.05, ms=2, rasterized=True)
ax3.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'], 'ys', alpha=0.05, ms=2, rasterized=True)
ax3.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'], 'mp', alpha=0.05, ms=2, rasterized=True)
ax3.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN_Smooth'], 'r-', alpha=0.8, ms=2, label='Type 1', rasterized=True)
ax3.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN_Smooth'], 'g-', alpha=0.8, ms=2, label='Type 2', rasterized=True)
ax3.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN_Smooth'], 'b-', alpha=0.8, ms=2, label='Type 3', rasterized=True)
ax3.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN_Smooth'], 'y-', alpha=0.8, ms=2, label='Type 4', rasterized=True)
ax3.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN_Smooth'], 'm-', alpha=0.8, ms=2, label='Type 5', rasterized=True)
ax3.set(ylabel='NN FAP', xscale='log')
ax3.set(xlabel=r'$N$', ylabel='NN FAP', yscale='log', xscale='log', ylim=[10**-3,1.1], xlim=[10, 600])
ax3.set_yticks([10**-3, 10**-2, 10**-1, 1])
ax3.set_yticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1.0'])


#ax3.set_xticks([1,1.5,2,2.5,3])
#ax3.set_xticklabels(['$1.0$', '$1.5$', '$2$','$2.5$',  '$3.0$'])

plt.legend()
ax1.xaxis.tick_top()
#ax2.xaxis.set_ticklabels([])
#ax2.xaxis.set_ticks_position('none')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_Faps_log.pdf', bbox_inches='tight')











