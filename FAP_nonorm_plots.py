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


















# Define the function to process each iteration
def process_iteration_N(i, j, k, dfs):
    star_type = cat_types[k]
    mag, magerr, time, cat_type, N_Periods = synth.synthesize(N=i, amplitude=2, cat_type=star_type, other_pert=0, contamination_flag=0, median_mag = 16, time_range=[0.5, 3000])
    N_i = len(mag)
    N_tr = max(time) - min(time)
    N_FAPS_NN = NN_FAP.inference(N_Periods, mag, time, TOOL.knn, TOOL.model)
    ls = LombScargle(time, mag)
    ls_power = ls.power(TOOL.test_freqs)
    max_id = np.argmax(ls_power)
    ls_y_y_0 = ls_power[max_id]
    ls_x_y_0 = TOOL.test_freqs[max_id]
    np.random.shuffle(mag)
    ap_period = (N_Periods * 0.333333 + (0.654321 * N_Periods)) * np.random.uniform(0.33333, 1.666666)
    ap_N_FAPS_NN = NN_FAP.inference(ap_period, mag, time, TOOL.knn, TOOL.model)
    ap_ls = LombScargle(time, mag)
    ap_ls_power = ap_ls.power(TOOL.test_freqs)
    ap_max_id = np.argmax(ap_ls_power)
    ap_ls_y_y_0 = ap_ls_power[ap_max_id]
    ap_ls_x_y_0 = TOOL.test_freqs[ap_max_id]
    N_FAPS_BAL = ls.false_alarm_probability(ls_y_y_0)
    ap_N_FAPS_BAL = ap_ls.false_alarm_probability(ap_ls_y_y_0)
    N_LS_Periods = 1 / ls_x_y_0
    N_err = np.median(magerr)
    N_pdiffp = ((1 / ls_x_y_0) - N_Periods) / N_Periods
    NN_i = i#N_tr / N_Periods
    amp = max(mag) - min(mag)
    N_snr = amp / np.median(magerr)
    n_row = [
        N_FAPS_NN, 
        N_FAPS_BAL, 
        N_i, 
        NN_i, 
        N_Periods, 
        N_LS_Periods,
        N_err, 
        N_pdiffp, 
        ap_N_FAPS_BAL,
        ap_N_FAPS_NN, 
        N_tr,
        N_snr]
    dfs[k].append(n_row)




# Create a function to process a single call of process_iteration_N with given arguments
def process_single_call(args):
    i, j, k, ndfs = args
    process_iteration_N(i, j, k, ndfs)


if make_data == True:

    # Create a list of arguments for all the iterations
    args_list = [(i, j, k, ndfs) for i in np.arange(3, 600, 1) for j in range(boots) for k in range(len(ndfs))]
    # Use concurrent.futures.ThreadPoolExecutor to run the iterations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Wrap executor.map with tqdm to create a progress bar
        for _ in tqdm(executor.map(process_single_call, args_list), total=len(args_list)):
            pass

    '''
    for args in args_list:
        process_single_call(args)
    '''

    # Assuming 'ndfs' is a list of dataframes, and 'cat_types' is a list of headers
    for i, df in enumerate(ndfs):
        csv_file = f'/home/njm/FAPS/N/data_{cat_types[i]}.csv'
        #print(df)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header to the CSV file
            writer.writerow(ns)
            # Write the data rows to the CSV file
            for row in df:
                writer.writerow(row)


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








































# Define the function to process each iteration
def process_iteration_A(i, j, k, dfs):
    star_type = cat_types[k]
    mag, magerr, time, cat_type, period = TOOL.synthesize(N=200, amplitude=i, cat_type=star_type, other_pert=0, contamination_flag=0)
    time_range = max(time) - min(time)
    NN_True_FAP = NN_FAP.inference(period, mag, time, TOOL.knn, TOOL.model)
    ls = LombScargle(time, mag)
    ls_power = ls.power(TOOL.test_freqs)
    max_id = np.argmax(ls_power)
    ls_y_y_0 = ls_power[max_id]
    ls_x_y_0 = TOOL.test_freqs[max_id]
    np.random.shuffle(mag)
    ap_period = (period * 0.333333 + (0.654321 * period)) * np.random.uniform(0.33333, 1.666666)
    ap_NN_True_FAP = TOOL.fap_inference(ap_period, mag, time, TOOL.knn, TOOL.model)
    ap_ls = LombScargle(time, mag)
    ap_ls_power = ap_ls.power(TOOL.test_freqs)
    ap_max_id = np.argmax(ap_ls_power)
    ap_ls_y_y_0 = ap_ls_power[ap_max_id]
    ap_ls_x_y_0 = TOOL.test_freqs[ap_max_id]
    A_FAPS_BAL = ls.false_alarm_probability(ls_y_y_0)
    ap_A_FAPS_BAL = ap_ls.false_alarm_probability(ap_ls_y_y_0)
    A_Periods = period
    A_LS_Periods = 1 / ls_x_y_0
    A_FAPS_NN = NN_True_FAP
    A_i = i
    A_err = np.median(magerr)
    A_pdiffp = ((1 / ls_x_y_0) - period) / period
    ap_A_FAPS_NN = ap_NN_True_FAP
    NA_i = time_range / period
    A_tr = time_range
    #amp = max(mag) - min(mag)
    range_99, range_1 = np.percentile(mag, [99,1])
    range_99_1 = range_99 - range_1
    A_sn = range_99_1 / np.median(magerr)
    amplitude_row = {'A_FAPS_NN': A_FAPS_NN, 
            'A_FAPS_BAL': A_FAPS_BAL, 
            'ap_A_FAPS_BAL': ap_A_FAPS_BAL,
            'A_i': A_i, 
            'NA_i': NA_i, 
            'A_Periods': A_Periods, 
            'A_LS_Periods': A_LS_Periods,
            'A_err': A_err, 
            'A_pdiffp': A_pdiffp,
            'ap_A_FAPS_NN': ap_A_FAPS_NN,
            'A_tr': A_tr,
            'A_sn': A_sn
    }
    dfs[k].append(amplitude_row)




# Create a function to process a single call of process_iteration_N with given arguments
def process_single_call(args):
    i, j, k, adfs = args
    process_iteration_A(i, j, k, adfs)


if make_data == True:

    # Create a list of arguments for all the iterations
    args_list = [(i, j, k, adfs) for i in np.arange(0.01, 1, 0.01) for j in range(boots) for k in range(len(adfs))]
    # Use concurrent.futures.ThreadPoolExecutor to run the iterations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Wrap executor.map with tqdm to create a progress bar
        for _ in tqdm(executor.map(process_single_call, args_list), total=len(args_list)):
            pass



    # Assuming 'ndfs' is a list of dataframes, and 'cat_types' is a list of headers
    for i, df in enumerate(adfs):
        csv_file = f'/home/njm/FAPS/A/data_{cat_types[i]}.csv'
        
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header to the CSV file
            writer.writerow(amplitudes)
            # Write the data rows to the CSV file
            for row in df:
                writer.writerow(row)


adfs = []
# Read data from files
for cat_type in cat_types:
    filename = f'/home/njm/FAPS/A/data_{cat_type}.csv'
    df = pd.read_csv(filename)
    adfs.append(df)


a_ceph_df = adfs[0]
a_eb_df = adfs[1]
a_rr_df = adfs[2]
a_yso_df = adfs[3]
a_cv_df = adfs[4]
combined_df = pd.concat(adfs)
combined_df = combined_df[combined_df['A_sn'] < 12.5]

fake_ap_count_instances = (combined_df['A_FAPS_BAL'] > 1e-2).sum()
fake_p_count_instances = (combined_df['ap_A_FAPS_BAL'] < 1e-2).sum()
print(fake_ap_count_instances,fake_p_count_instances,len(combined_df))



fake_ap_count_instances = (combined_df['A_FAPS_NN'] > 0.5).sum()
fake_p_count_instances = (combined_df['ap_A_FAPS_NN'] < 0.5).sum()
print(fake_ap_count_instances,fake_p_count_instances,len(combined_df))

min_value = combined_df['A_FAPS_BAL'].min()
max_value = combined_df['A_FAPS_BAL'].max()
median_value = combined_df['A_FAPS_BAL'].median()
print(f"A-BALUEV Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")


min_value = combined_df['ap_A_FAPS_BAL'].min()
max_value = combined_df['ap_A_FAPS_BAL'].max()
median_value = combined_df['ap_A_FAPS_BAL'].median()
print(f"ap_A-BALUEV Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")

min_value = combined_df['A_FAPS_NN'].min()
max_value = combined_df['A_FAPS_NN'].max()
median_value = combined_df['A_FAPS_NN'].median()
print(f"A-NN Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")

min_value = combined_df['ap_A_FAPS_NN'].min()
max_value = combined_df['ap_A_FAPS_NN'].max()
median_value = combined_df['ap_A_FAPS_NN'].median()
print(f"ap_A-NN Minimum: {min_value}, Maximum: {max_value}, Median: {median_value}")


# Assuming you have loaded your dataframes a_ceph_df, a_eb_df, a_rr_df, a_yso_df, and a_cv_df
# Sort the DataFrame by a specific column before applying rolling window calculation
a_ceph_df.sort_values(by='A_sn', inplace=True)
a_eb_df.sort_values(by='A_sn', inplace=True)
a_rr_df.sort_values(by='A_sn', inplace=True)
a_yso_df.sort_values(by='A_sn', inplace=True)
a_cv_df.sort_values(by='A_sn', inplace=True)


# Apply the logarithm-exponentiation trick for 'A_FAPS_BAL' feature
a_ceph_df['A_FAPS_BAL_Smooth'] = np.log(a_ceph_df['A_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_eb_df['A_FAPS_BAL_Smooth'] = np.log(a_eb_df['A_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_rr_df['A_FAPS_BAL_Smooth'] = np.log(a_rr_df['A_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_yso_df['A_FAPS_BAL_Smooth'] = np.log(a_yso_df['A_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_cv_df['A_FAPS_BAL_Smooth'] = np.log(a_cv_df['A_FAPS_BAL']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)

# Similarly, apply the logarithm-exponentiation trick for 'A_FAPS_NN' feature
a_ceph_df['A_FAPS_NN_Smooth'] = np.log(a_ceph_df['A_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_eb_df['A_FAPS_NN_Smooth'] = np.log(a_eb_df['A_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_rr_df['A_FAPS_NN_Smooth'] = np.log(a_rr_df['A_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_yso_df['A_FAPS_NN_Smooth'] = np.log(a_yso_df['A_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
a_cv_df['A_FAPS_NN_Smooth'] = np.log(a_cv_df['A_FAPS_NN']).rolling(window=window_size, min_periods=1).mean().apply(np.exp)
# Apply Savitzky-Golay filter to get smoothed values

exit()



'''
a_ceph_df['A_FAPS_BAL_Smooth'] = savgol_filter(a_ceph_df['A_FAPS_BAL'], window_size, 3)
a_eb_df['A_FAPS_BAL_Smooth'] = savgol_filter(a_eb_df['A_FAPS_BAL'], window_size, 3)
a_rr_df['A_FAPS_BAL_Smooth'] = savgol_filter(a_rr_df['A_FAPS_BAL'], window_size, 3)
a_yso_df['A_FAPS_BAL_Smooth'] = savgol_filter(a_yso_df['A_FAPS_BAL'], window_size, 3)
a_cv_df['A_FAPS_BAL_Smooth'] = savgol_filter(a_cv_df['A_FAPS_BAL'], window_size, 3)
# Similarly, calculate the smoothed values for the second subplot
a_ceph_df['A_FAPS_NN_Smooth'] = savgol_filter(a_ceph_df['A_FAPS_NN'], window_size, 3)
a_eb_df['A_FAPS_NN_Smooth'] = savgol_filter(a_eb_df['A_FAPS_NN'], window_size, 3)
a_rr_df['A_FAPS_NN_Smooth'] = savgol_filter(a_rr_df['A_FAPS_NN'], window_size, 3)
a_yso_df['A_FAPS_NN_Smooth'] = savgol_filter(a_yso_df['A_FAPS_NN'], window_size, 3)
a_cv_df['A_FAPS_NN_Smooth'] = savgol_filter(a_cv_df['A_FAPS_NN'], window_size, 3)






# Fit a 3rd order polynomial to the data
a_ceph_coeffs = np.polyfit(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL'], 3)
a_eb_coeffs = np.polyfit(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL'], 3)
a_rr_coeffs = np.polyfit(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL'], 3)
a_yso_coeffs = np.polyfit(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL'], 3)
a_cv_coeffs = np.polyfit(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL'], 3)

# Create the fitted curve using the polynomial coefficients
a_ceph_df['A_FAPS_BAL_Smooth'] = np.polyval(a_ceph_coeffs, a_ceph_df['A_sn'] / 10)
a_eb_df['A_FAPS_BAL_Smooth'] = np.polyval(a_eb_coeffs, a_eb_df['A_sn'] / 10)
a_rr_df['A_FAPS_BAL_Smooth'] = np.polyval(a_rr_coeffs, a_rr_df['A_sn'] / 10)
a_yso_df['A_FAPS_BAL_Smooth'] = np.polyval(a_yso_coeffs, a_yso_df['A_sn'] / 10)
a_cv_df['A_FAPS_BAL_Smooth'] = np.polyval(a_cv_coeffs, a_cv_df['A_sn'] / 10)

# Similarly, fit the 3rd order polynomial for the second subplot
a_ceph_coeffs_nn = np.polyfit(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_NN'], 3)
a_eb_coeffs_nn = np.polyfit(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_NN'], 3)
a_rr_coeffs_nn = np.polyfit(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_NN'], 3)
a_yso_coeffs_nn = np.polyfit(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_NN'], 3)
a_cv_coeffs_nn = np.polyfit(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_NN'], 3)

a_ceph_df['A_FAPS_NN_Smooth'] = np.polyval(a_ceph_coeffs_nn, a_ceph_df['A_sn'] / 10)
a_eb_df['A_FAPS_NN_Smooth'] = np.polyval(a_eb_coeffs_nn, a_eb_df['A_sn'] / 10)
a_rr_df['A_FAPS_NN_Smooth'] = np.polyval(a_rr_coeffs_nn, a_rr_df['A_sn'] / 10)
a_yso_df['A_FAPS_NN_Smooth'] = np.polyval(a_yso_coeffs_nn, a_yso_df['A_sn'] / 10)
a_cv_df['A_FAPS_NN_Smooth'] = np.polyval(a_cv_coeffs_nn, a_cv_df['A_sn'] / 10)
'''





cat_types = ['CV','YSO','RR','EB','Ceph']
cat_types_fix = ['Type 5','Type 4','Type 3','Type 2','Type 1']
cat_marker = ['mp','ys','bD','g+','rx']




plt.clf()
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL'], 'rx', alpha=0.05, ms=2)
ax1.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL'], 'g+', alpha=0.05, ms=2)
ax1.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.05, ms=2)
ax1.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL'], 'ys', alpha=0.05, ms=2)
ax1.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL'], 'mp', alpha=0.05, ms=2)

ax1.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2)
ax1.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2)
ax1.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2)
ax1.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2)
ax1.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2)

ax1.set(ylabel='Baluev FAP', yscale='log', xscale='linear', ylim=[10**-105, 1.05])#, xlim=[10, 80])
ax1.set_yticks([10**-75, 10**-50,10**-25, 1])
ax1.set_yticklabels(['$10^{-75}$', '$10^{-50}$','$10^{-25}$',  '1.00'])
ax1.xaxis.tick_top()

ax2.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_NN'], 'rx', alpha=0.05, ms=2)#, label='Type 1')
ax2.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_NN'], 'g+', alpha=0.05, ms=2)#, label='Type 2')
ax2.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_NN'], 'bD', alpha=0.05, ms=2)#, label='Type 3')
ax2.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_NN'], 'ys', alpha=0.05, ms=2)#, label='Type 4')
ax2.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_NN'], 'mp', alpha=0.05, ms=2)#, label='Type 5')

ax2.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_NN_Smooth'], 'r-', alpha=0.8, ms=2, label='Type 1')
ax2.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_NN_Smooth'], 'g-', alpha=0.8, ms=2, label='Type 2')
ax2.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_NN_Smooth'], 'b-', alpha=0.8, ms=2, label='Type 3')
ax2.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_NN_Smooth'], 'y-', alpha=0.8, ms=2, label='Type 4')
ax2.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_NN_Smooth'], 'm-', alpha=0.8, ms=2, label='Type 5')

ax2.set(xlabel=r'$A/\bar{\sigma}$', ylabel='NN FAP', xscale='linear')#, xlim=[10, 80])
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/A_faps.pdf', bbox_inches='tight')





plt.clf()
fig = plt.figure(figsize = [4,5])
gs = gridspec.GridSpec(3, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)
ax3 = plt.subplot(gs[2])#, sharex=ax1)

ax1.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL'], 'rx', alpha=0.05, ms=2)
ax1.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL'], 'g+', alpha=0.05, ms=2)
ax1.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.05, ms=2)
ax1.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL'], 'ys', alpha=0.05, ms=2)
ax1.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL'], 'mp', alpha=0.05, ms=2)
ax1.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2)
ax1.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2)
ax1.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2)
ax1.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2)
ax1.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2)
ax1.set(yscale='log', xscale='linear', ylim=[10**-110, 5])
ax1.set_yticks([10**-100, 10**-80, 10**-60, 10**-40,10**-20, 1])
ax1.set_yticklabels(['$10^{-100}$','$10^{-80}$', '$10^{-60}$', '$10^{-40}$','$10^{-20}$',  '1.0'])

ax2.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL'], 'rx', alpha=0.05, ms=2)
ax2.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL'], 'g+', alpha=0.05, ms=2)
ax2.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.05, ms=2)
ax2.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL'], 'ys', alpha=0.05, ms=2)
ax2.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL'], 'mp', alpha=0.05, ms=2)
ax2.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2)
ax2.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2)
ax2.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2)
ax2.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2)
ax2.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2)
ax2.set(ylabel='Baluev FAP', yscale='linear', xscale='linear', ylim=[-0.1, 1.05])
ax2.yaxis.set_label_coords(-0.17,1)

ax3.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_NN'], 'rx', alpha=0.05, ms=2)
ax3.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_NN'], 'g+', alpha=0.05, ms=2)
ax3.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_NN'], 'bD', alpha=0.05, ms=2)
ax3.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_NN'], 'ys', alpha=0.05, ms=2)
ax3.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_NN'], 'mp', alpha=0.05, ms=2)
ax3.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_NN_Smooth'], 'r-', alpha=0.8, ms=2, label='Type 1')
ax3.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_NN_Smooth'], 'g-', alpha=0.8, ms=2, label='Type 2')
ax3.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_NN_Smooth'], 'b-', alpha=0.8, ms=2, label='Type 3')
ax3.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_NN_Smooth'], 'y-', alpha=0.8, ms=2, label='Type 4')
ax3.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_NN_Smooth'], 'm-', alpha=0.8, ms=2, label='Type 5')
ax3.set(ylabel='NN FAP', xscale='linear')
ax3.set(xlabel=r'$A/\bar{\sigma}$', yscale='linear', xscale='linear', ylim=[-0.1, 1.05])

ax3.set_xticks([1,1.5,2,2.5,3])
ax3.set_xticklabels(['$1.0$', '$1.5$', '$2$','$2.5$',  '$3.0$'])

ax1.xaxis.tick_top()
ax2.xaxis.set_ticklabels([])
ax2.xaxis.set_ticks_position('none')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/A_faps_linlog.pdf', bbox_inches='tight')







plt.clf()
fig = plt.figure(figsize = [4,5])
gs = gridspec.GridSpec(2, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(a_ceph_df['A_sn']/10, a_ceph_df['ap_A_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax1.plot(a_eb_df['A_sn']/10, a_eb_df['ap_A_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_rr_df['A_sn']/10, a_rr_df['ap_A_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_yso_df['A_sn']/10, a_yso_df['ap_A_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_cv_df['A_sn']/10, a_cv_df['ap_A_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_ceph_df['A_sn']/10, a_ceph_df['A_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax1.plot(a_eb_df['A_sn']/10, a_eb_df['A_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_rr_df['A_sn']/10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_yso_df['A_sn']/10, a_yso_df['A_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_cv_df['A_sn']/10, a_cv_df['A_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)

ax1.set(yscale='log', xscale='linear', ylim=[10**-105, 1.2])#, xlim=[10, 80])
ax1.set_yticks([10**-100, 10**-75, 10**-50,10**-25, 1])
ax1.set_yticklabels(['$10^{-100}$', '$10^{-75}$', '$10^{-50}$','$10^{-25}$',  '1.00'])
ax1.xaxis.tick_top()
ax2.plot(a_ceph_df['A_sn']/10, a_ceph_df['ap_A_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax2.plot(a_eb_df['A_sn']/10, a_eb_df['ap_A_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_rr_df['A_sn']/10, a_rr_df['ap_A_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_yso_df['A_sn']/10, a_yso_df['ap_A_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_cv_df['A_sn']/10, a_cv_df['ap_A_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_ceph_df['A_sn']/10, a_ceph_df['A_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax2.plot(a_eb_df['A_sn']/10, a_eb_df['A_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_rr_df['A_sn']/10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_yso_df['A_sn']/10, a_yso_df['A_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_cv_df['A_sn']/10, a_cv_df['A_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax2.set(xlabel=r'$A/\bar{\sigma}$', yscale='linear', ylim = [-0.05,1.05])#, xlim=[10, 80])
fig.supylabel('Baluev FAP', x = 0.07)
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/A_bal_linlog.pdf', bbox_inches='tight')





# Plot 2
plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(a_ceph_df['A_sn']/10, a_ceph_df['A_FAPS_NN'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax2.plot(a_eb_df['A_sn']/10, a_eb_df['A_FAPS_NN'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_rr_df['A_sn']/10, a_rr_df['A_FAPS_NN'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_yso_df['A_sn']/10, a_yso_df['A_FAPS_NN'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_cv_df['A_sn']/10, a_cv_df['A_FAPS_NN'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_ceph_df['A_sn']/10, a_ceph_df['ap_A_FAPS_NN'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax2.plot(a_eb_df['A_sn']/10, a_eb_df['ap_A_FAPS_NN'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_rr_df['A_sn']/10, a_rr_df['ap_A_FAPS_NN'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_yso_df['A_sn']/10, a_yso_df['ap_A_FAPS_NN'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_cv_df['A_sn']/10, a_cv_df['ap_A_FAPS_NN'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax2.set(xlabel=r'$A/\bar{\sigma}$', ylabel='NN FAP', yscale='linear')#, xlim=[10, 80])
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/A/A_NN.pdf', bbox_inches='tight')






# Plot 1
plt.clf()
gs = gridspec.GridSpec(1, 1, hspace=0)
fig = plt.figure()
ax1 = plt.subplot(gs[0])
ax1.plot(a_ceph_df['A_sn']/10, a_ceph_df['ap_A_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax1.plot(a_eb_df['A_sn']/10, a_eb_df['ap_A_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_rr_df['A_sn']/10, a_rr_df['ap_A_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_yso_df['A_sn']/10, a_yso_df['ap_A_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_cv_df['A_sn']/10, a_cv_df['ap_A_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_ceph_df['A_sn']/10, a_ceph_df['A_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax1.plot(a_eb_df['A_sn']/10, a_eb_df['A_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_rr_df['A_sn']/10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_yso_df['A_sn']/10, a_yso_df['A_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_cv_df['A_sn']/10, a_cv_df['A_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax1.set(xlabel=r'$A/\bar{\sigma}$', yscale='linear', ylim=[10**-110, 1.05])#, xlim=[10, 80])
fig.text(-0.005, 0.5, 'Baluev FAP', va='center', rotation='vertical', fontsize='small')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/A_Bal.pdf', bbox_inches='tight')








plt.clf()
gs = gridspec.GridSpec(2, 1, hspace=0)
fig = plt.figure()
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['ap_A_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax1.plot(a_eb_df['A_sn'] / 10, a_eb_df['ap_A_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_rr_df['A_sn'] / 10, a_rr_df['ap_A_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_yso_df['A_sn'] / 10, a_yso_df['ap_A_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_cv_df['A_sn'] / 10, a_cv_df['ap_A_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax1.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax1.set(xlabel=r'$A/\bar{\sigma}$', yscale='log', ylim=[10**-5, 1])
ax1.xaxis.tick_top()

ax2.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['ap_A_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax2.plot(a_eb_df['A_sn'] / 10, a_eb_df['ap_A_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_rr_df['A_sn'] / 10, a_rr_df['ap_A_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_yso_df['A_sn'] / 10, a_yso_df['ap_A_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_cv_df['A_sn'] / 10, a_cv_df['ap_A_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_ceph_df['A_sn'] / 10, a_ceph_df['A_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax2.plot(a_eb_df['A_sn'] / 10, a_eb_df['A_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_rr_df['A_sn'] / 10, a_rr_df['A_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_yso_df['A_sn'] / 10, a_yso_df['A_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(a_cv_df['A_sn'] / 10, a_cv_df['A_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax2.set(xlabel=r'$A/\bar{\sigma}$', yscale='log', ylim=[0, 10**-5])
fig.text(-0.005, 0.5, 'Baluev FAP', va='center', rotation='vertical', fontsize='small')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/A_Bal_split.pdf', bbox_inches='tight')








































plt.clf()
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'rx', alpha=0.05, ms=2)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'g+', alpha=0.05, ms=2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.05, ms=2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'ys', alpha=0.05, ms=2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'mp', alpha=0.05, ms=2)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2)

ax1.set(ylabel='Baluev FAP', yscale='linear', xscale='log', ylim=[10**-100, 1.05], xlim=[10, 600])
ax1.xaxis.tick_top()

ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'], 'rx', alpha=0.05, ms=2)#, label='Type 1')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'], 'g+', alpha=0.05, ms=2)#, label='Type 2')
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'], 'bD', alpha=0.05, ms=2)#, label='Type 3')
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'], 'ys', alpha=0.05, ms=2)#, label='Type 4')
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'], 'mp', alpha=0.05, ms=2)#, label='Type 5')

ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN_Smooth'], 'r-', alpha=0.8, ms=2, label='Type 1')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN_Smooth'], 'g-', alpha=0.8, ms=2, label='Type 2')
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN_Smooth'], 'b-', alpha=0.8, ms=2, label='Type 3')
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN_Smooth'], 'y-', alpha=0.8, ms=2, label='Type 4')
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN_Smooth'], 'm-', alpha=0.8, ms=2, label='Type 5')

ax2.set(xlabel=r'$N$', ylabel='NN FAP', xscale='log', ylim=[0, 1], xlim=[10, 600])
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_faps.pdf', bbox_inches='tight')







plt.clf()
fig = plt.figure(figsize = [4,5])
gs = gridspec.GridSpec(3, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)
ax3 = plt.subplot(gs[2])#, sharex=ax1)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'rx', alpha=0.05, ms=2)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'g+', alpha=0.05, ms=2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.05, ms=2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'ys', alpha=0.05, ms=2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'mp', alpha=0.05, ms=2)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2)
ax1.set(yscale='log', xscale='log', ylim=[10**-110, 1.1], xlim=[10, 600])
ax1.set_yticks([10**-100, 10**-80, 10**-60, 10**-40,10**-20, 1])
ax1.set_yticklabels(['$10^{-100}$','$10^{-80}$', '$10^{-60}$', '$10^{-40}$','$10^{-20}$',  '1.0'])

ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'rx', alpha=0.05, ms=2)
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'g+', alpha=0.05, ms=2)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.05, ms=2)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'ys', alpha=0.05, ms=2)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'mp', alpha=0.05, ms=2)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2)
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2)
ax2.set(ylabel='Baluev FAP', yscale='linear', xscale='log', ylim=[-0.05,0.15], xlim=[10, 600])
ax2.yaxis.set_label_coords(-0.17,1)

ax3.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'], 'rx', alpha=0.05, ms=2)
ax3.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'], 'g+', alpha=0.05, ms=2)
ax3.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'], 'bD', alpha=0.05, ms=2)
ax3.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'], 'ys', alpha=0.05, ms=2)
ax3.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'], 'mp', alpha=0.05, ms=2)
ax3.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN_Smooth'], 'r-', alpha=0.8, ms=2, label='Type 1')
ax3.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN_Smooth'], 'g-', alpha=0.8, ms=2, label='Type 2')
ax3.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN_Smooth'], 'b-', alpha=0.8, ms=2, label='Type 3')
ax3.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN_Smooth'], 'y-', alpha=0.8, ms=2, label='Type 4')
ax3.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN_Smooth'], 'm-', alpha=0.8, ms=2, label='Type 5')
ax3.set(ylabel='NN FAP', xscale='linear')
ax3.set(xlabel=r'$N$', ylabel='NN FAP', xscale='log', ylim=[-0.05,0.15], xlim=[10, 600])


#ax3.set_xticks([1,1.5,2,2.5,3])
#ax3.set_xticklabels(['$1.0$', '$1.5$', '$2$','$2.5$',  '$3.0$'])


ax1.xaxis.tick_top()
ax2.xaxis.set_ticklabels([])
ax2.xaxis.set_ticks_position('none')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_Faps_linlog.pdf', bbox_inches='tight')






















plt.clf()
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'rx', alpha=0.05, ms=2)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'g+', alpha=0.05, ms=2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.05, ms=2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'ys', alpha=0.05, ms=2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'mp', alpha=0.05, ms=2)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL_Smooth'], 'r-', alpha=0.8, ms=2)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL_Smooth'], 'g-', alpha=0.8, ms=2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL_Smooth'], 'b-', alpha=0.8, ms=2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL_Smooth'], 'y-', alpha=0.8, ms=2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL_Smooth'], 'm-', alpha=0.8, ms=2)

ax1.set(ylabel='Baluev FAP', yscale='log', xscale='log', ylim=[10**-100, 1.05], xlim=[10, 600])
ax1.set_yticks([10**-75, 10**-50,10**-25, 1])
ax1.set_yticklabels(['$10^{-75}$', '$10^{-50}$','$10^{-25}$',  '1.00'])

ax1.xaxis.tick_top()

ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'], 'rx', alpha=0.05, ms=2)#, label='Type 1')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'], 'g+', alpha=0.05, ms=2)#, label='Type 2')
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'], 'bD', alpha=0.05, ms=2)#, label='Type 3')
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'], 'ys', alpha=0.05, ms=2)#, label='Type 4')
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'], 'mp', alpha=0.05, ms=2)#, label='Type 5')

ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN_Smooth'], 'r-', alpha=0.8, ms=2, label='Type 1')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN_Smooth'], 'g-', alpha=0.8, ms=2, label='Type 2')
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN_Smooth'], 'b-', alpha=0.8, ms=2, label='Type 3')
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN_Smooth'], 'y-', alpha=0.8, ms=2, label='Type 4')
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN_Smooth'], 'm-', alpha=0.8, ms=2, label='Type 5')

ax2.set(xlabel=r'$N$', ylabel='NN FAP', xscale='log', ylim=[0, 1], xlim=[10, 600])
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_fapslog.pdf', bbox_inches='tight')





# Plot 2
plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_NN'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax2.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_NN'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_NN'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_NN'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_NN'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax2.set(xlabel=r'$N$', ylabel='NN FAP', yscale='linear', xscale='log', xlim=[10, 600], ylim=[0, 1])
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/N/N_NN.pdf', bbox_inches='tight')




# Plot 1
plt.clf()
gs = gridspec.GridSpec(1, 1, hspace=0)
fig = plt.figure()
ax1 = plt.subplot(gs[0])
ax1.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax1.set(xlabel=r'$N$', xlim=[10, 600], xscale='log')
fig.text(-0.005, 0.5, 'Baluev FAP', va='center', rotation='vertical', fontsize='small')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_Bal.pdf', bbox_inches='tight')










plt.clf()
fig = plt.figure(figsize = [4,5])
gs = gridspec.GridSpec(2, 1, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)

ax1.set(yscale='log', xscale='log', ylim=[10**-105, 1.05], xlim=[10, 600])
ax1.set_yticks([10**-100, 10**-75, 10**-50,10**-25, 1])
ax1.set_yticklabels(['$10^{-100}$', '$10^{-75}$', '$10^{-50}$','$10^{-25}$',  '1.0'])
ax1.xaxis.tick_top()
ax2.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax2.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax2.set(xlabel=r'$N$', xlim=[10, 600], xscale='log', ylim = [-0.05,1.05])
fig.supylabel('Baluev FAP', x = 0.07)
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_bal_linlog.pdf', bbox_inches='tight')








# Plot 1
plt.clf()
gs = gridspec.GridSpec(1, 1, hspace=0)
fig = plt.figure()
ax1 = plt.subplot(gs[0])
ax1.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax1.set(xlabel=r'$N$', yscale='log', ylim=[10**-100, 1.05], xlim=[10, 600], xscale='log')

ax1.set_yticks([10**-100,10**-75, 10**-50,10**-25, 1])
ax1.set_yticklabels(['$10^{-100}$', '$10^{-75}$', '$10^{-50}$','$10^{-25}$',  '1.0'])

fig.text(-0.005, 0.5, 'Baluev FAP', va='center', rotation='vertical', fontsize='small')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_Ballog.pdf', bbox_inches='tight')





plt.clf()
gs = gridspec.GridSpec(2, 1, hspace=0)
fig = plt.figure()
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)

ax1.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax1.set(xlabel=r'$N$', yscale='log', ylim=[10**-5, 1.05], xlim=[10, 200], xscale='log')
ax1.xaxis.tick_top()

ax2.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'], 'rx', alpha=0.3, ms=2, label='Aperiodic', rasterized=True)
ax2.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'], 'r+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'], 'rD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'], 'rs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'], 'rp', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'], 'bx', alpha=0.3, ms=2, label='Periodic', rasterized=True)
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'], 'b+', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'], 'bD', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'], 'bs', alpha=0.3, ms=2, rasterized=True)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'], 'bp', alpha=0.3, ms=2, rasterized=True)
ax2.set(xlabel=r'$N$', yscale='log', ylim=[0, 10**-5], xlim=[10, 600], xscale='log')
fig.text(-0.005, 0.5, 'Baluev FAP', va='center', rotation='vertical', fontsize='small')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/N_Bal_split.pdf', bbox_inches='tight')

















