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
warnings.filterwarnings("ignore")


dpi = 200  # 200-300 as per guidelines
maxpix = 670  # max pixels of plot
width = maxpix / dpi  # max allowed with
matplotlib.rcParams.update({'axes.labelsize': 'small', 'axes.titlesize': 'small',  # the size of labels and title
                 'xtick.labelsize': 'small', 'ytick.labelsize': 'small',  # the size of the axes ticks
                 'legend.fontsize': 'x-small', 'legend.frameon': False,  # legend font size, no frame
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
                 'text.usetex': True, 'xtick.labelsize':'medium',
                 'ytick.labelsize':'medium'})  # process text with LaTeX instead of matplotlib math mode




dpi = 200  # 200-300 as per guidelines
maxpix = 670  # max pixels of plot
width = maxpix / dpi  # max allowed with
matplotlib.rcParams.update({'axes.labelsize': 'small', 'axes.titlesize': 'small',  # the size of labels and title
                 'xtick.labelsize': 'large', 'ytick.labelsize': 'large',  # the size of the axes ticks
                 'legend.fontsize': 'x-small', 'legend.frameon': False,  # legend font size, no frame
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













big_data_fp = '/beegfs/car/njm/OUTPUT/vars/Redux_vars_filled.csv'
big_data = pd.read_csv(big_data_fp)
big_data = big_data.fillna(0)
big_df = big_data

boots = 10

star_type = 'Ceph'

cat_types = ['Ceph','EB','RR','YSO','CV']

#for star_type in cat_types:




cat_types = ['Ceph','EB','RR','YSO','CV']

periods = ['P_FAPS_NN','P_FAPS_BAL','ap_P_FAPS_BAL','P_i','NP_i','P_Periods','P_LS_Periods','P_err','P_pdiffp','ap_P_FAPS_NN','P_tr']
ns = ['N_FAPS_NN','N_FAPS_BAL','N_i','NN_i','N_Periods','N_LS_Periods','N_err','N_pdiffp','N_tr','ap_N_FAPS_BAL','ap_N_FAPS_NN']
amplitudes = ['A_FAPS_NN','A_FAPS_BAL','ap_A_FAPS_BAL','A_i','NA_i','A_Periods','A_LS_Periods','A_err','A_pdiffp','ap_A_FAPS_NN','A_tr','A_sn']

ceph_df = pd.read_csv('/home/njm/FAPS/Ceph/Ceph_FAPDATA.csv')  
eb_df = pd.read_csv('/home/njm/FAPS/EB/EB_FAPDATA.csv')  
rr_df = pd.read_csv('/home/njm/FAPS/RR/RR_FAPDATA.csv')  
yso_df = pd.read_csv('/home/njm/FAPS/YSO/YSO_FAPDATA.csv')  
cv_df = pd.read_csv('/home/njm/FAPS/CV/CV_FAPDATA.csv')  


p_ceph_df = ceph_df[periods].copy()
n_ceph_df = ceph_df[ns].copy()
a_ceph_df = ceph_df[amplitudes].copy()

p_eb_df = eb_df[periods].copy()
n_eb_df = eb_df[ns].copy()
a_eb_df = eb_df[amplitudes].copy()

p_rr_df = rr_df[periods].copy()
n_rr_df = rr_df[ns].copy()
a_rr_df = rr_df[amplitudes].copy()

p_yso_df = yso_df[periods].copy()
n_yso_df = yso_df[ns].copy()
a_yso_df = yso_df[amplitudes].copy()

p_cv_df = cv_df[periods].copy()
n_cv_df = cv_df[ns].copy()
a_cv_df = cv_df[amplitudes].copy()




cat_types = ['Ceph','EB','RR','YSO','CV']
dfs = [a_ceph_df, a_eb_df, a_rr_df, a_yso_df, a_cv_df]

for i in tqdm(np.arange(0.001, 0.5, 0.005)):
	for j in range(boots):
		for k, df in enumerate(dfs):
			star_type = cat_types[k]
			mag, magerr, time, cat_type, period = TOOL.synthesize(N = 200, amplitude = i, cat_type = star_type,other_pert = 0, contamination_flag = 0)
			time_range = max(time) - min(time)
			NN_True_FAP = TOOL.fap_inference(period, mag, time, TOOL.knn, TOOL.model)
			ls = LombScargle(time, mag)
			ls_power = ls.power(TOOL.test_freqs)
			max_id = np.argmax(ls_power)
			ls_y_y_0 = ls_power[max_id]
			ls_x_y_0 = TOOL.test_freqs[max_id]
			random.shuffle(mag)
			ap_period = (period * 0.333333 + (0.654321 * period)) * random.uniform(0.33333,1.666666)
			ap_NN_True_FAP = TOOL.fap_inference(ap_period, mag, time, TOOL.knn, TOOL.model)
			ap_ls = LombScargle(time, mag)
			ap_ls_power = ap_ls.power(TOOL.test_freqs)
			ap_max_id = np.argmax(ap_ls_power)
			ap_ls_y_y_0 = ap_ls_power[ap_max_id]
			ap_ls_x_y_0 = TOOL.test_freqs[ap_max_id]
			A_FAPS_BAL = ls.false_alarm_probability(ls_y_y_0)
			ap_A_FAPS_BAL = ap_ls.false_alarm_probability(ap_ls_y_y_0)
			A_Periods = period
			A_LS_Periods = 1/ls_x_y_0
			A_FAPS_NN = NN_True_FAP
			A_i = i
			A_err = np.median(magerr)
			A_pdiffp = ((1/ls_x_y_0)-period)/period
			ap_A_FAPS_NN = ap_NN_True_FAP
			NA_i = time_range/period
			A_tr = time_range
			amp = max(mag) - min(mag)
			A_sn = amp/np.median(magerr)
			amplitude_row = {'A_FAPS_NN':A_FAPS_NN,'A_FAPS_BAL':A_FAPS_BAL,'ap_A_FAPS_BAL':ap_A_FAPS_BAL,'A_i':A_i,'NA_i':NA_i,'A_Periods':A_Periods,'A_LS_Periods':A_LS_Periods,'A_err':A_err,'A_pdiffp':A_pdiffp,'ap_A_FAPS_NN':ap_A_FAPS_NN,'A_tr':A_tr,'A_sn':A_sn}
			dfs[k] = df.append(amplitude_row, ignore_index=True)

a_ceph_df = dfs[0]
a_eb_df = dfs[1]
a_rr_df = dfs[2]
a_yso_df = dfs[3]
a_cv_df = dfs[4]


bal_ticks = [10**-10,10**-50,10**-90]



p_ceph_df = ceph_df[periods].copy()
n_ceph_df = ceph_df[ns].copy()
a_ceph_df = ceph_df[amplitudes].copy()

p_eb_df = eb_df[periods].copy()
n_eb_df = eb_df[ns].copy()
a_eb_df = eb_df[amplitudes].copy()

p_rr_df = rr_df[periods].copy()
n_rr_df = rr_df[ns].copy()
a_rr_df = rr_df[amplitudes].copy()

p_yso_df = yso_df[periods].copy()
n_yso_df = yso_df[ns].copy()
a_yso_df = yso_df[amplitudes].copy()

p_cv_df = cv_df[periods].copy()
n_cv_df = cv_df[ns].copy()
a_cv_df = cv_df[amplitudes].copy()



def normalise_FAP(A,AP):
	A = np.log10(A)
	AP = np.log10(AP)
	A.loc[lambda x : x > -200]
	AP.loc[lambda x : x > -200]
	print(type(A))
	print(np.max(A),np.max(AP))
	print(np.min(A),np.min(AP))
	A = (A + 200)/(200)
	AP = (AP + 200)/(200)
	return A, AP



a_ceph_df['A_FAPS_BAL'], a_ceph_df['ap_A_FAPS_BAL'] = normalise_FAP(a_ceph_df['A_FAPS_BAL'], a_ceph_df['ap_A_FAPS_BAL'])
a_eb_df['A_FAPS_BAL'], a_eb_df['ap_A_FAPS_BAL'] = normalise_FAP(a_eb_df['A_FAPS_BAL'], a_eb_df['ap_A_FAPS_BAL'])
a_rr_df['A_FAPS_BAL'], a_rr_df['ap_A_FAPS_BAL'] = normalise_FAP(a_rr_df['A_FAPS_BAL'], a_rr_df['ap_A_FAPS_BAL'])
a_yso_df['A_FAPS_BAL'], a_yso_df['ap_A_FAPS_BAL'] = normalise_FAP(a_yso_df['A_FAPS_BAL'], a_yso_df['ap_A_FAPS_BAL'])
a_cv_df['A_FAPS_BAL'], a_cv_df['ap_A_FAPS_BAL'] = normalise_FAP(a_cv_df['A_FAPS_BAL'], a_cv_df['ap_A_FAPS_BAL'])



n_ceph_df['N_FAPS_BAL'], n_ceph_df['ap_N_FAPS_BAL'] = normalise_FAP(n_ceph_df['N_FAPS_BAL'], n_ceph_df['ap_N_FAPS_BAL'])
n_eb_df['N_FAPS_BAL'], n_eb_df['ap_N_FAPS_BAL'] = normalise_FAP(n_eb_df['N_FAPS_BAL'], n_eb_df['ap_N_FAPS_BAL'])
n_rr_df['N_FAPS_BAL'], n_rr_df['ap_N_FAPS_BAL'] = normalise_FAP(n_rr_df['N_FAPS_BAL'], n_rr_df['ap_N_FAPS_BAL'])
n_yso_df['N_FAPS_BAL'], n_yso_df['ap_N_FAPS_BAL'] = normalise_FAP(n_yso_df['N_FAPS_BAL'], n_yso_df['ap_N_FAPS_BAL'])
n_cv_df['N_FAPS_BAL'], n_cv_df['ap_N_FAPS_BAL'] = normalise_FAP(n_cv_df['N_FAPS_BAL'], n_cv_df['ap_N_FAPS_BAL'])


plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(a_ceph_df['A_FAPS_NN'], a_ceph_df['A_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax1.plot(a_eb_df['A_FAPS_NN'], a_eb_df['A_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax1.plot(a_rr_df['A_FAPS_NN'], a_rr_df['A_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(a_yso_df['A_FAPS_NN'], a_yso_df['A_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax1.plot(a_cv_df['A_FAPS_NN'], a_cv_df['A_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
#ax1.set(xscale = 'linear', yscale = 'log', ylim = [10**-100,1], yticks = bal_ticks)
ax1.xaxis.tick_top()
ax2.plot(a_ceph_df['ap_A_FAPS_NN'], a_ceph_df['ap_A_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax2.plot(a_eb_df['ap_A_FAPS_NN'], a_eb_df['ap_A_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax2.plot(a_rr_df['ap_A_FAPS_NN'], a_rr_df['ap_A_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax2.plot(a_yso_df['ap_A_FAPS_NN'], a_yso_df['ap_A_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax2.plot(a_cv_df['ap_A_FAPS_NN'], a_cv_df['ap_A_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
#ax2.set(xlabel = 'NN FAP', xscale = 'linear', yscale = 'log', yticks = [10**-2,10**-4,10**-6])
#fig.text(0.01, 0.5, 'Baluev FAP', va='center', rotation='vertical', fontsize = 'small')
plt.savefig('/home/njm/FAPS/A/A_fap_vs_fap.jpg', bbox_inches = 'tight')


plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(a_ceph_df['A_sn'], a_ceph_df['A_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax1.plot(a_eb_df['A_sn'], a_eb_df['A_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax1.plot(a_rr_df['A_sn'], a_rr_df['A_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(a_yso_df['A_sn'], a_yso_df['A_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax1.plot(a_cv_df['A_sn'], a_cv_df['A_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
ax1.set(ylabel='Baluev FAP', yscale = 'log', xscale = 'linear', ylim = [10**-100,1], yticks = bal_ticks, xlim = [10,80])
ax1.xaxis.tick_top()
ax2.plot(a_ceph_df['A_sn'], a_ceph_df['A_FAPS_NN'],'rx', alpha = 0.3, ms = 2, label = 'Cepheid')
ax2.plot(a_eb_df['A_sn'], a_eb_df['A_FAPS_NN'],'g+', alpha = 0.3, ms = 2, label = 'EB')
ax2.plot(a_rr_df['A_sn'], a_rr_df['A_FAPS_NN'],'bD', alpha = 0.3, ms = 2, label = 'W UMa')
ax2.plot(a_yso_df['A_sn'], a_yso_df['A_FAPS_NN'],'ys', alpha = 0.3, ms = 2, label = 'YSO')
ax2.plot(a_cv_df['A_sn'], a_cv_df['A_FAPS_NN'],'mp', alpha = 0.3, ms = 2, label = 'CV')
ax2.set(xlabel = r'$A/\bar{\sigma}$', ylabel='NN FAP', xscale = 'linear', xlim = [10,80])
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/A_faps.jpg', bbox_inches = 'tight')


plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
fig = plt.figure()
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(a_ceph_df['A_sn'], a_ceph_df['ap_A_FAPS_BAL'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax1.plot(a_eb_df['A_sn'], a_eb_df['ap_A_FAPS_BAL'],'r+', alpha = 0.3, ms = 2)
ax1.plot(a_rr_df['A_sn'], a_rr_df['ap_A_FAPS_BAL'],'rD', alpha = 0.3, ms = 2)
ax1.plot(a_yso_df['A_sn'], a_yso_df['ap_A_FAPS_BAL'],'rs', alpha = 0.3, ms = 2)
ax1.plot(a_cv_df['A_sn'], a_cv_df['ap_A_FAPS_BAL'],'rp', alpha = 0.3, ms = 2)
ax1.plot(a_ceph_df['A_sn'], a_ceph_df['A_FAPS_BAL'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax1.plot(a_eb_df['A_sn'], a_eb_df['A_FAPS_BAL'],'b+', alpha = 0.3, ms = 2)
ax1.plot(a_rr_df['A_sn'], a_rr_df['A_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(a_yso_df['A_sn'], a_yso_df['A_FAPS_BAL'],'bs', alpha = 0.3, ms = 2)
ax1.plot(a_cv_df['A_sn'], a_cv_df['A_FAPS_BAL'],'bp', alpha = 0.3, ms = 2)
ax1.set(xlabel = r'$A/\bar{\sigma}$', yscale = 'log', ylim = [0.000000001,1], xlim = [10,80], yticks = [10**-3,10**-6,10**-9])
ax1.xaxis.tick_top()
ax2.plot(a_ceph_df['A_sn'], a_ceph_df['ap_A_FAPS_BAL'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(a_eb_df['A_sn'], a_eb_df['ap_A_FAPS_BAL'],'r+', alpha = 0.3, ms = 2)
ax2.plot(a_rr_df['A_sn'], a_rr_df['ap_A_FAPS_BAL'],'rD', alpha = 0.3, ms = 2)
ax2.plot(a_yso_df['A_sn'], a_yso_df['ap_A_FAPS_BAL'],'rs', alpha = 0.3, ms = 2)
ax2.plot(a_cv_df['A_sn'], a_cv_df['ap_A_FAPS_BAL'],'rp', alpha = 0.3, ms = 2)
ax2.plot(a_ceph_df['A_sn'], a_ceph_df['A_FAPS_BAL'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(a_eb_df['A_sn'], a_eb_df['A_FAPS_BAL'],'b+', alpha = 0.3, ms = 2)
ax2.plot(a_rr_df['A_sn'], a_rr_df['A_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax2.plot(a_yso_df['A_sn'], a_yso_df['A_FAPS_BAL'],'bs', alpha = 0.3, ms = 2)
ax2.plot(a_cv_df['A_sn'], a_cv_df['A_FAPS_BAL'],'bp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$A/\bar{\sigma}$', yscale = 'log', ylim = [10**-100,1], xlim = [10,80], yticks = [10**-30,10**-60,10**-90])
fig.text(-0.005, 0.5, 'Baluev FAP', va='center', rotation='vertical', fontsize = 'small')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/A_Bal.jpg', bbox_inches = 'tight')


plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(a_ceph_df['A_sn'], a_ceph_df['A_FAPS_NN'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(a_eb_df['A_sn'], a_eb_df['A_FAPS_NN'],'b+', alpha = 0.3, ms = 2)
ax2.plot(a_rr_df['A_sn'], a_rr_df['A_FAPS_NN'],'bD', alpha = 0.3, ms = 2)
ax2.plot(a_yso_df['A_sn'], a_yso_df['A_FAPS_NN'],'bs', alpha = 0.3, ms = 2)
ax2.plot(a_cv_df['A_sn'], a_cv_df['A_FAPS_NN'],'bp', alpha = 0.3, ms = 2)
ax2.plot(a_ceph_df['A_sn'], a_ceph_df['ap_A_FAPS_NN'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(a_eb_df['A_sn'], a_eb_df['ap_A_FAPS_NN'],'r+', alpha = 0.3, ms = 2)
ax2.plot(a_rr_df['A_sn'], a_rr_df['ap_A_FAPS_NN'],'rD', alpha = 0.3, ms = 2)
ax2.plot(a_yso_df['A_sn'], a_yso_df['ap_A_FAPS_NN'],'rs', alpha = 0.3, ms = 2)
ax2.plot(a_cv_df['A_sn'], a_cv_df['ap_A_FAPS_NN'],'rp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$A/\bar{\sigma}$', ylabel='NN FAP', yscale = 'linear', xlim = [10,80])
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/A/A_NN.jpg', bbox_inches = 'tight')


plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(abs(np.array(a_ceph_df['A_pdiffp'])), a_ceph_df['A_sn'],'rx', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(a_eb_df['A_pdiffp'])), a_eb_df['A_sn'],'g+', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(a_rr_df['A_pdiffp'])), a_rr_df['A_sn'],'bD', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(a_yso_df['A_pdiffp'])), a_yso_df['A_sn'],'ys', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(a_cv_df['A_pdiffp'])), a_cv_df['A_sn'],'mp', alpha = 0.5, ms = 2)
ax1.set(ylabel=r'$A/\bar{\sigma}$', ylim = [10,80])
ax1.xaxis.tick_top()
ax2.plot(abs(np.array(a_ceph_df['A_pdiffp'])), a_ceph_df['A_FAPS_BAL'],'rx', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(a_eb_df['A_pdiffp'])), a_eb_df['A_FAPS_BAL'],'g+', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(a_rr_df['A_pdiffp'])), a_rr_df['A_FAPS_BAL'],'bD', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(a_yso_df['A_pdiffp'])), a_yso_df['A_FAPS_BAL'],'ys', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(a_cv_df['A_pdiffp'])), a_cv_df['A_FAPS_BAL'],'mp', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'Period Percentage Difference', ylabel='BAL FAP', yscale = 'log', ylim = [10**-100,1], yticks = bal_ticks)
fig.tight_layout()
plt.savefig('/home/njm/FAPS/A/Each_A_BAL_Periods.jpg', bbox_inches = 'tight')





a_bal_sens_yso = [];a_bal_sens_eb = [];a_bal_sens_rr = [];a_bal_sens_cv = [];a_bal_sens_ceph = []
a_nn_sens_yso = [];a_nn_sens_eb = [];a_nn_sens_rr = [];a_nn_sens_cv = [];a_nn_sens_ceph = []
a_bal_spec_yso = [];a_bal_spec_eb = [];a_bal_spec_rr = [];a_bal_spec_cv = [];a_bal_spec_ceph = []
a_nn_spec_yso = [];a_nn_spec_eb = [];a_nn_spec_rr = [];a_nn_spec_cv = [];a_nn_spec_ceph = []





for ranges in [[10**-500,10**-200],[10**-200,0.1],[0.1,1]]:
    for fap in tqdm(np.linspace(ranges[0],ranges[1],50000)):








a_NN_P_FAPS = list(a_yso_df['A_FAPS_NN']) + list(a_eb_df['A_FAPS_NN']) + list(a_rr_df['A_FAPS_NN']) + list(a_cv_df['A_FAPS_NN']) + list(a_ceph_df['A_FAPS_NN'])
a_BAL_P_FAPS = list(a_yso_df['A_FAPS_BAL']) + list(a_eb_df['A_FAPS_BAL']) + list(a_rr_df['A_FAPS_BAL']) + list(a_cv_df['A_FAPS_BAL']) + list(a_ceph_df['A_FAPS_BAL'])
a_NN_AP_FAPS = list(a_yso_df['ap_A_FAPS_NN']) + list(a_eb_df['ap_A_FAPS_NN']) + list(a_rr_df['ap_A_FAPS_NN']) + list(a_cv_df['ap_A_FAPS_NN']) + list(a_ceph_df['ap_A_FAPS_NN'])
a_BAL_AP_FAPS = list(a_yso_df['ap_A_FAPS_BAL']) + list(a_eb_df['ap_A_FAPS_BAL']) + list(a_rr_df['ap_A_FAPS_BAL']) + list(a_cv_df['ap_A_FAPS_BAL']) + list(a_ceph_df['ap_A_FAPS_BAL'])

bal_n_spec = []
bal_n_sens = []
nn_n_spec = []
nn_n_sens = []

idxs = []
idxs = list(a_NN_P_FAPS) + list(a_NN_AP_FAPS) + list(a_BAL_P_FAPS) + list(a_BAL_AP_FAPS)
idxs.sort()
for fap in tqdm(idxs):
	    a_bal_sens_yso.append(len(np.where(a_yso_df['A_FAPS_BAL'] < fap)[0])/len(a_yso_df['A_FAPS_BAL']))
	    a_bal_sens_eb.append(len(np.where(a_eb_df['A_FAPS_BAL'] < fap)[0])/len(a_eb_df['A_FAPS_BAL']))
	    a_bal_sens_rr.append(len(np.where(a_rr_df['A_FAPS_BAL'] < fap)[0])/len(a_rr_df['A_FAPS_BAL']))
	    a_bal_sens_cv.append(len(np.where(a_cv_df['A_FAPS_BAL'] < fap)[0])/len(a_cv_df['A_FAPS_BAL']))
	    a_bal_sens_ceph.append(len(np.where(a_ceph_df['A_FAPS_BAL'] < fap)[0])/len(a_ceph_df['A_FAPS_BAL']))
	    a_nn_sens_yso.append(len(np.where(a_yso_df['A_FAPS_NN'] < fap)[0])/len(a_yso_df['A_FAPS_NN']))
	    a_nn_sens_eb.append(len(np.where(a_eb_df['A_FAPS_NN'] < fap)[0])/len(a_eb_df['A_FAPS_NN']))
	    a_nn_sens_rr.append(len(np.where(a_rr_df['A_FAPS_NN'] < fap)[0])/len(a_rr_df['A_FAPS_NN']))
	    a_nn_sens_cv.append(len(np.where(a_cv_df['A_FAPS_NN'] < fap)[0])/len(a_cv_df['A_FAPS_NN']))
	    a_nn_sens_ceph.append(len(np.where(a_ceph_df['A_FAPS_NN'] < fap)[0])/len(a_ceph_df['A_FAPS_NN']))
	    a_bal_spec_yso.append(len(np.where(a_yso_df['ap_A_FAPS_BAL'] > fap)[0])/len(a_yso_df['ap_A_FAPS_BAL']))
	    a_bal_spec_eb.append(len(np.where(a_eb_df['ap_A_FAPS_BAL'] > fap)[0])/len(a_eb_df['ap_A_FAPS_BAL']))
	    a_bal_spec_rr.append(len(np.where(a_rr_df['ap_A_FAPS_BAL'] > fap)[0])/len(a_rr_df['ap_A_FAPS_BAL']))
	    a_bal_spec_cv.append(len(np.where(a_cv_df['ap_A_FAPS_BAL'] > fap)[0])/len(a_cv_df['ap_A_FAPS_BAL']))
	    a_bal_spec_ceph.append(len(np.where(a_ceph_df['ap_A_FAPS_BAL'] > fap)[0])/len(a_ceph_df['ap_A_FAPS_BAL']))
	    a_nn_spec_yso.append(len(np.where(a_yso_df['ap_A_FAPS_NN'] > fap)[0])/len(a_yso_df['ap_A_FAPS_NN']))
	    a_nn_spec_eb.append(len(np.where(a_eb_df['ap_A_FAPS_NN'] > fap)[0])/len(a_eb_df['ap_A_FAPS_NN']))
	    a_nn_spec_rr.append(len(np.where(a_rr_df['ap_A_FAPS_NN'] > fap)[0])/len(a_rr_df['ap_A_FAPS_NN']))
	    a_nn_spec_cv.append(len(np.where(a_cv_df['ap_A_FAPS_NN'] > fap)[0])/len(a_cv_df['ap_A_FAPS_NN']))
	    a_nn_spec_ceph.append(len(np.where(a_ceph_df['ap_A_FAPS_NN'] > fap)[0])/len(a_ceph_df['ap_A_FAPS_NN']))


split_point = 0.05
plt.clf()
gs=gridspec.GridSpec(1,2, hspace = 0, wspace = 0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharey=ax1)
ax1.plot(1-np.array(a_bal_spec_yso),a_bal_sens_yso, color = 'r')
ax1.plot(1-np.array(a_bal_spec_eb),a_bal_sens_eb, color = 'r')
ax1.plot(1-np.array(a_bal_spec_rr),a_bal_sens_rr, color = 'r')
ax1.plot(1-np.array(a_bal_spec_cv),a_bal_sens_cv, color = 'r')
ax1.plot(1-np.array(a_bal_spec_ceph),a_bal_sens_ceph, color = 'r')
ax1.plot(1-np.array(a_nn_spec_yso),a_nn_sens_yso, color = 'g')
ax1.plot(1-np.array(a_nn_spec_eb),a_nn_sens_eb, color = 'g')
ax1.plot(1-np.array(a_nn_spec_rr),a_nn_sens_rr, color = 'g')
ax1.plot(1-np.array(a_nn_spec_cv),a_nn_sens_cv, color = 'g')
ax1.plot(1-np.array(a_nn_spec_ceph),a_nn_sens_ceph, color = 'g')
ax1.set(xlim = [-0.0001,split_point],ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
ax2.yaxis.tick_right()
ax2.plot(1-np.array(a_bal_spec_yso),a_bal_sens_yso, color = 'r')
ax2.plot(1-np.array(a_bal_spec_eb),a_bal_sens_eb, color = 'r')
ax2.plot(1-np.array(a_bal_spec_rr),a_bal_sens_rr, color = 'r')
ax2.plot(1-np.array(a_bal_spec_cv),a_bal_sens_cv, color = 'r')
ax2.plot(1-np.array(a_bal_spec_ceph),a_bal_sens_ceph, color = 'r')
ax2.plot(1-np.array(a_nn_spec_yso),a_nn_sens_yso, color = 'g')
ax2.plot(1-np.array(a_nn_spec_eb),a_nn_sens_eb, color = 'g')
ax2.plot(1-np.array(a_nn_spec_rr),a_nn_sens_rr, color = 'g')
ax2.plot(1-np.array(a_nn_spec_cv),a_nn_sens_cv, color = 'g')
ax2.plot(1-np.array(a_nn_spec_ceph),a_nn_sens_ceph, color = 'g')
ax2.set(xlim = [split_point,1], xscale = 'linear')
plt.savefig('/home/njm/FAPS/A/AUC_split.jpg', bbox_inches = 'tight', dpi = 666)


plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(1-np.array(a_bal_spec_yso),a_bal_sens_yso, color = 'r')
ax1.plot(1-np.array(a_bal_spec_eb),a_bal_sens_eb, color = 'r')
ax1.plot(1-np.array(a_bal_spec_rr),a_bal_sens_rr, color = 'r')
ax1.plot(1-np.array(a_bal_spec_cv),a_bal_sens_cv, color = 'r')
ax1.plot(1-np.array(a_bal_spec_ceph),a_bal_sens_ceph, color = 'r')
ax1.plot(1-np.array(a_nn_spec_yso),a_nn_sens_yso, color = 'g')
ax1.plot(1-np.array(a_nn_spec_eb),a_nn_sens_eb, color = 'g')
ax1.plot(1-np.array(a_nn_spec_rr),a_nn_sens_rr, color = 'g')
ax1.plot(1-np.array(a_nn_spec_cv),a_nn_sens_cv, color = 'g')
ax1.plot(1-np.array(a_nn_spec_ceph),a_nn_sens_ceph, color = 'g')
ax1.set(ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
plt.savefig('/home/njm/FAPS/A/AUC.jpg', bbox_inches = 'tight', dpi = 666)


plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(1-np.array(a_bal_spec_yso),a_bal_sens_yso, color = 'r', label = 'Baluev')
ax1.plot(1-np.array(a_bal_spec_eb),a_bal_sens_eb, color = 'r')
ax1.plot(1-np.array(a_bal_spec_rr),a_bal_sens_rr, color = 'r')
ax1.plot(1-np.array(a_bal_spec_cv),a_bal_sens_cv, color = 'r')
ax1.plot(1-np.array(a_bal_spec_ceph),a_bal_sens_ceph, color = 'r')
ax1.plot(1-np.array(a_nn_spec_yso),a_nn_sens_yso, color = 'g', label = 'NN')
ax1.plot(1-np.array(a_nn_spec_eb),a_nn_sens_eb, color = 'g')
ax1.plot(1-np.array(a_nn_spec_rr),a_nn_sens_rr, color = 'g')
ax1.plot(1-np.array(a_nn_spec_cv),a_nn_sens_cv, color = 'g')
ax1.plot(1-np.array(a_nn_spec_ceph),a_nn_sens_ceph, color = 'g')
ax1.plot([0,0.196],[0.9,0.37],color='k', alpha = 0.5)
ax1.plot([0.2,0.625],[1,0.795],color='k', alpha = 0.5)
ax1.legend()
ax1.set(ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')

# this is an inset axes over the main axes
ax2 = fig.add_axes([.3, .4, .3, .3], facecolor='w', alpha = 0.3)
ax2.set(xlim = [-0.01,0.2], ylim=[0.9,1.01], xticks = [], yticks = [])
ax2.plot(1-np.array(a_bal_spec_yso),a_bal_sens_yso, color = 'r')
ax2.plot(1-np.array(a_bal_spec_eb),a_bal_sens_eb, color = 'r')
ax2.plot(1-np.array(a_bal_spec_rr),a_bal_sens_rr, color = 'r')
ax2.plot(1-np.array(a_bal_spec_cv),a_bal_sens_cv, color = 'r')
ax2.plot(1-np.array(a_bal_spec_ceph),a_bal_sens_ceph, color = 'r')
ax2.plot(1-np.array(a_nn_spec_yso),a_nn_sens_yso, color = 'g')
ax2.plot(1-np.array(a_nn_spec_eb),a_nn_sens_eb, color = 'g')
ax2.plot(1-np.array(a_nn_spec_rr),a_nn_sens_rr, color = 'g')
ax2.plot(1-np.array(a_nn_spec_cv),a_nn_sens_cv, color = 'g')
ax2.plot(1-np.array(a_nn_spec_ceph),a_nn_sens_ceph, color = 'g')

plt.savefig('/home/njm/FAPS/A/AUC_sub.jpg', bbox_inches = 'tight', dpi = 666)




















ns = ['N_FAPS_NN','N_FAPS_BAL','N_i','NN_i','N_Periods','N_LS_Periods','N_err','N_pdiffp','N_tr','ap_N_FAPS_BAL','ap_N_FAPS_NN']

plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(n_ceph_df['N_FAPS_NN'], n_ceph_df['N_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax1.plot(n_eb_df['N_FAPS_NN'], n_eb_df['N_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax1.plot(n_rr_df['N_FAPS_NN'], n_rr_df['N_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(n_yso_df['N_FAPS_NN'], n_yso_df['N_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax1.plot(n_cv_df['N_FAPS_NN'], n_cv_df['N_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
ax1.set(ylabel='Baluev FAP', xscale = 'linear', yscale = 'log', ylim = [10**-100,1], yticks = bal_ticks)
ax1.xaxis.tick_top()
ax2.plot(n_ceph_df['ap_N_FAPS_NN'], n_ceph_df['ap_N_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax2.plot(n_eb_df['ap_N_FAPS_NN'], n_eb_df['ap_N_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['ap_N_FAPS_NN'], n_rr_df['ap_N_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['ap_N_FAPS_NN'], n_yso_df['ap_N_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['ap_N_FAPS_NN'], n_cv_df['ap_N_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
ax2.set(xlabel = 'NN FAP', ylabel='Baluev FAP', xscale = 'linear', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/Each_N_fapfap.jpg', bbox_inches = 'tight')


plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
ax1.set(ylabel='Baluev FAP', yscale = 'log', xscale = 'log', xlim = [10,600], ylim = [10**-100,1], yticks = bal_ticks)
ax1.xaxis.tick_top()
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'],'rx', alpha = 0.3, ms = 2, label = 'Cepheid')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'],'g+', alpha = 0.3, ms = 2, label = 'EB')
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'],'bD', alpha = 0.3, ms = 2, label = 'W UMa')
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'],'ys', alpha = 0.3, ms = 2, label = 'YSO')
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'],'mp', alpha = 0.3, ms = 2, label = 'CV')
ax2.set(xlabel = r'$N$', ylabel='NN FAP', xscale = 'log', xlim = [10,600])
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/Each_N_faps.jpg', bbox_inches = 'tight')



plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'],'b+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'],'bs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'],'bp', alpha = 0.3, ms = 2)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'],'r+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'],'rD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'],'rs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'],'rp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$N$', ylabel='Baluev FAP', yscale = 'log', xscale = 'log', xlim = [10,600], ylim = [10**-100,1])
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/N/Each_ap_N_BAL_faps.jpg', bbox_inches = 'tight')



plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
fig = plt.figure()
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax1.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'],'r+', alpha = 0.3, ms = 2)
ax1.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'],'rD', alpha = 0.3, ms = 2)
ax1.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'],'rs', alpha = 0.3, ms = 2)
ax1.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'],'rp', alpha = 0.3, ms = 2)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'],'b+', alpha = 0.3, ms = 2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'],'bs', alpha = 0.3, ms = 2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'],'bp', alpha = 0.3, ms = 2)
ax1.set(xlabel = r'$N$', yscale = 'log', ylim = [0.000000001,1], xlim = [10,600], xscale = 'log', yticks = [10**-3,10**-6,10**-9])
ax1.xaxis.tick_top()

ax2.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'],'r+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'],'rD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'],'rs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'],'rp', alpha = 0.3, ms = 2)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'],'b+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'],'bs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'],'bp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$N$', yscale = 'log', ylim = [10**-100,1], xlim = [10,600], xscale = 'log', yticks = [10**-30,10**-60,10**-90])

fig.text(-0.005, 0.5, 'Baluev FAP', va='center', rotation='vertical')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/Each_ap_N_BAL_faps.jpg', bbox_inches = 'tight')















plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_NN'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_NN'],'r+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_NN'],'rD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_NN'],'rs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_NN'],'rp', alpha = 0.3, ms = 2)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'],'b+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'],'bD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'],'bs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'],'bp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$N$', ylabel='NN FAP', yscale = 'linear', xscale = 'log', xlim = [5,600])
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/N/Each_ap_N_NN_faps.jpg', bbox_inches = 'tight')






matplotlib.rcParams.update({'ytick.right': False})  # process text with LaTeX instead of matplotlib math mode
plt.clf()
gs=gridspec.GridSpec(1,2,vspace=0, hspace = 0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1])
ax1.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_BAL'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax1.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_BAL'],'b+', alpha = 0.3, ms = 2)
ax1.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_BAL'],'bs', alpha = 0.3, ms = 2)
ax1.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_BAL'],'bp', alpha = 0.3, ms = 2)
ax1.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_BAL'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax1.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_BAL'],'r+', alpha = 0.3, ms = 2)
ax1.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_BAL'],'rD', alpha = 0.3, ms = 2)
ax1.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_BAL'],'rs', alpha = 0.3, ms = 2)
ax1.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_BAL'],'rp', alpha = 0.3, ms = 2)
ax1.set(xlabel = r'$N$', ylabel='Baluev FAP', yscale = 'log', xscale = 'log', ylim = [10**-100,1])
ax1.yaxis.set_ticks_position("left")
ax1.yaxis.set_label_position("left")
plt.legend()

ax2.plot(n_ceph_df['N_i'], n_ceph_df['N_FAPS_NN'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['N_FAPS_NN'],'b+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['N_FAPS_NN'],'bD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['N_FAPS_NN'],'bs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['N_FAPS_NN'],'bp', alpha = 0.3, ms = 2)
ax2.plot(n_ceph_df['N_i'], n_ceph_df['ap_N_FAPS_NN'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(n_eb_df['N_i'], n_eb_df['ap_N_FAPS_NN'],'r+', alpha = 0.3, ms = 2)
ax2.plot(n_rr_df['N_i'], n_rr_df['ap_N_FAPS_NN'],'rD', alpha = 0.3, ms = 2)
ax2.plot(n_yso_df['N_i'], n_yso_df['ap_N_FAPS_NN'],'rs', alpha = 0.3, ms = 2)
ax2.plot(n_cv_df['N_i'], n_cv_df['ap_N_FAPS_NN'],'rp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$N$', ylabel='NN FAP', yscale = 'linear', xscale = 'log')
ax2.yaxis.set_ticks_position("right")
ax2.yaxis.set_label_position("right")
plt.savefig('/home/njm/FAPS/N/Each_ap_N_NN_BAL_faps.jpg', bbox_inches = 'tight')
matplotlib.rcParams.update({'ytick.right': True})  # process text with LaTeX instead of matplotlib math mode





plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(abs(np.array(n_ceph_df['N_pdiffp'])), n_ceph_df['N_i'],'rx', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(n_eb_df['N_pdiffp'])), n_eb_df['N_i'],'g+', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(n_rr_df['N_pdiffp'])), n_rr_df['N_i'],'bD', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(n_yso_df['N_pdiffp'])), n_yso_df['N_i'],'ys', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(n_cv_df['N_pdiffp'])), n_cv_df['N_i'],'mp', alpha = 0.5, ms = 2)
ax1.set(ylabel=r'$N$', yscale = 'log', ylim = [5,600])
ax1.xaxis.tick_top()
ax2.plot(abs(np.array(n_ceph_df['N_pdiffp'])), n_ceph_df['N_FAPS_BAL'],'rx', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(n_eb_df['N_pdiffp'])), n_eb_df['N_FAPS_BAL'],'g+', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(n_rr_df['N_pdiffp'])), n_rr_df['N_FAPS_BAL'],'bD', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(n_yso_df['N_pdiffp'])), n_yso_df['N_FAPS_BAL'],'ys', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(n_cv_df['N_pdiffp'])), n_cv_df['N_FAPS_BAL'],'mp', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'Period Percentage Difference', ylabel='BAL FAP', yscale = 'log', ylim = [10**-100,1], yticks = bal_ticks)
fig.tight_layout()
plt.savefig('/home/njm/FAPS/N/Each_N_BAL_Periods.jpg', bbox_inches = 'tight')
















n_bal_sens_yso = [];n_bal_sens_eb = [];n_bal_sens_rr = [];n_bal_sens_cv = [];n_bal_sens_ceph = []
n_nn_sens_yso = [];n_nn_sens_eb = [];n_nn_sens_rr = [];n_nn_sens_cv = [];n_nn_sens_ceph = []
n_bal_spec_yso = [];n_bal_spec_eb = [];n_bal_spec_rr = [];n_bal_spec_cv = [];n_bal_spec_ceph = []
n_nn_spec_yso = [];n_nn_spec_eb = [];n_nn_spec_rr = [];n_nn_spec_cv = [];n_nn_spec_ceph = []

n_NN_P_FAPS = list(n_yso_df['N_FAPS_NN']) + list(n_eb_df['N_FAPS_NN']) + list(n_rr_df['N_FAPS_NN']) + list(n_cv_df['N_FAPS_NN']) + list(n_ceph_df['N_FAPS_NN'])
n_BAL_P_FAPS = list(n_yso_df['N_FAPS_BAL']) + list(n_eb_df['N_FAPS_BAL']) + list(n_rr_df['N_FAPS_BAL']) + list(n_cv_df['N_FAPS_BAL']) + list(n_ceph_df['N_FAPS_BAL'])
n_NN_AP_FAPS = list(n_yso_df['ap_N_FAPS_NN']) + list(n_eb_df['ap_N_FAPS_NN']) + list(n_rr_df['ap_N_FAPS_NN']) + list(n_cv_df['ap_N_FAPS_NN']) + list(n_ceph_df['ap_N_FAPS_NN'])
n_BAL_AP_FAPS = list(n_yso_df['ap_N_FAPS_BAL']) + list(n_eb_df['ap_N_FAPS_BAL']) + list(n_rr_df['ap_N_FAPS_BAL']) + list(n_cv_df['ap_N_FAPS_BAL']) + list(n_ceph_df['ap_N_FAPS_BAL'])

bal_n_spec = []
bal_n_sens = []
nn_n_spec = []
nn_n_sens = []




n_bal_sens_yso = [];n_bal_sens_eb = [];n_bal_sens_rr = [];n_bal_sens_cv = [];n_bal_sens_ceph = []
n_nn_sens_yso = [];n_nn_sens_eb = [];n_nn_sens_rr = [];n_nn_sens_cv = [];n_nn_sens_ceph = []
n_bal_spec_yso = [];n_bal_spec_eb = [];n_bal_spec_rr = [];n_bal_spec_cv = [];n_bal_spec_ceph = []
n_nn_spec_yso = [];n_nn_spec_eb = [];n_nn_spec_rr = [];n_nn_spec_cv = [];n_nn_spec_ceph = []

idxs = []
idxs = list(n_NN_P_FAPS) + list(n_NN_AP_FAPS) + list(n_BAL_P_FAPS) + list(n_BAL_AP_FAPS)
idxs.sort()
for fap in tqdm(idxs):




for ranges in [[10**-500,10**-200],[10**-200,0.1],[0.1,1]]:
    for fap in tqdm(np.linspace(ranges[0],ranges[1],50000)):



idxs = []
idxs = list(n_NN_P_FAPS) + list(n_NN_AP_FAPS) + list(n_BAL_P_FAPS) + list(n_BAL_AP_FAPS)
idxs.sort()
for fap in tqdm(idxs):
	n_bal_sens_yso.append(len(np.where(n_yso_df['N_FAPS_BAL'] < fap)[0])/len(n_yso_df['N_FAPS_BAL']))
	n_bal_sens_eb.append(len(np.where(n_eb_df['N_FAPS_BAL'] < fap)[0])/len(n_eb_df['N_FAPS_BAL']))
	n_bal_sens_rr.append(len(np.where(n_rr_df['N_FAPS_BAL'] < fap)[0])/len(n_rr_df['N_FAPS_BAL']))
	n_bal_sens_cv.append(len(np.where(n_cv_df['N_FAPS_BAL'] < fap)[0])/len(n_cv_df['N_FAPS_BAL']))
	n_bal_sens_ceph.append(len(np.where(n_ceph_df['N_FAPS_BAL'] < fap)[0])/len(n_ceph_df['N_FAPS_BAL']))
	n_nn_sens_yso.append(len(np.where(n_yso_df['N_FAPS_NN'] < fap)[0])/len(n_yso_df['N_FAPS_NN']))
	n_nn_sens_eb.append(len(np.where(n_eb_df['N_FAPS_NN'] < fap)[0])/len(n_eb_df['N_FAPS_NN']))
	n_nn_sens_rr.append(len(np.where(n_rr_df['N_FAPS_NN'] < fap)[0])/len(n_rr_df['N_FAPS_NN']))
	n_nn_sens_cv.append(len(np.where(n_cv_df['N_FAPS_NN'] < fap)[0])/len(n_cv_df['N_FAPS_NN']))
	n_nn_sens_ceph.append(len(np.where(n_ceph_df['N_FAPS_NN'] < fap)[0])/len(n_ceph_df['N_FAPS_NN']))
	n_bal_spec_yso.append(len(np.where(n_yso_df['ap_N_FAPS_BAL'] > fap)[0])/len(n_yso_df['ap_N_FAPS_BAL']))
	n_bal_spec_eb.append(len(np.where(n_eb_df['ap_N_FAPS_BAL'] > fap)[0])/len(n_eb_df['ap_N_FAPS_BAL']))
	n_bal_spec_rr.append(len(np.where(n_rr_df['ap_N_FAPS_BAL'] > fap)[0])/len(n_rr_df['ap_N_FAPS_BAL']))
	n_bal_spec_cv.append(len(np.where(n_cv_df['ap_N_FAPS_BAL'] > fap)[0])/len(n_cv_df['ap_N_FAPS_BAL']))
	n_bal_spec_ceph.append(len(np.where(n_ceph_df['ap_N_FAPS_BAL'] > fap)[0])/len(n_ceph_df['ap_N_FAPS_BAL']))
	n_nn_spec_yso.append(len(np.where(n_yso_df['ap_N_FAPS_NN'] > fap)[0])/len(n_yso_df['ap_N_FAPS_NN']))
	n_nn_spec_eb.append(len(np.where(n_eb_df['ap_N_FAPS_NN'] > fap)[0])/len(n_eb_df['ap_N_FAPS_NN']))
	n_nn_spec_rr.append(len(np.where(n_rr_df['ap_N_FAPS_NN'] > fap)[0])/len(n_rr_df['ap_N_FAPS_NN']))
	n_nn_spec_cv.append(len(np.where(n_cv_df['ap_N_FAPS_NN'] > fap)[0])/len(n_cv_df['ap_N_FAPS_NN']))
	n_nn_spec_ceph.append(len(np.where(n_ceph_df['ap_N_FAPS_NN'] > fap)[0])/len(n_ceph_df['ap_N_FAPS_NN']))



split_point = 0.05
plt.clf()
gs=gridspec.GridSpec(1,2, hspace = 0, wspace = 0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharey=ax1)
ax1.plot(1-np.array(n_bal_spec_yso),n_bal_sens_yso, color = 'r')
ax1.plot(1-np.array(n_bal_spec_eb),n_bal_sens_eb, color = 'r')
ax1.plot(1-np.array(n_bal_spec_rr),n_bal_sens_rr, color = 'r')
ax1.plot(1-np.array(n_bal_spec_cv),n_bal_sens_cv, color = 'r')
ax1.plot(1-np.array(n_bal_spec_ceph),n_bal_sens_ceph, color = 'r')
ax1.plot(1-np.array(n_nn_spec_yso),n_nn_sens_yso, color = 'g')
ax1.plot(1-np.array(n_nn_spec_eb),n_nn_sens_eb, color = 'g')
ax1.plot(1-np.array(n_nn_spec_rr),n_nn_sens_rr, color = 'g')
ax1.plot(1-np.array(n_nn_spec_cv),n_nn_sens_cv, color = 'g')
ax1.plot(1-np.array(n_nn_spec_ceph),n_nn_sens_ceph, color = 'g')
ax1.set(xlim = [-0.0001,split_point],ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
ax2.yaxis.tick_right()
ax2.plot(1-np.array(n_bal_spec_yso),n_bal_sens_yso, color = 'r')
ax2.plot(1-np.array(n_bal_spec_eb),n_bal_sens_eb, color = 'r')
ax2.plot(1-np.array(n_bal_spec_rr),n_bal_sens_rr, color = 'r')
ax2.plot(1-np.array(n_bal_spec_cv),n_bal_sens_cv, color = 'r')
ax2.plot(1-np.array(n_bal_spec_ceph),n_bal_sens_ceph, color = 'r')
ax2.plot(1-np.array(n_nn_spec_yso),n_nn_sens_yso, color = 'g')
ax2.plot(1-np.array(n_nn_spec_eb),n_nn_sens_eb, color = 'g')
ax2.plot(1-np.array(n_nn_spec_rr),n_nn_sens_rr, color = 'g')
ax2.plot(1-np.array(n_nn_spec_cv),n_nn_sens_cv, color = 'g')
ax2.plot(1-np.array(n_nn_spec_ceph),n_nn_sens_ceph, color = 'g')
ax2.set(xlim = [split_point,1], xscale = 'linear')
plt.savefig('/home/njm/FAPS/N/AUC_split.jpg', bbox_inches = 'tight', dpi = 666)


plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(1-np.array(n_bal_spec_yso),n_bal_sens_yso, color = 'r')
ax1.plot(1-np.array(n_bal_spec_eb),n_bal_sens_eb, color = 'r')
ax1.plot(1-np.array(n_bal_spec_rr),n_bal_sens_rr, color = 'r')
ax1.plot(1-np.array(n_bal_spec_cv),n_bal_sens_cv, color = 'r')
ax1.plot(1-np.array(n_bal_spec_ceph),n_bal_sens_ceph, color = 'r')
ax1.plot(1-np.array(n_nn_spec_yso),n_nn_sens_yso, color = 'g')
ax1.plot(1-np.array(n_nn_spec_eb),n_nn_sens_eb, color = 'g')
ax1.plot(1-np.array(n_nn_spec_rr),n_nn_sens_rr, color = 'g')
ax1.plot(1-np.array(n_nn_spec_cv),n_nn_sens_cv, color = 'g')
ax1.plot(1-np.array(n_nn_spec_ceph),n_nn_sens_ceph, color = 'g')
ax1.set(ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
plt.savefig('/home/njm/FAPS/N/AUC.jpg', bbox_inches = 'tight', dpi = 666)


plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(1-np.array(n_bal_spec_yso),n_bal_sens_yso, color = 'r', label = 'Baluev')
ax1.plot(1-np.array(n_bal_spec_eb),n_bal_sens_eb, color = 'r')
ax1.plot(1-np.array(n_bal_spec_rr),n_bal_sens_rr, color = 'r')
ax1.plot(1-np.array(n_bal_spec_cv),n_bal_sens_cv, color = 'r')
ax1.plot(1-np.array(n_bal_spec_ceph),n_bal_sens_ceph, color = 'r')
ax1.plot(1-np.array(n_nn_spec_yso),n_nn_sens_yso, color = 'g', label = 'NN')
ax1.plot(1-np.array(n_nn_spec_eb),n_nn_sens_eb, color = 'g')
ax1.plot(1-np.array(n_nn_spec_rr),n_nn_sens_rr, color = 'g')
ax1.plot(1-np.array(n_nn_spec_cv),n_nn_sens_cv, color = 'g')
ax1.plot(1-np.array(n_nn_spec_ceph),n_nn_sens_ceph, color = 'g')
ax1.plot([0,0.196],[0.9,0.37],color='k', alpha = 0.5)
ax1.plot([0.2,0.625],[1,0.795],color='k', alpha = 0.5)
ax1.legend()
ax1.set(ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')

# this is an inset axes over the main axes
ax2 = fig.add_axes([.3, .4, .3, .3], facecolor='w', alpha = 0.3)
ax2.set(xlim = [-0.01,0.2], ylim=[0.9,1.01], xticks = [], yticks = [])
ax2.plot(1-np.array(n_bal_spec_yso),n_bal_sens_yso, color = 'r')
ax2.plot(1-np.array(n_bal_spec_eb),n_bal_sens_eb, color = 'r')
ax2.plot(1-np.array(n_bal_spec_rr),n_bal_sens_rr, color = 'r')
ax2.plot(1-np.array(n_bal_spec_cv),n_bal_sens_cv, color = 'r')
ax2.plot(1-np.array(n_bal_spec_ceph),n_bal_sens_ceph, color = 'r')
ax2.plot(1-np.array(n_nn_spec_yso),n_nn_sens_yso, color = 'g')
ax2.plot(1-np.array(n_nn_spec_eb),n_nn_sens_eb, color = 'g')
ax2.plot(1-np.array(n_nn_spec_rr),n_nn_sens_rr, color = 'g')
ax2.plot(1-np.array(n_nn_spec_cv),n_nn_sens_cv, color = 'g')
ax2.plot(1-np.array(n_nn_spec_ceph),n_nn_sens_ceph, color = 'g')

plt.savefig('/home/njm/FAPS/N/AUC_sub.jpg', bbox_inches = 'tight', dpi = 666)









cp Aperiodic_Variables/Figures/Light_Curve/*.jpg FAP_paper/TEST/AP/




TOOL.OUTPUT_redux_load(load_file_path = '/beegfs/car/njm/OUTPUT/vars/Redux_vars_filled.csv')






ps = glob.glob('/beegfs/car/njm/FAP_paper/TEST/P/test/*')
p_idx = []
random_samp = np.arange(0,len(ps))
random.shuffle(random_samp)
random_samp = random_samp#[0:10000]
for idx in random_samp:
    p = ps[idx].split('/')[-1].replace('__','_')
    p_idx.append(int(p.split('.')[0]))

FAPS_NN = [];FAPS_BAL = [];N = [];A_sn = [];name = []
for id_x in p_idx:
    idx = np.where(int(id_x) == TOOL.list_name)[0]
    TOOL.OUTPUT_redux_index_assign(idx)
    A_sn.append(TOOL.true_amplitude/TOOL.magerr_avg)
    FAPS_NN.append(TOOL.best_fap)
    FAPS_BAL.append(TOOL.ls_bal_fap)
    N.append(TOOL.ks_n_detections)
    name.append(TOOL.name)

real_df_p = pd.DataFrame({'Name':name,'FAPS_NN':FAPS_NN,'FAPS_BAL':FAPS_BAL,'N':N,'A_sn':A_sn})
aps = glob.glob('/beegfs/car/njm/FAP_paper/TEST/AP/test/*')
ap_idx = []
random_samp = np.arange(0,len(aps))
random.shuffle(random_samp)
random_samp = random_samp#[0:10000]
for idx in random_samp:
    ap = aps[idx].split('/')[-1].replace('__','_')
    ap_idx.append(int(ap.split('.')[0]))

FAPS_NN = [];FAPS_BAL = [];N = [];A_sn = [];name = []
for id_x in ap_idx:
    idx = np.where(int(id_x) == TOOL.list_name)[0]
    TOOL.OUTPUT_redux_index_assign(idx)
    A_sn.append(TOOL.true_amplitude/TOOL.magerr_avg)
    FAPS_NN.append(TOOL.best_fap)
    FAPS_BAL.append(TOOL.ls_bal_fap)
    N.append(TOOL.ks_n_detections)
    name.append(TOOL.name)

real_df_ap = pd.DataFrame({'Name':name,'FAPS_NN':FAPS_NN,'FAPS_BAL':FAPS_BAL,'N':N,'A_sn':A_sn})





bal_real_spec = []
bal_real_sens = []
nn_real_spec = []
nn_real_sens = []
idxs = []
idxs = list(real_df_p['FAPS_BAL']) + list(real_df_ap['FAPS_BAL']) + list(real_df_p['FAPS_NN']) + list(real_df_ap['FAPS_NN'])
idxs.sort()
for fap in tqdm(idxs):
        bal_real_sens.append(len(np.where(real_df_p['FAPS_BAL'] < fap)[0])/len(real_df_p['FAPS_BAL']))
        nn_real_sens.append(len(np.where(real_df_p['FAPS_NN'] < fap)[0])/len(real_df_p['FAPS_NN']))
        bal_real_spec.append(len(np.where(real_df_ap['FAPS_BAL'] > fap)[0])/len(real_df_ap['FAPS_BAL']))
        nn_real_spec.append(len(np.where(real_df_ap['FAPS_NN'] > fap)[0])/len(real_df_ap['FAPS_NN']))


a_nn_spec = (np.array(a_nn_spec_yso) + np.array(a_nn_spec_eb) + np.array(a_nn_spec_rr) + np.array(a_nn_spec_cv) + np.array(a_nn_spec_ceph))/5

n_nn_spec = (np.array(n_nn_spec_yso) + np.array(n_nn_spec_eb + np.array(n_nn_spec_rr) + np.array(n_nn_spec_cv) + np.array(n_nn_spec_ceph))/5

a_bal_spec = (np.array(a_bal_spec_yso) + np.array(a_bal_spec_eb) + np.array(a_bal_spec_rr) + np.array(a_bal_spec_cv) + np.array(a_bal_spec_ceph))/5
n_bal_spec = (np.array(n_bal_spec_yso) + np.array(n_bal_spec_eb) + np.array(n_bal_spec_rr) + np.array(n_bal_spec_cv) + np.array(n_bal_spec_ceph))/5

a_nn_sens = (np.array(a_nn_sens_yso) + np.array(a_nn_sens_eb) + np.array(a_nn_sens_rr) + np.array(a_nn_sens_cv) + np.array(a_nn_sens_ceph))/5
n_nn_sens = (np.array(n_nn_sens_yso) + np.array(n_nn_sens_eb) + np.array(n_nn_sens_rr) + np.array(n_nn_sens_cv) + np.array(n_nn_sens_ceph))/5

a_bal_sens = (np.array(a_bal_sens_yso) + np.array(a_bal_sens_eb) + np.array(a_bal_sens_rr) + np.array(a_bal_sens_cv) + np.array(a_bal_sens_ceph))/5
n_bal_sens = (np.array(n_bal_sens_yso) + np.array(n_bal_sens_eb) + np.array(n_bal_sens_rr) + np.array(n_bal_sens_cv) + np.array(n_bal_sens_ceph))/5


a_ceph_df.to_pickle('/home/njm/FAPS/a_ceph.csv')
n_ceph_df.to_pickle('/home/njm/FAPS/b_ceph.csv')

a_eb_df.to_pickle('/home/njm/FAPS/a_eb.csv')
n_eb_df.to_pickle('/home/njm/FAPS/n_eb.csv')

a_yso_df.to_pickle('/home/njm/FAPS/a_yso.csv')
n_yso_df.to_pickle('/home/njm/FAPS/n_yso.csv')

a_cv_df.to_pickle('/home/njm/FAPS/a_cv.csv')
n_cv_df.to_pickle('/home/njm/FAPS/n_cv.csv')

a_rr_df.to_pickle('/home/njm/FAPS/a_rr.csv')
n_rr_df.to_pickle('/home/njm/FAPS/n_rr.csv')

real_df_ap.to_pickle('/home/njm/FAPS/real.csv')




OUTPUT = [a_nn_spec, n_nn_spec, nn_real_spec, a_bal_spec, n_bal_spec, bal_real_spec, a_nn_sens, n_nn_sens, nn_real_sens, a_bal_sens, n_bal_sens, bal_real_sens]
with open('/home/njm/FAPS/ROC.csv', "a") as fp:
	wr = csv.writer(fp, dialect='excel')
	wr.writerow(OUTPUT)
	
a_nn_spec, a_bal_spec, a_nn_sens, a_bal_sens = np.genfromtxt('/home/njm/FAPS/ROC.csv', delimiter = ',', unpack = True, usecols = [0,3,6,9])
n_nn_spec, n_bal_spec, n_nn_sens, n_bal_sens = np.genfromtxt('/home/njm/FAPS/ROC.csv', delimiter = ',', unpack = True, usecols = [1,3,7,10])
nn_real_spec, bal_real_spec, nn_real_sens, bal_real_sens = np.genfromtxt('/home/njm/FAPS/ROC.csv', delimiter = ',', unpack = True, usecols = [2,4,8,11])
	
	
	

split_point = 0.05
plt.clf()
gs=gridspec.GridSpec(1,2, hspace = 0, wspace = 0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharey=ax1)
ax1.plot(1-np.array(bal_real_spec),bal_real_sens, color = 'r')
ax1.plot(1-np.array(nn_real_spec),nn_real_sens, color = 'g')
ax1.set(xlim = [-0.0001,split_point],ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
ax2.yaxis.tick_right()
ax2.plot(1-np.array(bal_real_spec),bal_real_sens, color = 'r')
ax2.plot(1-np.array(nn_real_spec),nn_real_sens, color = 'g')
ax2.set(xlim = [split_point,1], xscale = 'linear')
plt.savefig('/home/njm/FAPS/Real/AUC_split.jpg', bbox_inches = 'tight', dpi = 666)


plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(1-np.array(bal_real_spec),bal_real_sens, color = 'r')
ax1.plot(1-np.array(nn_real_spec),nn_real_sens, color = 'g')
ax1.set(ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
plt.savefig('/home/njm/FAPS/Real/AUC.jpg', bbox_inches = 'tight', dpi = 666)




plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(1-np.array(bal_real_spec),bal_real_sens, color = 'r', label = 'Baluev')
ax1.plot(1-np.array(nn_real_spec),nn_real_sens, color = 'g', label = 'NN')
ax1.plot([0,0.196],[0.9,0.37],color='k', alpha = 0.5)
ax1.plot([0.2,0.625],[1,0.795],color='k', alpha = 0.5)
ax1.legend()
ax1.set(ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
ax2 = fig.add_axes([.3, .4, .3, .3], facecolor='w', alpha = 0.3)
ax2.set(xlim = [-0.01,0.2], ylim=[0.9,1.01], xticks = [], yticks = [])
ax2.plot(1-np.array(bal_real_spec),bal_real_sens, color = 'r')
ax2.plot(1-np.array(nn_real_spec),nn_real_sens, color = 'g')
plt.savefig('/home/njm/FAPS/Real/AUC_sub.jpg', bbox_inches = 'tight', dpi = 666)







nnn = list(np.argsort(n_nn_spec))
nbal = list(np.argsort(n_bal_spec))
ann = list(np.argsort(a_nn_spec))
abal = list(np.argsort(a_bal_spec))


np.convolve(np.array(n_bal_spec)[nbal], np.ones(N)/N, mode='valid')

N = 50

plt.clf()
fig, ax1 = plt.subplots()
line_with = 0.5
ax1.plot(1-np.array(n_bal_spec)[nbal],np.array(n_bal_sens)[nbal], color = 'navy', ls = '--', lw = line_with)#, label = 'N Baluev')
ax1.plot(1-np.array(n_nn_spec)[nnn],np.array(n_nn_sens)[nnn], color = 'forestgreen', ls = '--', lw = line_with)#, label = 'N NN')
ax1.plot(1-np.array(a_bal_spec)[abal],np.array(a_bal_sens)[abal], color = 'navy', ls = ':', lw = line_with)#, label = 'A Baluev')
ax1.plot(1-np.array(a_nn_spec)[ann],np.array(a_nn_sens)[ann], color = 'forestgreen', ls = ':', lw = line_with)#, label = 'A NN')
ax1.plot(1-np.array(nn_real_spec),nn_real_sens, color = 'forestgreen', label = 'NN', ls = '-', lw = line_with)
ax1.plot(1-np.array(bal_real_spec),bal_real_sens, color = 'navy', label = 'Baluev', ls = '-', lw = line_with)
ax1.plot([0,0.196],[0.9,0.37],color='k', alpha = 0.5)
ax1.plot([0.2,0.625],[1,0.795],color='k', alpha = 0.5)
ax1.legend()
ax1.set(ylabel = r'$Sensitivity$', xlabel = r'$1 - Specificity$', xscale = 'linear')
ax2 = fig.add_axes([.3, .4, .3, .3], facecolor='w', alpha = 0.3)
ax2.set(xlim = [-0.01,0.2], ylim=[0.9,1.01], xticks = [], yticks = [])

ax2.plot(1-np.array(n_bal_spec)[nbal],np.array(n_bal_sens)[nbal], color = 'navy', ls = '--', lw = line_with)
ax2.plot(1-np.array(n_nn_spec)[nnn],np.array(n_nn_sens)[nnn], color = 'forestgreen', ls = '--', lw = line_with)
ax2.plot(1-np.array(a_bal_spec)[abal],np.array(a_bal_sens)[abal], color = 'navy', ls = ':', lw = line_with)
ax2.plot(1-np.array(a_nn_spec)[ann],np.array(a_nn_sens)[ann], color = 'forestgreen', ls = ':', lw = line_with)
ax2.plot(1-np.array(nn_real_spec),nn_real_sens, color = 'forestgreen', ls = '-', lw = line_with)
ax2.plot(1-np.array(bal_real_spec),bal_real_sens, color = 'navy', ls = '-', lw = line_with)
plt.savefig('/home/njm/FAPS/AUC_sub.jpg', bbox_inches = 'tight', dpi = 666)




















































periods = ['P_FAPS_NN','P_FAPS_BAL','ap_P_FAPS_BAL','P_i','NP_i','P_Periods','P_LS_Periods','P_err','P_pdiffp','ap_P_FAPS_NN','P_tr']




plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(p_ceph_df['P_FAPS_NN'], p_ceph_df['P_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax1.plot(p_eb_df['P_FAPS_NN'], p_eb_df['P_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax1.plot(p_rr_df['P_FAPS_NN'], p_rr_df['P_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(p_yso_df['P_FAPS_NN'], p_yso_df['P_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax1.plot(p_cv_df['P_FAPS_NN'], p_cv_df['P_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
ax1.set(ylabel='Baluev FAP', xscale = 'linear', yscale = 'log', ylim = [10**-100,1], yticks = bal_ticks)
ax1.xaxis.tick_top()
ax2.plot(p_ceph_df['ap_P_FAPS_NN'], p_ceph_df['ap_P_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax2.plot(p_eb_df['ap_P_FAPS_NN'], p_eb_df['ap_P_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax2.plot(p_rr_df['ap_P_FAPS_NN'], p_rr_df['ap_P_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax2.plot(p_yso_df['ap_P_FAPS_NN'], p_yso_df['ap_P_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax2.plot(p_cv_df['ap_P_FAPS_NN'], p_cv_df['ap_P_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
ax2.set(xlabel = 'NN FAP', ylabel='Baluev FAP', xscale = 'linear', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/P/Each_P_fapfap.jpg', bbox_inches = 'tight')


plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(p_ceph_df['NP_i'], p_ceph_df['P_FAPS_BAL'],'rx', alpha = 0.3, ms = 2)
ax1.plot(p_eb_df['NP_i'], p_eb_df['P_FAPS_BAL'],'g+', alpha = 0.3, ms = 2)
ax1.plot(p_rr_df['NP_i'], p_rr_df['P_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax1.plot(p_yso_df['NP_i'], p_yso_df['P_FAPS_BAL'],'ys', alpha = 0.3, ms = 2)
ax1.plot(p_cv_df['NP_i'], p_cv_df['P_FAPS_BAL'],'mp', alpha = 0.3, ms = 2)
ax1.set(ylabel='Baluev FAP', yscale = 'log', xscale = 'log', ylim = [10**-100,1], yticks = bal_ticks)
ax1.xaxis.tick_top()
ax2.plot(p_ceph_df['NP_i'], p_ceph_df['P_FAPS_NN'],'rx', alpha = 0.3, ms = 2, label = 'Cepheid')
ax2.plot(p_eb_df['NP_i'], p_eb_df['P_FAPS_NN'],'g+', alpha = 0.3, ms = 2, label = 'EB')
ax2.plot(p_rr_df['NP_i'], p_rr_df['P_FAPS_NN'],'bD', alpha = 0.3, ms = 2, label = 'W UMa')
ax2.plot(p_yso_df['NP_i'], p_yso_df['P_FAPS_NN'],'ys', alpha = 0.3, ms = 2, label = 'YSO')
ax2.plot(p_cv_df['NP_i'], p_cv_df['P_FAPS_NN'],'mp', alpha = 0.3, ms = 2, label = 'CV')
ax2.set(xlabel = r'$N\lambda$', ylabel='NN FAP', xscale = 'log')
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/P/Each_P_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(p_ceph_df['NP_i'], p_ceph_df['P_FAPS_BAL'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(p_eb_df['NP_i'], p_eb_df['P_FAPS_BAL'],'b+', alpha = 0.3, ms = 2)
ax2.plot(p_rr_df['NP_i'], p_rr_df['P_FAPS_BAL'],'bD', alpha = 0.3, ms = 2)
ax2.plot(p_yso_df['NP_i'], p_yso_df['P_FAPS_BAL'],'bs', alpha = 0.3, ms = 2)
ax2.plot(p_cv_df['NP_i'], p_cv_df['P_FAPS_BAL'],'bp', alpha = 0.3, ms = 2)
ax2.plot(p_ceph_df['NP_i'], p_ceph_df['ap_P_FAPS_BAL'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(p_eb_df['NP_i'], p_eb_df['ap_P_FAPS_BAL'],'r+', alpha = 0.3, ms = 2)
ax2.plot(p_rr_df['NP_i'], p_rr_df['ap_P_FAPS_BAL'],'rD', alpha = 0.3, ms = 2)
ax2.plot(p_yso_df['NP_i'], p_yso_df['ap_P_FAPS_BAL'],'rs', alpha = 0.3, ms = 2)
ax2.plot(p_cv_df['NP_i'], p_cv_df['ap_P_FAPS_BAL'],'rp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$N\lambda$', ylabel='Baluev FAP', yscale = 'log', xscale = 'log', ylim = [10**-100,1])
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/P/Each_ap_P_BAL_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(p_ceph_df['NP_i'], p_ceph_df['P_FAPS_NN'],'bx', alpha = 0.3, ms = 2, label = 'Periodic')
ax2.plot(p_eb_df['NP_i'], p_eb_df['P_FAPS_NN'],'b+', alpha = 0.3, ms = 2)
ax2.plot(p_rr_df['NP_i'], p_rr_df['P_FAPS_NN'],'bD', alpha = 0.3, ms = 2)
ax2.plot(p_yso_df['NP_i'], p_yso_df['P_FAPS_NN'],'bs', alpha = 0.3, ms = 2)
ax2.plot(p_cv_df['NP_i'], p_cv_df['P_FAPS_NN'],'bp', alpha = 0.3, ms = 2)
ax2.plot(p_ceph_df['NP_i'], p_ceph_df['ap_P_FAPS_NN'],'rx', alpha = 0.3, ms = 2, label = 'Aperiodic')
ax2.plot(p_eb_df['NP_i'], p_eb_df['ap_P_FAPS_NN'],'r+', alpha = 0.3, ms = 2)
ax2.plot(p_rr_df['NP_i'], p_rr_df['ap_P_FAPS_NN'],'rD', alpha = 0.3, ms = 2)
ax2.plot(p_yso_df['NP_i'], p_yso_df['ap_P_FAPS_NN'],'rs', alpha = 0.3, ms = 2)
ax2.plot(p_cv_df['NP_i'], p_cv_df['ap_P_FAPS_NN'],'rp', alpha = 0.3, ms = 2)
ax2.set(xlabel = r'$N\lambda$', ylabel='NN FAP', xscale = 'log')
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/P/Each_ap_P_NP_faps.jpg', bbox_inches = 'tight')


plt.clf()
gs=gridspec.GridSpec(2,1,hspace=0)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1],sharex=ax1)
ax1.plot(abs(np.array(p_ceph_df['P_pdiffp'])), p_ceph_df['NP_i'],'rx', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(p_eb_df['P_pdiffp'])), p_eb_df['NP_i'],'g+', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(p_rr_df['P_pdiffp'])), p_rr_df['NP_i'],'bD', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(p_yso_df['P_pdiffp'])), p_yso_df['NP_i'],'ys', alpha = 0.5, ms = 2)
ax1.plot(abs(np.array(p_cv_df['P_pdiffp'])), p_cv_df['NP_i'],'mp', alpha = 0.5, ms = 2)
ax1.set(ylabel=r'$N\lambda$', yscale = 'log')
ax1.xaxis.tick_top()
ax2.plot(abs(np.array(p_ceph_df['P_pdiffp'])), p_ceph_df['P_FAPS_BAL'],'rx', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(p_eb_df['P_pdiffp'])), p_eb_df['P_FAPS_BAL'],'g+', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(p_rr_df['P_pdiffp'])), p_rr_df['P_FAPS_BAL'],'bD', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(p_yso_df['P_pdiffp'])), p_yso_df['P_FAPS_BAL'],'ys', alpha = 0.5, ms = 2)
ax2.plot(abs(np.array(p_cv_df['P_pdiffp'])), p_cv_df['P_FAPS_BAL'],'mp', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'Period Percentage Difference', ylabel='BAL FAP', yscale = 'log', ylim = [10**-100,1], yticks = bal_ticks)
fig.tight_layout()
plt.savefig('/home/njm/FAPS/P/Each_P_BAL_Periods.jpg', bbox_inches = 'tight')




