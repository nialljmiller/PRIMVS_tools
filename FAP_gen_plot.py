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
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from astropy.timeseries import LombScargle
from random import shuffle
from matplotlib import gridspec
import warnings
warnings.filterwarnings("ignore")




P_FAPS_NN = []
P_FAPS_BAL = []
ap_P_FAPS_BAL = []
P_i = []
NP_i = []
P_Periods = []
P_LS_Periods = []
P_err = []
P_pdiffp = []
ap_P_FAPS_NN = []
P_tr = []

for i in tqdm(np.arange(1, 1000, 0.96)):
	for j in range(boots):
		mag, magerr, time, cat_type, period = TOOL.synthesize(N = 200, period = i, cat_type = star_type,other_pert = 0, contamination_flag = 0)
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
		P_FAPS_BAL.append(ls.false_alarm_probability(ls_y_y_0))
		ap_P_FAPS_BAL.append(ap_ls.false_alarm_probability(ap_ls_y_y_0))
		P_Periods.append(period)
		P_LS_Periods.append(1/ls_x_y_0)
		P_FAPS_NN.append(NN_True_FAP)
		P_i.append(i)
		#P_Ni.append(20*j)
		P_err.append(np.median(magerr))
		P_pdiffp.append(((1/ls_x_y_0)-period)/period)
		ap_P_FAPS_NN.append(ap_NN_True_FAP)
		NP_i.append(time_range/period)
		P_tr.append(time_range)


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(NP_i, P_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel='Baluev FAP', ylim = [min(P_FAPS_BAL)*0.95,max(P_FAPS_BAL)*1.05], yscale = 'log', xscale = 'log')
ax1.xaxis.tick_top()
ax2.plot(NP_i, P_FAPS_NN,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = r'$N\lambda$', ylabel='NN FAP', ylim = [min(P_FAPS_NN)*0.95,max(P_FAPS_NN)*1.05], xscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_P_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(NP_i, P_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(NP_i, ap_P_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$N\lambda$', ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_P_BAL_faps.jpg', bbox_inches = 'tight')



plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(NP_i, P_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax1.plot(NP_i, ap_P_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax1.set(ylabel='Baluev FAP', ylim = [0.0000001,1], xscale = 'log', yscale = 'log')
ax2.plot(NP_i, P_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(NP_i, ap_P_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$N\lambda$', ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_P_BAL2_faps.jpg', bbox_inches = 'tight')



plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(NP_i, P_FAPS_NN,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(NP_i, ap_P_FAPS_NN,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$N\lambda$', ylabel='NN FAP', xscale = 'log')
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_P_NN_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(abs(np.array(P_pdiffp)), NP_i,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel=r'$N\lambda$', yscale = 'log')
#ax1.legend()
ax1.xaxis.tick_top()
ax2.plot(abs(np.array(P_pdiffp)), P_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'Period Percentage Difference', ylabel='BAL FAP', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_P_BAL_Periods.jpg', bbox_inches = 'tight')


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(P_FAPS_NN, P_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
ax1.xaxis.tick_top()
ax2.plot(ap_P_FAPS_NN, ap_P_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'NN FAP', ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_P_fapfap.jpg', bbox_inches = 'tight')












N_FAPS_NN = []
N_FAPS_BAL = []
N_i = []
NN_i = []
N_Periods = []
N_LS_Periods = []
N_err = []
N_pdiffp = []
N_tr = []
ap_N_FAPS_BAL = []
ap_N_FAPS_NN = []


for i in tqdm(np.arange(5, 1000, 2)):
	for j in range(boots):
		mag, magerr, time, cat_type, period = TOOL.synthesize(N = i, cat_type = star_type,other_pert = 0, contamination_flag = 0)
		NN_True_FAP = TOOL.fap_inference(period, mag, time, TOOL.knn, TOOL.model)
		time_range = max(time) - min(time)
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
		N_FAPS_BAL.append(ls.false_alarm_probability(ls_y_y_0))
		ap_N_FAPS_BAL.append(ap_ls.false_alarm_probability(ap_ls_y_y_0))
		N_Periods.append(period)
		N_LS_Periods.append(1/ls_x_y_0)
		N_FAPS_NN.append(NN_True_FAP)
		N_i.append(i)
		N_err.append(np.median(magerr))
		N_pdiffp.append(((1/ls_x_y_0)-period)/period)
		ap_N_FAPS_NN.append(ap_NN_True_FAP)
		NN_i.append(time_range/period)
		N_tr.append(time_range)


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(N_i, N_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel='Baluev FAP', yscale = 'log', xscale = 'log')
ax1.xaxis.tick_top()
ax2.plot(N_i, N_FAPS_NN,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = r'$N$', ylabel='NN FAP', xscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_N_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(N_i, N_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(N_i, ap_N_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$N$', ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_N_BAL_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(N_i, N_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax1.plot(N_i, ap_N_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax1.set(ylabel='Baluev FAP', ylim = [0.0000001,1], xscale = 'log', yscale = 'log')
ax2.plot(N_i, N_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(N_i, ap_N_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$N$', ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_N_BAL2_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(N_i, N_FAPS_NN,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(N_i, ap_N_FAPS_NN,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$N$', ylabel='NN FAP', xscale = 'log')
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_N_NN_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(abs(np.array(N_pdiffp)), N_i,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel=r'$N$', yscale = 'log')
#ax1.legend()
ax1.xaxis.tick_top()
ax2.plot(abs(np.array(N_pdiffp)), N_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'Period Percentage Difference', ylabel='BAL FAP', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_N_BAL_Periods.jpg', bbox_inches = 'tight')


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(N_FAPS_NN, N_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
ax1.xaxis.tick_top()
ax2.plot(ap_N_FAPS_NN, ap_N_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'NN FAP', ylabel='Baluev FAP', xscale = 'log', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_N_fapfap.jpg', bbox_inches = 'tight')











A_FAPS_NN = []
A_FAPS_BAL = []
ap_A_FAPS_BAL = []
A_i = []
NA_i = []
A_Periods = []
A_LS_Periods = []
A_err = []
A_pdiffp = []
ap_A_FAPS_NN = []
A_tr = []
A_sn=[]

for i in tqdm(np.arange(0.0001, 0.005, 0.0001)):
	for j in range(boots):
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
		A_FAPS_BAL.append(ls.false_alarm_probability(ls_y_y_0))
		ap_A_FAPS_BAL.append(ap_ls.false_alarm_probability(ap_ls_y_y_0))
		A_Periods.append(period)
		A_LS_Periods.append(1/ls_x_y_0)
		A_FAPS_NN.append(NN_True_FAP)
		A_i.append(i)
		A_err.append(np.median(magerr))
		A_pdiffp.append(((1/ls_x_y_0)-period)/period)
		ap_A_FAPS_NN.append(ap_NN_True_FAP)
		NA_i.append(time_range/period)
		A_tr.append(time_range)
		amp = max(mag) - min(mag)
		A_sn.append(amp/np.median(magerr))

plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(A_sn, A_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel='Baluev FAP', ylim = [min(A_FAPS_BAL)*0.95,max(A_FAPS_BAL)*1.05], yscale = 'log', xscale = 'linear')
ax1.xaxis.tick_top()
ax2.plot(A_sn, A_FAPS_NN,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = r'$A/\bar{\sigma}$', ylabel='NN FAP', ylim = [min(A_FAPS_NN)*0.95,max(A_FAPS_NN)*1.05], xscale = 'linear')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_A_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(A_sn, A_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(A_sn, ap_A_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$A/\bar{\sigma}$', ylabel='Baluev FAP', yscale = 'log')
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_A_BAL_faps.jpg', bbox_inches = 'tight')



plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(A_sn, A_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax1.plot(A_sn, ap_A_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax1.set(ylabel='Baluev FAP', ylim = [0.0000001,1], yscale = 'log')
ax2.plot(A_sn, A_FAPS_BAL,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(A_sn, ap_A_FAPS_BAL,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$A/\bar{\sigma}$', ylabel='Baluev FAP', yscale = 'log')
plt.legend()
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_A_BAL2_faps.jpg', bbox_inches = 'tight')




plt.clf()
fig, ax2 = plt.subplots()
ax2.plot(A_sn, A_FAPS_NN,'bx', alpha = 0.5, label = 'Periodic', ms = 2)
ax2.plot(A_sn, ap_A_FAPS_NN,'r+', alpha = 0.5, label = 'Aperiodic', ms = 2)
ax2.set(xlabel = r'$A/\bar{\sigma}$', ylabel='NN FAP')
fig.tight_layout()
plt.legend()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_ap_A_NN_faps.jpg', bbox_inches = 'tight')


plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(abs(np.array(A_pdiffp)), A_sn,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel=r'$A/\bar{\sigma}$', yscale = 'log')
#ax1.legend()
ax1.xaxis.tick_top()
ax2.plot(abs(np.array(A_pdiffp)), A_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'Period Percentage Difference', ylabel='BAL FAP', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_A_BAL_Periods.jpg', bbox_inches = 'tight')




plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(A_FAPS_NN, A_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax1.set(ylabel='Baluev FAP', xscale = 'linear', yscale = 'log')
ax1.xaxis.tick_top()
ax2.plot(ap_A_FAPS_NN, ap_A_FAPS_BAL,'kx', alpha = 0.5, ms = 2)
ax2.set(xlabel = 'NN FAP', ylabel='Baluev FAP', xscale = 'linear', yscale = 'log')
fig.tight_layout()
plt.savefig('/home/njm/FAPS/'+star_type+'/'+star_type+'_A_fapfap.jpg', bbox_inches = 'tight')




columns = ['P_FAPS_NN','P_FAPS_BAL','ap_P_FAPS_BAL','P_i','NP_i','P_Periods','P_LS_Periods','P_err','P_pdiffp','ap_P_FAPS_NN','P_tr',
'N_FAPS_NN','N_FAPS_BAL','N_i','NN_i','N_Periods','N_LS_Periods','N_err','N_pdiffp','N_tr','ap_N_FAPS_BAL','ap_N_FAPS_NN',
'A_FAPS_NN','A_FAPS_BAL','ap_A_FAPS_BAL','A_i','NA_i','A_Periods','A_LS_Periods','A_err','A_pdiffp','ap_A_FAPS_NN','A_tr','A_sn']
faps_df = pd.DataFrame(list(zip(P_FAPS_NN ,P_FAPS_BAL ,ap_P_FAPS_BAL ,P_i ,NP_i ,P_Periods ,P_LS_Periods ,P_err ,P_pdiffp ,ap_P_FAPS_NN ,P_tr ,N_FAPS_NN ,N_FAPS_BAL ,N_i ,NN_i ,N_Periods ,N_LS_Periods ,N_err ,N_pdiffp ,N_tr ,ap_N_FAPS_BAL ,ap_N_FAPS_NN ,A_FAPS_NN ,A_FAPS_BAL ,ap_A_FAPS_BAL ,A_i ,NA_i ,A_Periods ,A_LS_Periods ,A_err ,A_pdiffp ,ap_A_FAPS_NN ,A_tr ,A_sn,)), columns = columns)
faps_df.to_csv('/home/njm/FAPS/'+star_type+'/'+star_type+'_FAPDATA.csv', index=False)  




