import os
from astropy.io import fits
import numpy as np
import Virac as Virac
import glob
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import TSA_Stats as TSA_Stats
#import box_finder as boxy
from scipy.signal import savgol_filter
from tqdm import tqdm
import csv
import sys
#This stuff is for recalculating FAP
from tensorflow.keras.models import model_from_json
from sklearn import neighbors
import gc
import pandas as pd



def phaser(time, period):
    # this is to mitigate against div 0
    if period == 0:
        period = 1 
    phase = np.array(time) * 0.0
    for i in range(0, len(time)):
         phase[i] = (time[i])/period - np.floor((time[i])/period)
         if (phase[i] >= 1):
           phase[i] = phase[i]-1.
         if (phase[i] <= 0):
           phase[i] = phase[i]+1.
    return phase





def plot_lightcurves(file_path, output_fp,select):
    df = pd.read_csv(file_path)
    names = df['sourceid'].values[select]
    periods = df['true_period'].values[select]
    n = len(names)
    ncols = 3  # Define the number of columns in your grid
    nrows = n // ncols + (n % ncols > 0)  # Calculate the number of rows needed

    


    plt.figure(figsize=(15, 5 * nrows))  # Adjust the figure size as needed
    print(names, periods)
    for i, name in enumerate(names, start=1):
        period = periods[i-1]
        lightcurve = Virac.run_sourceid(int(name))
        filters = (lightcurve['filter'].astype(str) == 'Ks')
        mag_gt_0 = (lightcurve['hfad_mag'].astype(float) > 0)
        emag_gt_0 = (lightcurve['hfad_emag'].astype(float) > 0)
        ast_res_chisq_lt_20 = (lightcurve['ast_res_chisq'].astype(float) < 20)
        chi_lt_10 = (lightcurve['chi'].astype(float) < 10)
        filtered_indices = np.where(filters & mag_gt_0 & emag_gt_0 & ast_res_chisq_lt_20 & chi_lt_10)[0]
        lightcurve = lightcurve[filtered_indices]
        mag, magerr, time, chi, astchi = lightcurve['hfad_mag'], lightcurve['hfad_emag'], lightcurve['mjdobs'], lightcurve['chi'], lightcurve['ast_res_chisq']
        sigma = np.std(magerr)
        filtered_indices = np.where(magerr <= 4 * sigma)[0]
        mag, magerr, time, chi, astchi = mag[filtered_indices], magerr[filtered_indices], time[filtered_indices], chi[filtered_indices], astchi[filtered_indices]
        phase = phaser(time, period)

        plt.subplot(nrows, ncols, i)
        plt.scatter(phase, mag, c=time, s=20, alpha=1, cmap='viridis')  # Color-code by time
        plt.scatter(phase+1, mag, c=time, s=20, alpha=1, cmap='viridis')  # Color-code by time
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.gca().invert_yaxis()  # Invert y-axis for astronomical convention

    plt.tight_layout()
    plt.savefig(f'{output_fp}.jpg', dpi=300)
    plt.close()

#files = ['Cepheid.csv', 'Delta Scuti.csv', 'Eclipsing Binary.csv', 'Ellipsoidal.csv', 'Long-period Variable.csv', 'RR Lyrae.csv', 'Solar-like.csv']
#files = [['Cepheid',[1,3,5,8]], ['Delta Scuti',[1,5,8,9]], ['Eclipsing Binary',[3,5,6,7]], ['Ellipsoidal',[1,4,6,8]], ['Long-period Variable',[4,5,7,8]], ['RR Lyrae',[0,3,5,8]], ['Solar-like',[0,1,3,6]]]

parent_fp = '/beegfs/car/njm/PRIMVS/dtree/gaia/'

for ffile in range(10):
    #ffile = 'extreme_group_'+str(ffile)
    file_path = f'{parent_fp}{ffile}.csv'
    plot_lightcurves(file_path, file_path[:-4], [0,1,2,3,4,5,6,7,8,9])#,10,11])

