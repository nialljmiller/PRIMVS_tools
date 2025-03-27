import os
from astropy.io import fits
import numpy as np
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
import matplotlib
from matplotlib.gridspec import GridSpec
import NN_FAP
from astropy.timeseries import LombScargle


def get_model(model_path = '/beegfs/car/njm/models/final_12l_dp_all/'):
    #model_path = '/beegfs/car/njm/models/final_better/'
    print("Opening model from here :", model_path)
    json_file = open(model_path+'_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"_model.h5")
    history=np.load(model_path+'_model_history.npy',allow_pickle='TRUE').item()
    model = loaded_model
    N = 200
    knn_N = int(N / 20)
    knn = neighbors.KNeighborsRegressor(knn_N, weights='distance')
    return knn, model




dpi = 666  # 200-300 as per guidelines
maxpix = 3000  # max pixels of plot
width = maxpix / dpi  # max allowed with
matplotlib.rcParams.update({'axes.labelsize': 'x-large', 'axes.titlesize': 'xx-large',  # the size of labels and title
                 'xtick.labelsize': 'xx-large', 'ytick.labelsize': 'x-large',  # the size of the axes ticks
                 'legend.fontsize': 'large', 'legend.frameon': False,  # legend font size, no frame
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
                 'text.usetex': True, 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'})  # process text with LaTeX instead of matplotlib math mode




def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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






knn, model = get_model()
N = 200

# Define the directory path
directory = '/home/njm/externalc/'
periodics = []
ls_periodics = []
folders = os.listdir(directory)

import random

folders = os.listdir(directory)






for sampy in [0]:#[0,1,2,3]:
    random.shuffle(folders)

    # Iterate through each folder
    for folder_name in folders:
        print(folder_name)
        folder_path = os.path.join(directory, folder_name)
        integer_part = folder_name[:-3]
        decimal_part = folder_name[-3:]
        period = float(integer_part + '.' + decimal_part)
        files = os.listdir(folder_path)
        data = []
        ls_data = []
        for i in [0,1]:
            filename = files[i]
            file_path = os.path.join(folder_path, filename)
            time, mag, gabagooo = np.genfromtxt(file_path, delimiter = ' ', unpack = True)
            fap = NN_FAP.inference(period, mag, time, knn, model)    
            mag = norm_data(mag)



            # Calculate the number of elements to shuffle (30%)
            num_elements_to_shuffle = int(0.7 * len(mag))

            # Get indices of elements to shuffle
            indices_to_shuffle = np.random.choice(len(mag), size=num_elements_to_shuffle, replace=False)

            # Shuffle selected elements
            subset_to_shuffle = np.array(mag)[indices_to_shuffle]
            np.random.shuffle(subset_to_shuffle)

            # Update original list with shuffled elements
            mag = np.array(mag)
            #mag[indices_to_shuffle] = subset_to_shuffle



            phase = phaser(time, period)

            ls = LombScargle(time, mag)
            ls_freq, ls_power = ls.autopower()
            max_id = np.argmax(ls_power)
            ls_y_y_0 = ls_power[max_id]
            ls_period = 1/ls_freq[max_id]
            FAP_BAL = ls.false_alarm_probability(ls_y_y_0)
            ls_phase = phaser(time, ls_period)
            
            data.append([NN_FAP.gen_chan(mag, phase, knn, N),fap])
            ls_data.append([NN_FAP.gen_chan(mag, ls_phase, knn, N),FAP_BAL])
            print(period, ls_period, fap, FAP_BAL)
        periodics.append(data)
        ls_periodics.append(ls_data)





    # Create the figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # Hide axis labels, ticks, and markers for each subplot
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)
        ax.invert_yaxis()  # Flip the y-axis        
    for i in range(2):
        for j in range(2):
            axs[i, j].scatter(periodics[(2*i)+j][1][0][3],periodics[(2*i)+j][1][0][0], s=10, c='#6500bf')  # Sample scatter plot with a single red point
            axs[i, j].scatter(periodics[(2*i)+j][1][0][3]+1,periodics[(2*i)+j][1][0][0], s=10, c='#6500bf')  # Sample scatter plot with a single red point
            axs[i, j].scatter(periodics[(2*i)+j][0][0][3],periodics[(2*i)+j][0][0][0], s=10, c='#14bf00')  # Sample scatter plot with a single red point
            axs[i, j].scatter(periodics[(2*i)+j][0][0][3]+1,periodics[(2*i)+j][0][0][0], s=10, c='#14bf00')  # Sample scatter plot with a single red point
            axs[i, j].set_title('', fontsize=20, ha='center', va='center')
            axs[i, j].text(0.4, -0.1, 'FAP = ' +str(round(periodics[(2*i)+j][1][1],3)), fontsize=30, color='#6500bf', ha='center', va='center')
            axs[i, j].text(1.3, -0.1, r'$\&$' + ' ' + str(round(periodics[(2*i)+j][0][1], 3)), fontsize=30, color='#14bf00', ha='center', va='center')


    # Adjust layout to avoid overlapping titles
    plt.tight_layout()
    # Show the plot
    plt.savefig('/home/njm/grids/oogle.pdf')







