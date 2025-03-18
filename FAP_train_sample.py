import os
from astropy.io import fits
import numpy as np
import Virac as Virac
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


meta_col_names = ['sourceid','ra','ra_error','dec','dec_error','l','b','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
        'chisq','uwe','ks_n_detections','ks_n_observations','ks_n_ambiguous','ks_n_chilt5','ks_med_mag','ks_mean_mag','ks_ivw_mean_mag',
        'ks_chilt5_ivw_mean_mag','ks_std_mag','ks_mad_mag','ks_ivw_err_mag','ks_chilt5_ivw_err_mag','z_n_observations','z_med_mag',
        'z_mean_mag','z_ivw_mean_mag','z_chilt5_ivw_mean_mag','z_std_mag','z_mad_mag','z_ivw_err_mag','z_chilt5_ivw_err_mag','y_n_detections',
        'y_n_observations','y_med_mag','y_mean_mag','y_ivw_mean_mag','y_chilt5_ivw_mean_mag','y_std_mag','y_mad_mag','y_ivw_err_mag',
        'y_chilt5_ivw_err_mag','j_n_detections','j_n_observations','j_med_mag','j_mean_mag','j_ivw_mean_mag','j_chilt5_ivw_mean_mag',
        'j_std_mag','j_mad_mag','j_ivw_err_mag','j_chilt5_ivw_err_mag','h_n_detections','h_n_observations','h_med_mag','h_mean_mag',
        'h_ivw_mean_mag','h_chilt5_ivw_mean_mag','h_std_mag','h_mad_mag','h_ivw_err_mag','h_chilt5_ivw_err_mag',
             'mag_n','mag_avg','magerr_avg','Cody_M','stet_k','eta','eta_e','med_BRP',
        'range_cum_sum','max_slope','MAD','mean_var','percent_amp','true_amplitude','roms','p_to_p_var',
        'lag_auto','AD','std_nxs','weight_mean','weight_std','weight_skew','weight_kurt','mean','std','skew',
        'kurt','time_range','true_period','true_class','best_fap','best_method','trans_flag','ls_p','ls_y_y_0',
        'ls_peak_width_0','ls_period1','ls_y_y_1','ls_peak_width_1','ls_period2','ls_y_y_2','ls_peak_width_2',
        'ls_q001','ls_q01','ls_q1','ls_q25','ls_q50','ls_q75','ls_q99','ls_q999','ls_q9999','ls_fap','ls_bal_fap',
        'Cody_Q_ls','pdm_p','pdm_y_y_0','pdm_peak_width_0','pdm_period1','pdm_y_y_1','pdm_peak_width_1','pdm_period2',
        'pdm_y_y_2','pdm_peak_width_2','pdm_q001','pdm_q01','pdm_q1','pdm_q25','pdm_q50','pdm_q75','pdm_q99',
        'pdm_q999','pdm_q9999','pdm_fap','Cody_Q_pdm','ce_p','ce_y_y_0','ce_peak_width_0','ce_period1','ce_y_y_1',
        'ce_peak_width_1','ce_period2','ce_y_y_2','ce_peak_width_2','ce_q001','ce_q01','ce_q1','ce_q25','ce_q50',
        'ce_q75','ce_q99','ce_q999','ce_q9999','ce_fap','Cody_Q_ce','gp_lnlike','gp_b','gp_c','gp_p','gp_fap','Cody_Q_gp']



def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def abs_norm_data(data, val):
    return (data - np.min(data)) / val

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

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



def lc_plot(mag, magerr, phase, time, period, amplitude, best_fap, outputfp):
    plt.clf()               
    norm = mplcol.Normalize(vmin=min(time), vmax=max(time), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='brg')
    date_color = np.array(mapper.to_rgba(time))
    line_widths = 1
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(r'$Period$:{:.2f}  $\Delta T:${:.2f}  $\Delta m:${:.2f}  $Amplitude:${:.2f}  $FAP:${:.2f}'.format(round(period, 2), round(max(time)-min(time), 2), round(max(mag)-min(mag), 2), round(amplitude, 2), round(best_fap, 2)), fontsize=8, y=0.99)
    ax1.vlines(x = min(time) + period, ymin=min(mag), ymax=max(mag), color = 'g', ls='--', lw=1, alpha = 0.5)
    ax1.scatter(time, mag, c = 'k', s = 1)
    for x, y, e, colour in zip(time, mag, magerr, date_color):
        ax1.errorbar(x, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
        ax1.errorbar(x+1, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
    ax1.set_xlabel(r"$mjd$")
    ax1.set_xlim(min(time)-1,max(time)+1)
    ax1.xaxis.tick_top()
    ax1.invert_yaxis()
    ax2.scatter(phase, mag, c = 'k', s = 1)
    ax2.scatter(phase+1, mag, c = 'k', s = 1)
    for x, y, e, colour in zip(phase, mag, magerr, date_color):
        ax2.errorbar(x, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
        ax2.errorbar(x+1, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
    ax2.set_xlabel(r"$\phi$")
    ax2.set_xlim(0,2)
    ax2.invert_yaxis()
    plt.ylabel(r"    $Magnitude [Ks]$")
    plt.savefig(outputfp, dpi=300, bbox_inches='tight')
    plt.clf()



def lc_debug_plot(mag, magerr, time, chi, ast_chi, outputfp):
  
    plt.clf()               
    norm = mplcol.Normalize(vmin=min(time), vmax=max(time), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='brg')
    date_color = np.array(mapper.to_rgba(time))
    line_widths = 1
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(time, chi, c = 'k', s = 1)
    for x, y, colour in zip(time, chi, date_color):
        ax1.scatter(x, y, color=colour,s=1)
        ax1.scatter(x+1, y, color=colour,s=1)
    ax1.set_xlabel(r"$mjd$")
    ax1.set_ylabel(r"$chi$")
    ax1.set_xlim(min(time)-1,max(time)+1)
    ax1.xaxis.tick_top()

    for x, y, colour in zip(ast_chi, chi, date_color):
        ax2.scatter(x, y, color=colour,s=1)
        ax2.scatter(x+1, y, color=colour,s=1)
    ax2.set_xlabel(r"$astrometric residual chi$")
    ax2.set_ylabel(r"$chi$")
    plt.savefig(outputfp, dpi=300, bbox_inches='tight')
    plt.clf()



filename = '/beegfs/car/njm/OUTPUT/PRIMVS.fits'
chunk_size = 10000  # number of rows to read at a time
knn, model = get_model()
N = 200

#python FAP_train_sample.py 0 & python FAP_train_sample.py 1 & python FAP_train_sample.py 2 & python FAP_train_sample.py 3 & python FAP_train_sample.py 4 & python FAP_train_sample.py 5 & python FAP_train_sample.py 6 & python FAP_train_sample.py 7 & 

for samps in range(10):
    # List all the files in the directory
    np.random.seed(42)
    files = []#os.listdir('/beegfs/car/njm/PRIMVS/LC_FIG/')
    periodics = []
    aperiodics = []
    periods = []
    names = []

    # Open the fits file with memmap to conserve memory space
    with fits.open(filename, memmap=True) as hdulist:
        data = Table.read(hdulist[1], format='fits')

        # Get the number of rows in the table
        num_rows = len(data)

        # Generate a random permutation of row indices
        shuffled_indices = np.random.permutation(num_rows)

        # Reorder the table using the shuffled indices
        shuffled_data = data[shuffled_indices]

        header = hdulist[1].header
        period_col = 'true_period'
        amplitude_col = 'true_amplitude'
        fap_col = 'best_fap'
        name_col = 'sourceid'
        ra_col = 'ra'
        dec_col = 'dec'
        codyM_col = 'Cody_M'

        if sys.argv[1]:
            version = int(sys.argv[1])
            data = np.array_split(data,8)[version]

        for row in data:
            
            if len(periodics) > 5 and len(aperiodics) > 5:
                break

            period, amplitude, fap, name, ra, dec, codyM = row[period_col], row[amplitude_col], row[fap_col], row[name_col], row[ra_col], row[dec_col], row[codyM_col]
            name_str = str(name)

            if name in names or period in periods:
                pass
            else:

                names.append(name)
                periods.append(period)

                lightcurve = Virac.run_sourceid(int(name))
                filters = (lightcurve['filter'].astype(str) == 'Ks')
                mag_gt_0 = (lightcurve['hfad_mag'].astype(float) > 0)
                emag_gt_0 = (lightcurve['hfad_emag'].astype(float) > 0)
                ast_res_chisq_lt_20 = (lightcurve['ast_res_chisq'].astype(float) < 20)
                chi_lt_10 = (lightcurve['chi'].astype(float) < 10)
                filtered_indices = np.where(filters & mag_gt_0 & emag_gt_0 & ast_res_chisq_lt_20 & chi_lt_10)[0]
                lightcurve = lightcurve[filtered_indices]
                mag, magerr, time, chi, astchi = lightcurve['hfad_mag'], lightcurve['hfad_emag'], lightcurve['mjdobs'], lightcurve['chi'], lightcurve['ast_res_chisq']
                #print(mag, magerr, time, period)
                # Remove points with magerr > 2 sigma
                sigma = np.std(magerr)
                filtered_indices = np.where(magerr <= 4 * sigma)[0]
                mag, magerr, time, chi, astchi = mag[filtered_indices], magerr[filtered_indices], time[filtered_indices], chi[filtered_indices], astchi[filtered_indices]

                phase = phaser(time, period)
                smoothed_magnitude = mag

                if mag.size > 40:

                    new_fap = TSA_Stats.inference(period, mag, time, knn, model)

                    if len(mag) < N:
                        clip_pad = 1
                        mag = np.pad(mag, (0,int((N - len(mag)))), 'wrap')
                        time = np.pad(time, (0,int((N - len(time)))), 'wrap')
                    else:
                        clip_pad = 0
                        mag, mag_shite, time = TSA_Stats.delete_rand_items(mag, mag, time, len(mag)-N)

                    phase = phaser(time, period)
                    mag = norm_data(mag)
                  
                    if new_fap < 0.3 and period < 100:
                        periodics.append([TSA_Stats.gen_chan(mag, phase, knn, N),new_fap])

                    if new_fap > 0.7:
                        aperiodics.append([TSA_Stats.gen_chan(mag, phase, knn, N),new_fap])


    # Create the figure and axes
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

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
            axs[i, j].scatter(aperiodics[(2*i)+j][0][3],aperiodics[(2*i)+j][0][0], s=10, c='red')  # Sample scatter plot with a single red point
            axs[i, j].scatter(aperiodics[(2*i)+j][0][3]+1,aperiodics[(2*i)+j][0][0], s=10, c='red')  # Sample scatter plot with a single red point
            axs[i, j].scatter(np.linspace(0,1, num = N), aperiodics[(2*i)+j][0][4], c='black', marker='x', s=8)  # Line plot with red markers            
            axs[i, j].scatter(np.linspace(0,1, num = N)+1, aperiodics[(2*i)+j][0][4], c='black', marker='x', s=8)  # Line plot with red markers
            axs[i, j].set_title(f'{aperiodics[(2*i)+j][1]:.3f}', fontsize=10, ha='center', va='center')
        for j in range(2):
            axs[i, j+3].scatter(periodics[(2*i)+j][0][3],periodics[(2*i)+j][0][0], s=10, c='green')  # Sample scatter plot with a single red point
            axs[i, j+3].scatter(periodics[(2*i)+j][0][3]+1,periodics[(2*i)+j][0][0], s=10, c='green')  # Sample scatter plot with a single red point
            axs[i, j+3].scatter(np.linspace(0,1, num = N), periodics[(2*i)+j][0][4], c='black', marker='x', s=8)  # Line plot with red markers            
            axs[i, j+3].scatter(np.linspace(0,1, num = N)+1, periodics[(2*i)+j][0][4], c='black', marker='x', s=8)  # Line plot with red markers            
            axs[i, j+3].set_title(f'{periodics[(2*i)+j][1]:.3f}', fontsize=10, ha='center', va='center')

    # Adjust layout to avoid overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.savefig('/home/njm/grids/fap_periodic_grid_examples' + str(version) + str(samps) + '.jpg')






