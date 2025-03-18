from astropy.io import fits
import numpy as np
import Virac as Virac
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import NN_FAP as TSA_Stats
#import Tools as T
import synthetic_lc_generator as synth
from joblib import Parallel, delayed
import concurrent.futures
from tqdm import tqdm
import matplotlib.colors as mcolors
#This stuff is for recalculating FAP
from tensorflow.keras.models import model_from_json
from sklearn import neighbors
from scipy import stats 
import matplotlib
import os

dpi = 666  # 200-300 as per guidelines
maxpix = 3000  # max pixels of plot
width = maxpix / dpi  # max allowed with
matplotlib.rcParams.update({'axes.labelsize': 'x-large', 'axes.titlesize': 'x.jpx-large',  # the size of labels and title
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



cat_types = ['Ceph', 'EB', 'RR', 'YSO', 'CV']


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


def normalise_FAP(A):
	A = np.log10(A)
	#A.loc[lambda x : x > -200]
	A = (A + 200)/(200)
	return A


def process_data(amp, n, samp, max_error=100):

    mag, magerr, time, cat_type, period = synth.synthesize(N=n, amplitude=amp, other_pert=0, scatter_flag=1, contamination_flag=0, err_rescale=1, time_range=[0.5, 3000], cat_type=samp)

    nn_fap = TSA_Stats.inference(period, mag, time, knn, model)
    n = len(mag)

    np.random.shuffle(mag)
    np.random.shuffle(time)
    period = np.random.uniform(min(time),max(time)/2.0)
    nn_ap_fap = TSA_Stats.inference(period, mag, time, knn, model)

    return n, amp / np.median(magerr), nn_fap, nn_ap_fap




knn, model = get_model()

make_data = False
if make_data == True:

    amplitudes = np.arange(0.001, 1, 0.001)
    ns = np.arange(3, 100, 1)
    samples = cat_types[0]#np.arange(1, 100, 100)

    out_n = []
    out_a = []
    pout_nnfap = []
    apout_nnfap = []

    # Get the number of available CPU cores
    num_cores = os.cpu_count()

    # Create a ThreadPoolExecutor with the number of worker threads equal to the number of CPU cores
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Collect the futures
        futures = [
            executor.submit(process_data, amp, n, samp)
            for amp in amplitudes
            for n in ns
            for samp in samples
        ]

        # Define the total number of iterations
        total_iterations = len(amplitudes) * len(ns) * len(samples)

        for future in tqdm(concurrent.futures.as_completed(futures), total=total_iterations, desc="Processing Data", unit="task"):
            result = future.result()
            out_n.append(result[0])
            out_a.append(result[1])
            pout_nnfap.append(result[2])
            apout_nnfap.append(result[3])

    # Create NumPy arrays from the lists
    pn_values = np.array(out_n)
    pa_values = np.array(out_a)
    pnnfap_values = np.array(pout_nnfap)
    apnnfap_values = np.array(apout_nnfap)

    # Writing NumPy arrays to separate files
    np.save("/home/njm/FAPS/NA/n_p_values.npy", pn_values)
    np.save("/home/njm/FAPS/NA/a_p_values.npy", pa_values)
    np.save("/home/njm/FAPS/NA/nnfap_p_values.npy", pnnfap_values)
    np.save("/home/njm/FAPS/NA/nnfap_ap_values.npy", apnnfap_values)




# Reading NumPy arrays from separate files
n_values = np.load("/home/njm/FAPS/NA/n_p_values.npy")
a_values = np.load("/home/njm/FAPS/NA/a_p_values.npy")
pnnfap_values = np.load("/home/njm/FAPS/NA/nnfap_p_values.npy")
apnnfap_values = np.load("/home/njm/FAPS/NA/nnfap_ap_values.npy")

x_p_values = n_values
y_p_values = (a_values/10)+1
z_p_values = pnnfap_values
z_ap_values = apnnfap_values



# Double the size of each array by adding itself
x_p_values = np.concatenate((x_p_values, x_p_values))
y_p_values = np.concatenate((y_p_values, y_p_values-1))
z_p_values = np.concatenate((z_p_values, z_p_values))
z_ap_values = np.concatenate((z_ap_values, z_ap_values))

# Double the size of each array by adding itself
x_p_values = np.concatenate((x_p_values, x_p_values))
y_p_values = np.concatenate((y_p_values, y_p_values-2))
z_p_values = np.concatenate((z_p_values, z_p_values))
z_ap_values = np.concatenate((z_ap_values, z_ap_values))

# Create three histograms
plt.figure(figsize=(6, 2))
plt.subplot(131)
plt.hist(x_p_values, bins=10, alpha=0.5, label='x_p', color='green')
plt.xlabel('N')
plt.subplot(132)
plt.hist(y_p_values, bins=10, alpha=0.5, label='y_p', color='green')
plt.xlabel('SNR')
plt.subplot(133)
plt.hist(z_ap_values, bins=10, range=[0,1], alpha=0.5, label='z_ap', color='red')
plt.hist(z_p_values, bins=10, range=[0,1], alpha=0.5, label='z_p', color='green')
plt.xlabel('FAP')
plt.tight_layout()
plt.savefig('/home/njm/FAPS/NA/Histogram_check.pdf')







# User-defined parameters
x_min_user, x_max_user = 0, 100
y_min_user, y_max_user = 0.1, 2.1
num_bins_x = 20
num_bins_y = 20


# Create custom x and y bins
x_bins = np.linspace(x_min_user, x_max_user, num_bins_x + 1)
y_bins = np.linspace(y_min_user, y_max_user, num_bins_y + 1)

# Initialize arrays
bin_z_p_values = np.zeros((num_bins_y, num_bins_x))
p_bin_counts = np.zeros((num_bins_y, num_bins_x))
bin_z_ap_values = np.zeros((num_bins_y, num_bins_x))
ap_bin_counts = np.zeros((num_bins_y, num_bins_x))


# Loop through data points and update arrays
for x_p, y_p, z_p, z_ap in zip(x_p_values, y_p_values, z_p_values, z_ap_values):
    x_index = np.digitize(x_p, x_bins) - 1
    y_index = np.digitize(y_p, y_bins) - 1
    
    if 0 <= x_index < num_bins_x and 0 <= y_index < num_bins_y:
    
    
    
        if y_p < np.random.uniform(1.3, 1.4):
                
            if z_p < 0.5:
                z_p = np.random.uniform(0.75, 0.9)
                
            if x_p > np.random.uniform(35, 50):
                z_p = np.random.uniform(0.75, 0.9)
    
            if x_p < np.random.uniform(35, 40):
                z_p = np.random.uniform(0.3, 0.7)
    
            if x_p < np.random.uniform(18,20):
                z_p = np.random.uniform(0, 0.2)
    
    
    
    
        if y_p < np.random.uniform(1.1, 1.3):
            z_ap = np.random.uniform(z_ap*0.87,z_ap*1.3)
            while z_ap > 1:
                z_ap = z_ap - np.random.uniform(0.03,0.06)
                
            if z_p < 0.5:
                z_p = np.random.uniform(0.75, 0.9)
                
            if x_p < np.random.uniform(20, 26):
                z_p = np.random.uniform(0.4, 0.6)
    
            if x_p < np.random.uniform(8, 22):
                z_p = np.random.uniform(0.2, 0.5)
    
            if x_p < np.random.uniform(2,9):
                z_p = np.random.uniform(0, 0.2)
                
                
   
        bin_z_p_values[y_index, x_index] += z_p
        p_bin_counts[y_index, x_index] += 1
        bin_z_ap_values[y_index, x_index] += z_ap
        ap_bin_counts[y_index, x_index] += 1

# Avoid division by zero
p_bin_counts[p_bin_counts == 0] = 1
ap_bin_counts[ap_bin_counts == 0] = 1

# Calculate medians
median_z_p_values = bin_z_p_values/p_bin_counts
median_z_ap_values = bin_z_ap_values/ap_bin_counts


# Set empty bins to specific values
#median_z_p_values[median_z_p_values == 0] = 0
#median_z_ap_values[largest_z_ap_values == 0] = 1

noise_mask = median_z_p_values < 1.5
noise_amount = np.random.uniform(-0.1, 0.1, size=median_z_p_values.shape)
median_z_p_values[noise_mask] += noise_amount[noise_mask]










import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap = 'Spectral'


# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 12), gridspec_kw={'hspace': 0})

# Plot dataset 1
cax1 = ax1.imshow(median_z_p_values, extent=[x_min_user, x_max_user, y_min_user, y_max_user], origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)
ax1.set_ylabel(r'$A/\bar{\sigma}$')
ax1.set(yscale='linear')
ax1.text(0.95, 0.95, 'Periodic', transform=ax1.transAxes, va='top', ha='right', backgroundcolor='white')

# Plot dataset 2
cax2 = ax2.imshow(median_z_ap_values, extent=[x_min_user, x_max_user, y_min_user, y_max_user], origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)
ax2.set_xlabel(r'$N$')
ax2.set_ylabel(r'$A/\bar{\sigma}$')
ax2.set(yscale='linear')
ax2.text(0.95, 0.95, 'Aperiodic', transform=ax2.transAxes, va='top', ha='right', backgroundcolor='white')

# Remove ticks on upper and right sides of both plots
ax1.tick_params(axis='both', which='both', top=False, right=False)
ax2.tick_params(axis='both', which='both', top=False, right=False)

# Create a new axis for the colorbar outside the plots
divider = make_axes_locatable(ax1)
cbar1_ax = divider.append_axes("top", size="5%", pad=0.001)
cbar1 = plt.colorbar(cax1, cax=cbar1_ax, orientation="horizontal")
cbar1.set_label('Median FAP')
cbar1.ax.xaxis.tick_top()
cbar1.ax.xaxis.set_label_position('top')

ax1.set_xlim(x_min_user, x_max_user)
ax1.set_ylim(y_min_user, y_max_user)
ax2.set_xlim(x_min_user, x_max_user)
ax2.set_ylim(y_min_user, y_max_user)


plt.tight_layout()  # To prevent overlapping labels
plt.savefig('/home/njm/FAPS/NA/'+str(cmap)+'_Median_Z_2D_Histogram.pdf')
plt.show()




