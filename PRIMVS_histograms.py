import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.cluster import KMeans

# Load your painting image
image_path = 'art.jpg'  # Replace with your image path
image = Image.open(image_path)
image = np.array(image, dtype=np.float64) / 255

# Reshape the image array into a 2D array where each row is a color
pixels = image.reshape(-1, 3)

# Use k-means clustering to find distinct colors
# Adjust n_clusters to change the number of distinct colors
kmeans = KMeans(n_clusters=10)
kmeans.fit(pixels)
cluster_centers = kmeans.cluster_centers_

# Sort the colors to ensure a gradual gradient
# This example sorts by the sum of the RGB components, but you can choose other sorting criteria
sorted_colors = cluster_centers[np.argsort(cluster_centers.sum(axis=1))]

# Create a custom colormap
custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", sorted_colors)




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
                 'xtick.minor.visible': False, 'ytick.minor.visible': False,  # show minor ticks
                 'text.usetex': True, 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'})  # process text with LaTeX instead of matplotlib math mode






# Open the fits file with memmap to conserve memory space
filename = '/beegfs/car/njm/OUTPUT/PRIMVS_SAMPLE.fits'
with fits.open(filename, memmap=True) as hdulist:
    data = Table.read(hdulist[1], format='fits')
    header = hdulist[1].header

# Assuming 'data' is your Astropy Table
df = data.to_pandas()

# Convert 'best_fap' column to numeric type
df['best_fap'] = pd.to_numeric(df['best_fap'], errors='coerce')
df['time_range'] = pd.to_numeric(df['time_range'], errors='coerce')
df['true_period'] = pd.to_numeric(df['true_period'], errors='coerce')
df['true_amplitude'] = pd.to_numeric(df['true_amplitude'], errors='coerce')
df['magerr_avg'] = pd.to_numeric(df['magerr_avg'], errors='coerce')
# Now create the masks with the column as a numeric type
apmask = df['best_fap'] > 0.5
pmask = df['best_fap'] < 0.1






# Define the percentage of scatter you want to add
scatter_percentage = 0.01 # 10% scatter

# Add random scatter to each feature
Pcyclefeat = (df.loc[pmask, 'time_range'] / df.loc[pmask, 'true_period'])
Psnrfeat = (df.loc[pmask, 'true_amplitude'] / df.loc[pmask, 'magerr_avg'])
Pnfeat = df.loc[pmask, 'mag_n']

APsnrfeat = (df.loc[apmask, 'true_amplitude'] / df.loc[apmask, 'magerr_avg'])
APnfeat = df.loc[apmask, 'mag_n']  # Replace 'feature3_column_name' with the actual column name



APsnrfeat = np.array(APsnrfeat, dtype=float)

num_points_to_change = int(len(APsnrfeat) * 0.01)  # For example, 10% of the points
indices_to_change = np.random.choice(len(APsnrfeat), size=num_points_to_change, replace=False)
APsnrfeat[indices_to_change] = APsnrfeat[indices_to_change] + np.random.normal(50,30, size=num_points_to_change)




#Pnfeat = np.array(Pnfeat)  * (1 + np.random.normal(0, scatter_percentage, len(Pnfeat)))
plt.figure(figsize=(5, 3))  # Larger figure size

your_lower_clip_value = 2
your_upper_clip_value = 100
# Define the boundaries for clipping
boundaries = list(np.linspace(your_lower_clip_value, your_upper_clip_value, 480))
norm = BoundaryNorm(boundaries, ncolors=480, clip=True)

# Define the number of bins for the histograms
num_bins = 40
# Define linearly spaced bins
cycle_bins = np.logspace(np.log10(1),np.log10(10000), num_bins)#Pcyclefeat.max(), num_bins)
snr_bins = np.linspace(1, 110, num_bins)#Psnrfeat.max(), num_bins)
n_bins = np.linspace(40, 300, num_bins)#Pnfeat.max(), num_bins)



# Assuming you have the data in the variables Pnfeat, APnfeat, Psnrfeat, APsnrfeat, Pcyclefeat

def calculate_histogram_percentage(data, bins):
    hist_percentage = []
    total_data = len(data)
    for i in range(len(bins)-1):
        bin_start = bins[i]
        #print(bin_start, np.min(data), np.max(data))
        if i == len(bins):
            bin_end = np.max(data)
        else:
            bin_end = bins[i + 1]
        data_in_bin = [x for x in data if bin_start <= x < bin_end]
        #print(len(data_in_bin) , total_data)
        bin_percentage = len(data_in_bin) / total_data
        hist_percentage.append(bin_percentage)
    return hist_percentage

# Calculate histograms and percentages
hist_Psnr_percentage = calculate_histogram_percentage(Psnrfeat, snr_bins)
hist_APsnr_percentage = calculate_histogram_percentage(APsnrfeat, snr_bins)
hist_Pcycle_percentage = calculate_histogram_percentage(Pcyclefeat, cycle_bins)
hist_Pn_percentage = calculate_histogram_percentage(Pnfeat, n_bins)
hist_APn_percentage = calculate_histogram_percentage(APnfeat, n_bins)



'''


plt.clf()
sc = plt.scatter(Pcyclefeat, Pnfeat, c=Psnrfeat, cmap='plasma', marker='.', s=10, alpha=0.7, norm=norm)

plt.xlabel('Cycles')
plt.ylabel('N')

# Set scale of axes
plt.yscale('linear')
plt.xscale('log')

# Set limits for x and y axes
plt.xlim(left=5, right=3000)
plt.ylim(bottom=50, top=400)

# Custom colorbar ticks
colorbar_ticks = [your_lower_clip_value, 20, 40, 60, 80, your_upper_clip_value]  # Adjust these values as necessary

# Add a colorbar to show the scale of Psnrfeat without minor ticks
cbar = plt.colorbar(sc, label=r'$A/\bar{\sigma}$', ticks=colorbar_ticks, pad=-0.0005, alpha=0.5)
cbar.ax.minorticks_off()  # Disable minor ticks on the colorbar

# Gridlines can improve readability
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Remove right-side ticks
plt.tick_params(right=False, labelright=False)  # Turn off right side ticks

plt.savefig('/home/njm/FAPS/hist/N_vs_C_plot.jpg', bbox_inches='tight', dpi=300)
plt.clf()




# Create the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 4), sharey=True)

# Plot the leftmost subplot
ax1.bar(n_bins[:-1], hist_Pn_percentage, width=np.diff(n_bins), color='b', alpha=0.4, label='Periodic')
ax1.bar(n_bins[:-1], hist_APn_percentage, width=np.diff(n_bins), color='r', alpha=0.4, label='Aperiodic')
ax1.set_ylabel(r'Log(Percentage)')
ax1.set_yticklabels([])
ax1.set_xticks([50,150,250])
ax1.legend()
ax1.set_xlabel(r'N')
ax1.set_yscale('log')

# Plot the middle subplot
ax2.bar(snr_bins[:-1], hist_Psnr_percentage, width=np.diff(snr_bins), color='b', alpha=0.4, label='Psnrfeat')
ax2.bar(snr_bins[:-1], hist_APsnr_percentage, width=np.diff(snr_bins), color='r', alpha=0.4, label='APsnrfeat')
ax2.set_yticklabels([])
ax2.set_xticks([2,50,100])
ax2.set_xlabel(r'$A/\bar{\sigma}$')
ax2.set_yscale('log')

# Plot the rightmost subplot
ax3.bar(cycle_bins[:-1], hist_Pcycle_percentage, width=np.diff(cycle_bins), color='b', alpha=0.7, label='Pcyclefeat')
ax3.set_xlabel(r'Cycles')
ax3.set_yticklabels([])

ax3.set_xticks([10,100,1000])
ax3.set_xscale('log')
ax3.set_yscale('log')

# Remove the vertical space between subplots
plt.subplots_adjust(wspace=0)

# Ensure that the bars in each histogram add up to 1
plt.yscale('log')

# Save the plot
plt.savefig('/home/njm/FAPS/hist/training_data_hist.pdf', bbox_inches='tight')

'''

plt.clf()


# Figure setup
fig = plt.figure(figsize=(5, 4))

# Scatter plot on top
scatter_axis = fig.add_axes([0.1, 0.5, 1.034, 0.5])  # [left, bottom, width, height]
sc = scatter_axis.scatter(Pcyclefeat, Pnfeat, c=Psnrfeat, cmap='Spectral_r', marker='.', s=20, alpha=0.5, norm=norm)
scatter_axis.set_ylabel('N')
scatter_axis.set_xlabel('Cycles')
scatter_axis.set_yscale('linear')
scatter_axis.set_xscale('log')
scatter_axis.set_xlim(left=5, right=3000)
scatter_axis.set_ylim(bottom=50, top=400)
scatter_axis.xaxis.tick_top()
scatter_axis.xaxis.set_label_position('top')
scatter_axis.set_yticks([100, 200, 300, 400])
colorbar_ticks = [10, 25, 50, 75, your_upper_clip_value]
cbar = fig.colorbar(sc, ax=scatter_axis, label=r'$A/\bar{\sigma}$', ticks=colorbar_ticks, pad=0.)
cbar.ax.minorticks_off()
scatter_axis.tick_params(right=False, labelright=False)

# First histogram below scatter plot
hist1_axis = fig.add_axes([0.1, 0., 0.31, 0.5])  # Adjusted for alignment and spacing
hist1_axis.bar(n_bins[:-1], hist_Pn_percentage, width=np.diff(n_bins), color='b', alpha=0.4, label='Periodic')
hist1_axis.bar(n_bins[:-1], hist_APn_percentage, width=np.diff(n_bins), color='r', alpha=0.4, label='Aperiodic')
hist1_axis.set_ylabel(r'Log(Percentage)')
hist1_axis.set_xlabel(r'N')
hist1_axis.set_xticks([50,150,250])
hist1_axis.set_yscale('log')
hist1_axis.set_ylim(bottom=0., top=0.6)

# Second histogram
hist2_axis = fig.add_axes([0.4, 0., 0.3, 0.5])  # Adjusted for alignment and spacing
hist2_axis.bar(snr_bins[:-1], hist_Psnr_percentage, width=np.diff(snr_bins), color='b', alpha=0.4, label='Periodic')
hist2_axis.bar(snr_bins[:-1], hist_APsnr_percentage, width=np.diff(snr_bins), color='r', alpha=0.4, label='Aperiodic')
hist2_axis.set_xlabel(r'$A/\bar{\sigma}$')
hist2_axis.set_yscale('log')
hist2_axis.legend()
hist2_axis.set_yticks([])
hist2_axis.set_ylim(bottom=0., top=0.6)
# Third histogram
hist3_axis = fig.add_axes([0.7, 0., 0.3, 0.5])  # Adjusted for alignment and spacing
hist3_axis.bar(cycle_bins[:-1], hist_Pcycle_percentage, width=np.diff(cycle_bins), color='b', alpha=0.7, label='Pcyclefeat')
hist3_axis.set_xlabel(r'Cycles')
hist3_axis.set_xscale('log')
hist3_axis.set_yscale('log')
hist3_axis.set_yticks([])
hist3_axis.set_ylim(bottom=0., top=0.6)

plt.savefig('/home/njm/FAPS/hist/combined_plot.jpg', bbox_inches='tight', dpi=300)



