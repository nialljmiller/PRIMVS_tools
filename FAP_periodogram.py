import numpy as np
import Tools as T
#matplotlib
import matplotlib;matplotlib.rcParams['agg.path.chunksize'] = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib import gridspec
from tqdm import tqdm
import random



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

TOOL = T.Tools(do_stats = 1)


for cat_type in ['a','b','c','d','e','f','g','h']:

	TOOL.mag, TOOL.magerr, TOOL.time, cattype, god_period = TOOL.synthesize(N = 200, amplitude = 0.666, period = random.uniform(0, 500), cat_type = 'CV', median_mag = 12, other_pert = 1, contamination_flag = 1)

	periods = np.linspace(0.1,1000,5000)
	FAPS = []
	for period in tqdm(periods):
		FAP = TOOL.fap_inference(period, TOOL.mag, TOOL.time, TOOL.knn, TOOL.model)
		FAPS.append(FAP)

	period_true = periods[np.argmin(FAPS)]
	print(period_true, god_period)

	plt.clf()
	fig, ax = plt.subplots()
	#fig.suptitle('MSE: '+str(round(mse,3))+'  RMSE: '+str(round(sqrt(mse),3))+'  Accuracy: '+str(round(acc,3)))
	ax.plot(periods, FAPS, 'k', alpha = 0.9, ms = 2)

	ax.axvline(x = god_period, ls = '-', color = 'blue', label = 'True Period = '+str(round(god_period, 8)))
	ax.axvline(x = period_true, ls = '--', color = 'orange', label = 'Extracted Period = '+str(round(period_true, 8)))
	ax.set(xlabel = 'Period', ylabel='FAP', xlim = [0,1000], ylim = [0,1])
	fig.tight_layout()
	plt.savefig('/home/njm/hard_periodogram_'+str(cat_type)+'.jpg', bbox_inches = 'tight')

	TOOL.pltpsd(periods, FAPS, 'hard_FAPS_periodogram_'+str(cat_type), 'Period [mins]', 'FAP', xlim = period_true, peak_min = 1, fp = '/home/njm/')

