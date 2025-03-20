# -*##- coding: utf-8 -*-
import os#; os.system('module load cuda-10.0')
import glob
from os import listdir
from art import *
import csv
import sys
import traceback
import matplotlib;matplotlib.rcParams['agg.path.chunksize'] = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
import random
import numpy as np
import pathlib
import gc;gc.enable()
from time import time as timer
from time import sleep

import Tools as T
import PDM as PDM
import LS as LS
from os.path import isfile
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io.fits import writeto
from astropy.io import fits
import math as maths
#import Launcher
import Virac as Virac
from functools import reduce
from sklearn import neighbors
import utils
import random
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from tqdm import tqdm
#print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\t hubworld STARTED!\n')
#-------------------Random crap and initilisations------------------#



#python hubworld.py 0 & python hubworld.py 1 & python hubworld.py 2 & python hubworld.py 3 & python hubworld.py 7 & python hubworld.py 8 & python hubworld.py 9 & python hubworld.py 10 & python hubworld.py 11 & python hubworld.py 13 & python hubworld.py 15
#python hubworld.py 0 & python hubworld.py 1 & python hubworld.py 2 & python hubworld.py 3 & python hubworld.py 4 & python hubworld.py 5 & python hubworld.py 6 & python hubworld.py 7 & 


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


def lc_plot(mag, magerr, phase, time, period, method, best_fap, outputfp):
	plt.clf()				
	norm = mplcol.Normalize(vmin=min(time), vmax=max(time), clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap='brg')
	date_color = np.array(mapper.to_rgba(time))
	line_widths = 1
	fig, ax1 = plt.subplots()
	fig.suptitle(r'$Period$:'+str(round(period,3)) + '  $\Delta T:$' + str(round(max(time)-min(time),3)) + '  $\Delta m:$' + str(round(max(mag)-min(mag),3)) + '  $Method:$' + method + '  $FAP:$' + best_fap)
	ax1.scatter(phase, mag, c = 'k', s = 1)
	ax1.scatter(phase+1, mag, c = 'k', s = 1)
	for x, y, e, colour in zip(phase, mag, magerr, date_color):
		ax1.errorbar(x, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
		ax1.errorbar(x+1, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
	ax1.set_xlabel(r"$\phi$")
	ax1.set_xlim(0,2)
	ax1.invert_yaxis()
	plt.ylabel(r"	$Magnitude [Ks]$")
	plt.savefig(outputfp, dpi=300, bbox_inches='tight')
	plt.clf()

def remove_core_files():
	files_in_directory = os.listdir(pathlib.Path(__file__).parent.absolute())
	filtered_files = [file for file in files_in_directory if file.startswith("core.")]
	if filtered_files:
		for file in filtered_files:
			path_to_file = os.path.join(pathlib.Path(__file__).parent.absolute(), file)
			os.remove(path_to_file)	



def clean_lightcurve(lightcurve):
	# Precompute masks
	mask_filter = lightcurve['filter'].astype(str) == 'Ks'
	mask_hfad_mag = lightcurve['hfad_mag'].astype(float) > 0
	mask_hfad_emag = lightcurve['hfad_emag'].astype(float) > 0
	mask_ast_res_chisq = lightcurve['ast_res_chisq'].astype(float) < 20
	mask_chis = lightcurve['chi'].astype(float) < 10
	mask_ambiguous_match = lightcurve['ambiguous_match'].astype(float) == 0

	# Combine masks
	combined_mask = mask_filter & mask_hfad_mag & mask_hfad_emag & mask_ast_res_chisq & mask_chis & mask_ambiguous_match
	
	# Apply combined mask
	lightcurve = lightcurve[combined_mask]
	return lightcurve

def check_variability(lightcurve):
	# Check if the lightcurve has more than 100 points
	if len(lightcurve) <= 100:
		return False

	med_error = np.nanmedian(lightcurve['hfad_emag'])
	snr = (np.nanmax(lightcurve['hfad_mag']) - np.nanmin(lightcurve['hfad_mag'])) / med_error
	return snr > 4 and (np.nanmax(lightcurve['hfad_mag']) - np.nanmin(lightcurve['hfad_mag'])) > 0.1



def process_code(process_num, version_files):

	TOOL = T.Tools(data_name = data_name, output_dir = output_dir, parent_name = parent_name, img_dir = img_dir,  IO = IO, FAP_type = FAP_type, np_check = np_check, do_fap = do_fap, do_stats = do_stats, s_fit = s_fit, error_clip = error_clip, time_res = time_res, do_pdm = do_pdm, do_ls = do_ls, do_ce = do_ce, do_gp = do_gp)
	TOOL.par_version = int(sys.argv[1]) + (4 * process_num)

	for version_file in tqdm(version_files):
		names = np.genfromtxt('/beegfs/car/njm/OUTPUT/next/'+version_file, delimiter = ',', unpack = True)
		for star_name in names:

			lightcurve=Virac.run_sourceid(int(star_name))
			lightcurve = clean_lightcurve(lightcurve)
			is_variable = check_variability(lightcurve)
			if is_variable:
				try:
					time_range = max(np.array(lightcurve["mjdobs"])) - min(np.array(lightcurve["mjdobs"]))
					F_start = 2.0 / time_range
					F_stop = 100
					F_N = 10000000

					TOOL.lightcurve(name = lightcurve["sourceid"][0],
								mag = lightcurve["hfad_mag"],
								magerr = lightcurve["hfad_emag"],
								time = lightcurve["mjdobs"],
								ast_res_chi = lightcurve["ast_res_chisq"],
								chi = lightcurve["chi"],
								ambi_match = lightcurve["ambiguous_match"],
								true_period = 0, 
								true_amplitude = 0, 
								true_mag = 0, 
								true_class = 'Aperiodic',#TOOL.true_class, 
								F_start = F_start, 
								F_stop = F_stop, 
								F_N = F_N)
				except:
					with open(output_dir + str(version) + '_.FAILED', "a") as fp:
						wr = csv.writer(fp, dialect='excel')
						wr.writerow([star_name])

				TOOL.period_analysis()
				output_file_path = output_dir + version_file + '.csv'
				TOOL.ts_write(output_file_path, overwrite = False)

				remove_core_files()

		os.rename('/beegfs/car/njm/OUTPUT/next/'+version_file, '/beegfs/car/njm/OUTPUT/next/complete/'+version_file)







IO = 0			# 1 for prints, 2 for folded LC, 3 for everything
np_check = 0					#Research at range - 1:all P. 2:all P, P/2. 3: all P, 2P. 4: all 2P, P/2. 5: all P, P/2, 2P. 6: all P, P(second peak). 7: all P, P(second peak), P(third Peak).
normalise = 0					#normalise psd
s_fit = 1					#spline fit
error_clip = 1					#remove contamination and high error
time_res = 0#0.5/24.				#minimum seperation between datapoints (all within range of 'time_res' will be median binned)
do_stats = 1
do_fap = 1					#Calculate AM Cody Q and M values https://arxiv.org/pdf/1401.6582.pdf	(pg 25-26)
															#FAP type (0 is dummy gen. 1 is random order)
FAP_type = None

do_pdm = 1
do_ls = 1
do_ce = 0
do_gp = 0
																	#unfolded LC, 1 = yes
data_name = 'vars'
parent_name = 'vars'

img_dir = '/Figures_redo/'												#output image directory
output_dir = '/beegfs/car/njm/OUTPUT/next_vars/'													#output file and directory
LC_source_dir = '/beegfs/car/njm/LC/'+data_name+'/'



#attain versions for paralalelle

multi_version = int(os.getcwd().split('/')[-1])
version = int(sys.argv[1])
files = os.listdir('/beegfs/car/njm/OUTPUT/next/')
#files.reverse()
files = np.array_split(files,4)[multi_version]
version_files = np.array_split(files,4)[version]



import concurrent.futures

num_iterations = 16
file_chunks = np.array_split(version_files,num_iterations)

with concurrent.futures.ThreadPoolExecutor(max_workers=num_iterations) as executor:
	for i, chunk in enumerate(file_chunks):
		executor.submit(process_code, i, chunk)





