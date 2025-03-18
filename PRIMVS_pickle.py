import sys
import Tools as T
import numpy as np
import pickle
from tqdm import tqdm
import glob
from sklearn import neighbors
from random import randint, randrange

classes = []
classes_source = []
sequences = []
fucked_count = 0

tier = '/beegfs/car/njm/PRIMVS/LC_TOKEN/*.csv'
pkl_dir = '/beegfs/car/njm/PRIMVS/LC_PKL/'

files = glob.glob(tier)
for fi in tqdm(files):
	try:
		mag, mag_smooth, mag_err, time, phase, chi, astchi = np.genfromtxt(fi, dtype='float', converters = None, comments = '#', delimiter = ',').T
		# Filtering out invalid data points
		valid_indices = np.isfinite(mag) & np.isfinite(mag_err) & np.isfinite(time) & np.isfinite(phase)
		mag = mag[valid_indices]
		mag_err = mag_err[valid_indices]
		time = time[valid_indices]
		phase = phase[valid_indices]
		if mag.size > 40:
		    star_name = fi.split('/')[-1].replace('__', '_')
		    star_name = star_name.split('_')[1]
		    sequence = np.stack((mag, mag_err, phase, time), axis=-1)
		    classes.append(star_name)
		    sequences.append(sequence)
	except Exception as e:
		fucked_count = fucked_count + 1
		print(f"this file {fi} does not work")
		print(star_name)
		print(e)
	if fucked_count == 5:
		exit()
	else:
		continue



with open(pkl_dir+'sequences.pkl', 'wb') as f:
    pickle.dump(sequences, f)

with open(pkl_dir+'classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
