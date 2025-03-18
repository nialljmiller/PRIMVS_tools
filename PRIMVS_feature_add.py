import pandas as pd
import glob
import random
import csv
import os
import tqdm as tqdm
import numpy as np
import itertools
# List of column names
col_names = ['name','mag_n','mag_avg','magerr_avg','Cody_M','stet_k','eta','eta_e','med_BRP',
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


csv_files = glob.glob("/beegfs/car/njm/OUTPUT/vars/complete/rescan/*.csv") + glob.glob("/beegfs/car/njm/OUTPUT/vars/complete_lower_vars/rescan/*.csv")

# iterate over a list of file names
for filename in tqdm.tqdm(csv_file):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = []
        for row in reader:

    # specify the column names explicitly
    period_col = 'true_period'
    amplitude_col = 'true_amplitude'
    fap_col = 'best_fap'

'mag_n'
'mag_avg'
'magerr_avg'
'Cody_M'
'stet_k'
'eta'
'eta_e'
'med_BRP'
'range_cum_sum'
'max_slope'
'MAD'
'mean_var'
'percent_amp'
'true_amplitude'
'roms'
'p_to_p_var'
'lag_auto'
'AD'
'std_nxs'
'weight_mean'
'weight_std'
'weight_skew'
'weight_kurt'
'mean'
'std'
'skew'
'kurt'
'time_range'
'true_period'
'true_class'
'best_fap'








            # extract values from the row using column names
            values = [row[header.index(name)] for name in col_names]
            # calculate new values and insert them at specified positions
            new_values = calculate_new_values(values)

            for pos, val in zip(insert_positions, new_values):
                row.insert(pos, val)
            # append the updated row to the list of rows
            rows.append(row)


	# access the relevant columns for each row
	period = row[period_col]
	amplitude = row[amplitude_col]
	fap = row[fap_col]
	name = row[name_col]
	ra = row[ra_col]
	dec = row[dec_col]

	# Use the Virac module to obtain the lightcurve data for this row (⊙ω⊙)
	lightcurve = Virac.run_sourceid(int(name))

	lightcurve = lightcurve[np.where(lightcurve['filter'].astype(str) == 'Ks')[0]]
	lightcurve = lightcurve[np.where(lightcurve['hfad_mag'].astype(float) > 0)[0]]
	lightcurve = lightcurve[np.where(lightcurve['hfad_emag'].astype(float) > 0)[0]]
	lightcurve = lightcurve[np.where(lightcurve['ast_res_chisq'].astype(float) < 20)[0]]
	lightcurve = lightcurve[np.where(lightcurve['chi'].astype(float) < 10)[0]]

	mag = lightcurve['hfad_mag']
	magerr = lightcurve['hfad_emag']
	time = lightcurve['mjdobs']
	chi = lightcurve['chi']
	astchi = lightcurve['ast_res_chisq']
	phase = phaser(time, period)

	new_fap = TSA_Stats.inference(period, mag, time, knn, model)

