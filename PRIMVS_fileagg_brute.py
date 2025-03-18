import pandas as pd
from astropy.table import Table, vstack, hstack
import glob
import random
import csv
import os
from tqdm import tqdm
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

meta_col_names = ['sourceid','ra','ra_error','dec','dec_error','l','b','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
		'chisq','uwe','ks_n_detections','ks_n_observations','ks_n_ambiguous','ks_n_chilt5','ks_med_mag','ks_mean_mag','ks_ivw_mean_mag',
		'ks_chilt5_ivw_mean_mag','ks_std_mag','ks_mad_mag','ks_ivw_err_mag','ks_chilt5_ivw_err_mag','z_n_observations','z_med_mag',
		'z_mean_mag','z_ivw_mean_mag','z_chilt5_ivw_mean_mag','z_std_mag','z_mad_mag','z_ivw_err_mag','z_chilt5_ivw_err_mag','y_n_detections',
		'y_n_observations','y_med_mag','y_mean_mag','y_ivw_mean_mag','y_chilt5_ivw_mean_mag','y_std_mag','y_mad_mag','y_ivw_err_mag',
		'y_chilt5_ivw_err_mag','j_n_detections','j_n_observations','j_med_mag','j_mean_mag','j_ivw_mean_mag','j_chilt5_ivw_mean_mag',
		'j_std_mag','j_mad_mag','j_ivw_err_mag','j_chilt5_ivw_err_mag','h_n_detections','h_n_observations','h_med_mag','h_mean_mag',
		'h_ivw_mean_mag','h_chilt5_ivw_mean_mag','h_std_mag','h_mad_mag','h_ivw_err_mag','h_chilt5_ivw_err_mag']		

fap_cut = 0.5

fits_col_types = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float','float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		  'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		   'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		    'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		    'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'str', 'float', 'str', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']


csv_files = ["/beegfs/car/njm/OUTPUT/vars/Periodics_completed.csv"]


def get_sub_file_name(source_id):
    return f'subfile_{source_id[:4]}.csv'

# Assume output_file is the path to the CSV file you want to append to
output_file = "/beegfs/car/njm/OUTPUT/PRIMVS_AuxSource.csv"

# Open the file in append mode outside of the loop
file = open(output_file, "a", newline="")
writer = csv.writer(file)


processed_idx = []
processed_fap = []
old_filename = ''

for csv_file in csv_files:

    # Assuming csv_file is the path to your CSV file
    data = np.genfromtxt(csv_file, delimiter=',', unpack = True, invalid_raise=False)
    idxs = data[0]
    sorted_data = data[np.lexsort(idxs)]

    for row in tqdm(data):
        idx = row[0]
        fap = row[31]
        cont = False

        if idx in processed_idx:
            if fap < processed_fap[idx]:
                cont = True
        else:
            cont = True

        if cont:
            processed_idx.append(idx)
            processed_fap.append(fap)
            filename = get_sub_file_name(idx)
            if filename not in old_filename:
                data = np.genfromtxt("/beegfs/car/njm/OUTPUT/split_by_id/multi/"+filename, delimiter = ',', unpack = True)
                meta_idx = data[0]
                old_filename = filename
            index_meta_idx = np.where(meta_idx == idx)[0]
            meta_row = data.T[index_meta_idx]
            output_row = list(meta_row) + list(row)[1:]
            # Write the output_row to the CSV file
            writer.writerow(output_row)

# Close the file after the loop
file.close()

