import pandas as pd
from astropy.table import Table, vstack, hstack
import glob
import random
import csv
import os
import tqdm as tqdm
import numpy as np
import itertools
import StochiStats as ss
import Virac
import NN_FAP
import concurrent.futures
import numpy as np
#This stuff is for recalculating FAP
from tensorflow.keras.models import model_from_json
from sklearn import neighbors
import multiprocessing

import threading  # Import the threading module
# Define a lock for synchronization
data_lock = threading.Lock()


import concurrent.futures
import pandas as pd
from tqdm import tqdm


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

knn, model = get_model()

def phaser(time, period):
    period = 1 if period == 0 else period
    time = np.array(time)
    phase = (time / period) - np.floor(time / period)
    phase = np.where(phase >= 1, phase - 1, phase)
    phase = np.where(phase <= 0, phase + 1, phase)
    return phase



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

cut_col_names = ['cut_mag_n','cut_mag_avg','cut_magerr_avg','cut_Cody_M','cut_stet_k','cut_eta','cut_eta_e',
            'cut_med_BRP','cut_range_cum_sum','cut_max_slope','cut_MAD','cut_mean_var',
            'cut_percent_amp','cut_true_amplitude','cut_roms','cut_p_to_p_var','cut_lag_auto',
            'cut_std_nxs','cut_weight_mean','cut_weight_std','cut_weight_skew',
            'cut_weight_kurt','cut_mean','cut_std','cut_skew','cut_kurt','cut_time_range','cut_fap']


meta_col_names = ['sourceid','ra','ra_error','dec','dec_error','l','b','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
		'chisq','uwe','ks_n_detections','ks_n_observations','ks_n_ambiguous','ks_n_chilt5','ks_med_mag','ks_mean_mag','ks_ivw_mean_mag',
		'ks_chilt5_ivw_mean_mag','ks_std_mag','ks_mad_mag','ks_ivw_err_mag','ks_chilt5_ivw_err_mag','z_n_observations','z_med_mag',
		'z_mean_mag','z_ivw_mean_mag','z_chilt5_ivw_mean_mag','z_std_mag','z_mad_mag','z_ivw_err_mag','z_chilt5_ivw_err_mag','y_n_detections',
		'y_n_observations','y_med_mag','y_mean_mag','y_ivw_mean_mag','y_chilt5_ivw_mean_mag','y_std_mag','y_mad_mag','y_ivw_err_mag',
		'y_chilt5_ivw_err_mag','j_n_detections','j_n_observations','j_med_mag','j_mean_mag','j_ivw_mean_mag','j_chilt5_ivw_mean_mag',
		'j_std_mag','j_mad_mag','j_ivw_err_mag','j_chilt5_ivw_err_mag','h_n_detections','h_n_observations','h_med_mag','h_mean_mag',
		'h_ivw_mean_mag','h_chilt5_ivw_mean_mag','h_std_mag','h_mad_mag','h_ivw_err_mag','h_chilt5_ivw_err_mag']		

# Define a lock for file access
file_lock = multiprocessing.Lock()


def process_csv_file(csv_fp):
    data = []
    # Open the CSV file in 'r' mode (read)
    with open(csv_fp, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        qualifying_rows1 = list(itertools.dropwhile(lambda row: float(row[31]) >= fap_cut, csv_reader))
        print('')
        for row in qualifying_rows1:
            period = float(row[29])
           # Add the combined row to the output table
            lightcurve = Virac.run_sourceid(int(row[0]))
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
            smoothed_magnitude = mag
            cut_fap = NN_FAP.inference(period, mag, time, knn, model)                        
            q1, q50, q99 = np.percentile(mag, [1, 50, 99])
            cut_mag_n = len(mag)
            cut_mag_avg = q50
            cut_magerr_avg = np.median(magerr) 
            cut_Cody_M = ss.cody_M(mag, time)
            cut_stet_k = ss.Stetson_K(mag, magerr)
            cut_eta = ss.Eta(mag, time)
            cut_eta_e = ss.Eta_e(mag,time)
            cut_med_BRP = ss.medianBRP(mag, magerr)
            cut_range_cum_sum = ss.RangeCumSum(mag)
            cut_max_slope = ss.MaxSlope(mag, time)
            cut_MAD = ss.MedianAbsDev(mag, magerr)
            cut_mean_var = ss.Meanvariance(mag)
            cut_percent_amp = ss.PercentAmplitude(mag)
            cut_true_amplitude = abs(q99-q1)
            cut_roms = ss.RoMS(mag, magerr)
            cut_p_to_p_var = ss.ptop_var(mag, magerr)
            cut_lag_auto = ss.lagauto(mag)
            #cut_AD = ss.AndersonDarling(mag)
            cut_std_nxs = ss.stdnxs(mag, magerr)
            cut_weight_mean = ss.weighted_mean(mag,magerr)
            cut_weight_std = ss.weighted_variance(mag,magerr)
            cut_weight_skew = ss.weighted_skew(mag,magerr)
            cut_weight_kurt = ss.weighted_kurtosis(mag,magerr)
            cut_mean = ss.mu(mag)
            cut_std = ss.sigma(mag)
            cut_skew = ss.skewness(mag)
            cut_kurt = ss.kurtosis(mag)
            cut_time_range = np.ptp(time)
            cut_row = [cut_mag_n,cut_mag_avg,cut_magerr_avg,cut_Cody_M,cut_stet_k,cut_eta,cut_eta_e,cut_med_BRP,cut_range_cum_sum,
                        cut_max_slope,cut_MAD,cut_mean_var,cut_percent_amp,cut_true_amplitude,cut_roms,cut_p_to_p_var,cut_lag_auto,
                        cut_std_nxs,cut_weight_mean,cut_weight_std,cut_weight_skew,cut_weight_kurt,cut_mean,cut_std,cut_skew,
                        cut_kurt,cut_time_range,cut_fap]

            data_line = [row[0]] + cut_row + row[1:]

            # Write the processed data to the output file
            with file_lock:
                with open('/beegfs/car/njm/OUTPUT/PRIMVS_periodics.csv', 'a') as output_csv:
                    csv_writer = csv.writer(output_csv)
                    csv_writer.writerow(data_line)
            '''
            if cut_fap < 0:#fap_cut:
                name_str = str(row[0])
                period_str = str('{:0>8d}'.format(int(period * 100000)))
                fap_str = str('{:0>3d}'.format(int(cut_fap * 100)))
                output_npy_fp = f'/beegfs/car/njm/PRIMVS/LC_TOKEN/LC_{name_str}_period-{period_str}_fap-{fap_str}.npy'
                if output_npy_fp not in completed_lcs:
                    # Combine the arrays into a structured numpy array
                    lightcurves = np.array(list(zip(mag, magerr, time, phase)), dtype=[('mag', float), ('magerr', float), ('time', float), ('phase', float)])
                    np.save(output_npy_fp, lightcurves)
            '''
            
    # Convert the data list to a NumPy array for efficient manipulation
    np_data = np.array(data)

    # Convert the NumPy array to a Pandas DataFrame
    df = pd.DataFrame(data=np_data)

    return df




def compute(csv_files, num_processes, output_csv):


    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_csv_file, csv_file) for csv_file in csv_files]
        for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    output_table = Table.from_pandas(combined_df)#, dtype=fits_col_types)

    # Write the table to a FITS file
    try:
        os.system('rm ' + output_csv_fp + '_old.fits')  
        os.rename(output_csv_fp + '.fits', output_csv_fp + '_old.fits')
    except:
        pass
    output_table.write(output_csv_fp + '.fits')




fits_col_types = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 	 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		  'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		   'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		    'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		    'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'str', 'float', 'str', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 		     'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']


completed_lcs = glob.glob("/beegfs/car/njm/PRIMVS/LC_TOKEN/*")
csv_meta_files = glob.glob("/beegfs/car/njm/OUTPUT/results/complete/*")
csv_files = glob.glob("/beegfs/car/njm/OUTPUT/vars/complete_lower_var/rescan_lp/*.csv") + glob.glob("/beegfs/car/njm/OUTPUT/vars/complete_lower_var/rescan_sp/*.csv") + glob.glob("/beegfs/car/njm/OUTPUT/vars/complete/rescan_lp/*.csv") + glob.glob("/beegfs/car/njm/OUTPUT/vars/complete/rescan_sp/*.csv") + glob.glob("/beegfs/car/njm/OUTPUT/vars/complete_lower_var/*.csv") + glob.glob("/beegfs/car/njm/OUTPUT/vars/complete/*.csv")
output_csv_fp = "/beegfs/car/njm/OUTPUT/PRIMVS_periodics"
num_threads = 16
fap_cut = 0.2
csv_files_sample = [csv_files[i] for i in random.sample(range(1, len(csv_files)), 42)] # make small sample

compute(csv_files, num_threads, output_csv_fp)




