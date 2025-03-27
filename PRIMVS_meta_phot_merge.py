import pandas as pd
import numpy as np
from astropy.table import Table

def infer_and_apply_dtypes(df):
    for col in df.columns:
        col_data = df[col]

        if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            if len(col_data.unique()) / len(col_data) < 0.5:
                df[col] = col_data.astype('category')
            continue

        col_numeric = pd.to_numeric(col_data, errors='coerce')

        if not pd.isnull(col_numeric).all():
            min_val, max_val = col_numeric.min(), col_numeric.max()

            if pd.api.types.is_float_dtype(col_numeric) and (col_numeric % 1 == 0).all():
                if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    df[col] = col_numeric.astype(np.int8)
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    df[col] = col_numeric.astype(np.int16)
                elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    df[col] = col_numeric.astype(np.int32)
                else:
                    df[col] = col_numeric.astype(np.int64)
            else:
                if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
                    df[col] = col_numeric.astype(np.float32)
                else:
                    df[col] = col_numeric.astype(np.float64)
    return df



file_path1 = '/beegfs/car/njm/OUTPUT/PRIMVS_03.csv'
file_path2 = '/beegfs/car/njm/OUTPUT/VIRAC_PRIMVS_03.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

merged_df = pd.merge(df1, df2, on='sourceid')


# Loop through each column to check if it ends with '_med_mag'
for column in merged_df.columns:
    if column.endswith('_med_mag') and column != 'ks_med_mag':
        new_col_name = f"{column}-ks_med_mag"
        merged_df[new_col_name] = merged_df[column] - merged_df['ks_med_mag']

    if column.endswith('_mean_mag') and column != 'ks_mean_mag':
        new_col_name = f"{column}-ks_mean_mag"
        merged_df[new_col_name] = merged_df[column] - merged_df['ks_mean_mag']

    if column.endswith('_mad_mag') and column != 'ks_mad_mag':
        new_col_name = f"{column}-ks_mad_mag"
        merged_df[new_col_name] = merged_df[column] - merged_df['ks_mad_mag']


new_column_order = ['uniqueid', 'sourceid', 'ra', 'ra_error', 'dec', 'dec_error', 'l', 'b', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'chisq', 'uwe', 'ks_n_detections', 'ks_n_observations', 'ks_n_ambiguous', 'ks_n_chilt5', 'ks_med_mag', 'ks_mean_mag', 'ks_ivw_mean_mag', 'ks_chilt5_ivw_mean_mag', 'ks_std_mag', 'ks_mad_mag', 'ks_ivw_err_mag', 'ks_chilt5_ivw_err_mag', 'z_n_observations', 'z_med_mag', 'z_mean_mag', 'z_ivw_mean_mag', 'z_chilt5_ivw_mean_mag', 'z_std_mag', 'z_mad_mag', 'z_ivw_err_mag', 'z_chilt5_ivw_err_mag', 'y_n_detections', 'y_n_observations', 'y_med_mag', 'y_mean_mag', 'y_ivw_mean_mag', 'y_chilt5_ivw_mean_mag', 'y_std_mag', 'y_mad_mag', 'y_ivw_err_mag', 'y_chilt5_ivw_err_mag', 'j_n_detections', 'j_n_observations', 'j_med_mag', 'j_mean_mag', 'j_ivw_mean_mag', 'j_chilt5_ivw_mean_mag', 'j_std_mag', 'j_mad_mag', 'j_ivw_err_mag', 'j_chilt5_ivw_err_mag', 'h_n_detections', 'h_n_observations', 'h_med_mag', 'h_mean_mag', 'h_ivw_mean_mag', 'h_chilt5_ivw_mean_mag', 'h_std_mag', 'h_mad_mag', 'h_ivw_err_mag', 'h_chilt5_ivw_err_mag', 'mag_n', 'mag_avg', 'magerr_avg', 'Cody_M', 'stet_k', 'eta', 'eta_e', 'med_BRP', 'range_cum_sum', 'max_slope', 'MAD', 'mean_var', 'percent_amp', 'true_amplitude', 'roms', 'p_to_p_var', 'lag_auto', 'AD', 'std_nxs', 'weight_mean', 'weight_std', 'weight_skew', 'weight_kurt', 'mean', 'std', 'skew', 'kurt', 'time_range', 'true_period', 'true_class', 'best_fap', 'best_method', 'trans_flag', 'ls_p', 'ls_y_y_0', 'ls_peak_width_0', 'ls_period1', 'ls_y_y_1', 'ls_peak_width_1', 'ls_period2', 'ls_y_y_2', 'ls_peak_width_2', 'ls_q001', 'ls_q01', 'ls_q1', 'ls_q25', 'ls_q50', 'ls_q75', 'ls_q99', 'ls_q999', 'ls_q9999', 'ls_fap', 'ls_bal_fap', 'Cody_Q_ls', 'pdm_p', 'pdm_y_y_0', 'pdm_peak_width_0', 'pdm_period1', 'pdm_y_y_1', 'pdm_peak_width_1', 'pdm_period2', 'pdm_y_y_2', 'pdm_peak_width_2', 'pdm_q001', 'pdm_q01', 'pdm_q1', 'pdm_q25', 'pdm_q50', 'pdm_q75', 'pdm_q99', 'pdm_q999', 'pdm_q9999', 'pdm_fap', 'Cody_Q_pdm', 'ce_p', 'ce_y_y_0', 'ce_peak_width_0', 'ce_period1', 'ce_y_y_1', 'ce_peak_width_1', 'ce_period2', 'ce_y_y_2', 'ce_peak_width_2', 'ce_q001', 'ce_q01', 'ce_q1', 'ce_q25', 'ce_q50', 'ce_q75', 'ce_q99', 'ce_q999', 'ce_q9999', 'ce_fap', 'Cody_Q_ce', 'gp_lnlike', 'gp_b', 'gp_c', 'gp_p', 'gp_fap', 'Cody_Q_gp']

final_df = merged_df[new_column_order]

final_df.to_csv('/beegfs/car/njm/OUTPUT/PRIMVS_P.csv', index=False)



# Adding the new column names to the original new_column_order list just after 'uwe'
new_columns_after_uwe = [
    'z_med_mag-ks_med_mag', 'z_mean_mag-ks_mean_mag', 'z_mad_mag-ks_mad_mag', 
    'y_med_mag-ks_med_mag', 'y_mean_mag-ks_mean_mag', 'y_mad_mag-ks_mad_mag', 
    'j_med_mag-ks_med_mag', 'j_mean_mag-ks_mean_mag', 'j_mad_mag-ks_mad_mag', 
    'h_med_mag-ks_med_mag', 'h_mean_mag-ks_mean_mag', 'h_mad_mag-ks_mad_mag'
]

# Find the index of 'uwe' to insert the new columns after it
uwe_index = new_column_order.index('uwe') + 1

# Insert the new columns into the new_column_order list
new_column_order = new_column_order[:uwe_index] + new_columns_after_uwe + new_column_order[uwe_index:]

final_df = merged_df[new_column_order]

final_df.to_csv('/beegfs/car/njm/OUTPUT/PRIMVS_P.csv', index=False)

output_table = Table.from_pandas(final_df)
output_table.write('/beegfs/car/njm/OUTPUT/PRIMVS_P.fits',overwrite=True)






