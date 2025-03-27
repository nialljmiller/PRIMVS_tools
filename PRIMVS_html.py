import pandas as pd
from astropy.table import Table



col_names = ['unique_id', 'sourceid', 'period_id',
		'mag_n','mag_avg','magerr_avg','Cody_M','stet_k','eta','eta_e','med_BRP',
		'range_cum_sum','max_slope','MAD','mean_var','percent_amp','true_amplitude','roms','p_to_p_var',
		'lag_auto','AD','std_nxs','weight_mean','weight_std','weight_skew','weight_kurt','mean','std','skew',
		'kurt','time_range','true_period','best_fap','trans_flag','ls_p','ls_fap','ls_bal_fap',
		'Cody_Q_ls','pdm_p','pdm_fap','Cody_Q_pdm']





# Assuming 'fits_file.fits' is your FITS file path
fits_data = Table.read('/beegfs/car/njm/OUTPUT/PRIMVS_P01.fits')

# Convert Astropy table to Pandas DataFrame
df = fits_data.to_pandas()
df = df[col_names]
# Assuming your DataFrame is named 'df'
# Convert DataFrame to CSV
#df.to_csv('astro_catalog.csv', index=False)

# Generating HTML table
with open('/beegfs/car/njm/PRIMVS/PRIMVS.html', 'w') as f:
    f.write('<!DOCTYPE html>\n')
    f.write('<html>\n')
    f.write('<head>\n')
    f.write('<title>Astro Catalog</title>\n')
    f.write('</head>\n')
    f.write('<body>\n')
    f.write('<h1>Astronomy Catalog</h1>\n')
    f.write('<table border="1">\n')
    
    # Writing header row
    f.write('<tr>\n')
    for col in df.columns:
        f.write(f'<th>{col}</th>\n')
    f.write('</tr>\n')
    
    # Writing data rows
    for index, row in df.iterrows():
        f.write('<tr>\n')
        for value in row:
            f.write(f'<td>{value}</td>\n')
        f.write('</tr>\n')
    
    f.write('</table>\n')
    f.write('</body>\n')
    f.write('</html>\n')

