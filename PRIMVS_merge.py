from astropy.io import fits
import pandas as pd
import numpy as np
import glob
from multiprocessing import Pool, cpu_count

# Path to the FITS files
data_path = "/beegfs/car/njm/OUTPUT/"
master_fits = data_path + "PRIMVS_P.fits"  # Master file
fits_files = glob.glob(data_path + "PRIMVS_*.fits")  # All FITS files
fits_files.remove(master_fits)  # Remove master file from the list

# Read Master FITS
with fits.open(master_fits, memmap=True) as hdul:
    master_data = hdul[1].data
    master_df = pd.DataFrame(np.array(master_data))

# Function to process a FITS file
def process_fits(fits_file):
    with fits.open(fits_file, memmap=True) as hdul:
        print(f"Checking {fits_file}")
        data = hdul[1].data  # Assuming data is in the first extension
        df = pd.DataFrame(np.array(data))  # Convert FITS table to DataFrame
        
        # Ensure structure is compatible
        if set(df.columns) == set(master_df.columns):
            print(f"Appending {fits_file}")
            return df
        else:
            print(f"Skipping {fits_file} due to structure mismatch")
            return None

# Utilize multiprocessing to read FITS files in parallel
num_cores = min(96, cpu_count())  # Use up to 96 cores
with Pool(num_cores) as pool:
    results = pool.map(process_fits, fits_files)

# Filter out None results and merge
data_frames = [master_df] + [df for df in results if df is not None]
merged_df = pd.concat(data_frames, ignore_index=True)

# Deduplicate based on 'sourceid' or 'uniqueid'
id_col = 'sourceid' if 'sourceid' in merged_df.columns else 'uniqueid'
merged_df = merged_df.drop_duplicates(subset=id_col)

# Convert back to FITS
cols = [fits.Column(name=col, format='E', array=merged_df[col].values) for col in merged_df.columns]
hdu = fits.BinTableHDU.from_columns(cols)

# Save the unified FITS file
output_fits = data_path + "PRIMVS.fits"
hdu.writeto(output_fits, overwrite=True)

print(f"Merged FITS file saved to {output_fits}")