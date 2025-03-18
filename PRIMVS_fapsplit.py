from astropy.table import Table

# Load the FITS file

fits_fp = "/beegfs/car/njm/OUTPUT/PRIMVS_RAW"
catalog = Table.read(fits_fp+".fits")

# Splitting the data based on "best_fap"
fap_less_than_03 = catalog[catalog['best_fap'] < 0.3]
fap_between_03_and_06 = catalog[(catalog['best_fap'] >= 0.3) & (catalog['best_fap'] < 0.6)]
fap_greater_than_06 = catalog[catalog['best_fap'] > 0.6]

# Save the split data into three separate FITS files
fap_less_than_03.write(fits_fp + "_P.fits", overwrite=True)
fap_between_03_and_06.write(fits_fp + "_QP.fits", overwrite=True)
fap_greater_than_06.write(fits_fp + "_AP.fits", overwrite=True)

