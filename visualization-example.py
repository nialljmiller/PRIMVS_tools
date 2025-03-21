import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import sys

# Import the visualization module
# Assuming you've saved the visualization code as cv_visualization.py
sys.path.append('/path/to/code')
from cv_visualization import create_all_visualizations

# Set paths


output_dir = "../PRIMVS/cv_results"
primvs_file = "../PRIMVS/cv_results/cv_candidates.fits"
known_cv_file = "../PRIMVS/PRIMVS_CC_CV.fits"

# Load CV candidates from FITS file
with fits.open(primvs_cv_file) as hdul:
    cv_candidates = Table(hdul[1].data).to_pandas()

# Load known CVs if available
try:
    with fits.open(known_cv_file) as hdul:
        known_cvs = Table(hdul[1].data).to_pandas()
except:
    known_cvs = None
    print("No known CVs file found or could not be loaded")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Generate all visualizations
create_all_visualizations(
    cv_candidates=cv_candidates,
    known_cvs=known_cvs,
    output_dir=output_dir
)

print(f"All visualizations completed. Results saved to {output_dir}")
