from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import glob
import pandas as pd
import Virac as Virac
import sys
import random
from PIL import Image
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from reproject import reproject_interp
import time
from scipy.ndimage import shift
import logging
from astropy.table import Table
import os

# Disable all logging messages because astropy is really annoying
logging.disable(logging.CRITICAL)

# TESS pixel size in arcseconds (21 arcseconds)
TESS_PIXEL_SIZE = 21.0

def phaser(time, period):
    """Calculate phase values from time series and period."""
    # This is to mitigate against div 0
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


def gif_genny_phase(primus_line):
    """Generate a phase-folded GIF for a given source."""
    def update(frame):
        lc_line = lightcurve[frame]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'
        hdu = fits.open(image_fp)[lc_line['ext']]
        wcs = WCS(hdu.header)
        data = hdu.data
        p1, p5, p50, p95, p99 = np.percentile(data, (1, 5, 50, 95, 99))
        data = data - p50

        # Define the desired WCS information for the reprojected image
        reproj_image, reproj_footprint = reproject_interp(hdu, master_header)
        
        # Compute the cross-correlation between the reprojected image and the master image
        corr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(reproj_image) * np.fft.fft2(master_data).conj()))
        dy, dx = np.unravel_index(np.argmax(corr), corr.shape)

        # Shift the reprojected image based on the cross-correlation
        data = shift(reproj_image, (dy, dx))
        position = SkyCoord(ra, dec, unit="deg")
        
        # Use 4x TESS pixel size as the cutout size
        size = int(TESS_PIXEL_SIZE * 4 / vvv_pixel_scale)
        cutout = Cutout2D(reproj_image, position, size, wcs=master_wcs, mode='partial', fill_value=np.nan, copy=True)
        newdata = cutout.data
        newdata = (newdata - np.mean(newdata))/np.std(newdata)

        # Update the image plot
        image_ax.clear()
        image_ax.imshow(newdata, origin='lower', cmap='rainbow')
        
        # Draw TESS pixel overlay
        tess_pixel_size_vvv = TESS_PIXEL_SIZE / vvv_pixel_scale  # Convert TESS pixel size to VVV pixels
        center_x = size // 2
        center_y = size // 2
        
        # Create a TESS pixel centered on the target
        rect = Rectangle((center_x - tess_pixel_size_vvv/2, center_y - tess_pixel_size_vvv/2), 
                         tess_pixel_size_vvv, tess_pixel_size_vvv, 
                         linewidth=2, edgecolor='white', facecolor='none', alpha=0.8)
        image_ax.add_patch(rect)
        
        # Add a scale indicator
        image_ax.text(5, 5, f"TESS pixel: {TESS_PIXEL_SIZE}\"", color='white', fontsize=10,
                      bbox=dict(facecolor='black', alpha=0.5))
        
        filename = str(lc_line["filename"]).split("'")[1]
        image_ax.set_title(f'{name} - {frame+1}/{len(lightcurve)}')
        image_ax.axis('off')

        # Update the light curve plot
        lc_ax.clear()
        lc_ax.invert_yaxis()  # Invert the y-axis
        lc_ax.scatter(phase, lightcurve['hfad_mag'], s=2, color='k')
        lc_ax.scatter(phase + 1, lightcurve['hfad_mag'], s=2, color='k')

        for x, y, e, colour in zip(phase, lightcurve['hfad_mag'], lightcurve['hfad_emag'], date_color):
            lc_ax.errorbar(x, y, yerr=e, fmt='o', lw=0.5, capsize=0.8, color=colour, markersize=3)
            lc_ax.errorbar(x+1, y, yerr=e, fmt='o', lw=0.5, capsize=0.8, color=colour, markersize=3)

        lc_ax.errorbar(phase[frame], lightcurve['hfad_mag'][frame], color='k', markersize=10, marker='x')
        lc_ax.errorbar(phase[frame], lightcurve['hfad_mag'][frame], color='grey', markersize=10, marker='+')
        lc_ax.set_title(f"M = {round(float(lightcurve['hfad_mag'][frame]),3)} - P = {round(period,3)} [d] - FAP = {round(fap,2)}")
        lc_ax.set_xlabel('Phase')
        lc_ax.set_ylabel('Magnitude')

    name = primus_line['sourceid']
    fap = primus_line['best_fap']
    period = primus_line['true_period']
    ra = primus_line['ra']
    dec = primus_line['dec']
    amplitude = primus_line['true_amplitude']

    lightcurve = Virac.run_sourceid(int(name))
    lightcurve = lightcurve[np.where(lightcurve['filter'].astype(str) == 'Ks')[0]]
    lightcurve = lightcurve[np.where(lightcurve['hfad_mag'].astype(float) > 0)[0]]
    lightcurve = lightcurve[np.where(lightcurve['hfad_emag'].astype(float) > 0)[0]]
    lightcurve = lightcurve[np.where(lightcurve['ast_res_chisq'].astype(float) < 100)[0]]
    lightcurve = lightcurve[np.where(lightcurve['chi'].astype(float) < 10)[0]]
    lightcurve = lightcurve[np.where(lightcurve['seeing'].astype(float) < 1)[0]]
    lightcurve = lightcurve[np.where(lightcurve['hfad_emag'].astype(float) < 0.15)[0]]
 
    if len(lightcurve['mjdobs']) > 10:
        phase = phaser(lightcurve['mjdobs'], period)  # Calculate phases
        phase_idx = np.argsort(phase)  # Sort the phases
        lightcurve = lightcurve[phase_idx]  # Sort lightcurve based on the sorted phases
        phase = phase[phase_idx]  # Update phases based on sorted lightcurve

        # Create figure with 2 subplots
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))  # Slightly larger figure
        lc_ax = ax[0]  # Light curve axis
        lc_ax.set_xlabel('Phase')
        lc_ax.set_ylabel('Magnitude')
        lc_ax.set_title('Light Curve')

        # Color mapping for time
        norm = mplcol.Normalize(vmin=min(lightcurve['mjdobs']), vmax=max(lightcurve['mjdobs']), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='brg')
        date_color = np.array(mapper.to_rgba(lightcurve['mjdobs']))

        image_ax = ax[1]  # Image axis
        image_ax.set_title('Image')

        # Find the 'best' image (lowest astrometric residual chi-squared)
        lc_line = lightcurve[np.where(lightcurve['ast_res_chisq'].astype(float) == min(lightcurve['ast_res_chisq'].astype(float)))[0]]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'
        master_hdu = fits.open(image_fp)[int(lc_line['ext'])]
        master_header = master_hdu.header
        master_data = master_hdu.data
        master_wcs = WCS(master_hdu.header)
        p1, p5, p50, p95, p99 = np.percentile(master_data, (1, 5, 50, 95, 99))
        master_data = master_data - p50  # Normalize the image data by subtracting its median

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(lightcurve), interval=1, repeat=True)

        # Define the output directory and file path
        base_dir = '/beegfs/car/njm/PRIMVS/cv_results/target_list_gif/'
        file_path = os.path.join(base_dir, f"{str(name)}_phase.gif")

        # Save the animation
        anim.save(file_path, writer='imagemagick', fps=6)


def gif_genny_mjd(name):
    """Generate a GIF sorted by MJD for a given source."""
    def update(frame):
        lc_line = lightcurve[frame]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'
        hdu = fits.open(image_fp)[lc_line['ext']]
        wcs = WCS(hdu.header)
        data = hdu.data
        p1, p5, p50, p95, p99 = np.percentile(data, (1, 5, 50, 95, 99))
        data = data - p50

        # Define the desired WCS information for the reprojected image
        reproj_image, reproj_footprint = reproject_interp(hdu, master_header)
        
        # Compute the cross-correlation between the reprojected image and the master image
        corr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(reproj_image) * np.fft.fft2(master_data).conj()))
        dy, dx = np.unravel_index(np.argmax(corr), corr.shape)

        # Shift the reprojected image based on the cross-correlation
        data = shift(reproj_image, (dy, dx))
        position = SkyCoord(ra, dec, unit="deg")
        
        # Use 4x TESS pixel size as the cutout size
        size = int(TESS_PIXEL_SIZE * 4 / vvv_pixel_scale)
        cutout = Cutout2D(reproj_image, position, size, wcs=master_wcs, mode='partial', fill_value=np.nan, copy=True)
        newdata = cutout.data
        newdata = (newdata - np.mean(newdata))/np.std(newdata)

        # Update the image plot
        image_ax.clear()
        image_ax.imshow(newdata, origin='lower', cmap='rainbow')
        
        # Draw TESS pixel overlay
        tess_pixel_size_vvv = TESS_PIXEL_SIZE / vvv_pixel_scale  # Convert TESS pixel size to VVV pixels
        center_x = size // 2
        center_y = size // 2
        
        # Create a TESS pixel centered on the target
        rect = Rectangle((center_x - tess_pixel_size_vvv/2, center_y - tess_pixel_size_vvv/2), 
                         tess_pixel_size_vvv, tess_pixel_size_vvv, 
                         linewidth=2, edgecolor='white', facecolor='none', alpha=0.8)
        image_ax.add_patch(rect)
        
        # Add a scale indicator
        image_ax.text(5, 5, f"TESS pixel: {TESS_PIXEL_SIZE}\"", color='white', fontsize=10,
                      bbox=dict(facecolor='black', alpha=0.5))
        
        filename = str(lc_line["filename"]).split("'")[1]
        image_ax.set_title(f'{name} - {frame+1}/{len(lightcurve)}')
        image_ax.axis('off')

        # Update the light curve plot
        lc_ax.clear()
        lc_ax.invert_yaxis()  # Invert the y-axis
        lc_ax.scatter(mjd, lightcurve['hfad_mag'], s=2, color='k')

        for x, y, e, colour in zip(mjd, lightcurve['hfad_mag'], lightcurve['hfad_emag'], date_color):
            lc_ax.errorbar(x, y, yerr=e, fmt='o', lw=0.5, capsize=0.8, color=colour, markersize=3)

        lc_ax.errorbar(mjd[frame], lightcurve['hfad_mag'][frame], color='k', markersize=10, marker='x')
        lc_ax.errorbar(mjd[frame], lightcurve['hfad_mag'][frame], color='grey', markersize=10, marker='+')
        lc_ax.set_title(f"M = {round(float(lightcurve['hfad_mag'][frame]),3)}")
        lc_ax.set_xlabel('MJD')
        lc_ax.set_ylabel('Magnitude')

    lightcurve = Virac.run_sourceid(int(name))
    lightcurve = lightcurve[np.where(lightcurve['filter'].astype(str) == 'Ks')[0]]
    lightcurve = lightcurve[np.where(lightcurve['hfad_mag'].astype(float) > 0)[0]]
    lightcurve = lightcurve[np.where(lightcurve['hfad_emag'].astype(float) > 0)[0]]
    lightcurve = lightcurve[np.where(lightcurve['ast_res_chisq'].astype(float) < 100)[0]]
    lightcurve = lightcurve[np.where(lightcurve['chi'].astype(float) < 10)[0]]
    lightcurve = lightcurve[np.where(lightcurve['seeing'].astype(float) < 1)[0]]
    lightcurve = lightcurve[np.where(lightcurve['hfad_emag'].astype(float) < 0.15)[0]]
    ra, dec = Virac.run_get_coords(int(name))
 
    if len(lightcurve['mjdobs']) > 10:
        mjd = lightcurve['mjdobs']
        mjd_idx = np.argsort(mjd)  # Sort by MJD
        lightcurve = lightcurve[mjd_idx]  # Sort lightcurve by MJD
        mjd = mjd[mjd_idx]  # Update MJD based on sorted lightcurve

        # Create figure with 2 subplots
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))  # Slightly larger figure
        lc_ax = ax[0]  # Light curve axis
        lc_ax.set_xlabel('MJD')
        lc_ax.set_ylabel('Magnitude')
        lc_ax.set_title('Light Curve')

        # Color mapping for time
        norm = mplcol.Normalize(vmin=min(lightcurve['mjdobs']), vmax=max(lightcurve['mjdobs']), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='brg')
        date_color = np.array(mapper.to_rgba(lightcurve['mjdobs']))

        image_ax = ax[1]  # Image axis
        image_ax.set_title('Image')

        # Find the 'best' image (lowest astrometric residual chi-squared)
        lc_line = lightcurve[np.where(lightcurve['ast_res_chisq'].astype(float) == min(lightcurve['ast_res_chisq'].astype(float)))[0]]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'
        master_hdu = fits.open(image_fp)[int(lc_line['ext'])]
        master_header = master_hdu.header
        master_data = master_hdu.data
        master_wcs = WCS(master_hdu.header)
        p1, p5, p50, p95, p99 = np.percentile(master_data, (1, 5, 50, 95, 99))
        master_data = master_data - p50  # Normalize the image data by subtracting its median

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(lightcurve), interval=1, repeat=True)

        # Define the output directory and file path
        base_dir = '/beegfs/car/njm/PRIMVS/cv_results/target_list_gif/'
        file_path = os.path.join(base_dir, f"{str(name)}_mjd.gif")
        
        # Save the animation
        anim.save(file_path, writer='imagemagick', fps=6)


def read_fits_data(fits_file):
    """Read data from a FITS file and convert to a pandas DataFrame."""
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        df = Table(data).to_pandas()  # Convert to a pandas DataFrame
        for column in df.columns:
            if df[column].dtype.byteorder == '>':  # Big-endian
                df[column] = df[column].byteswap().newbyteorder()
    return df


# Main execution block
if len(sys.argv) >= 2:
    sysarg = int(sys.argv[1])
    print("Starting to make a phase folded GIF for star", sysarg)
    
    # Set file paths
    primus_fp = '/beegfs/car/njm/OUTPUT/PRIMVS.fits'  # PRIMVS data file path
    fits_fp = '/beegfs/car/lsmith/vvv_data/images/'  # FITS images file path
    vvv_pixel_scale = 0.339  # VVV pixel scale in arcseconds per pixel

    try:
        PRIMVS_hdu = fits.open(primus_fp)  # Open the FITS file
        # Find the index of the source
        print(PRIMVS_hdu[1].data['sourceid'])
        index = list(PRIMVS_hdu[1].data['sourceid']).index(sysarg)
        
        # Get source data
        primus_line = PRIMVS_hdu[1].data[index]
        name = primus_line['sourceid']
        fap = primus_line['best_fap']
        period = primus_line['true_period']
        amplitude = primus_line['true_amplitude']
        N = primus_line['mag_n']
        
        print("Found star in PRIMVS catalog with the following properties:")
        print('FAP:', fap)
        print('Period:', period)
        print('Amplitude:', amplitude)
        print('N:', N)
        
        # Generate phase-folded GIF
        gif_genny_phase(primus_line)
        print("Created phase-folded GIF. Now creating MJD-sorted GIF...")
    except Exception as e:
        print(repr(e))
        print("Star not found in the PRIMVS catalog, but will still try to make an MJD GIF")

    try:
        # Generate MJD-sorted GIF
        gif_genny_mjd(sysarg)
        print(f"Light curve GIFs for star {sysarg} successfully created and stored at '/beegfs/car/njm/PRIMVS/cv_results/target_list_gif/'")
    except Exception as e:
        print(repr(e))
        print(f"Failed to create GIFs for star {sysarg}")
        print("\n\nIt did not work")
        exit()
else:
    # Process list of sources from a file
    vvv_pixel_scale = 0.339  # VVV pixel scale in arcseconds per pixel
    fits_fp = '/beegfs/car/lsmith/vvv_data/images/'  # FITS images file path
    fits_file = 'PRIMVS_P_CLASS_GAIAnew'
    output_fp = '/beegfs/car/njm/PRIMVS/cv_results/target_list_gif/'
    csv_file_path = '/beegfs/car/njm/PRIMVS/cv_results/tess_crossmatch_results/tess_big_targets.csv'
    df = pd.read_csv(csv_file_path)

    sampled_file_path = output_fp + fits_file + '_sampled.csv'

    for index, primus_line in df.iterrows():
        print(primus_line)
        source_id = primus_line['sourceid']

        name = primus_line['sourceid']
        fap = primus_line['best_fap']
        period = primus_line['true_period']
        amplitude = primus_line['true_amplitude']
        N = primus_line['mag_n']
        
        print("Processing star from CSV with the following properties:")
        print('FAP:', fap)
        print('Period:', period)
        print('Amplitude:', amplitude)
        print('N:', N)
        
        # Generate GIFs
        gif_genny_phase(primus_line)
        print("Created phase-folded GIF. Now creating MJD-sorted GIF...")
        
        gif_genny_mjd(source_id)
        print(f"Light curve GIFs for star {source_id} successfully created and stored at '/beegfs/car/njm/PRIMVS/cv_results/target_list_gif/'")