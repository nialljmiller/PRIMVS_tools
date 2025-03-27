from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import glob
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
from reproject import reproject_interp
import time
from scipy.ndimage import shift
import logging
import glob

# Disable all logging messages because astropy is really fucking annoying.
logging.disable(logging.CRITICAL)

def phaser(time, period):
    # this is to mitigate against div 0
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
    def update(frame):
        lc_line = lightcurve[frame]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'
        hdu = fits.open(image_fp)[lc_line['ext']]
        wcs = WCS(hdu.header)
        data = hdu.data
        p1, p5, p50, p95, p99 = np.percentile(data, (1, 5, 50, 95, 99))
        data = data - p50
        
        # Define the desired WCS information for the reprojected image, partner
        reproj_image, reproj_footprint = reproject_interp(hdu, master_header)
        # Compute the rootin'-tootin' cross-correlation between the reprojected image and the master image
        corr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(reproj_image) * np.fft.fft2(master_data).conj()))
        dy, dx = np.unravel_index(np.argmax(corr), corr.shape)

        # Corral the reprojected image based on the cross-correlation
        data = shift(reproj_image, (dy, dx))
        position = SkyCoord(ra, dec, unit="deg")
        size = int(0.5*60 / vvv_pixel_scale)
        cutout = Cutout2D(reproj_image,position,size,wcs=master_wcs,mode='partial',fill_value=np.nan,copy=True)
        newdata = cutout.data
                
        # Define the size of the central square to exclude
        central_square_size = 10  # Adjust the size as needed
        half_size = central_square_size // 2
        center_y, center_x = newdata.shape[0] // 2, newdata.shape[1] // 2

        # Create a mask for the central square
        mask = np.ones(newdata.shape, dtype=bool)
        mask[center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size] = False

        # Calculate background level (median) excluding the central square
        background_level = np.median(newdata[mask])

        # Subtract background level from the cutout data
        newdata -= background_level
        newdata = (newdata - np.mean(newdata))/np.std(newdata)
                
        # Yeehaw! update the image plot
        image_ax.imshow(newdata, origin='lower', cmap='rainbow')#, norm=norm)
        filename = str(lc_line["filename"]).split("'")[1]
        image_ax.set_title(f'{name} - {frame+1}/{len(lightcurve)}')
        image_ax.axis('off')


        lc_ax.clear()
        lc_ax.invert_yaxis()  # Invert the y-axis, just for kicks
        lc_ax.scatter(phase, lightcurve['hfad_mag'], s=2, color='k')
        lc_ax.scatter(phase + 1, lightcurve['hfad_mag'], s=2, color='k')

        for x, y, e, colour in zip(phase, lightcurve['hfad_mag'], lightcurve['hfad_emag'], date_color):
            lc_ax.errorbar(x, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
            lc_ax.errorbar(x+1, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
            # update the light curve plot

        lc_ax.errorbar(phase[frame], lightcurve['hfad_mag'][frame], color='k', markersize = 10, marker = 'x')
        lc_ax.errorbar(phase[frame], lightcurve['hfad_mag'][frame], color='grey', markersize = 10, marker = '+')
        lc_ax.set_title(f"M = {round(float(lightcurve['hfad_mag'][frame]),3)} - P = {round(period,3)} [d] - FAP = {round(fap,2)}")


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

        phase = phaser(lightcurve['mjdobs'], period)  # Wrangle up the phases
        phase_idx = np.argsort(phase)  # Sort the phases
        lightcurve = lightcurve[phase_idx]  # Round up the lightcurve based on the sorted phases
        phase = phase[phase_idx]  # Update phases based on sorted lightcurve

        # Time to get fancy with the graphics
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))  # Creating a fine-looking figure with 2 axes
        lc_ax = ax[0]  # Yeehaw! Let's ride on the first axis
        lc_ax.set_xlabel('Phase')  # Label the x-axis with 'Phase'
        lc_ax.set_ylabel('Magnitude')  # Label the y-axis with 'Magnitude'
        lc_ax.set_title('Light Curve')  # Set the title of the light curve to 'Light Curve'


        norm = mplcol.Normalize(vmin=min(lightcurve['mjdobs']), vmax=max(lightcurve['mjdobs']), clip=True)  # Some normilzin' for the lightcurve
        mapper = cm.ScalarMappable(norm=norm, cmap='brg')  # Get a mapper for the lightcurve
        date_color = np.array(mapper.to_rgba(lightcurve['mjdobs']))  # Pick a color based on the MJDobs for each point

        image_ax = ax[1]  # Giddyup, time to move on to the second axis
        image_ax.set_title('Image')  # Set the title of the image to 'Image'

        # Let's find the 'best' image
        lc_line = lightcurve[np.where(lightcurve['ast_res_chisq'].astype(float) == min(lightcurve['ast_res_chisq'].astype(float)))[0]]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'  # Lasso up the path to the image file
        master_hdu = fits.open(image_fp)[int(lc_line['ext'])]  # Open up the master image file
        master_header = master_hdu.header  # Round up the header
        master_data = master_hdu.data  # Lasso up the image data
        master_wcs = WCS(master_hdu.header)  # Get the WCS
        p1, p5, p50, p95, p99 = np.percentile(master_data, (1, 5, 50, 95, 99))  # Compute percentiles of the image data
        master_data = master_data - p50  # Normalize the image data by subtracting its median

        anim = FuncAnimation(fig, update, frames=len(lightcurve), interval=1, repeat=True)  # Make an animation using the update function and 10 frames
        anim.save('/beegfs/car/njm/PRIMVS/LC_GIF/highdm/phase/'+str(name)+"_phase.gif", writer='imagemagick', fps=6) 





def gif_genny_mjd(name):
    def update(frame):
        lc_line = lightcurve[frame]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'
        hdu = fits.open(image_fp)[lc_line['ext']]
        wcs = WCS(hdu.header)
        data = hdu.data
        p1, p5, p50, p95, p99 = np.percentile(data, (1, 5, 50, 95, 99))
        data = data - p50

        # Define the desired WCS information for the reprojected image, partner
        reproj_image, reproj_footprint = reproject_interp(hdu, master_header)
        # Compute the rootin'-tootin' cross-correlation between the reprojected image and the master image
        corr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(reproj_image) * np.fft.fft2(master_data).conj()))
        dy, dx = np.unravel_index(np.argmax(corr), corr.shape)

        # Corral the reprojected image based on the cross-correlation
        data = shift(reproj_image, (dy, dx))
        position = SkyCoord(ra, dec, unit="deg")
        size = int(0.5*60 / vvv_pixel_scale)
        cutout = Cutout2D(reproj_image,position,size,wcs=master_wcs,mode='partial',fill_value=np.nan,copy=True)
        newdata = cutout.data
      
        # Define the size of the central square to exclude
        central_square_size = 10  # Adjust the size as needed
        half_size = central_square_size // 2
        center_y, center_x = newdata.shape[0] // 2, newdata.shape[1] // 2

        # Create a mask for the central square
        mask = np.ones(newdata.shape, dtype=bool)
        mask[center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size] = False

        # Calculate background level (median) excluding the central square
        background_level = np.median(newdata[mask])

        # Subtract background level from the cutout data
        newdata -= background_level
        newdata = (newdata - np.mean(newdata))/np.std(newdata)
               

        # Yeehaw! update the image plot
        image_ax.imshow(newdata, origin='lower', cmap='rainbow')#, norm=norm)
        filename = str(lc_line["filename"]).split("'")[1]
        image_ax.set_title(f'{name} - {frame+1}/{len(lightcurve)}')
        image_ax.axis('off')

        lc_ax.clear()
        lc_ax.invert_yaxis()  # Invert the y-axis, just for kicks
        lc_ax.scatter(mjd, lightcurve['hfad_mag'], s=2, color='k')

        for x, y, e, colour in zip(mjd, lightcurve['hfad_mag'], lightcurve['hfad_emag'], date_color):
            lc_ax.errorbar(x, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
            lc_ax.errorbar(x+1, y, yerr = e, fmt = 'o', lw=0.5, capsize=0.8, color=colour, markersize = 3)
            # update the light curve plot
        lc_ax.errorbar(mjd[frame], lightcurve['hfad_mag'][frame], color='k', markersize = 10, marker = 'x')
        lc_ax.errorbar(mjd[frame], lightcurve['hfad_mag'][frame], color='grey', markersize = 10, marker = '+')
        lc_ax.set_title(f"M = {round(float(lightcurve['hfad_mag'][frame]),3)}")


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
        mjd_idx = np.argsort(mjd)  # Sort the phases
        lightcurve = lightcurve[mjd_idx]  # Round up the lightcurve based on the sorted phases
        mjd = mjd[mjd_idx]  # Update phases based on sorted lightcurve

        # Time to get fancy with the graphics
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))  # Creating a fine-looking figure with 2 axes
        lc_ax = ax[0]  # Yeehaw! Let's ride on the first axis
        lc_ax.set_xlabel('mjd')  # Label the x-axis with 'Phase'
        lc_ax.set_ylabel('Magnitude')  # Label the y-axis with 'Magnitude'
        lc_ax.set_title('Light Curve')  # Set the title of the light curve to 'Light Curve'


        norm = mplcol.Normalize(vmin=min(lightcurve['mjdobs']), vmax=max(lightcurve['mjdobs']), clip=True)  # Some normilzin' for the lightcurve
        mapper = cm.ScalarMappable(norm=norm, cmap='brg')  # Get a mapper for the lightcurve
        date_color = np.array(mapper.to_rgba(lightcurve['mjdobs']))  # Pick a color based on the MJDobs for each point

        image_ax = ax[1]  # Giddyup, time to move on to the second axis
        image_ax.set_title('Image')  # Set the title of the image to 'Image'

        # Let's find the 'best' image
        lc_line = lightcurve[np.where(lightcurve['ast_res_chisq'].astype(float) == min(lightcurve['ast_res_chisq'].astype(float)))[0]]
        image_fp = fits_fp + str(lc_line['filename']).split("'")[1] + '.fit'  # Lasso up the path to the image file
        master_hdu = fits.open(image_fp)[int(lc_line['ext'])]  # Open up the master image file
        master_header = master_hdu.header  # Round up the header
        master_data = master_hdu.data  # Lasso up the image data
        master_wcs = WCS(master_hdu.header)  # Get the WCS
        p1, p5, p50, p95, p99 = np.percentile(master_data, (1, 5, 50, 95, 99))  # Compute percentiles of the image data
        master_data = master_data - p50  # Normalize the image data by subtracting its median

        anim = FuncAnimation(fig, update, frames=len(lightcurve), interval=1, repeat=True)  # Make an animation using the update function and 10 frames
        anim.save('/beegfs/car/njm/PRIMVS/LC_GIF/highdm/mjd/'+str(name)+"_mjd.gif", writer='imagemagick', fps=6) 
    else:
        print(lightcurve['hfad_mag'].max() - lightcurve['hfad_mag'].min())




#python PRIMVS_gif_remp.py 0 & python PRIMVS_gif_remp.py 1 & python PRIMVS_gif_remp.py 2 & python PRIMVS_gif_remp.py 3 & python PRIMVS_gif_remp.py 4 & python PRIMVS_gif_remp.py 5 & python PRIMVS_gif_remp.py 6 & python PRIMVS_gif_remp.py 7 & 


version = int(sys.argv[1])

primus_fp = '/beegfs/car/njm/OUTPUT/PRIMVS_03.fits'

PRIMVS_hdu = fits.open(primus_fp)  # Opening that there FITS file


# Calculate the threshold value for the top 10%


mask = PRIMVS_hdu[1].data['true_amplitude'] > 2.
newdata = PRIMVS_hdu[1].data[mask]

name_list = np.array_split(newdata['sourceid'],8)[version]
print('Version:',version)
print(name_list)


for sysarg in name_list:

    print("Hold on tight, we're startin' to make a phase folded GIF for star ", sysarg)
    # Yeehaw, let's load ourselves some data!
    fits_fp = '/beegfs/car/lsmith/vvv_data/images/'  # File path for them FITS images
    vvv_pixel_scale = 0.339  # The pixel scale in arcseconds per pixel

    try:
        # Alright cowboy, let's rustle up a random index if the arg is 1, otherwise we'll use the arg as the index
        index = list(PRIMVS_hdu[1].data['sourceid']).index(sysarg)
        # Yippee ki-yay, let's get down to business and generate that GIF!
        primus_line = PRIMVS_hdu[1].data[index]  # Gettin' the data for the selected index
        name = primus_line['sourceid']  # Grabbin' the source ID
        fap = primus_line['best_fap']  # Gettin' the best false alarm probability
        period = primus_line['true_period']  # Grabbin' the true period
        amplitude = primus_line['true_amplitude']  # Grabbin' the true period
        N = primus_line['mag_n']  # Fetchin' the number of magnitudes
        print("Found that star we been chasin' in PRIMVS. Got the secrets right here. Gather 'round, cowboys, let me show ya what this catalogue has to say...")
        print('FAP : ', fap)  # Displayin' that there false alarm probability
        print('Period : ', period)  # Showin' off the true period
        print('Amplitude : ', amplitude)  # Showin' off the true period
        print('N : ', N)  # Tellin' ya the number of magnitudes
        gif_genny_phase(primus_line)  # Time to wrangle that phase-folded GIF
        print("Whoop! We done made a GIF of the phase fold. Now, let's make the same GIF sorted by MJD...")
    except Exception as e:  # Whoa now, if there's an exception, we'll handle it like a true cowboy
        print(repr(e))  # Displayin' the exception message like a brand on a steer
        print("Hold your horses! The star ain't found in the PRIMVS catalogue, but we'll still try to make a mjd GIF")


    try:
        #gif_genny_mjd(sysarg)  # Let's round up that MJD GIF, partner
        print("Listen up, folks! Gather 'round, for I've got news to share. Them light curve gifs for star " + str(sysarg) + "? A triumph beyond measure! Stored safe 'n sound at '/beegfs/car/njm/PRIMVS/LC_GIF/'. So gather your spirits, embrace the darkness, and witness the cosmic ballet unfold. Find 'em there, in that vast digital realm.") 
    except Exception as e:  # If there's another exception, we ain't lettin' it break our stride
        print(repr(e))  # Displayin' the exception message like a tumbleweed in the wind
        print("We tried, but the creation of them gifs, it didn't work. Our efforts fell short, lost amidst the unforgiving currents of this forsaken realm.")
        print("You reckon this star [" + str(sysarg) +"] truly exists out there in the boundless cosmos? Or is it nothin' but a specter, taunting us from the black void?")
        #print("The Judge's voice resonated, his tone as cold and unforgiving as the desolate plains they traversed. Listen well, kid, he spoke, his eyes gleaming with an unsettling fire. Our pursuit to create a phase gif and an MJD gif has faltered. The hands of fate have dealt their cruel hand, denying us the triumph we sought. He continued, his words carving through the air like a sharpened blade. In this vast and merciless universe, we have been bested. The cosmic secrets have eluded our grasp, slipping away like smoke in the wind. No amount of grit or sweat can mend this broken path. The Kid stood in silence, his spirit battered by the Judge's grim proclamation. The weight of failure settled upon their shoulders, heavy as the burden of their bloody journey. In that moment, they knew that some truths were destined to remain hidden, forever beyond their reach. The Judge's gaze pierced through the Kid, his voice an eerie whisper. Accept this defeat, young one, for it is the nature of our existence. The cosmic dance mocks our mortal desires, reminding us of our insignificance. Ride on, amidst the darkness and uncertainty, for there are yet mysteries that shall forever elude our grasp. And so, the Kid absorbed the Judges words, the bitter taste of failure lingering in their mouth. They rode onward, haunted by the knowledge that even the most relentless pursuit can end in naught but shadows and silence.")
        print("\n\nIt did not work")

