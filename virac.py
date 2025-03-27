# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
import os
import os.path
import random
import argparse
import sys
import gc;gc.enable()
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io.fits import writeto
from astropy.io import fits
import h5py
import healpy as hp
#import Synth_LC as Synth_LC
from time import time as timer
lc_meta_dir = "/beegfs/car/lsmith/virac_v2/data/output/agg_tables/"
lc_file_dir = "/beegfs/car/lsmith/virac_v2/data/output/ts_tables/"
lc_FITS_dir = "/home/njm/FITS/"


############################################################
#==========================================================#
############################################################


def LC_retrieve(LC_source_dir, current_sourceid):

    fits_fp = LC_source_dir+str(int(current_sourceid))+'.FITS'
    try:
        light_curve = fits_open(fits_fp)

    except Exception as e:
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(fits_fp)
        print(repr(e))
        exit()
        return None

    return light_curve










def Real_LC(TOOL,cat_dir):

    #sourceid, cat_ra, cat_dec, cat_period, cat_amplitude, cat_magnitude, cat_type = np.genfromtxt(LC_source_dir+set_name+'.META', dtype = 'float', converters = None, unpack = True, comments = '#', delimiter = ",")
    sourceid, cat_ra, cat_dec, cat_period, cat_amplitude, cat_magnitude, cat_type_nan = np.genfromtxt(cat_dir+'.META', dtype = 'float', converters = None, unpack = True, comments = '#', delimiter = ",")
    cat_type = np.genfromtxt(cat_dir+'.META', dtype = 'str', converters = None, unpack = True, comments = '#', delimiter = ",", usecols=(-1))

    for i,current_sourceid in enumerate(sourceid):

        fits_fp = TOOL.LC_source_dir+str(int(current_sourceid))+'.FITS'
        light_curve = fits_open(fits_fp)

        if len(light_curve['Ks_sourceid']) > 40 and light_curve['Ks_sourceid'][0] == current_sourceid: 
            try:
                light_curve['cat_ra'] = cat_ra[i]
                light_curve['cat_dec'] = cat_dec[i]
                light_curve['cat_mag'] = cat_magnitude[i]
                light_curve['cat_type'] = cat_type[i]
                light_curve['cat_period'] = cat_period[i]
                if cat_amplitude[i] == light_curve['cat_dec'] or cat_amplitude[i] == light_curve['cat_ra']:
                    light_curve['cat_amplitude'] = 0
                else:
                    light_curve['cat_amplitude'] = cat_amplitude[i]


                print('Given Period:', cat_period[i])
                print('Given Type:', cat_type[i])
                return light_curve
            except Exception as e:
                print(repr(e))
                TOOL.counter()
        else:
            TOOL.counter()
            print("Skipped because not enough dp")
            print("N:",len(light_curve['Ks_sourceid']))





###########################################################
#==========================================================
###########################################################
#open a fits file
def fits_open(fp):
    hdul = fits.open(fp)
    print(hdul[1].data)
    return ks_extract(hdul)

def ks_extract(hdul):
    l_mjdobs = hdul[1].data['mjdobs']
    l_mag = hdul[1].data['hfad_mag']
    l_emag = hdul[1].data['hfad_emag']
    l_filter = hdul[1].data['filter']
    l_sourceid= hdul[1].data['sourceid']
    l_filename= hdul[1].data['filename']
    l_tile= hdul[1].data['tile']
    l_seeing= hdul[1].data['seeing']
    l_exptime= hdul[1].data['exptime']
    l_skylevel= hdul[1].data['skylevel']
    l_tileloc= hdul[1].data['tileloc']
    l_ellipticity= hdul[1].data['ellipticity']
    l_ambiguous_match= hdul[1].data['ambiguous_match']
    l_ast_res_chisq= hdul[1].data['ast_res_chisq']
    l_chi= hdul[1].data['chi']
    l_cnf_ctr= hdul[1].data['cnf_ctr']
    l_diff_fit_ap= hdul[1].data['diff_fit_ap']
    l_ext= hdul[1].data['ext']
    l_objtype= hdul[1].data['ext']
    l_sky= hdul[1].data['sky']
    l_x= hdul[1].data['x']
    l_y= hdul[1].data['y']
    l_detected= hdul[1].data['detected']
    
    Ks_id=[i for i, e in enumerate(l_filter) if e == 'Ks']
    light_curve = {

        'Ks_mjdobs' : l_mjdobs[Ks_id],
        'Ks_mag' : l_mag[Ks_id],
        'Ks_emag' : l_emag[Ks_id],
        'Ks_filter' : l_filter[Ks_id],
        'Ks_sourceid' : l_sourceid[Ks_id],
        'Ks_filename' : l_filename[Ks_id],
        'Ks_tile' : l_tile[Ks_id],
        'Ks_seeing' : l_seeing[Ks_id],
        'Ks_exptime' : l_exptime[Ks_id],
        'Ks_skylevel' : l_skylevel[Ks_id],
        'Ks_tileloc' : l_tileloc[Ks_id],
        'Ks_ellipticity' : l_ellipticity[Ks_id],
        'Ks_ambiguous_match' : l_ambiguous_match[Ks_id],
        'Ks_ast_res_chisq' : l_ast_res_chisq[Ks_id],
        'Ks_chi' : l_chi[Ks_id],
        'Ks_cnf_ctr' : l_cnf_ctr[Ks_id],
        'Ks_diff_fit_ap' : l_diff_fit_ap[Ks_id],
        'Ks_ext' : l_ext[Ks_id],
        'Ks_objtype' : l_objtype[Ks_id],
        'Ks_sky' : l_sky[Ks_id],
        'Ks_x' : l_x[Ks_id],
        'Ks_y' : l_y[Ks_id],
        'Ks_detected' : l_detected[Ks_id],
    }

    return lc_nan_clean(light_curve)
    
    
def lc_chi_clean(lightcurve,chi_lim):


    Ks_id=[i for i, e in enumerate(lightcurve['ast_res_chisq']) if e < chi_lim]
    lightcurve_out = {
        'Ks_mjdobs' : lightcurve['Ks_mjdobs'][Ks_id],
        'Ks_mag' : lightcurve['Ks_mag'][Ks_id],
        'Ks_emag' : lightcurve['Ks_emag'][Ks_id],
        'Ks_filter' : lightcurve['Ks_filter'][Ks_id],
        'Ks_sourceid' : lightcurve['Ks_sourceid'][Ks_id],
        'Ks_filename' : lightcurve['Ks_filename'][Ks_id],
        'Ks_tile' : lightcurve['Ks_tile'][Ks_id],
        'Ks_seeing' : lightcurve['Ks_seeing'][Ks_id],
        'Ks_exptime' : lightcurve['Ks_exptime'][Ks_id],
        'Ks_skylevel' : lightcurve['Ks_skylevel'][Ks_id],
        'Ks_tileloc' : lightcurve['Ks_tileloc'][Ks_id],
        'Ks_ellipticity' : lightcurve['Ks_ellipticity'][Ks_id],
        'Ks_ambiguous_match' : lightcurve['Ks_ambiguous_match'][Ks_id],
        'Ks_ast_res_chisq' : lightcurve['Ks_ast_res_chisq'][Ks_id],
        'Ks_chi' : lightcurve['Ks_chi'][Ks_id],
        'Ks_cnf_ctr' : lightcurve['Ks_cnf_ctr'][Ks_id],
        'Ks_diff_fit_ap' : lightcurve['Ks_diff_fit_ap'][Ks_id],
        'Ks_ext' : lightcurve['Ks_ext'][Ks_id],
        'Ks_objtype' : lightcurve['Ks_objtype'][Ks_id],
        'Ks_sky' : lightcurve['Ks_sky'][Ks_id],
        'Ks_x' : lightcurve['Ks_x'][Ks_id],
        'Ks_y' : lightcurve['Ks_y'][Ks_id],
        'Ks_detected' : lightcurve['Ks_detected'][Ks_id]
    }

    return lightcurve_out




def lc_nan_clean(lightcurve):

    Ks_id=[i for i, e in enumerate(lightcurve['Ks_mag']) if (np.isnan(e) == False) & (np.isnan(lightcurve['Ks_emag'][i]) == False)]
    lightcurve_out = {
        'Ks_mjdobs' : lightcurve['Ks_mjdobs'][Ks_id],
        'Ks_mag' : lightcurve['Ks_mag'][Ks_id],
        'Ks_emag' : lightcurve['Ks_emag'][Ks_id],
        'Ks_filter' : lightcurve['Ks_filter'][Ks_id],
        'Ks_sourceid' : lightcurve['Ks_sourceid'][Ks_id],
        'Ks_filename' : lightcurve['Ks_filename'][Ks_id],
        'Ks_tile' : lightcurve['Ks_tile'][Ks_id],
        'Ks_seeing' : lightcurve['Ks_seeing'][Ks_id],
        'Ks_exptime' : lightcurve['Ks_exptime'][Ks_id],
        'Ks_skylevel' : lightcurve['Ks_skylevel'][Ks_id],
        'Ks_tileloc' : lightcurve['Ks_tileloc'][Ks_id],
        'Ks_ellipticity' : lightcurve['Ks_ellipticity'][Ks_id],
        'Ks_ambiguous_match' : lightcurve['Ks_ambiguous_match'][Ks_id],
        'Ks_ast_res_chisq' : lightcurve['Ks_ast_res_chisq'][Ks_id],
        'Ks_chi' : lightcurve['Ks_chi'][Ks_id],
        'Ks_cnf_ctr' : lightcurve['Ks_cnf_ctr'][Ks_id],
        'Ks_diff_fit_ap' : lightcurve['Ks_diff_fit_ap'][Ks_id],
        'Ks_ext' : lightcurve['Ks_ext'][Ks_id],
        'Ks_objtype' : lightcurve['Ks_objtype'][Ks_id],
        'Ks_sky' : lightcurve['Ks_sky'][Ks_id],
        'Ks_x' : lightcurve['Ks_x'][Ks_id],
        'Ks_y' : lightcurve['Ks_y'][Ks_id],
        'Ks_detected' : lightcurve['Ks_detected'][Ks_id]
    }

    return lc_zero_clean(lightcurve_out)



def lc_zero_clean(lightcurve):

    Ks_id=[i for i, e in enumerate(lightcurve['Ks_mag']) if (e > 1)]
    lightcurve_out = {
        'Ks_mjdobs' : lightcurve['Ks_mjdobs'][Ks_id],
        'Ks_mag' : lightcurve['Ks_mag'][Ks_id],
        'Ks_emag' : lightcurve['Ks_emag'][Ks_id],
        'Ks_filter' : lightcurve['Ks_filter'][Ks_id],
        'Ks_sourceid' : lightcurve['Ks_sourceid'][Ks_id],
        'Ks_filename' : lightcurve['Ks_filename'][Ks_id],
        'Ks_tile' : lightcurve['Ks_tile'][Ks_id],
        'Ks_seeing' : lightcurve['Ks_seeing'][Ks_id],
        'Ks_exptime' : lightcurve['Ks_exptime'][Ks_id],
        'Ks_skylevel' : lightcurve['Ks_skylevel'][Ks_id],
        'Ks_tileloc' : lightcurve['Ks_tileloc'][Ks_id],
        'Ks_ellipticity' : lightcurve['Ks_ellipticity'][Ks_id],
        'Ks_ambiguous_match' : lightcurve['Ks_ambiguous_match'][Ks_id],
        'Ks_ast_res_chisq' : lightcurve['Ks_ast_res_chisq'][Ks_id],
        'Ks_chi' : lightcurve['Ks_chi'][Ks_id],
        'Ks_cnf_ctr' : lightcurve['Ks_cnf_ctr'][Ks_id],
        'Ks_diff_fit_ap' : lightcurve['Ks_diff_fit_ap'][Ks_id],
        'Ks_ext' : lightcurve['Ks_ext'][Ks_id],
        'Ks_objtype' : lightcurve['Ks_objtype'][Ks_id],
        'Ks_sky' : lightcurve['Ks_sky'][Ks_id],
        'Ks_x' : lightcurve['Ks_x'][Ks_id],
        'Ks_y' : lightcurve['Ks_y'][Ks_id],
        'Ks_detected' : lightcurve['Ks_detected'][Ks_id]
    }

    return lightcurve_out


def get_source_id_coords(ra, dec):
    """
    Gets the row number in the time series file of a source closest to the given
    coordinates (dec.deg.) with a given radius (arcsec).
    """
    rad = 5
    filepath = coords_to_filename(ra,dec) # get the file path
    lc = h5py.File(filepath, "r") # open the file
    perfect_match = np.where((lc["sourceList/dec"][:] == dec) & (lc["sourceList/ra"][:] == ra))[0]    
    if len(perfect_match) == 1:
        return lc["sourceList/sourceid"][int(perfect_match)]
    else:
        _ct = SkyCoord(ra, dec, frame="icrs", unit="deg")
        ra_dec_cut = np.where((lc["sourceList/dec"][:] < dec + rad/3600)&(lc["sourceList/dec"][:] > dec - rad/3600) & (lc["sourceList/ra"][:] < ra + rad/3600)&(lc["sourceList/ra"][:] > ra - rad/3600))[0]    
        _cl = SkyCoord(lc["sourceList/ra"][ra_dec_cut], lc["sourceList/dec"][ra_dec_cut], frame="icrs", unit="deg")
        sep_arcsec = _ct.separation(_cl).arcsec
        if any(sep_arcsec<rad):
            return lc["sourceList/sourceid"][np.argmin(_ct.separation(_cl).arcsec)]
        else:
            print(ra, dec)
            return 0
            #raise IndexError("There are no sources within {:.1f} arcsec of the requested position.".format(rad))

    
def run_coords(ra,dec):
    filepath = coords_to_filename(ra,dec) # get the file path
    lc = h5py.File(filepath, "r") # open the file
    global idx
    idx = get_idx_coords(lc, ra,dec,1) # get the row index of the target
    return return_lc(lc, idx)

def run_sourceid(sourceid):
    filepath = sourceid_to_filename(sourceid)
    lc = h5py.File(filepath, "r") # open the file
    idx = get_idx_sourceid(lc, sourceid)
    return return_lc(lc, idx)

def run_get_coords(sourceid):
    filepath = sourceid_to_filename(sourceid)
    lc = h5py.File(filepath, "r") # open the file
    return get_coords_by_sourceid(lc, sourceid)

def run_coords_fits(ra,dec):

    filepath = coords_to_filename(ra,dec) # get the file path
    lc = h5py.File(filepath, "r") # open the file
    global idx
    idx = get_idx_coords(lc, ra,dec,1) # get the row index of the target
    #print('ID:',int(idx))

    global fp
    fp=lc_FITS_dir+str(idx)
    if os.path.isfile(fp) == False:
        write_FITS(lc, idx)
    return fits_open(fp)
    
def run_idx_fits(idd):

    filepath = sourceid_to_filename(idd) # get the file path
    lc = open_lc_file(filepath) # open the file
    global idx
    idx = get_idx_sourceid(lc, idd) # get the row index of the target

    #print('ID:',int(idx))
    global fp
    fp=lc_FITS_dir+str(idx)
    if os.path.isfile(fp) == False:
        write_FITS(lc, idx)
    return fits_open(fp)


def get_idx_coords(lc, ra, dec, rad):
    """
    Gets the row number in the time series file of a source closest to the given
    coordinates (dec.deg.) with a given radius (arcsec).
    """

    perfect_match = np.where((lc["sourceList/dec"][:] == dec) & (lc["sourceList/ra"][:] == ra))[0]    

    if len(perfect_match) == 1:
        return int(perfect_match)
    else:
        _ct = SkyCoord(ra, dec, frame="icrs", unit="deg")
        ra_dec_cut = np.where((lc["sourceList/dec"][:] < dec + rad/3600)&(lc["sourceList/dec"][:] > dec - rad/3600) & (lc["sourceList/ra"][:] < ra + rad/3600)&(lc["sourceList/ra"][:] > ra - rad/3600))[0]    
        _cl = SkyCoord(lc["sourceList/ra"][ra_dec_cut], lc["sourceList/dec"][ra_dec_cut], frame="icrs", unit="deg")
        sep_arcsec = _ct.separation(_cl).arcsec
        if any(sep_arcsec<rad):
            return np.argmin(_ct.separation(_cl).arcsec)
        else:
            raise IndexError("There are no sources within {:.1f} arcsec of the requested position.".format(rad))

def get_coords_by_sourceid(lc, sourceid):
    """
    Gets the "ra" and "dec" coordinates for a given sourceid.
    """
    row_idx = get_idx_sourceid(lc, sourceid)
    ra = lc["sourceList/ra"][row_idx]
    dec = lc["sourceList/dec"][row_idx]
    return ra, dec

def get_idx_sourceid(lc, sourceid):
    """
    Gets the row number in the time series file of a source with a given
    sourceid.
    """
    return np.argmin(np.abs(lc["sourceList/sourceid"][:] - sourceid))


def write_FITS(lc, idx): #potentially dont use anymore

    # grab time series data
    ci_idx = lc["timeSeries/catindexid"][idx]
    ci_idx_covered = lc["timeSeries/catindexidcovered"][idx]
    ci_idx_nondet = ci_idx_covered[np.isin(ci_idx_covered, ci_idx, invert=True)]

    # generate output table
    output = np.empty(ci_idx.size+ci_idx_nondet.size, dtype=[
    ("sourceid", np.int64),
    ("filename", "a20"),
    ("filter", "a2"),
    ("tile", "a4"),
    ("mjdobs", np.float64),
    ("seeing", np.float32),
    ("exptime", np.float32),
    ("skylevel", np.float32),
    ("tileloc", np.int32),
    ("ellipticity", np.float32),
    ("ambiguous_match", np.int),
    ("ast_res_chisq", np.float32),
    ("chi", np.float32),
    ("cnf_ctr", np.int),
    ("diff_fit_ap", np.float32),
    ("ext", np.int),
    ("objtype", np.int),
    ("sky", np.float32),
    ("x", np.float32),
    ("y", np.float32),
    ("hfad_mag", np.float32),
    ("hfad_emag", np.float32),
    ("detected", np.int)
    ])
    # fill with default values
    output[:] = tuple([lc["sourceList/sourceid"][idx]] + ['']*3 + [0]*19)
    # nans for floats
    for n in list(output.dtype.names)[ci_idx.size:]:
        if output.dtype[n] in [np.float32, np.float64]:
            output[n] = np.nan

    # catalogue index row number of all detections and non-detections
    ci_idx_all = np.concatenate((ci_idx, ci_idx_nondet))

    # fill catalogue related info for all relevent catalogues
    for col in ["seeing", "ellipticity", "exptime", "filename", "mjdobs",
                "skylevel", "tile", "tileloc", "filter"]:
        output[col] = lc["catIndex/"+col][:][ci_idx_all]

    # fill detection related info for all detections
    # selection of output rows corresponding to a detection
    det = slice(0,ci_idx.size,None)
    for col in ["ambiguous_match","ast_res_chisq","chi","cnf_ctr","diff_fit_ap",
                "hfad_mag","hfad_emag","ext","objtype","sky","x","y"]:
        output[col][det] = lc["timeSeries/"+col][idx]
    # flag where detected
    output["detected"][det] = 1



    # write to disk
    global star_id
    writeto(lc_FITS_dir+str(idx), output, overwrite=True)
    #fp=lc_FITS_dir+str(idx)
    #writeto(args.outfile, output)




def return_lc(lc, idx):

    # grab time series data
    ci_idx = lc["timeSeries/catindexid"][idx]
    ci_idx_covered = lc["timeSeries/catindexidcovered"][idx]
    ci_idx_nondet = ci_idx_covered[np.isin(ci_idx_covered, ci_idx, invert=True)]

    # generate output table
    output = np.empty(ci_idx.size+ci_idx_nondet.size, dtype=[
    ("sourceid", np.int64),
    ("filename", "a20"),
    ("filter", "a2"),
    ("tile", "a4"),
    ("mjdobs", np.float64),
    ("seeing", np.float32),
    ("exptime", np.float32),
    ("skylevel", np.float32),
    ("tileloc", np.int32),
    ("ellipticity", np.float32),
    ("ambiguous_match", np.int),
    ("ast_res_chisq", np.float32),
    ("chi", np.float32),
    ("cnf_ctr", np.int),
    ("diff_fit_ap", np.float32),
    ("ext", np.int),
    ("objtype", np.int),
    ("sky", np.float32),
    ("x", np.float32),
    ("y", np.float32),
    ("hfad_mag", np.float32),
    ("hfad_emag", np.float32),
    ("detected", np.int)
    ])
    # fill with default values
    output[:] = tuple([lc["sourceList/sourceid"][idx]] + ['']*3 + [0]*19)
    # nans for floats
    for n in list(output.dtype.names)[ci_idx.size:]:
        if output.dtype[n] in [np.float32, np.float64]:
            output[n] = np.nan

    # catalogue index row number of all detections and non-detections
    ci_idx_all = np.concatenate((ci_idx, ci_idx_nondet))

    # fill catalogue related info for all relevent catalogues
    for col in ["seeing", "ellipticity", "exptime", "filename", "mjdobs",
                "skylevel", "tile", "tileloc", "filter"]:
        output[col] = lc["catIndex/"+col][:][ci_idx_all]

    # fill detection related info for all detections
    # selection of output rows corresponding to a detection
    det = slice(0,ci_idx.size,None)
    for col in ["ambiguous_match","ast_res_chisq","chi","cnf_ctr","diff_fit_ap",
                "hfad_mag","hfad_emag","ext","objtype","sky","x","y"]:
        output[col][det] = lc["timeSeries/"+col][idx]
    # flag where detected
    output["detected"][det] = 1
    #lc_chi_clean(output,5)
    return output


def coords_to_filename(ra,dec):
    """
    Get a time series file path from a set of coordinates.
    """
    hpix = [hp.ang2pix(nside=n,theta=ra, phi=dec,lonlat=True) for n in [256,512,1024]]
    paths = np.array(["{0:s}/n{1:d}_{2:d}.hdf5".format(lc_file_dir, _n, _hp) for _hp, _n in zip(hpix, [256,512,1024])])
    exist = np.array([isfile(path) for path in paths])
    if not any(exist):
        #print(ra,dec)
        raise IndexError("There is no data for these coordinates.")
    else:
        return paths[np.argmax(exist)]

def sourceid_to_filename(sourceid):
    """
    Get a time series file path from a source id.
    """
    pix1024 = int(np.floor(1.0*sourceid/1000000))
    pix512 = get_sub_pixels(pix1024, 1024, 512)[0]
    pix256 = get_sub_pixels(pix1024, 1024, 256)[0]
    paths = np.array(["{0:s}/n{1:d}_{2:d}.hdf5".format(lc_file_dir, _n, _hp)
                      for _hp, _n
                      in zip([pix256, pix512, pix1024], [256,512,1024])])
    exist = np.array([isfile(path) for path in paths])
    if not any(exist):
        raise IndexError("There is no data for this sourceid.")
    else:
        return paths[np.argmax(exist)]


# Get HEALPix sub-pixels #
def get_sub_pixels(orig_pixel, orig_nsides, tgt_nsides, include_border=False):
    """
    Generate a list of sub pixels. Works in one of three ways depending on
    whether the target nsides is higher, the same, or lower than the original
    nsides.
    """

    ### the same ###
    if int(orig_nsides) == int(tgt_nsides):
        # the subpixel list is the input list
        subpixels = np.asarray(orig_pixel).flatten()

    ### higher ###
    if int(orig_nsides) < int(tgt_nsides):
        # get vertices of target pixel
        vecs = hp.boundaries(orig_nsides, orig_pixel, step=1).T
        # get sub pixels of finer grid
        subpixels = hp.query_polygon(tgt_nsides, vecs, inclusive=False)

    ### lower ###
    if int(orig_nsides) > int(tgt_nsides):
        # get coords of original pixel
        _ra, _dec = hp.pix2ang(orig_nsides, orig_pixel, lonlat=True)
        # find which coarser grid pixel contains them
        subpixels = np.asarray(hp.ang2pix(tgt_nsides, _ra, _dec,
                                          lonlat=True)
                              ).flatten()

    # include a 1 pixel border if requested
    if include_border:
        border = hp.get_all_neighbours(tgt_nsides, subpixels).flatten()
        subpixels = np.concatenate((subpixels,border))

    return np.unique(subpixels)


def open_lc_file(filepath):
    return h5py.File(filepath, "r")


