#!/usr/bin/env python3
"""
Simplified TESS CV Processor with VVV Comparison

This script processes TESS light curves for cataclysmic variable (CV) stars using
two approaches to NN_FAP periodogram analysis:
  1. VVV-like subsample: subsamples the TESS light curve to mimic VVV cadence/sparsity
  2. Full TESS data: uses a sliding window approach to handle the 200-point NN_FAP limitation

Features:
- Uses most recent available cycle for analysis
- Compares period finding capabilities between VVV-like and full TESS data
- Generates 4-panel visualization comparing periodograms and phase-folded light curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from tqdm import tqdm
import lightkurve as lk
from scipy import stats
from scipy.signal import savgol_filter
from astropy.io import fits
import NN_FAP  # Import the NN_FAP package
from functools import lru_cache
import time
from concurrent.futures import ProcessPoolExecutor

@lru_cache(maxsize=1)
def get_nn_fap_model(model_path='/home/njm/Period/NN_FAP/final_12l_dp_all/'):
    """Get the NN_FAP model, with caching to avoid reloading."""
    return NN_FAP.get_model(model_path)

def remove_units_from_fits(fits_file):
    """
    Remove all unit keywords (TUNIT*) from the FITS file header to avoid unit errors on reload.
    """
    with fits.open(fits_file, mode='update') as hdul:
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU):
                keys_to_remove = [key for key in hdu.header if key.startswith("TUNIT")]
                for key in keys_to_remove:
                    del hdu.header[key]
        hdul.flush()

def build_default_cv_targets():
    """
    Build a DEFAULT_CV_TARGETS dictionary by querying SIMBAD for objects with type 'CV*'
    and extracting their TIC IDs.
    
    Returns:
        dict: A dictionary of CV targets in TESS with TIC IDs.
    """
    from astroquery.simbad import Simbad
    import re

    # Increase the timeout to give SIMBAD more time to respond.
    Simbad.TIMEOUT = 120
    # Request additional fields: ids and object type.
    Simbad.add_votable_fields("ids", "otype")
    
    print("Querying SIMBAD for cataclysmic variables (CV*)...")
    try:
        result = Simbad.query_criteria("otype='CV*'")
    except Exception as e:
        print(f"Error querying SIMBAD: {e}")
        return {}

    if result is None:
        print("No cataclysmic variables found in SIMBAD.")
        return {}

    cv_targets = {}
    for row in result:
        main_id = row['MAIN_ID']
        # Ensure main_id is a proper string
        if isinstance(main_id, bytes):
            main_id = main_id.decode()
        
        # Get the identifiers field and split them
        ids_field = row['IDS']
        if isinstance(ids_field, bytes):
            ids_field = ids_field.decode()
        identifiers = [identifier.strip() for identifier in ids_field.split('|')]
        
        # Look for a TIC ID in the identifiers using a regex.
        tic_id = None
        for identifier in identifiers:
            # This regex looks for "TIC" optionally followed by a space then numbers.
            match = re.search(r'TIC\s*([\d]+)', identifier)
            if match:
                try:
                    tic_id = int(match.group(1))
                    break
                except ValueError:
                    continue
        
        if tic_id is not None:
            target_key = main_id.replace(" ", "_")
            cv_targets[target_key] = (tic_id, main_id, "Cataclysmic Variable")
    
    print(f"Found {len(cv_targets)} confirmed CV targets with TIC IDs.")
    return cv_targets

# Example usage:
DEFAULT_CV_TARGETS = build_default_cv_targets()
print("Default CV targets:", DEFAULT_CV_TARGETS)

def download_lightcurves(tic_id, output_dir, cadence="short", max_cycles=1):
    """
    Download TESS light curves for a given TIC ID.
    
    Parameters:
    -----------
    tic_id : int
        TIC ID of the target
    output_dir : str
        Directory to save the results
    cadence : str
        Cadence type: "short" or "long"
    max_cycles : int
        Maximum number of cycles to download (default: 1 for most recent)
        
    Returns:
    --------
    dict
        Dictionary mapping cycle numbers to light curve file paths
    """
    # Create directory for this target
    target_dir = os.path.join(output_dir, f"TIC_{tic_id}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Search for target
    print(f"Searching for TESS data for TIC {tic_id}...")
    search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    
    if len(search_result) == 0:
        print(f"No TESS data found for TIC {tic_id}")
        return {}
    
    # Filter by cadence
    if cadence == "short":
        search_result = search_result[search_result.exptime.value < 600]
    else:
        search_result = search_result[search_result.exptime.value >= 600]
    
    if len(search_result) == 0:
        print(f"No {cadence} cadence data found for TIC {tic_id}")
        return {}
    
    print(f"Found {len(search_result)} {cadence} cadence observations, using most recent {max_cycles}")
    
    # Get most recent cycles
    cycles = {}
    for idx in range(min(max_cycles, len(search_result))):
        cycle_num = idx + 1
        output_file = os.path.join(target_dir, f"cycle_{cycle_num}.fits")
        
        # Download if file doesn't exist
        if not os.path.exists(output_file):
            print(f"Downloading cycle {cycle_num} data...")
            try:
                lc = search_result[idx].download()
                lc.to_fits(output_file, overwrite=True)
                # Strip the astropy unit keywords
                remove_units_from_fits(output_file)
                print(f"Saved to {output_file}")
            except Exception as e:
                print(f"Error downloading data: {e}")
                continue
        else:
            print(f"Using existing file: {output_file}")

        cycles[cycle_num] = output_file
    
    return cycles

def process_lightcurve(lc_file):
    """Process a light curve FITS file into time, flux, and error arrays."""
    try:
        # Load the light curve and remove outliers
        lc = lk.read(lc_file)
        clean_lc = lc.remove_outliers(sigma=5)
        
        # Get the raw time and flux values
        time = clean_lc.time.value
        flux = clean_lc.flux.value
        
        # Get flux_err if it exists, otherwise assign a default
        if hasattr(clean_lc, 'flux_err') and clean_lc.flux_err is not None:
            error = clean_lc.flux_err.value
        else:
            error = np.ones_like(flux) * 0.001
        
        # Normalize flux by its median
        median_flux = np.median(flux)
        flux = flux / median_flux
        error = error / median_flux
        
        return clean_lc, time, flux, error
    
    except Exception as e:
        print(f"Error processing {lc_file}: {e}")
        return None, None, None, None

def create_vvv_subsample(time, flux, error, n_points=50):
    """
    Create a subsample of the light curve that approximates VVV cadence.
    
    Parameters:
    -----------
    time : array
        Time array (days)
    flux : array
        Flux array (normalized)
    error : array
        Error array
    n_points : int
        Approximate number of points to include in the subsample
        
    Returns:
    --------
    tuple
        - Subsampled time array
        - Subsampled flux array
        - Subsampled error array
    """
    # Determine the sampling rate
    if len(time) <= n_points:
        # If we already have fewer points than requested, use all of them
        return time, flux, error
    
    # Calculate the stride to get approximately n_points
    stride = max(1, len(time) // n_points)
    
    # Create the subsample with some randomness to better mimic VVV
    indices = np.arange(0, len(time), stride)
    
    # Add some random jitter to indices to avoid perfectly even sampling
    jitter = np.random.randint(-stride//4, stride//4, size=len(indices))
    indices = np.clip(indices + jitter, 0, len(time) - 1)
    
    # Ensure indices are unique and sorted
    indices = np.unique(indices)
    indices.sort()
    
    return time[indices], flux[indices], error[indices]

def create_nn_fap_periodogram(time, flux, periods, model_path, n_workers=4):
    """
    Create a periodogram using NN_FAP for a single light curve dataset.
    
    Parameters:
    -----------
    time : array
        Time array (days)
    flux : array
        Flux array (normalized)
    periods : array
        Periods to test
    model_path : str
        Path to the NN_FAP model
    n_workers : int
        Number of worker processes for parallel computation
        
    Returns:
    --------
    array
        Power array (1-FAP values for each period)
    """
    # For small light curves (<= 200 points), we can directly apply NN_FAP
    if len(time) <= 200:
        knn, model = get_nn_fap_model(model_path)
        power = np.array([
            1.0 - NN_FAP.inference(period, flux, time, knn, model)
            for period in periods
        ])
        return power
    
    # For larger light curves, use random subsampling approach
    # Function to compute NN_FAP for a set of periods using a random subsample
    def compute_batch(batch_periods):
        knn, model = get_nn_fap_model(model_path)
        batch_power = np.zeros(len(batch_periods))
        
        for i, period in enumerate(batch_periods):
            # Randomly select 200 points for each period evaluation
            # This introduces some randomness but ensures we use the same model constraints
            indices = np.random.choice(len(flux), 200, replace=False)
            indices.sort()  # Keep them in time order
            
            # Calculate NN_FAP
            fap = NN_FAP.inference(period, flux[indices], time[indices], knn, model)
            batch_power[i] = 1.0 - fap  # Convert to power (higher is better)
        
        return batch_power
    
    # Split periods into batches for parallel processing
    batch_size = max(1, len(periods) // n_workers)
    period_batches = [periods[i:i+batch_size] for i in range(0, len(periods), batch_size)]
    
    # Process batches in parallel
    power = np.zeros(len(periods))
    batch_results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        batch_results = list(executor.map(compute_batch, period_batches))
    
    # Combine results
    start_idx = 0
    for batch_power in batch_results:
        end_idx = start_idx + len(batch_power)
        power[start_idx:end_idx] = batch_power
        start_idx = end_idx
    
    return power

def sliding_window_worker(args):
    """Worker function for parallel processing of sliding window periodograms."""
    window_time, window_flux, periods, model_path = args
    knn, model = get_nn_fap_model(model_path)
    power = np.array([
        1.0 - NN_FAP.inference(period, window_flux, window_time, knn, model)
        for period in periods
    ])
    return power

def create_nn_fap_sliding_window_periodogram(time, flux, periods, model_path, window_size=200, step=50, n_workers=4):
    """
    Create a periodogram using a sliding window approach with NN_FAP.
    
    Parameters:
    -----------
    time : array
        Time array (days)
    flux : array
        Flux array (normalized)
    periods : array
        Periods to test
    model_path : str
        Path to the NN_FAP model
    window_size : int
        Size of each sliding window
    step : int
        Step size between windows
    n_workers : int
        Number of worker processes for parallel computation
        
    Returns:
    --------
    array
        Power array (1-FAP values for each period)
    """
    # If light curve is smaller than window_size, just use the direct approach
    if len(time) <= window_size:
        return create_nn_fap_periodogram(time, flux, periods, model_path, n_workers=1)
    
    # Create sliding windows
    windows = [
        (time[i:i+window_size], flux[i:i+window_size], periods, model_path)
        for i in range(0, len(time)-window_size+1, step) if len(time[i:i+window_size]) >= 50
    ]
    
    # Process windows in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(sliding_window_worker, windows), 
                           total=len(windows), desc="Sliding Window Periodogram"))
    
    # Average the results
    avg_power = np.mean(results, axis=0)
    return avg_power

def phase_fold_lightcurve(time, flux, error, period):
    """
    Phase fold a light curve.
    
    Parameters:
    -----------
    time : array
        Time array (days)
    flux : array
        Flux array (normalized)
    error : array
        Error array
    period : float
        Period to fold at (days)
        
    Returns:
    --------
    tuple
        - Phase array (0-1)
        - Flux array
        - Error array
    """
    phase = (time / period) % 1.0
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux = flux[sort_idx]
    error = error[sort_idx]
    
    return phase, flux, error

def bin_phased_lightcurve(phase, flux, error, bins=100):
    """
    Bin a phase-folded light curve.
    
    Parameters:
    -----------
    phase : array
        Phase array (0-1)
    flux : array
        Flux array
    error : array
        Error array
    bins : int
        Number of bins
        
    Returns:
    --------
    tuple
        - Binned phase array
        - Binned flux array
        - Binned error array
    """
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    binned_flux = np.zeros(bins)
    binned_error = np.zeros(bins)
    
    for i in range(bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_flux[i] = np.mean(flux[mask])
            # Error propagation
            binned_error[i] = np.sqrt(np.sum(error[mask]**2)) / np.sum(mask)
        else:
            binned_flux[i] = np.nan
            binned_error[i] = np.nan
    
    return bin_centers, binned_flux, binned_error

def find_orbital_period(time, flux, error):
    """
    Find orbital period using two methods:
    1. VVV-like subsample
    2. Full TESS data with sliding window
    
    Parameters:
    -----------
    time : array
        Time array (days)
    flux : array
        Flux array (normalized)
    error : array
        Error array
        
    Returns:
    --------
    dict
        Results dictionary containing periods, powers, and best periods
    """
    # Common parameters
    short_periods = np.linspace(0.001, 1, 100)  # Periods to search in days
    model_path = '/home/njm/Period/NN_FAP/final_12l_dp_all/'
    
    # Method 1: VVV-like subsample
    print("Creating VVV-like subsample...")
    vvv_time, vvv_flux, vvv_error = create_vvv_subsample(time, flux, error, n_points=50)
    print(f"VVV subsample has {len(vvv_time)} points")
    
    print("Computing VVV-like periodogram...")
    vvv_power = create_nn_fap_periodogram(vvv_time, vvv_flux, short_periods, model_path)
    
    # Method 2: Full TESS data with sliding window
    print("Computing full TESS periodogram with sliding window...")
    tess_power = create_nn_fap_sliding_window_periodogram(
        time, flux, short_periods, model_path, window_size=200, step=50)
    
    # Find best periods from each method
    vvv_best_idx = np.argmax(vvv_power)
    tess_best_idx = np.argmax(tess_power)
    
    vvv_best_period = short_periods[vvv_best_idx]
    tess_best_period = short_periods[tess_best_idx]
    
    print(f"VVV-like best period: {vvv_best_period*24:.6f} hours")
    print(f"TESS best period: {tess_best_period*24:.6f} hours")
    
    # Estimate uncertainties
    # For VVV
    try:
        high_power_idx = np.where(vvv_power > 0.5 * vvv_power[vvv_best_idx])[0]
        if len(high_power_idx) > 1:
            vvv_period_err = 0.5 * (short_periods[high_power_idx[-1]] - short_periods[high_power_idx[0]])
        else:
            vvv_period_err = 0.01 * vvv_best_period
    except:
        vvv_period_err = 0.01 * vvv_best_period
    
    # For TESS
    try:
        high_power_idx = np.where(tess_power > 0.5 * tess_power[tess_best_idx])[0]
        if len(high_power_idx) > 1:
            tess_period_err = 0.5 * (short_periods[high_power_idx[-1]] - short_periods[high_power_idx[0]])
        else:
            tess_period_err = 0.01 * tess_best_period
    except:
        tess_period_err = 0.01 * tess_best_period
    
    return {
        "periods": short_periods,
        "vvv_power": vvv_power,
        "tess_power": tess_power,
        "vvv_best_period": vvv_best_period,
        "vvv_period_err": vvv_period_err,
        "tess_best_period": tess_best_period,
        "tess_period_err": tess_period_err,
        "vvv_data": (vvv_time, vvv_flux, vvv_error),
        "tess_data": (time, flux, error)
    }

def create_comparison_plot(star_name, cv_type, result, output_dir):
    """
    Create a 4-panel comparison plot:
    - Top left: VVV-like periodogram
    - Top right: TESS periodogram
    - Bottom left: VVV-like phase-folded light curve
    - Bottom right: TESS phase-folded light curve
    
    Parameters:
    -----------
    star_name : str
        Name of the star
    cv_type : str
        CV type
    result : dict
        Results dictionary from find_orbital_period
    output_dir : str
        Directory to save the output plot
    """
    # Extract data from result
    periods = result["periods"]
    vvv_power = result["vvv_power"]
    tess_power = result["tess_power"]
    vvv_best_period = result["vvv_best_period"]
    vvv_period_err = result["vvv_period_err"]
    tess_best_period = result["tess_best_period"]
    tess_period_err = result["tess_period_err"]
    vvv_time, vvv_flux, vvv_error = result["vvv_data"]
    tess_time, tess_flux, tess_error = result["tess_data"]
    
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Top left: VVV-like periodogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(periods * 24, vvv_power, 'b-', linewidth=1.5)
    
    # Mark the best period
    ax1.axvline(vvv_best_period * 24, color='r', linestyle='--', alpha=0.7)
    ax1.scatter([vvv_best_period * 24], [vvv_power[np.argmax(vvv_power)]], 
                color='red', s=50, marker='o', zorder=5)
    
    ax1.set_xlabel('Period (hours)')
    ax1.set_ylabel('Power (1-FAP)')
    ax1.set_title('VVV-like Subsample Periodogram')
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f'Period: {vvv_best_period*24:.6f} ± {vvv_period_err*24:.6f} h', 
                xy=(0.05, 0.95), xycoords='axes fraction', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    
    # Top right: TESS periodogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(periods * 24, tess_power, 'g-', linewidth=1.5)
    
    # Mark the best period
    ax2.axvline(tess_best_period * 24, color='r', linestyle='--', alpha=0.7)
    ax2.scatter([tess_best_period * 24], [tess_power[np.argmax(tess_power)]], 
                color='red', s=50, marker='o', zorder=5)
    
    ax2.set_xlabel('Period (hours)')
    ax2.set_ylabel('Power (1-FAP)')
    ax2.set_title('Full TESS Sliding Window Periodogram')
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f'Period: {tess_best_period*24:.6f} ± {tess_period_err*24:.6f} h', 
                xy=(0.05, 0.95), xycoords='axes fraction', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    
    # Bottom left: VVV-like phase-folded light curve
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Phase fold and bin the VVV-like data
    vvv_phase, vvv_folded_flux, vvv_folded_error = phase_fold_lightcurve(
        vvv_time, vvv_flux, vvv_error, vvv_best_period)
    vvv_bin_phase, vvv_bin_flux, vvv_bin_error = bin_phased_lightcurve(
        vvv_phase, vvv_folded_flux, vvv_folded_error, bins=20)  # Fewer bins for sparse data
    
    # Plot the raw phase-folded data
    ax3.errorbar(vvv_phase, vvv_folded_flux, yerr=vvv_folded_error, fmt='.', 
                 color='blue', alpha=0.5, ecolor='lightblue', markersize=5)
    
    # Plot the binned data
    ax3.errorbar(vvv_bin_phase, vvv_bin_flux, yerr=vvv_bin_error, fmt='o', 
                 color='blue', alpha=0.8, markersize=6)
    
    # Add a second cycle
    ax3.errorbar(vvv_phase + 1, vvv_folded_flux, yerr=vvv_folded_error, fmt='.', 
                 color='blue', alpha=0.5, ecolor='lightblue', markersize=5)
    ax3.errorbar(vvv_bin_phase + 1, vvv_bin_flux, yerr=vvv_bin_error, fmt='o', 
                 color='blue', alpha=0.8, markersize=6)
    
    # Try to smooth the binned data
    try:
        # Filter out NaN values
        nan_mask = np.isnan(vvv_bin_flux)
        if not all(nan_mask) and sum(~nan_mask) > 3:
            x_valid = vvv_bin_phase[~nan_mask]
            y_valid = vvv_bin_flux[~nan_mask]
            
            # Sort x and y by x values
            sort_idx = np.argsort(x_valid)
            x_valid = x_valid[sort_idx]
            y_valid = y_valid[sort_idx]
            
            # Apply Savitzky-Golay filter if possible
            if len(y_valid) >= 5:
                window_length = min(5, len(y_valid) // 2 * 2 + 1)  # Ensure odd number
                if window_length >= 3:
                    y_smooth = savgol_filter(y_valid, window_length, 1)
                    ax3.plot(x_valid, y_smooth, '-', color='blue', linewidth=2)
                    ax3.plot(x_valid + 1, y_smooth, '-', color='blue', linewidth=2)
    except Exception as e:
        print(f"Smoothing error for VVV data: {e}")
    
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Relative Flux')
    ax3.set_title(f'VVV-like Phase-folded Light Curve (P = {vvv_best_period*24:.6f} h)')
    ax3.set_xlim(0, 2)
    ax3.grid(True, alpha=0.3)
    
    # Calculate and display SNR
    try:
        mean_flux = np.nanmedian(vvv_bin_flux)
        depth = mean_flux - np.nanmin(vvv_bin_flux)
        noise = np.nanmedian(vvv_bin_error)
        snr = depth / noise if noise > 0 else 0
        ax3.annotate(f'SNR: {snr:.2f}', xy=(0.05, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    except:
        pass
    
    # Bottom right: TESS phase-folded light curve
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Phase fold and bin the TESS data
    tess_phase, tess_folded_flux, tess_folded_error = phase_fold_lightcurve(
        tess_time, tess_flux, tess_error, tess_best_period)
    tess_bin_phase, tess_bin_flux, tess_bin_error = bin_phased_lightcurve(
        tess_phase, tess_folded_flux, tess_folded_error, bins=100)
    
    # Plot the binned data only (raw data would be too dense)
    ax4.errorbar(tess_bin_phase, tess_bin_flux, yerr=tess_bin_error, fmt='o', 
                 color='green', alpha=0.7, markersize=4)
    
    # Add a second cycle
    ax4.errorbar(tess_bin_phase + 1, tess_bin_flux, yerr=tess_bin_error, fmt='o', 
                 color='green', alpha=0.7, markersize=4)
    
    # Try to smooth the binned data
    try:
        # Filter out NaN values
        nan_mask = np.isnan(tess_bin_flux)
        if not all(nan_mask):
            x_valid = tess_bin_phase[~nan_mask]
            y_valid = tess_bin_flux[~nan_mask]
            
            # Apply Savitzky-Golay filter
            if len(y_valid) > 10:
                window_length = min(15, len(y_valid) // 4 * 2 + 1)  # Ensure odd number
                if window_length > 3:
                    y_smooth = savgol_filter(y_valid, window_length, 3)
                    ax4.plot(x_valid, y_smooth, '-', color='green', linewidth=2)
                    ax4.plot(x_valid + 1, y_smooth, '-', color='green', linewidth=2)
    except Exception as e:
        print(f"Smoothing error for TESS data: {e}")
    
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Relative Flux')
    ax4.set_title(f'TESS Phase-folded Light Curve (P = {tess_best_period*24:.6f} h)')
    ax4.set_xlim(0, 2)
    ax4.grid(True, alpha=0.3)
    
    # Calculate and display SNR
    try:
        mean_flux = np.nanmedian(tess_bin_flux)
        depth = mean_flux - np.nanmin(tess_bin_flux)
        noise = np.nanmedian(tess_bin_error)
        snr = depth / noise if noise > 0 else 0
        ax4.annotate(f'SNR: {snr:.2f}', xy=(0.05, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    except:
        pass
    
    # Set overall title
    plt.suptitle(f'{star_name} ({cv_type}) - VVV vs TESS Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    safe_name = star_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(output_dir, f"{safe_name}_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_cv_target(tic_id, common_name, cv_type, output_dir, cycles):
    """
    Analyze a CV target using VVV-like subsample vs full TESS data.
    
    Parameters:
    -----------
    tic_id : int
        TIC ID of the target
    common_name : str
        Common name of the target
    cv_type : str
        CV type
    output_dir : str
        Output directory
    cycles : dict
        Dictionary mapping cycle numbers to light curve file paths
        
    Returns:
    --------
    dict
        Results dictionary
    """
    # Create results directory for this target
    target_dir = os.path.join(output_dir, f"{common_name.replace(' ', '_')}")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\nAnalyzing {common_name} (TIC {tic_id}, {cv_type})")
    print("-" * 50)
    
    # Use the most recent cycle
    if not cycles:
        print(f"No cycles available for {common_name}")
        return None
        
    cycle_num = min(cycles.keys())  # Get the most recent cycle (cycle 1)
    lc_file = cycles[cycle_num]
    
    print(f"Using cycle {cycle_num} for analysis...")
    
    # Process light curve
    lc, time, flux, error = process_lightcurve(lc_file)
    if time is None:
        print(f"Error processing light curve for {common_name}")
        return None
    
    # Find orbital periods using both methods
    result = find_orbital_period(time, flux, error)
    
    # Create comparison plot
    create_comparison_plot(common_name, cv_type, result, target_dir)
    
    # Return results
    return {
        "tic_id": tic_id,
        "common_name": common_name,
        "cv_type": cv_type,
        "vvv_best_period": result["vvv_best_period"],
        "vvv_period_err": result["vvv_period_err"],
        "tess_best_period": result["tess_best_period"],
        "tess_period_err": result["tess_period_err"],
        "period_ratio": result["vvv_best_period"] / result["tess_best_period"],
        "n_vvv_points": len(result["vvv_data"][0]),
        "n_tess_points": len(result["tess_data"][0]),
    }

def create_summary_table(all_results, output_dir):
    """
    Create a summary table comparing VVV-like and TESS results.
    
    Parameters:
    -----------
    all_results : list
        List of results dictionaries
    output_dir : str
        Output directory
    """
    # Filter out None results
    all_results = [r for r in all_results if r is not None]
    
    if len(all_results) == 0:
        print("No valid results for summary table")
        return
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "Star": r["common_name"],
            "TIC ID": r["tic_id"],
            "Type": r["cv_type"],
            "VVV-like Period (h)": r["vvv_best_period"] * 24,
            "VVV-like Error (h)": r["vvv_period_err"] * 24,
            "TESS Period (h)": r["tess_best_period"] * 24,
            "TESS Error (h)": r["tess_period_err"] * 24,
            "Period Ratio": r["period_ratio"],
            "VVV-like Points": r["n_vvv_points"],
            "TESS Points": r["n_tess_points"],
        }
        for r in all_results
    ])
    
    # Add some summary statistics
    df["Period Difference (%)"] = 100 * abs(df["VVV-like Period (h)"] - df["TESS Period (h)"]) / df["TESS Period (h)"]
    df["Error Ratio"] = df["VVV-like Error (h)"] / df["TESS Error (h)"]
    
    # Save as CSV
    csv_file = os.path.join(output_dir, "vvv_tess_comparison.csv")
    df.to_csv(csv_file, index=False)
    print(f"Summary table saved to {csv_file}")
    
    # Create a summary plot
    plt.figure(figsize=(10, 8))
    
    # Plot period comparison
    plt.scatter(df["TESS Period (h)"], df["VVV-like Period (h)"], 
                s=50, alpha=0.7, c=df["Period Difference (%)"], cmap='viridis')
    
    # Add a diagonal line
    min_val = min(df["TESS Period (h)"].min(), df["VVV-like Period (h)"].min())
    max_val = max(df["TESS Period (h)"].max(), df["VVV-like Period (h)"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add color bar
    cbar = plt.colorbar()
    cbar.set_label('Period Difference (%)')
    
    # Annotate points with star names
    for i, row in df.iterrows():
        plt.annotate(row["Star"], 
                     (row["TESS Period (h)"], row["VVV-like Period (h)"]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.8)
    
    plt.xlabel('TESS Period (hours)')
    plt.ylabel('VVV-like Period (hours)')
    plt.title('Comparison of Period Determination: VVV-like vs. TESS')
    plt.grid(True, alpha=0.3)
    
    # Make axes equal
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "period_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the summary statistics
    return {
        "num_objects": len(df),
        "mean_period_diff": df["Period Difference (%)"].mean(),
        "median_period_diff": df["Period Difference (%)"].median(),
        "mean_error_ratio": df["Error Ratio"].mean(),
        "median_error_ratio": df["Error Ratio"].median(),
    }

def main():
    """Main function to run the analysis."""
    # Configuration parameters (hardcoded instead of using command-line arguments)
    config = {
        "output_dir": "../PRIMVS/cv_results/VVV_TESS_comparison/",
        "cadence": "short",
        "max_cycles": 1,  # Only use the most recent cycle
        "specific_tic_ids": [],  # Leave empty to use default targets, or add specific TIC IDs
        "num_workers": 4  # Number of worker processes for parallel computation
    }
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Get CV targets (either from specific_tic_ids or DEFAULT_CV_TARGETS)
    if config["specific_tic_ids"]:
        targets = []
        for tic in config["specific_tic_ids"]:
            # Try to match to known targets, otherwise use generic name
            for name, (known_tic, common_name, cv_type) in DEFAULT_CV_TARGETS.items():
                if tic == known_tic:
                    targets.append((tic, common_name, cv_type))
                    break
            else:
                targets.append((tic, f"TIC {tic}", "Unknown"))
    else:
        # Use default targets
        targets = [(tic, common_name, cv_type) for _, (tic, common_name, cv_type) in DEFAULT_CV_TARGETS.items()]
    
    print(f"Will process {len(targets)} CV targets")
    
    # Process each target
    all_results = []
    
    for tic_id, common_name, cv_type in targets:
        # Download light curves (only the most recent cycle)
        cycles = download_lightcurves(tic_id, config["output_dir"], config["cadence"], config["max_cycles"])
        
        if not cycles:
            print(f"No cycles found for {common_name} (TIC {tic_id})")
            continue
            
        print(f"Found {len(cycles)} cycles for {common_name} (TIC {tic_id})")
        
        # Analyze
        result = analyze_cv_target(tic_id, common_name, cv_type, config["output_dir"], cycles)
        if result:
            all_results.append(result)
    
    # Create summary table and comparison plot
    if all_results:
        stats = create_summary_table(all_results, config["output_dir"])
        
        print("\nAnalysis Summary:")
        print(f"Number of objects analyzed: {stats['num_objects']}")
        print(f"Mean period difference: {stats['mean_period_diff']:.2f}%")
        print(f"Median period difference: {stats['median_period_diff']:.2f}%")
        print(f"Mean VVV/TESS error ratio: {stats['mean_error_ratio']:.2f}")
        print(f"Median VVV/TESS error ratio: {stats['median_error_ratio']:.2f}")
    
    print(f"\nAnalysis complete. Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()