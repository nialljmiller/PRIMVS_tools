#!/usr/bin/env python3
"""
TESS CV Multi-Cycle Processor with Enhanced NN_FAP

This script processes TESS light curves for cataclysmic variable (CV) stars using
multiple approaches to NN_FAP periodogram analysis to better handle the 200-point limitation.

Features:
- Uses most recent available cycle for periodogram analysis
- Implements two different NN_FAP periodogram construction methods:
  1. Chunk method: splits the light curve into N chunks of 200 points and averages the periodograms
  2. Sliding window method: uses a 200-point sliding window focused on shorter periods
- Optional subtraction of chunk method from sliding window to enhance short period detection
- Generates visualization of light curves and period analysis
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
import NN_FAP

@lru_cache(maxsize=1)
def get_nn_fap_model(model_path='/home/njm/Period/NN_FAP/final_12l_dp_all/'):
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
    and extracting their TIC IDs. This returns a dictionary mapping a target key (the main ID with underscores)
    to a tuple: (TIC ID, main ID, "Cataclysmic Variable").

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


def create_nn_fap_single_periodogram(time, flux, periods, knn, model):
    """
    Create a periodogram for a single segment of data using NN_FAP.
    
    Parameters:
    -----------
    time : array
        Time array (days)
    flux : array
        Flux array (normalized)
    periods : array
        Periods to test
    knn : object
        KNN model from NN_FAP
    model : object
        Neural network model from NN_FAP
        
    Returns:
    --------
    array
        Power array (1-FAP values for each period)
    """
    power = np.zeros(len(periods))
    
    for i, period in enumerate(periods):
        fap = NN_FAP.inference(period, flux, time, knn, model)
        power[i] = 1.0 - fap  # Convert to power (higher is better)
    
    return power


from concurrent.futures import ProcessPoolExecutor

def chunk_periodogram_worker(args):
    chunk_time, chunk_flux, periods, model_path = args
    knn, model = get_nn_fap_model(model_path)
    power = np.array([
        1.0 - NN_FAP.inference(period, chunk_flux, chunk_time, knn, model)
        for period in periods
    ])
    return power

def create_nn_fap_chunk_periodogram(time, flux, periods, model_path, n_workers=32):
    chunk_size = 200
    chunks = [
        (time[i:i+chunk_size], flux[i:i+chunk_size], periods, model_path)
        for i in range(0, len(time), chunk_size) if len(time[i:i+chunk_size]) >= 50
    ]
    avg_power = np.zeros(len(periods))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(chunk_periodogram_worker, chunks), total=len(chunks), desc="Chunk Periodogram"))

    avg_power = np.mean(results, axis=0)
    return avg_power


def sliding_window_worker(args):
    window_time, window_flux, periods, model_path = args
    knn, model = get_nn_fap_model(model_path)
    power = np.array([
        1.0 - NN_FAP.inference(period, window_flux, window_time, knn, model)
        for period in periods
    ])
    return power

def create_nn_fap_sliding_window_periodogram(time, flux, periods, model_path, window_size=200, step=50, n_workers=32):
    windows = [
        (time[i:i+window_size], flux[i:i+window_size], periods, model_path)
        for i in range(0, len(time)-window_size+1, step) if len(time[i:i+window_size]) >= 50
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(sliding_window_worker, windows), total=len(windows), desc="Sliding Window Periodogram"))

    avg_power = np.mean(results, axis=0)
    return avg_power




def find_orbital_period(time, flux, error):
    """
    Find orbital period using multiple NN_FAP periodogram methods.
    
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
    tuple
        - Best period from method 1 (days)
        - Period uncertainty (days)
        - Periods array
        - Chunk method power array
        - Sliding window method power array
        - Subtraction method power array
    """


    
    # Create period grids to search
    long_periods = np.linspace(0.1, 10, 100)
    short_periods = np.linspace(0.001, 1, 100)
    
    # Method 1: Chunk periodogram
    model_path = '/home/njm/Period/NN_FAP/final_12l_dp_all/'
    sliding_power = create_nn_fap_sliding_window_periodogram(time, flux, short_periods, model_path, n_workers=80)
    
    chunk_power = create_nn_fap_chunk_periodogram(time, flux, long_periods, model_path, n_workers=80)
    chunk_power = np.interp(short_periods, long_periods, chunk_power)
    
    # Method 3 (2b): Subtraction method to enhance short periods
    # Clip negative values to zero after subtraction
    subtraction_power = np.clip(sliding_power - chunk_power, 0, None)
    
    # Find the best period from each method
    chunk_best_idx = np.argmax(chunk_power)
    sliding_best_idx = np.argmax(sliding_power)
    subtraction_best_idx = np.argmax(subtraction_power)
    
    chunk_best_period = short_periods[chunk_best_idx]
    sliding_best_period = short_periods[sliding_best_idx]
    subtraction_best_period = short_periods[subtraction_best_idx]
    
    print(f"Method 1 (Chunk): Best period = {chunk_best_period*24:.6f} hours")
    print(f"Method 2 (Sliding): Best period = {sliding_best_period*24:.6f} hours")
    print(f"Method 3 (Subtraction): Best period = {subtraction_best_period*24:.6f} hours")
    
    # Use the best period from the chunk method (Method 1) as the default
    best_period = subtraction_best_period
    best_idx = subtraction_best_idx
    
    # Estimate uncertainty based on the width of the peak
    try:
        # Find indices where power is greater than half the max power
        high_power_idx = np.where(chunk_power > 0.5 * chunk_power[best_idx])[0]
        if len(high_power_idx) > 1:
            # Use the width of the peak as the uncertainty
            period_uncertainty = 0.5 * (short_periods[high_power_idx[-1]] - short_periods[high_power_idx[0]])
        else:
            # If we can't determine the width, use 1% of the period as a default
            period_uncertainty = 0.01 * best_period
    except:
        period_uncertainty = 0.01 * best_period
    
    return best_period, period_uncertainty, short_periods, chunk_power, sliding_power, subtraction_power



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




def analyze_cv_target(tic_id, common_name, cv_type, output_dir, cycles):
    """
    Analyze a CV target using the most recent cycle.
    
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
        
    
    result = find_orbital_period(time, flux, error)
    period, period_err, periods, chunk_power, sliding_power, subtraction_power = result
    
    print(f"Best period from chunk method: {period*24:.6f} ± {period_err*24:.6f} hours")
    
    # Phase fold and bin using the best period from the chunk method
    phase, folded_flux, folded_error = phase_fold_lightcurve(time, flux, error, period)
    bin_phase, bin_flux, bin_error = bin_phased_lightcurve(phase, folded_flux, folded_error)
    
    # Calculate phase coverage
    phase_bins = np.linspace(0, 1, 20)
    phase_hist, _ = np.histogram(phase, bins=phase_bins)
    phase_coverage = np.sum(phase_hist > 0) / len(phase_bins)
    
    # Calculate SNR of the folded curve
    # For eclipse-like signals, use the ratio of depth to noise
    mean_flux = np.median(bin_flux)
    depth = mean_flux - np.min(bin_flux)
    noise = np.median(bin_error)
    snr = depth / noise if noise > 0 else 0
    
    # Create the analysis results
    analysis_result = {
        "tic_id": tic_id,
        "common_name": common_name,
        "cv_type": cv_type,
        "period": period,
        "period_err": period_err,
        "time_span": time.max() - time.min(),
        "n_points": len(time),
        "phase_coverage": phase_coverage,
        "snr": snr,
        "folded_data": (bin_phase, bin_flux, bin_error),
        "periods": periods,
        "chunk_power": chunk_power,
        "sliding_power": sliding_power,
        "subtraction_power": subtraction_power,
    }
    
    # Create summary plots
    create_summary_plots(common_name, cv_type, analysis_result, target_dir)
    
    return analysis_result




def create_summary_plots(star_name, cv_type, result, output_dir):
    """
    Create summary plots for a CV target.
    
    Parameters:
    -----------
    star_name : str
        Name of the star
    cv_type : str
        CV type
    result : dict
        Analysis result dictionary
    output_dir : str
        Output directory
    """
    # Extract data from result
    periods = result["periods"]
    chunk_power = result["chunk_power"]
    sliding_power = result["sliding_power"]
    subtraction_power = result["subtraction_power"]
    bin_phase, bin_flux, bin_error = result["folded_data"]
    period = result["period"]
    period_err = result["period_err"]
    
    # Create a figure with 2x2 grid
    plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Plot 1: Chunk Method Periodogram
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(periods * 24, chunk_power, 'b-', linewidth=1.5)
    
    # Mark the best period
    chunk_best_idx = np.argmax(chunk_power)
    chunk_best_period = periods[chunk_best_idx] * 24  # Hours
    ax1.axvline(chunk_best_period, color='r', linestyle='--', alpha=0.7)
    ax1.scatter([chunk_best_period], [chunk_power[chunk_best_idx]], color='red', s=50, marker='o', zorder=5)
    
    ax1.set_xlabel('Period (hours)')
    ax1.set_ylabel('Power (1-FAP)')
    ax1.set_title('Method 1: Chunk Periodogram')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sliding Window Periodogram
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(periods * 24, sliding_power, 'g-', linewidth=1.5)
    
    # Mark the best period
    sliding_best_idx = np.argmax(sliding_power)
    sliding_best_period = periods[sliding_best_idx] * 24  # Hours
    ax2.axvline(sliding_best_period, color='r', linestyle='--', alpha=0.7)
    ax2.scatter([sliding_best_period], [sliding_power[sliding_best_idx]], color='red', s=50, marker='o', zorder=5)
    
    ax2.set_xlabel('Period (hours)')
    ax2.set_ylabel('Power (1-FAP)')
    ax2.set_title('Method 2: Sliding Window Periodogram')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Subtraction Method Periodogram
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(periods * 24, subtraction_power, 'purple', linewidth=1.5)
    
    # Mark the best period
    subtraction_best_idx = np.argmax(subtraction_power)
    subtraction_best_period = periods[subtraction_best_idx] * 24  # Hours
    ax3.axvline(subtraction_best_period, color='r', linestyle='--', alpha=0.7)
    ax3.scatter([subtraction_best_period], [subtraction_power[subtraction_best_idx]], color='red', s=50, marker='o', zorder=5)
    
    ax3.set_xlabel('Period (hours)')
    ax3.set_ylabel('Power (1-FAP)')
    ax3.set_title('Method 3: Subtraction (Method 2 - Method 1)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Phase-folded Light Curve
    ax4 = plt.subplot(gs[1, 1])
    
    # Plot the binned data
    ax4.errorbar(bin_phase, bin_flux, yerr=bin_error, fmt='o', color='blue', 
                 alpha=0.7, ecolor='lightblue', markersize=4)
    
    # Add a second cycle
    ax4.errorbar(bin_phase + 1, bin_flux, yerr=bin_error, fmt='o', color='blue', 
                 alpha=0.7, ecolor='lightblue', markersize=4)
    
    # Try to fit a smoothing spline or savgol filter
    try:
        # Fill in any gaps first
        nan_mask = np.isnan(bin_flux)
        if not all(nan_mask):
            x_valid = bin_phase[~nan_mask]
            y_valid = bin_flux[~nan_mask]
            
            # Smooth with Savitzky-Golay filter
            if len(y_valid) > 10:
                window_length = min(15, len(y_valid) // 4 * 2 + 1)  # Odd number
                if window_length > 3:
                    y_smooth = savgol_filter(y_valid, window_length, 3)
                    ax4.plot(x_valid, y_smooth, 'r-', linewidth=2)
                    ax4.plot(x_valid + 1, y_smooth, 'r-', linewidth=2)
    except Exception as e:
        print(f"Error in smoothing: {e}")
    
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Relative Flux')
    ax4.set_title(f'Phase-folded Light Curve (P = {period*24:.6f} h)')
    ax4.set_xlim(0, 2)
    ax4.grid(True, alpha=0.3)
    
    # Add a note with the best orbital period
    hours_per_day = 24
    period_hours = period * hours_per_day
    period_err_hours = period_err * hours_per_day
    ax4.annotate(f'Period: {period_hours:.6f} ± {period_err_hours:.6f} h', 
                xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    
    # Add overall title
    plt.suptitle(f'{star_name} ({cv_type}) - NN_FAP Periodogram Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{star_name.replace(' ', '_')}_periodogram_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed comparison of the three methods
    plt.figure(figsize=(12, 8))
    
    # Normalize all powers to the same scale for easier comparison
    if np.max(chunk_power) > 0:
        norm_chunk = chunk_power / np.max(chunk_power)
    else:
        norm_chunk = chunk_power
        
    if np.max(sliding_power) > 0:
        norm_sliding = sliding_power / np.max(sliding_power)
    else:
        norm_sliding = sliding_power
        
    if np.max(subtraction_power) > 0:
        norm_subtraction = subtraction_power / np.max(subtraction_power)
    else:
        norm_subtraction = subtraction_power
    
    # Plot all methods together
    plt.plot(periods * 24, norm_chunk, 'b-', linewidth=2, alpha=0.7, label='Method 1: Chunk')
    plt.plot(periods * 24, norm_sliding, 'g-', linewidth=2, alpha=0.7, label='Method 2: Sliding Window')
    plt.plot(periods * 24, norm_subtraction, 'purple', linewidth=2, alpha=0.7, label='Method 3: Subtraction')
    
    # Mark the best periods
    chunk_best_idx = np.argmax(chunk_power)
    sliding_best_idx = np.argmax(sliding_power)
    subtraction_best_idx = np.argmax(subtraction_power)
    
    chunk_best_period = periods[chunk_best_idx] * 24
    sliding_best_period = periods[sliding_best_idx] * 24
    subtraction_best_period = periods[subtraction_best_idx] * 24
    
    plt.axvline(chunk_best_period, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(sliding_best_period, color='green', linestyle='--', alpha=0.5)
    plt.axvline(subtraction_best_period, color='purple', linestyle='--', alpha=0.5)
    
    plt.xlabel('Period (hours)')
    plt.ylabel('Normalized Power')
    plt.title(f'{star_name} - Comparison of NN_FAP Periodogram Methods')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate the best periods
    y_pos = 0.9
    plt.annotate(f'Method 1: {chunk_best_period:.6f} h', 
                xy=(0.65, y_pos), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    plt.annotate(f'Method 2: {sliding_best_period:.6f} h', 
                xy=(0.65, y_pos-0.07), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    plt.annotate(f'Method 3: {subtraction_best_period:.6f} h', 
                xy=(0.65, y_pos-0.14), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8))
    
    plt.savefig(os.path.join(output_dir, f"{star_name.replace(' ', '_')}_method_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Configuration parameters (hardcoded instead of using command-line arguments)
    config = {
        "output_dir": "../PRIMVS/cv_results/NN_TESS/",
        "cadence": "short",
        "max_cycles": 1,  # Only use the most recent cycle
        "specific_tic_ids": [],  # Leave empty to use default targets, or add specific TIC IDs
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
    
    print(f"\nAnalysis complete. Results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()