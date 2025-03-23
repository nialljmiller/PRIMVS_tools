#!/usr/bin/env python3
"""
TESS CV Multi-Cycle Processor

This script processes TESS light curves for cataclysmic variable (CV) stars across multiple observation cycles.
It demonstrates how additional cycles improve characterization of these complex variable systems.

Features:
- Downloads and processes light curves for specified CV stars
- Analyzes period determination improvement with additional cycles
- Characterizes flickering properties and how they vary by cycle
- Generates visualization of light curves and period analysis

Usage:
    python tess_cv_processor.py --output OUTPUT_DIR [--tics TIC_IDS [TIC_IDS ...]]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from tqdm import tqdm
import lightkurve as lk
from scipy import stats
from scipy.signal import savgol_filter

# Well-known CVs with multiple TESS cycles
DEFAULT_CV_TARGETS = {
    # Target: (TIC ID, Common Name, CV Type)
    "SS_Cyg": (149253887, "SS Cyg", "Dwarf Nova"),
    "V2051_Oph": (309953133, "V2051 Oph", "Eclipsing Dwarf Nova"),
    "AM_Her": (118327563, "AM Her", "Polar"),
    "TX_Col": (233187779, "TX Col", "Intermediate Polar"),
    "TW_Pic": (260130483, "TW Pic", "Intermediate Polar")
}

def setup_args():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="Process TESS CV light curves across multiple cycles")
    parser.add_argument("--output", type=str, default="./cv_results", 
                        help="Output directory for results")
    parser.add_argument("--tics", type=int, nargs="+", 
                        help="Specific TIC IDs to process (default: preset CVs)")
    parser.add_argument("--max-cycles", type=int, default=8,
                        help="Maximum number of cycles to process (default: 8)")
    parser.add_argument("--cadence", type=str, default="short", choices=["short", "long"],
                        help="TESS cadence to download (default: short)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading data (use existing files only)")
    return parser.parse_args()


def get_cv_targets(args):
    """Get list of CV targets to process, either from args or default list."""
    if args.tics:
        # Use specified TIC IDs
        targets = []
        for tic in args.tics:
            # Try to match to known targets, otherwise use generic name
            for name, (known_tic, common_name, cv_type) in DEFAULT_CV_TARGETS.items():
                if tic == known_tic:
                    targets.append((tic, common_name, cv_type))
                    break
            else:
                targets.append((tic, f"TIC {tic}", "Unknown"))
        return targets
    else:
        # Use default targets
        return [(tic, common_name, cv_type) for _, (tic, common_name, cv_type) in DEFAULT_CV_TARGETS.items()]


def download_lightcurves(tic_id, output_dir, cadence="short", max_cycles=8):
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
        Maximum number of cycles to download
        
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
        search_result = search_result[search_result.exptime.astype(int) < 600]
    else:
        search_result = search_result[search_result.exptime.astype(int) >= 600]
    
    if len(search_result) == 0:
        print(f"No {cadence} cadence data found for TIC {tic_id}")
        return {}
    
    print(f"Found {len(search_result)} {cadence} cadence observations")
    
    # Group by cycle/sector
    cycles = {}
    for idx, row in enumerate(search_result):
        if idx >= max_cycles:
            break
            
        cycle_num = idx + 1
        output_file = os.path.join(target_dir, f"cycle_{cycle_num}.fits")
        
        # Download if file doesn't exist
        if not os.path.exists(output_file):
            print(f"Downloading cycle {cycle_num} data...")
            try:
                lc = row.download()
                lc.to_fits(output_file, overwrite=True)
                print(f"Saved to {output_file}")
            except Exception as e:
                print(f"Error downloading data: {e}")
                continue
        else:
            print(f"Using existing file: {output_file}")
            
        cycles[cycle_num] = output_file
    
    return cycles


def process_lightcurve(lc_file):
    """
    Process a TESS light curve file.
    
    Parameters:
    -----------
    lc_file : str
        Path to the light curve FITS file
        
    Returns:
    --------
    tuple
        - Processed light curve object
        - Time array (days)
        - Flux array (normalized)
        - Error array (normalized)
    """
    try:
        # Load light curve
        lc = lk.read(lc_file)
        
        # Basic cleaning
        clean_lc = lc.remove_outliers(sigma=5)
        
        # Normalize flux
        median_flux = np.median(clean_lc.flux.value)
        time = clean_lc.time.value
        flux = clean_lc.flux.value / median_flux
        error = clean_lc.flux_err.value / median_flux if hasattr(clean_lc, 'flux_err') else np.ones_like(flux) * 0.001
        
        return clean_lc, time, flux, error
    
    except Exception as e:
        print(f"Error processing {lc_file}: {e}")
        return None, None, None, None


def find_orbital_period(time, flux, error, min_period=0.01, max_period=1.0, n_periods=10000):
    """
    Find orbital period using Lomb-Scargle periodogram.
    
    Parameters:
    -----------
    time : array
        Time array (days)
    flux : array
        Flux array (normalized)
    error : array
        Error array
    min_period : float
        Minimum period to search (days)
    max_period : float
        Maximum period to search (days)
    n_periods : int
        Number of periods to test
        
    Returns:
    --------
    tuple
        - Best period (days)
        - Period uncertainty (days)
        - Periodogram object
        - Frequency array
        - Power array
    """
    # Prepare frequency grid
    frequency = np.linspace(1/max_period, 1/min_period, n_periods)
    
    # Compute periodogram
    ls = LombScargle(time, flux, error)
    power = ls.power(frequency)
    
    # Find the best period
    best_idx = np.argmax(power)
    best_frequency = frequency[best_idx]
    best_period = 1.0 / best_frequency
    
    # Estimate uncertainty
    # This is a simple estimate based on the FWHM of the periodogram peak
    # For a proper uncertainty, a bootstrap method would be better
    try:
        fwhm_idx = np.where(power > 0.5 * power[best_idx])[0]
        if len(fwhm_idx) > 1:
            period_uncertainty = 0.5 * (1.0 / frequency[fwhm_idx[0]] - 1.0 / frequency[fwhm_idx[-1]])
        else:
            # Default to 1% uncertainty if FWHM method fails
            period_uncertainty = 0.01 * best_period
    except:
        period_uncertainty = 0.01 * best_period
    
    return best_period, period_uncertainty, ls, frequency, power


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


def combine_cycles(cycle_data, max_cycles=None):
    """
    Combine data from multiple cycles.
    
    Parameters:
    -----------
    cycle_data : dict
        Dictionary mapping cycle numbers to tuples of (time, flux, error)
    max_cycles : int, optional
        Maximum number of cycles to combine
        
    Returns:
    --------
    list
        List of tuples (cycle_count, combined_time, combined_flux, combined_error)
    """
    if max_cycles is None:
        max_cycles = len(cycle_data)
    
    combined_data = []
    
    for n_cycles in range(1, min(max_cycles+1, len(cycle_data)+1)):
        all_time = np.concatenate([cycle_data[i+1][0] for i in range(n_cycles)])
        all_flux = np.concatenate([cycle_data[i+1][1] for i in range(n_cycles)])
        all_error = np.concatenate([cycle_data[i+1][2] for i in range(n_cycles)])
        
        combined_data.append((n_cycles, all_time, all_flux, all_error))
    
    return combined_data


def analyze_cv_target(tic_id, common_name, cv_type, output_dir, cycles, args):
    """
    Analyze a CV target across multiple cycles.
    
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
    args : argparse.Namespace
        Command line arguments
        
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
    
    # Process individual cycles
    cycle_data = {}
    cycle_results = {}
    
    for cycle_num, lc_file in cycles.items():
        print(f"Processing cycle {cycle_num}...")
        
        # Process light curve
        lc, time, flux, error = process_lightcurve(lc_file)
        if time is None:
            print(f"Skipping cycle {cycle_num} due to processing error")
            continue
            
        cycle_data[cycle_num] = (time, flux, error)
        
        # Find orbital period
        result = find_orbital_period(time, flux, error)
        period, period_err, ls, frequency, power = result
        
        cycle_results[cycle_num] = {
            "period": period,
            "period_err": period_err,
            "time_span": time.max() - time.min(),
            "n_points": len(time),
            "phase_coverage": None,  # Will compute later
            "snr": None,  # Will compute later
        }
        
        print(f"  Period: {period*24:.6f} ± {period_err*24:.6f} hours")
    
    if not cycle_data:
        print(f"No valid cycles for {common_name}")
        return None
        
    # Now do the combined analysis
    combined_results = []
    combined_data = combine_cycles(cycle_data)
    
    for idx, (n_cycles, all_time, all_flux, all_error) in enumerate(combined_data):
        print(f"Analyzing combined data: cycles 1-{n_cycles}...")
        
        # Find orbital period
        result = find_orbital_period(all_time, all_flux, all_error)
        period, period_err, ls, frequency, power = result
        
        # Phase fold and bin
        phase, folded_flux, folded_error = phase_fold_lightcurve(all_time, all_flux, all_error, period)
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
        
        result = {
            "n_cycles": n_cycles,
            "period": period,
            "period_err": period_err,
            "time_span": all_time.max() - all_time.min(),
            "n_points": len(all_time),
            "phase_coverage": phase_coverage,
            "snr": snr,
            "folded_data": (bin_phase, bin_flux, bin_error),
            "frequency": frequency,
            "power": power,
        }
        
        combined_results.append(result)
        
        print(f"  Period: {period*24:.6f} ± {period_err*24:.6f} hours")
        print(f"  SNR: {snr:.2f}, Phase coverage: {phase_coverage:.2f}")
    
    # Create summary plots
    create_summary_plots(common_name, cv_type, combined_results, target_dir)
    
    return {
        "tic_id": tic_id,
        "common_name": common_name,
        "cv_type": cv_type,
        "cycle_results": cycle_results,
        "combined_results": combined_results,
    }


def create_summary_plots(star_name, cv_type, results, output_dir):
    """
    Create summary plots for a CV target.
    
    Parameters:
    -----------
    star_name : str
        Name of the star
    cv_type : str
        CV type
    results : list
        List of combined results dictionaries
    output_dir : str
        Output directory
    """
    # Plot 1: Period error vs. number of cycles
    plt.figure(figsize=(12, 9))
    
    # Create a 2x2 grid
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Plot 1: Period precision improvement
    ax1 = plt.subplot(gs[0, 0])
    
    n_cycles = [r["n_cycles"] for r in results]
    period_err = [r["period_err"] * 24 * 3600 for r in results]  # Convert to seconds
    
    ax1.plot(n_cycles, period_err, 'o-', color='blue', linewidth=2, markersize=8)
    
    # Add the theoretical 1/sqrt(N) improvement
    if len(n_cycles) > 1:
        initial_err = period_err[0]
        theoretical = [initial_err / np.sqrt(n) for n in n_cycles]
        ax1.plot(n_cycles, theoretical, 'k--', label='Theoretical: 1/√N', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Number of Cycles')
    ax1.set_ylabel('Period Uncertainty (seconds)')
    ax1.set_title('Period Precision Improvement')
    ax1.grid(True, alpha=0.3)
    if len(n_cycles) > 1:
        ax1.legend()
    
    # Plot 2: SNR vs. number of cycles
    ax2 = plt.subplot(gs[0, 1])
    
    snr = [r["snr"] for r in results]
    
    ax2.plot(n_cycles, snr, 'o-', color='green', linewidth=2, markersize=8)
    
    # Add the theoretical sqrt(N) improvement
    if len(n_cycles) > 1 and snr[0] > 0:
        initial_snr = snr[0]
        theoretical = [initial_snr * np.sqrt(n) for n in n_cycles]
        ax2.plot(n_cycles, theoretical, 'k--', label='Theoretical: √N', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Number of Cycles')
    ax2.set_ylabel('Signal-to-Noise Ratio')
    ax2.set_title('SNR Improvement')
    ax2.grid(True, alpha=0.3)
    if len(n_cycles) > 1 and snr[0] > 0:
        ax2.legend()
    
    # Plot 3: Phase-folded light curve for best period (using all cycles)
    ax3 = plt.subplot(gs[1, :])
    
    # Get the folded data from the last result (all cycles)
    bin_phase, bin_flux, bin_error = results[-1]["folded_data"]
    
    # Plot the binned data
    ax3.errorbar(bin_phase, bin_flux, yerr=bin_error, fmt='o', color='blue', 
                 alpha=0.7, ecolor='lightblue', markersize=4)
    
    # Add a second cycle
    ax3.errorbar(bin_phase + 1, bin_flux, yerr=bin_error, fmt='o', color='blue', 
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
                    ax3.plot(x_valid, y_smooth, 'r-', linewidth=2)
                    ax3.plot(x_valid + 1, y_smooth, 'r-', linewidth=2)
    except Exception as e:
        print(f"Error in smoothing: {e}")
    
    period = results[-1]["period"]
    period_err = results[-1]["period_err"]
    
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Relative Flux')
    ax3.set_title(f'Phase-folded Light Curve: P = {period*24:.6f} ± {period_err*24:.6f} hours')
    ax3.set_xlim(0, 2)
    ax3.grid(True, alpha=0.3)
    
    # Add a note with the best orbital period
    hours_per_day = 24
    period_hours = period * hours_per_day
    period_err_hours = period_err * hours_per_day
    ax3.annotate(f'Orbital Period: {period_hours:.6f} ± {period_err_hours:.6f} hours', 
                xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    
    # Add overall title
    plt.suptitle(f'{star_name} ({cv_type}) - Multi-cycle Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{star_name.replace(' ', '_')}_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the periodogram
    plt.figure(figsize=(10, 6))
    frequency = results[-1]["frequency"]
    power = results[-1]["power"]
    
    periods = 1/frequency
    in_range = (periods > 0.01) & (periods < 2.0)  # Limit range for clarity
    
    plt.plot(periods[in_range] * 24, power[in_range])  # Convert to hours
    
    # Mark the detected period
    best_period = results[-1]["period"] * 24  # Hours
    best_idx = np.argmin(np.abs(periods * 24 - best_period))
    plt.axvline(best_period, color='r', linestyle='--', alpha=0.7)
    plt.scatter([best_period], [power[best_idx]], color='red', s=100, marker='o', zorder=5)
    
    plt.xlabel('Period (hours)')
    plt.ylabel('Power')
    plt.title(f'{star_name} - Lomb-Scargle Periodogram (All Cycles)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, f"{star_name.replace(' ', '_')}_periodogram.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_comparative_plot(all_results, output_dir):
    """
    Create a comparative plot showing period precision improvement vs. number of cycles.
    
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
        print("No valid results for comparative plot")
        return
    
    # Extract data for plotting
    cv_names = [r["common_name"] for r in all_results]
    cv_types = [r["cv_type"] for r in all_results]
    
    plt.figure(figsize=(12, 8))
    
    type_colors = {
        "Dwarf Nova": "blue",
        "Eclipsing Dwarf Nova": "green",
        "Polar": "red",
        "Intermediate Polar": "purple",
        "Unknown": "gray"
    }
    
    # Create legend entries to avoid duplicates
    legend_entries = {}
    
    # Plot relative period precision improvement for each CV
    for result in all_results:
        if not result["combined_results"]:
            continue
            
        name = result["common_name"]
        cv_type = result["cv_type"]
        
        n_cycles = [r["n_cycles"] for r in result["combined_results"]]
        if len(n_cycles) <= 1:
            continue
            
        # Convert period errors to relative improvement
        period_err = [r["period_err"] for r in result["combined_results"]]
        relative_err = [err / period_err[0] for err in period_err]
        
        # Plot with color based on CV type
        color = type_colors.get(cv_type, "gray")
        
        # Only add to legend if type not already there
        if cv_type not in legend_entries:
            plt.plot(n_cycles, relative_err, 'o-', color=color, linewidth=2, markersize=8, 
                     label=cv_type, alpha=0.8)
            legend_entries[cv_type] = True
        else:
            plt.plot(n_cycles, relative_err, 'o-', color=color, linewidth=2, markersize=8, 
                     alpha=0.8)
        
        # Add star name as annotation
        plt.annotate(name, (n_cycles[-1], relative_err[-1]), 
                     xytext=(5, 0), textcoords='offset points', 
                     fontsize=9, alpha=0.8)
    
    # Add the theoretical 1/sqrt(N) improvement
    max_cycles = max([max([r["n_cycles"] for r in result["combined_results"]]) 
                     for result in all_results if result["combined_results"]])
    
    n_values = np.arange(1, max_cycles+1)
    theoretical = [1 / np.sqrt(n) for n in n_values]
    
    plt.plot(n_values, theoretical, 'k--', linewidth=2, label='Theoretical: 1/√N', alpha=0.7)
    
    plt.xlabel('Number of Cycles')
    plt.ylabel('Relative Period Uncertainty')
    plt.title('Period Precision Improvement with Multiple TESS Cycles')
    plt.yscale('log')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(title="CV Type")
    
    plt.savefig(os.path.join(output_dir, "period_precision_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Parse arguments
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get CV targets
    targets = get_cv_targets(args)
    
    print(f"Will process {len(targets)} CV targets")
    
    # Process each target
    all_results = []
    
    for tic_id, common_name, cv_type in targets:
        # Download light curves
        if not args.no_download:
            cycles = download_lightcurves(tic_id, args.output, args.cadence, args.max_cycles)
        else:
            # Look for existing files
            target_dir = os.path.join(args.output, f"TIC_{tic_id}")
            if not os.path.exists(target_dir):
                print(f"No data directory found for TIC {tic_id}")
                continue
                
            cycles = {}
            for cycle_num in range(1, args.max_cycles + 1):
                cycle_file = os.path.join(target_dir, f"cycle_{cycle_num}.fits")
                if os.path.exists(cycle_file):
                    cycles[cycle_num] = cycle_file
        
        if not cycles:
            print(f"No cycles found for {common_name} (TIC {tic_id})")
            continue
            
        print(f"Found {len(cycles)} cycles for {common_name} (TIC {tic_id})")
        
        # Analyze
        result = analyze_cv_target(tic_id, common_name, cv_type, args.output, cycles, args)
        all_results.append(result)
    
    # Create comparative plot
    create_comparative_plot(all_results, args.output)
    
    print(f"\nAnalysis complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()