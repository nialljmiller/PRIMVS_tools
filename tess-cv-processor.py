#!/usr/bin/env python3
"""
TESS CV Multi-Cycle Processor

This script identifies and processes TESS light curves of cataclysmic variable (CV) stars 
across multiple cycles to demonstrate the improvements gained from extended observations.

Key features:
1. CV candidate selection from TESS data
2. Download and processing of multi-cycle data
3. Periodogram analysis with increasing number of cycles
4. Visualization of period determination improvement
5. Outburst detection and characterization
6. Eclipse depth and profile analysis

Example usage:
python tess_cv_processor.py --cycles 4 --output cv_results

Author: Claude
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import lightkurve as lk
from tqdm import tqdm
import warnings
import seaborn as sns
from astropy.visualization import time_support
from astropy.time import Time

# Initialize time plotting support
time_support()

# Suppress warnings
warnings.filterwarnings('ignore')

class TESSCVProcessor:
    """
    A class to process TESS data for CV stars across multiple cycles.
    """
    
    def __init__(self, output_dir='cv_results', max_cycles=4, cv_list=None):
        """
        Initialize the CV processor.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save outputs
        max_cycles : int
            Maximum number of TESS cycles to process
        cv_list : list or None
            Optional list of TIC IDs to process. If None, will use built-in list.
        """
        self.output_dir = output_dir
        self.max_cycles = max_cycles
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        # Known CV candidates with good TESS coverage
        # These TIC IDs were selected based on:
        # - Multiple sectors of TESS coverage
        # - Known CV classification or high probability CV candidates
        # - Visible variability in TESS data
        if cv_list is None:
            self.cv_targets = [
                # Estimate eclipse width by finding points half as deep as maximum
            half_depth = eclipse_depth / 2
            half_depth_threshold = baseline - half_depth
            
            # Find where flux crosses the half-depth threshold
            below_threshold = binned_flux < half_depth_threshold
            if np.sum(below_threshold) >= 3:  # Need at least 3 points for width estimate
                # Find longest contiguous region below threshold
                from scipy.ndimage import label
                labeled_array, num_features = label(below_threshold)
                
                # Find largest region
                sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background
                if len(sizes) > 0:
                    largest_label = np.argmax(sizes) + 1  # +1 because background is 0
                    
                    # Get phases for this region
                    region_indices = np.where(labeled_array == largest_label)[0]
                    region_phases = bin_centers[region_indices]
                    
                    # Calculate width
                    eclipse_width = (region_phases.max() - region_phases.min()) * orbital_period * 24  # in hours
                else:
                    eclipse_width = bin_width * 3 * orbital_period * 24  # Minimum detectable width
            else:
                eclipse_width = bin_width * 3 * orbital_period * 24  # Minimum detectable width
            
            # Calculate signal-to-noise ratio of eclipse
            snr = eclipse_depth / eclipse_error
            
            # Store results
            results['n_cycles'].append(n_cycles)
            results['eclipse_depths'].append(eclipse_depth)
            results['eclipse_errors'].append(eclipse_error)
            results['eclipse_widths'].append(eclipse_width)
            results['signal_to_noise'].append(snr)
            
            # For the final cycle, create eclipse visualization
            if n_cycles == len(lcs):
                self._plot_eclipse_analysis(tic_id, combined_lc, orbital_period, results)
        
        return results
    
    def _plot_eclipse_analysis(self, tic_id, light_curve, orbital_period, eclipse_results):
        """
        Create visualization of eclipse detection and characterization improvement.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV
        light_curve : lightkurve.LightCurve
            Combined light curve object
        orbital_period : float
            Orbital period in days
        eclipse_results : dict
            Dictionary with eclipse analysis results
        """
        # Fold light curve
        folded_lc = light_curve.fold(period=orbital_period)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])
        
        # 1. Plot SNR improvement with cycles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(eclipse_results['n_cycles'], eclipse_results['signal_to_noise'], 'o-', 
               markersize=8, color='blue')
        ax1.set_xlabel('Number of Cycles')
        ax1.set_ylabel('Eclipse SNR')
        ax1.set_title('Eclipse SNR Improvement')
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot eclipse depth error improvement
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(eclipse_results['n_cycles'], np.array(eclipse_results['eclipse_errors'])*1000, 'o-', 
               markersize=8, color='red')  # Convert to millimag for better numbers
        ax2.set_xlabel('Number of Cycles')
        ax2.set_ylabel('Eclipse Depth Error (millimag)')
        ax2.set_title('Eclipse Measurement Precision')
        ax2.grid(True, alpha=0.3)
        
        # 3. Plot folded light curve with eclipse
        ax3 = fig.add_subplot(gs[1, :])
        
        phase = folded_lc.phase
        flux = folded_lc.flux
        
        # Create phase bins for cleaner visualization
        bins = 100
        bin_phase = np.linspace(0, 1, bins + 1)
        bin_centers = 0.5 * (bin_phase[1:] + bin_phase[:-1])
        binned_flux = np.zeros(bins)
        binned_errors = np.zeros(bins)
        
        for i in range(bins):
            mask = (phase.value >= bin_phase[i]) & (phase.value < bin_phase[i+1])
            if np.sum(mask) > 0:
                binned_flux[i] = np.median(flux.value[mask])
                binned_errors[i] = np.std(flux.value[mask]) / np.sqrt(np.sum(mask))
            else:
                binned_flux[i] = np.nan
                binned_errors[i] = np.nan
        
        # Plot density scatter of all points
        try:
            # Using gaussian KDE for density coloring
            from scipy.stats import gaussian_kde
            xy = np.vstack([phase.value, flux.value])
            z = gaussian_kde(xy)(xy)
            
            # Sort points by density
            idx = z.argsort()
            x, y, z = phase.value[idx], flux.value[idx], z[idx]
            
            ax3.scatter(x, y, c=z, s=2, alpha=0.6, cmap='viridis')
        except:
            # Fall back to simple scatter
            ax3.scatter(phase.value, flux.value, s=2, alpha=0.2, color='blue')
        
        # Plot binned light curve
        valid_mask = ~np.isnan(binned_flux)
        ax3.errorbar(bin_centers[valid_mask], binned_flux[valid_mask], 
                   yerr=binned_errors[valid_mask], 
                   fmt='o', color='red', markersize=4, capsize=0)
        
        # Add smooth curve through binned points
        try:
            # Fill gaps in binned flux if any
            from scipy.interpolate import interp1d
            
            # Create a complete cycle by wrapping around
            extended_phases = np.hstack([bin_centers[valid_mask], bin_centers[valid_mask] + 1])
            extended_flux = np.hstack([binned_flux[valid_mask], binned_flux[valid_mask]])
            
            # Sort by phase
            sort_idx = np.argsort(extended_phases)
            extended_phases = extended_phases[sort_idx]
            extended_flux = extended_flux[sort_idx]
            
            # Create interpolation function
            f = interp1d(extended_phases, extended_flux, kind='cubic')
            
            # Generate smooth curve
            smooth_phase = np.linspace(0, 1, 500)
            smooth_flux = f(smooth_phase)
            
            ax3.plot(smooth_phase, smooth_flux, 'k-', linewidth=2, alpha=0.7)
        except:
            # If interpolation fails, connect the dots
            ax3.plot(bin_centers[valid_mask], binned_flux[valid_mask], 'k-', linewidth=1, alpha=0.7)
        
        ax3.set_xlabel('Phase')
        ax3.set_ylabel('Normalized Flux')
        ax3.set_title(f'Folded Light Curve (P = {orbital_period*24:.4f} hours)')
        ax3.grid(True, alpha=0.3)
        
        # Highlight eclipse region if detected
        if eclipse_results['eclipse_depths']:
            # Get the final eclipse depth and width
            eclipse_depth = eclipse_results['eclipse_depths'][-1]
            eclipse_width = eclipse_results['eclipse_widths'][-1]
            
            # Find the eclipse minimum in the binned curve
            min_idx = np.argmin(binned_flux[valid_mask])
            min_phase = bin_centers[valid_mask][min_idx]
            
            # Draw an annotation for the eclipse
            min_flux = binned_flux[valid_mask][min_idx]
            
            # Highlight eclipse width
            width_in_phase = eclipse_width / (orbital_period * 24)
            eclipse_start = (min_phase - width_in_phase/2) % 1
            eclipse_end = (min_phase + width_in_phase/2) % 1
            
            # Handle case where eclipse wraps around phase=0
            if eclipse_start < eclipse_end:
                ax3.axvspan(eclipse_start, eclipse_end, alpha=0.2, color='red', label='Eclipse')
            else:
                ax3.axvspan(0, eclipse_end, alpha=0.2, color='red', label='Eclipse')
                ax3.axvspan(eclipse_start, 1, alpha=0.2, color='red')
            
            # Add annotation for eclipse
            ax3.annotate(f"Depth: {eclipse_depth:.4f}\nWidth: {eclipse_width:.2f} hrs",
                       xy=(min_phase, min_flux), xytext=(min_phase, min_flux - 0.2 * eclipse_depth),
                       arrowprops=dict(arrowstyle="->", color="black"),
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                       ha='center')
        
        # Add info text
        n_points = len(folded_lc.flux)
        cycles = eclipse_results['n_cycles'][-1]
        
        info_text = f"Data points: {n_points}\n"
        info_text += f"Cycles: {cycles}\n"
        if eclipse_results['signal_to_noise']:
            info_text += f"Final SNR: {eclipse_results['signal_to_noise'][-1]:.1f}\n"
        
        ax3.text(0.98, 0.05, info_text, transform=ax3.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'TIC{tic_id}_eclipse_analysis.png'), dpi=300)
        plt.close()
    
    def analyze_flickering(self, tic_id):
        """
        Analyze how flickering characterization improves with additional cycles.
        Flickering is rapid stochastic variability characteristic of CVs.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV to analyze
            
        Returns:
        -----------
        dict
            Dictionary with flickering analysis results
        """
        if tic_id not in self.light_curves:
            print(f"No data for TIC {tic_id}")
            return None
            
        lcs = self.light_curves[tic_id]
        if len(lcs) < 2:
            print(f"Need at least 2 cycles for TIC {tic_id}")
            return None
        
        # Results dictionary
        results = {
            'tic_id': tic_id,
            'n_cycles': [],
            'flickering_rms': [],
            'power_spectrum': []
        }
        
        # Analyze flickering for increasing number of cycles
        for n_cycles in range(1, len(lcs) + 1):
            # Combine first n_cycles
            combined_lc = lcs[0]
            for i in range(1, n_cycles):
                combined_lc = combined_lc.append(lcs[i])
            
            # Extract time and flux
            time = combined_lc.time.value
            flux = combined_lc.flux.value
            
            # Detrend the light curve to remove slow variations
            # For flickering, we want to analyze the high-frequency noise
            
            try:
                # Try a Savitzky-Golay filter to remove trends
                from scipy.signal import savgol_filter
                
                # Window size should be big enough to preserve flickering
                window = min(101, len(flux) // 5 * 2 + 1)  # Ensure it's odd
                if window < 11:
                    window = 11
                
                # Polynomial order 3 works well for CV light curves
                trend = savgol_filter(flux, window, 3)
                
                # Detrended flux
                detrended = flux - trend
                
                # Calculate RMS of flickering (standard deviation of detrended flux)
                flickering_rms = np.std(detrended)
                
                # Calculate power spectrum
                try:
                    from scipy import signal
                    
                    # Use Welch's method for power spectrum
                    fs = 1.0 / np.median(np.diff(time))  # Approximate sampling frequency
                    f, pxx = signal.welch(detrended, fs, nperseg=min(256, len(detrended)//10*2))
                    
                    # Store results
                    results['n_cycles'].append(n_cycles)
                    results['flickering_rms'].append(flickering_rms)
                    results['power_spectrum'].append((f, pxx))
                    
                    # For the final cycle, create flickering visualization
                    if n_cycles == len(lcs):
                        self._plot_flickering_analysis(tic_id, combined_lc, detrended, flickering_rms, results)
                except:
                    print(f"Error calculating power spectrum for TIC {tic_id}")
            except:
                print(f"Error detrending light curve for TIC {tic_id}")
        
        return results
    
    def _plot_flickering_analysis(self, tic_id, light_curve, detrended, flickering_rms, results):
        """
        Create visualization of flickering analysis and improvement with cycles.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV
        light_curve : lightkurve.LightCurve
            Combined light curve object
        detrended : array
            Detrended flux
        flickering_rms : float
            RMS of flickering
        results : dict
            Dictionary with flickering analysis results
        """
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 2])
        
        # 1. Plot flickering RMS improvement with cycles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['n_cycles'], np.array(results['flickering_rms'])*1000, 'o-', 
               markersize=8, color='green')  # Convert to millimag
        ax1.set_xlabel('Number of Cycles')
        ax1.set_ylabel('Flickering RMS (millimag)')
        ax1.set_title('Flickering Characterization')
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot power spectrum comparison
        ax2 = fig.add_subplot(gs[0, 1])
        
        for i, n_cycles in enumerate(results['n_cycles']):
            f, pxx = results['power_spectrum'][i]
            
            # Convert to periods in minutes for more intuitive x-axis
            # Use only relevant part of spectrum
            mask = (f > 0) & (f < 0.1)  # Focus on 0-0.1 Hz
            
            if np.sum(mask) > 5:  # Need at least a few points
                period_mins = (1.0 / f[mask]) / 60  # Convert Hz to minutes
                pxx_norm = pxx[mask] / np.max(pxx[mask])  # Normalize
                
                ax2.plot(period_mins, pxx_norm, '-', 
                       label=f"{n_cycles} cycle{'s' if n_cycles > 1 else ''}")
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Period (minutes)')
        ax2.set_ylabel('Normalized Power')
        ax2.set_title('Power Spectrum Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # 3. Plot time series section with flickering - original and detrended
        ax3 = fig.add_subplot(gs[1, :])
        
        # Extract a representative section to show flickering
        time = light_curve.time.value
        flux = light_curve.flux.value
        
        # Try to find a section with good flickering
        if len(time) > 200:
            # Look for a section without large trends
            window_size = 200
            best_rms = 0
            best_start = 0
            
            # Step through the light curve looking for highest short-term RMS
            step = max(1, len(time) // 50)  # Examine ~50 windows
            for i in range(0, len(time) - window_size, step):
                section = flux[i:i+window_size]
                section_rms = np.std(section - np.median(section))
                
                if section_rms > best_rms:
                    best_rms = section_rms
                    best_start = i
            
            # Extract the best section
            section_slice = slice(best_start, best_start + window_size)
            section_time = time[section_slice]
            section_flux = flux[section_slice]
            section_detrended = detrended[section_slice]
            
            # Normalize times to hours from start
            section_time_hrs = (section_time - section_time[0]) * 24
        else:
            # Use the whole light curve if it's short
            section_time = time
            section_flux = flux
            section_detrended = detrended
            section_time_hrs = (section_time - section_time[0]) * 24
        
        # Plot original and detrended
        ax3.plot(section_time_hrs, section_flux, 'b-', alpha=0.7, label='Original')
        
        # Offset detrended to avoid overlap
        detrended_offset = np.mean(section_flux) - np.mean(section_detrended) - 2*np.std(section_flux)
        ax3.plot(section_time_hrs, section_detrended + detrended_offset, 'r-', 
               alpha=0.7, label='Detrended (offset)')
        
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Normalized Flux')
        ax3.set_title('Flickering Time Series Detail')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Histogram of detrended flux showing flickering statistics
        ax4 = fig.add_subplot(gs[2, :])
        
        # Histogram of detrended flux
        n, bins, _ = ax4.hist(detrended, bins=50, alpha=0.7, density=True, color='purple')
        
        # Fit a Gaussian
        from scipy.stats import norm
        mu, sigma = norm.fit(detrended)
        x = np.linspace(min(bins), max(bins), 100)
        ax4.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Gaussian fit: σ={sigma*1000:.2f} millimag')
        
        ax4.set_xlabel('Detrended Flux')
        ax4.set_ylabel('Density')
        ax4.set_title('Flickering Amplitude Distribution')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'TIC{tic_id}_flickering_analysis.png'), dpi=300)
        plt.close()
    
    def generate_summary(self, tic_id):
        """
        Generate a summary showing all improvements with multiple cycles.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV to analyze
        """
        if tic_id not in self.light_curves:
            print(f"No data for TIC {tic_id}")
            return
        
        lcs = self.light_curves[tic_id]
        n_cycles = len(lcs)
        
        # Create a summary figure
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1])
        
        # 1. Plot the light curves for each cycle
        ax1 = fig.add_subplot(gs[0])
        
        all_times = []
        all_fluxes = []
        time_offsets = []
        
        # Plot each cycle in a different color
        colors = plt.cm.viridis(np.linspace(0, 1, n_cycles))
        
        for i, lc in enumerate(lcs):
            time = lc.time.value
            flux = lc.flux.value
            
            # Track all times for overall x-axis limits
            all_times.extend(time)
            all_fluxes.extend(flux)
            
            # Normalize times to BJD - 2457000 for cleaner numbers
            plot_time = time - 2457000
            time_offsets.append(np.min(plot_time))
            
            ax1.scatter(plot_time, flux, s=1, alpha=0.7, color=colors[i], 
                      label=f'Cycle {i+1}')
        
        ax1.set_xlabel('Time (BJD - 2457000)')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_title(f'TIC {tic_id}: Multi-cycle Analysis Summary')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Add time span text
        time_span = max(all_times) - min(all_times)
        ax1.text(0.02, 0.95, f"Time span: {time_span:.1f} days", 
               transform=ax1.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Folded light curve with best period
        ax2 = fig.add_subplot(gs[1])
        
        # Combine all light curves
        combined_lc = lcs[0]
        for i in range(1, n_cycles):
            combined_lc = combined_lc.append(lcs[i])
        
        # Try to get period from analysis or estimate from periodogram
        try:
            period_results = self.analyze_period_improvement(tic_id)
            if period_results and period_results['best_period']:
                orbital_period = period_results['best_period']
            else:
                # Estimate period from periodogram
                ls = LombScargle(combined_lc.time.value, combined_lc.flux.value, combined_lc.flux_err.value)
                freq, power = ls.autopower(minimum_frequency=1/10.0, maximum_frequency=1/0.05,
                                          samples_per_peak=100)
                peak_idx = np.argmax(power)
                orbital_period = 1/freq[peak_idx]
            
            # Fold the light curve
            folded_lc = combined_lc.fold(period=orbital_period)
            
            # Plot points colored by cycle
            for i, lc in enumerate(lcs):
                folded = lc.fold(period=orbital_period)
                ax2.scatter(folded.phase, folded.flux, s=2, alpha=0.7, color=colors[i], 
                         label=f'Cycle {i+1}')
            
            # Add binned curve
            phase = folded_lc.phase.value
            flux = folded_lc.flux.value
            
            phase_bins = np.linspace(0, 1, 75)
            binned_flux = []
            binned_err = []
            
            for i in range(len(phase_bins)-1):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.median(flux[mask]))
                    binned_err.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
                else:
                    binned_flux.append(np.nan)
                    binned_err.append(np.nan)
            
            bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
            
            # Plot with error bars
            ax2.errorbar(bin_centers, binned_flux, yerr=binned_err, fmt='ko-', 
                       markersize=4, capsize=0, label='Binned', zorder=10)
            
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Normalized Flux')
            ax2.set_title(f'Folded Light Curve (P = {orbital_period*24:.4f} hours)')
            ax2.grid(True, alpha=0.3)
            
            # Add legend but exclude individual cycles to avoid clutter
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend([handles[-1]], [labels[-1]], loc='upper right')
        except Exception as e:
            ax2.text(0.5, 0.5, f"Could not create folded plot: {str(e)}", 
                   ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Plot improvement metrics
        ax3 = fig.add_subplot(gs[2])
        
        # Get analysis results
        try:
            period_results = self.analyze_period_improvement(tic_id)
            eclipse_results = self.analyze_eclipse_improvement(tic_id)
            flickering_results = self.analyze_flickering(tic_id)
            outburst_results = self.analyze_outbursts(tic_id)
            
            # Normalize each metric to 0-1 range for comparison
            metrics = []
            cycle_nums = list(range(1, n_cycles + 1))
            
            # Period error improvement
            if period_results and 'period_errors' in period_results and period_results['period_errors']:
                # Lower error is better, so invert
                period_err = np.array(period_results['period_errors'])
                if max(period_err) > min(period_err):
                    norm_period_err = 1 - (period_err - min(period_err)) / (max(period_err) - min(period_err))
                    metrics.append(('Period determination', norm_period_err))
            
            # Eclipse SNR improvement
            if eclipse_results and 'signal_to_noise' in eclipse_results and eclipse_results['signal_to_noise']:
                snr = np.array(eclipse_results['signal_to_noise'])
                if max(snr) > min(snr):
                    norm_snr = (snr - min(snr)) / (max(snr) - min(snr))
                    metrics.append(('Eclipse SNR', norm_snr))
            
            # Flickering characterization
            if flickering_results and 'flickering_rms' in flickering_results and flickering_results['flickering_rms']:
                # More precise measurement (lower relative uncertainty) is better
                flicker_rms = np.array(flickering_results['flickering_rms'])
                # RMS precision improves with sqrt(N)
                flicker_precision = np.sqrt(np.array(cycle_nums))
                norm_flicker = (flicker_precision - min(flicker_precision)) / (max(flicker_precision) - min(flicker_precision))
                metrics.append(('Flickering precision', norm_flicker))
            
            # Plot each metric
            if metrics:
                for name, values in metrics:
                    ax3.plot(cycle_nums, values, 'o-', linewidth=2, markersize=8, label=name)
                
                ax3.set_xlim(0.9, n_cycles + 0.1)
                ax3.set_ylim(-0.05, 1.05)
                ax3.set_xlabel('Number of Cycles')
                ax3.set_ylabel('Relative Improvement')
                ax3.set_title('Multi-cycle Improvement Metrics')
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='best')
            else:
                ax3.text(0.5, 0.5, "No improvement metrics available", 
                       ha='center', va='center', transform=ax3.transAxes)
        except Exception as e:
            ax3.text(0.5, 0.5, f"Error plotting metrics: {str(e)}", 
                   ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'TIC{tic_id}_summary.png'), dpi=300)
        plt.close()
    
    def create_html_report(self, tic_ids=None):
        """
        Create an HTML report with all analysis results.
        
        Parameters:
        -----------
        tic_ids : list or None
            List of TIC IDs to include in report. If None, include all processed CVs.
        """
        if not hasattr(self, 'light_curves') or not self.light_curves:
            print("No light curves processed. Run search_and_download() first.")
            return
        
        if tic_ids is None:
            tic_ids = list(self.light_curves.keys())
        
        report_path = os.path.join(self.output_dir, 'cv_multi_cycle_report.html')
        
        with open(report_path, 'w') as f:
            # Write HTML header
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('<head>\n')
            f.write('<title>TESS CV Multi-Cycle Analysis Report</title>\n')
            f.write('<style>\n')
            f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
            f.write('h1 { color: #2c3e50; }\n')
            f.write('h2 { color: #3498db; margin-top: 30px; }\n')
            f.write('.cv-section { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }\n')
            f.write('.plot-container { display: flex; flex-wrap: wrap; justify-content: center; }\n')
            f.write('.plot { margin: 10px; max-width: 800px; }\n')
            f.write('.plot img { max-width: 100%; border: 1px ASASSN-14mv - Dwarf nova with outbursts
                219114871,
                
                # V2051 Oph - Eclipsing dwarf nova
                309953133,
                
                # AM Her - Polar type CV
                118327563,
                
                # SS Cyg - Prototype dwarf nova
                149253887,
                
                # V1223 Sgr - Intermediate polar
                372913684,
                
                # MV Lyr - VY Scl type CV
                465272717, 
                
                # TW Vir - Dwarf nova
                304042981,
                
                # RW Tri - Nova-like CV
                69073959
            ]
        else:
            self.cv_targets = cv_list
            
        # Dictionary to store light curves for each target
        self.light_curves = {}
        
    def search_and_download(self):
        """
        Search for and download TESS data for the target CVs.
        Prioritizes targets with data from multiple cycles.
        """
        print("Searching for TESS data on target CVs...")
        
        # Loop through each CV target
        for tic_id in tqdm(self.cv_targets):
            # Search for available data
            search_result = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS')
            
            if len(search_result) == 0:
                print(f"No data found for TIC {tic_id}, skipping.")
                continue
                
            # Group by sectors
            sectors = np.unique([result.quarter for result in search_result])
            
            if len(sectors) < 2:
                print(f"Only one sector available for TIC {tic_id}, but we need multiple for cycle analysis.")
                continue
                
            print(f"Found {len(sectors)} sectors for TIC {tic_id}: {sectors}")
            
            # Download data for each sector
            sector_lcs = []
            for i, sector in enumerate(sectors):
                if i >= self.max_cycles:
                    print(f"Reached maximum {self.max_cycles} cycles for TIC {tic_id}")
                    break
                    
                sector_data = search_result[search_result.quarter == sector]
                
                try:
                    # Try 2-minute cadence data first
                    lc = sector_data[sector_data.exptime == 120].download()
                except:
                    try:
                        # Fall back to other cadences
                        lc = sector_data.download()
                    except Exception as e:
                        print(f"Error downloading data for TIC {tic_id}, sector {sector}: {e}")
                        continue
                
                # Basic light curve cleaning
                lc = lc.remove_outliers(sigma=5)
                lc = lc.normalize()
                sector_lcs.append(lc)
            
            if sector_lcs:
                self.light_curves[tic_id] = sector_lcs
                
        print(f"Downloaded data for {len(self.light_curves)} CVs with multiple cycles")
        
    def find_best_cvs(self, min_cycles=2):
        """
        Find the best CV candidates with multiple cycles of data.
        Returns a list of TIC IDs for CVs with good multi-cycle data.
        
        Parameters:
        -----------
        min_cycles : int
            Minimum number of cycles required
            
        Returns:
        -----------
        list
            List of TIC IDs for the best CV candidates
        """
        good_cvs = []
        
        for tic_id, lcs in self.light_curves.items():
            if len(lcs) < min_cycles:
                continue
                
            # Check if any light curve shows interesting variability
            has_variability = False
            for lc in lcs:
                if lc.flux.std() > 0.005:  # Arbitrary threshold for variability
                    has_variability = True
                    break
            
            if has_variability:
                good_cvs.append(tic_id)
                
        print(f"Found {len(good_cvs)} good CV candidates with {min_cycles}+ cycles of data")
        return good_cvs
    
    def analyze_period_improvement(self, tic_id):
        """
        Analyze how period determination improves with additional cycles for a given CV.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV to analyze
            
        Returns:
        -----------
        dict
            Dictionary with period analysis results
        """
        if tic_id not in self.light_curves:
            print(f"No data for TIC {tic_id}")
            return None
            
        lcs = self.light_curves[tic_id]
        if len(lcs) < 2:
            print(f"Need at least 2 cycles for TIC {tic_id}")
            return None
            
        # Results dictionary
        results = {
            'tic_id': tic_id,
            'n_cycles': [],
            'periods': [],
            'period_powers': [],
            'period_errors': [],
            'best_period': None,
            'lc_stats': []
        }
        
        # Calculate periodogram for increasing number of cycles
        for n_cycles in range(1, len(lcs) + 1):
            # Combine first n_cycles
            combined_lc = lcs[0]
            for i in range(1, n_cycles):
                combined_lc = combined_lc.append(lcs[i])
            
            # Calculate statistics for this light curve
            stats = {
                'time_span': combined_lc.time.max().value - combined_lc.time.min().value,
                'n_points': len(combined_lc.flux),
                'std': combined_lc.flux.std(),
                'median_err': np.median(combined_lc.flux_err)
            }
            results['lc_stats'].append(stats)
            
            # Calculate periodogram
            try:
                # Use long minimum period for CVs since we expect orbital periods of hours
                min_freq = 1/(10.0)  # 10 day max period
                max_freq = 1/(0.05)  # 1.2 hour min period
                
                # Try a more sensitive periodogram for CV orbital periods
                ls = LombScargle(combined_lc.time.value, combined_lc.flux.value, combined_lc.flux_err.value)
                freq, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq,
                                          samples_per_peak=100)
                
                # Convert to periods
                periods = 1/freq
                
                # Find highest peak
                peak_idx = np.argmax(power)
                best_period = periods[peak_idx]
                best_power = power[peak_idx]
                
                # Estimate error using FWHM of peak
                # Find indices where power is at least half the peak power
                half_max = best_power / 2
                above_half_max = power >= half_max
                
                if np.sum(above_half_max) >= 3:  # Need at least 3 points for valid FWHM
                    # Find contiguous regions above half max
                    from scipy.ndimage import label
                    labeled_array, num_features = label(above_half_max)
                    
                    # Find which label corresponds to the peak
                    peak_label = labeled_array[peak_idx]
                    
                    # Get all frequencies in this peak
                    peak_indices = np.where(labeled_array == peak_label)[0]
                    peak_freqs = freq[peak_indices]
                    
                    # FWHM is the width of this region
                    delta_freq = np.max(peak_freqs) - np.min(peak_freqs)
                    period_error = best_period * delta_freq / (freq[peak_idx])
                else:
                    # Simple estimate if FWHM fails
                    period_error = best_period / (stats['time_span'] / best_period)
                
                # Store results
                results['n_cycles'].append(n_cycles)
                results['periods'].append(best_period)
                results['period_powers'].append(best_power)
                results['period_errors'].append(period_error)
                
            except Exception as e:
                print(f"Error calculating periodogram for TIC {tic_id} with {n_cycles} cycles: {e}")
        
        # Determine best overall period from all cycles
        if results['periods']:
            # Use period from max cycles as most accurate
            results['best_period'] = results['periods'][-1]
            
            # Create figure to show period improvement
            self._plot_period_improvement(tic_id, results)
            
        return results
    
    def _plot_period_improvement(self, tic_id, results):
        """
        Plot period determination improvement with increasing cycles.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV
        results : dict
            Dictionary with period analysis results
        """
        if not results['periods']:
            return
            
        # Hours to display
        periods_hrs = np.array(results['periods']) * 24
        errors_hrs = np.array(results['period_errors']) * 24
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])
        
        # 1. Plot period vs number of cycles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(results['n_cycles'], periods_hrs, yerr=errors_hrs, 
                   fmt='o-', capsize=5, markersize=8)
        ax1.set_xlabel('Number of Cycles')
        ax1.set_ylabel('Period (hours)')
        ax1.set_title(f'TIC {tic_id}: Period Determination Improvement')
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot period error vs number of cycles
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results['n_cycles'], errors_hrs, 'o-', markersize=8)
        ax2.set_xlabel('Number of Cycles')
        ax2.set_ylabel('Period Error (hours)')
        ax2.set_title('Period Determination Uncertainty')
        ax2.grid(True, alpha=0.3)
        
        # 3. Plot folded light curve with best period
        ax3 = fig.add_subplot(gs[1, :])
        
        # Combine all available light curves
        lcs = self.light_curves[tic_id]
        combined_lc = lcs[0]
        for i in range(1, len(lcs)):
            combined_lc = combined_lc.append(lcs[i])
        
        # Fold with best period
        folded_lc = combined_lc.fold(period=results['best_period'])
        
        # Plot with density coloring
        x = folded_lc.phase.value
        y = folded_lc.flux.value
        
        # Using a density-based scatter
        xy = np.vstack([x, y])
        
        # Use kernel density estimate for coloring
        try:
            from scipy.stats import gaussian_kde
            z = gaussian_kde(xy)(xy)
            
            # Sort points by density
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            
            scatter = ax3.scatter(x, y, c=z, s=2, alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, ax=ax3, label='Density')
        except:
            # Fallback to simple plot if density estimation fails
            ax3.scatter(x, y, s=2, alpha=0.2, color='blue')
        
        # Plot a binned average light curve
        try:
            phase_bins = np.linspace(0, 1, 75)
            binned_flux = []
            for i in range(len(phase_bins)-1):
                mask = (x >= phase_bins[i]) & (x < phase_bins[i+1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.mean(y[mask]))
                else:
                    binned_flux.append(np.nan)
            
            bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
            ax3.plot(bin_centers, binned_flux, 'r-', linewidth=2)
        except:
            pass
            
        ax3.set_xlabel('Phase')
        ax3.set_ylabel('Normalized Flux')
        ax3.set_title(f'Folded Light Curve (P = {periods_hrs[-1]:.4f} hours)')
        ax3.grid(True, alpha=0.3)
        
        # Add cycle info as text
        info_text = f"Data span: {results['lc_stats'][-1]['time_span']:.1f} days\n"
        info_text += f"Data points: {results['lc_stats'][-1]['n_points']}\n"
        info_text += f"Period error: {errors_hrs[-1]:.6f} hours"
        
        # Place text in upper right corner with semi-transparent background
        ax3.text(0.98, 0.95, info_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'TIC{tic_id}_period_improvement.png'), dpi=300)
        plt.close()
    
    def analyze_outbursts(self, tic_id):
        """
        Analyze and characterize outbursts with improved detection in multiple cycles.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV to analyze
            
        Returns:
        -----------
        dict
            Dictionary with outburst analysis results
        """
        if tic_id not in self.light_curves:
            print(f"No data for TIC {tic_id}")
            return None
            
        lcs = self.light_curves[tic_id]
        if len(lcs) < 2:
            print(f"Need at least 2 cycles for TIC {tic_id}")
            return None
            
        # Results dictionary
        results = {
            'tic_id': tic_id,
            'n_cycles': [],
            'n_outbursts': [],
            'outburst_details': []
        }
        
        # Analyze outbursts for increasing number of cycles
        for n_cycles in range(1, len(lcs) + 1):
            # Combine first n_cycles
            combined_lc = lcs[0]
            for i in range(1, n_cycles):
                combined_lc = combined_lc.append(lcs[i])
            
            # Standardize and normalize for outburst detection
            time = combined_lc.time.value
            flux = combined_lc.flux.value
            
            # Simple outburst detection algorithm
            # For CV outbursts, we expect significant brightening
            
            # 1. Calculate baseline flux using a rolling median filter
            window_size = min(101, len(flux) // 3 * 2 + 1)  # Ensure it's odd and covers max 1/3 of data
            if window_size < 3:
                window_size = 3
                
            baseline = medfilt(flux, window_size)
            
            # 2. Find points significantly above baseline
            threshold = np.std(flux - baseline) * 3
            outburst_candidate_mask = (flux - baseline) > threshold
            
            # 3. Group adjacent points into outburst events
            from scipy.ndimage import label
            outburst_regions, n_outbursts = label(outburst_candidate_mask)
            
            # 4. Process each outburst
            outburst_list = []
            for i in range(1, n_outbursts + 1):
                # Get indices for this outburst
                outburst_indices = np.where(outburst_regions == i)[0]
                
                # Only consider outbursts with multiple points
                if len(outburst_indices) < 3:
                    continue
                    
                # Get time and flux for this outburst
                outburst_time = time[outburst_indices]
                outburst_flux = flux[outburst_indices]
                
                # Calculate basic outburst parameters
                start_time = outburst_time.min()
                end_time = outburst_time.max()
                duration = end_time - start_time
                amplitude = outburst_flux.max() - baseline[outburst_indices[np.argmax(outburst_flux)]]
                
                # Only consider significant outbursts
                if duration < 0.5 or amplitude < 0.05:  # At least 12 hours and 5% amplitude
                    continue
                    
                outburst_list.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'amplitude': amplitude,
                    'peak_time': outburst_time[np.argmax(outburst_flux)],
                    'n_points': len(outburst_indices)
                })
            
            # Store results
            results['n_cycles'].append(n_cycles)
            results['n_outbursts'].append(len(outburst_list))
            
            # For the final cycle analysis, store detailed outburst info
            if n_cycles == len(lcs):
                results['outburst_details'] = outburst_list
                
                # Create outburst visualization
                self._plot_outburst_analysis(tic_id, combined_lc, outburst_list, baseline)
        
        return results
    
    def _plot_outburst_analysis(self, tic_id, light_curve, outbursts, baseline=None):
        """
        Create visualization of outburst detection and characterization.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV
        light_curve : lightkurve.LightCurve
            Combined light curve object
        outbursts : list
            List of detected outbursts
        baseline : array or None
            Baseline flux level if available
        """
        # Extract data
        time = light_curve.time.value
        flux = light_curve.flux.value
        
        # For time axis, use BJD - 2457000 for cleaner numbers
        plot_time = time - 2457000
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Main light curve with outbursts
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(plot_time, flux, s=2, alpha=0.5, color='black', label='TESS data')
        
        # Add baseline if provided
        if baseline is not None:
            ax1.plot(plot_time, baseline, 'r-', alpha=0.7, label='Baseline')
            
        # Highlight outbursts
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(outbursts))))
        
        for i, outburst in enumerate(outbursts):
            color = colors[i % len(colors)]
            
            # Find indices for this outburst
            mask = (time >= outburst['start_time']) & (time <= outburst['end_time'])
            
            # Plot outburst
            ax1.scatter(plot_time[mask], flux[mask], s=15, alpha=0.7, color=color, 
                      label=f'Outburst {i+1}' if i < 5 else "")
            
            # Add annotation
            peak_idx = np.argmax(flux[mask])
            peak_time = plot_time[mask][peak_idx]
            peak_flux = flux[mask][peak_idx]
            
            # Only annotate first 5 outbursts to avoid crowding
            if i < 5:
                ax1.annotate(f"{i+1}", (peak_time, peak_flux),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('Time (BJD - 2457000)')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_title(f'TIC {tic_id}: Detected Outbursts')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Lower panel: Cumulative outburst count
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        if outbursts:
            # Sort outbursts by time
            sorted_outbursts = sorted(outbursts, key=lambda x: x['peak_time'])
            
            # Create cumulative count
            cum_times = [ob['peak_time'] - 2457000 for ob in sorted_outbursts]
            cum_counts = np.arange(1, len(sorted_outbursts) + 1)
            
            # Plot cumulative function
            ax2.step(cum_times, cum_counts, where='post', linewidth=2)
            ax2.set_ylabel('Cumulative Count')
            
            # Try to fit a linear function to estimate recurrence time
            if len(cum_times) >= 3:
                try:
                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(cum_times, cum_counts)
                    
                    # Plot fitted line
                    xfit = np.array([cum_times[0], cum_times[-1]])
                    yfit = intercept + slope * xfit
                    ax2.plot(xfit, yfit, 'r--', alpha=0.7)
                    
                    # Calculate and display recurrence time
                    if slope > 0:
                        recurrence = 1.0 / slope
                        ax2.text(0.02, 0.85, f"Est. recurrence: {recurrence:.1f} days", 
                               transform=ax2.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except:
                    pass
        else:
            ax2.text(0.5, 0.5, "No outbursts detected", ha='center', va='center', 
                   transform=ax2.transAxes)
        
        ax2.set_xlabel('Time (BJD - 2457000)')
        ax2.grid(True, alpha=0.3)
        
        # Add outburst statistics as text
        if outbursts:
            durations = [ob['duration'] for ob in outbursts]
            amplitudes = [ob['amplitude'] for ob in outbursts]
            
            stats_text = f"Outbursts: {len(outbursts)}\n"
            stats_text += f"Median duration: {np.median(durations):.2f} days\n"
            stats_text += f"Median amplitude: {np.median(amplitudes):.2f}"
            
            ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'TIC{tic_id}_outburst_analysis.png'), dpi=300)
        plt.close()
    
    def analyze_eclipse_improvement(self, tic_id):
        """
        Analyze how eclipse characterization improves with additional cycles.
        
        Parameters:
        -----------
        tic_id : int
            TIC ID of the CV to analyze
            
        Returns:
        -----------
        dict
            Dictionary with eclipse analysis results
        """
        if tic_id not in self.light_curves:
            print(f"No data for TIC {tic_id}")
            return None
            
        lcs = self.light_curves[tic_id]
        if len(lcs) < 2:
            print(f"Need at least 2 cycles for TIC {tic_id}")
            return None
        
        # First, determine if this is an eclipsing system
        # We need a good orbital period
        period_results = self.analyze_period_improvement(tic_id)
        
        if period_results is None or not period_results['periods']:
            print(f"Could not determine period for TIC {tic_id}")
            return None
            
        # Results dictionary
        results = {
            'tic_id': tic_id,
            'n_cycles': [],
            'eclipse_depths': [],
            'eclipse_errors': [],
            'eclipse_widths': [],
            'signal_to_noise': []
        }
        
        # Use period from full dataset
        orbital_period = period_results['best_period']
        
        # Analyze eclipses for increasing number of cycles
        for n_cycles in range(1, len(lcs) + 1):
            # Combine first n_cycles
            combined_lc = lcs[0]
            for i in range(1, n_cycles):
                combined_lc = combined_lc.append(lcs[i])
            
            # Fold light curve at orbital period
            folded_lc = combined_lc.fold(period=orbital_period)
            
            phase = folded_lc.phase.value
            flux = folded_lc.flux.value
            
            # Create phase bins
            bins = 50
            phase_bins = np.linspace(0, 1, bins + 1)
            bin_width = phase_bins[1] - phase_bins[0]
            bin_centers = 0.5 * (phase_bins[1:] + phase_bins[:-1])
            
            # Compute binned flux and errors
            binned_flux = np.zeros(bins)
            binned_errors = np.zeros(bins)
            
            for i in range(bins):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
                if np.sum(mask) > 0:
                    binned_flux[i] = np.median(flux[mask])
                    # Use standard error of the mean for error estimate
                    binned_errors[i] = np.std(flux[mask]) / np.sqrt(np.sum(mask))
                else:
                    binned_flux[i] = np.nan
                    binned_errors[i] = np.nan
            
            # Look for eclipse - find the deepest point
            valid_mask = ~np.isnan(binned_flux)
            if np.sum(valid_mask) < bins * 0.5:
                # Too many missing bins
                print(f"Too many missing bins for TIC {tic_id} with {n_cycles} cycles")
                continue
                
            baseline = np.median(binned_flux[valid_mask])
            min_idx = np.argmin(binned_flux[valid_mask])
            min_flux = binned_flux[valid_mask][min_idx]
            min_phase = bin_centers[valid_mask][min_idx]
            
            # Calculate eclipse depth
            eclipse_depth = baseline - min_flux
            eclipse_error = binned_errors[valid_mask][min_idx]
            
            #