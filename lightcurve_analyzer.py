import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle
from lightkurve import search_lightcurve
from sklearn.preprocessing import StandardScaler
import os
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm import tqdm

# Get number of available cores
N_JOBS = os.cpu_count()
print(f"Detected {N_JOBS} CPU cores for parallel processing")

warnings.filterwarnings('ignore')

class LightCurveAnalyzer:
    """
    A class to analyze TESS light curves for CV candidate verification.
    
    This performs detailed light curve analysis to look for CV-specific 
    characteristics including:
    
    1. Outburst detection
    2. Flickering measurement
    3. Orbital period verification
    4. Superhump detection
    5. Eclipse characterization
    """
    
    def __init__(self, candidates_file, output_dir='./lc_analysis'):
        """
        Initialize the light curve analyzer with CV candidates.
        
        Parameters:
        -----------
        candidates_file : str
            Path to the CSV or FITS file with CV candidates
        output_dir : str
            Directory to save output files
        """
        self.candidates_file = candidates_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # CV light curve characteristics
        self.cv_lc_features = [
            'has_outburst',           # Presence of dwarf nova outbursts
            'outburst_amplitude',     # Amplitude of outbursts (if present)
            'flickering_strength',    # Measure of rapid variability (flickering)
            'eclipse_depth',          # Depth of eclipses (if present)
            'eclipse_duration',       # Duration of eclipses as fraction of period
            'has_superhump',          # Presence of superhumps (slight period offset)
            'superhump_excess',       # Superhump period excess
            'orbital_period_hr',      # Orbital period in hours
            'phase_skewness',         # Skewness of the phase-folded light curve
            'phase_kurtosis',         # Kurtosis of the phase-folded light curve
            'cv_probability'          # Final CV probability based on light curve analysis
        ]
        
        # Initialize storage for data
        self.candidates = None
        self.lc_features = pd.DataFrame(columns=self.cv_lc_features)
    
    def load_candidates(self):
        """Load CV candidates from file"""
        print("Loading CV candidates...")
        
        try:
            # Determine file type and load accordingly
            if self.candidates_file.endswith('.fits'):
                with fits.open(self.candidates_file) as hdul:
                    self.candidates = Table(hdul[1].data).to_pandas()
            elif self.candidates_file.endswith('.csv'):
                self.candidates = pd.read_csv(self.candidates_file)
            else:
                raise ValueError(f"Unsupported file format: {self.candidates_file}")
                
            print(f"Loaded {len(self.candidates)} CV candidates")
            
            # Ensure we have the necessary columns
            required_cols = ['tess_id', 'primvs_id', 'true_period']
            missing_cols = [col for col in required_cols if col not in self.candidates.columns]
            
            if missing_cols:
                # Try to find alternative column names
                if 'primvs_id' in missing_cols and 'sourceid' in self.candidates.columns:
                    self.candidates['primvs_id'] = self.candidates['sourceid']
                    missing_cols.remove('primvs_id')
                
                if 'tess_id' in missing_cols and 'ID' in self.candidates.columns:
                    self.candidates['tess_id'] = self.candidates['ID']
                    missing_cols.remove('tess_id')
                    
                if missing_cols:
                    print(f"Warning: Missing columns in candidates file: {missing_cols}")
            
            return True
            
        except Exception as e:
            print(f"Error loading candidates: {str(e)}")
            return False
    
    def fetch_lightcurve(self, tess_id):
        """
        Fetch TESS light curve for a given TIC ID
        
        Parameters:
        -----------
        tess_id : int or str
            TESS Input Catalog ID
            
        Returns:
        --------
        lk.LightCurve or None
            Light curve object if found, None otherwise
        """
        try:
            # Search for light curves
            search_result = search_lightcurve(f"TIC {tess_id}", mission='TESS')
            
            if len(search_result) == 0:
                print(f"No light curves found for TIC {tess_id}")
                return None
            
            # Download the first light curve (most recent by default)
            lc = search_result[0].download()
            
            # Basic quality filtering
            if hasattr(lc, 'remove_outliers'):
                lc = lc.remove_outliers()
                
            if hasattr(lc, 'remove_nans'):
                lc = lc.remove_nans()
                
            print(f"Retrieved light curve for TIC {tess_id} with {len(lc.time)} points")
            return lc
            
        except Exception as e:
            print(f"Error fetching light curve for TIC {tess_id}: {str(e)}")
            return None
    
    def detect_outbursts(self, time, flux, flux_err=None):
        """
        Detect outbursts in a light curve
        
        Parameters:
        -----------
        time : array-like
            Time array
        flux : array-like
            Flux array
        flux_err : array-like, optional
            Flux error array
            
        Returns:
        --------
        dict
            Dictionary with outburst characteristics
        """
        # Default result if no outbursts
        result = {
            'has_outburst': False,
            'outburst_amplitude': 0.0
        }
        
        # Normalize flux to median = 1
        median_flux = np.median(flux)
        norm_flux = flux / median_flux
        
        # Calculate robust statistics
        q25, q50, q75 = np.percentile(norm_flux, [25, 50, 75])
        iqr = q75 - q25
        
        # Define outburst threshold - significantly above the typical range
        # For CVs, outbursts are typically >0.7-4 magnitudes (~2-40x in flux)
        outburst_threshold = q75 + 3 * iqr
        
        # Look for points significantly above the typical range
        potential_outburst = norm_flux > outburst_threshold
        
        # Require at least 3 consecutive points to be an outburst
        # (to avoid isolated outliers)
        outburst_regions = []
        if np.any(potential_outburst):
            # Find continuous outburst regions
            outburst_idx = np.where(potential_outburst)[0]
            
            # Group consecutive indices
            groups = []
            current_group = [outburst_idx[0]]
            
            for i in range(1, len(outburst_idx)):
                if outburst_idx[i] == outburst_idx[i-1] + 1:
                    current_group.append(outburst_idx[i])
                else:
                    if len(current_group) >= 3:  # Require at least 3 consecutive points
                        groups.append(current_group)
                    current_group = [outburst_idx[i]]
            
            # Add the last group if it meets the criteria
            if len(current_group) >= 3:
                groups.append(current_group)
            
            # Extract outburst regions
            for group in groups:
                start_idx = group[0]
                end_idx = group[-1]
                duration = time[end_idx] - time[start_idx]
                max_flux = np.max(norm_flux[start_idx:end_idx+1])
                
                outburst_regions.append({
                    'start_time': time[start_idx],
                    'end_time': time[end_idx],
                    'duration': duration,
                    'max_amplitude': max_flux - 1.0  # Relative to median=1
                })
        
        # Update result
        if outburst_regions:
            result['has_outburst'] = True
            result['outburst_amplitude'] = max([r['max_amplitude'] for r in outburst_regions])
            result['outburst_regions'] = outburst_regions
        
        return result
    
    def measure_flickering(self, time, flux, flux_err=None, window_size=0.5):
        """
        Measure flickering (rapid random variations) in a light curve
        
        Parameters:
        -----------
        time : array-like
            Time array (assumed to be in days)
        flux : array-like
            Flux array
        flux_err : array-like, optional
            Flux error array
        window_size : float
            Window size in days for computing local scatter
            
        Returns:
        --------
        dict
            Dictionary with flickering characteristics
        """
        result = {
            'flickering_strength': 0.0
        }
        
        # Check if we have enough points
        if len(time) < 10:
            return result
            
        # Normalize flux
        median_flux = np.median(flux)
        norm_flux = flux / median_flux
        
        # Compute typical measurement error (if available)
        if flux_err is not None:
            typical_error = np.median(flux_err) / median_flux
        else:
            typical_error = 0.001  # Assume 0.1% if not provided
        
        # Compute high-frequency variability
        # We'll use the median absolute deviation of differences between consecutive points
        diffs = np.diff(norm_flux)
        mad_diffs = np.median(np.abs(diffs - np.median(diffs)))
        
        # Compute ratio of observed scatter to expected scatter from errors
        expected_diffs_scatter = typical_error * np.sqrt(2)
        flickering_ratio = mad_diffs / expected_diffs_scatter
        
        # Also compute local scatter in sliding windows
        local_scatter = []
        for i in range(len(time)):
            # Define window boundaries
            window_start = time[i] - window_size/2
            window_end = time[i] + window_size/2
            
            # Find points in the window
            in_window = (time >= window_start) & (time <= window_end)
            
            # Need at least 5 points
            if np.sum(in_window) >= 5:
                # Compute local MAD
                local_flux = norm_flux[in_window]
                local_mad = np.median(np.abs(local_flux - np.median(local_flux)))
                local_scatter.append(local_mad)
        
        # Compute median local scatter
        if local_scatter:
            median_local_scatter = np.median(local_scatter)
            # Normalize by typical error
            local_flickering_ratio = median_local_scatter / typical_error
        else:
            local_flickering_ratio = 1.0
        
        # Combine both measures, emphasizing the one that shows more flickering
        result['flickering_strength'] = max(flickering_ratio, local_flickering_ratio)
        
        return result
    
    def characterize_eclipses(self, time, flux, period):
        """
        Look for eclipses in a phased light curve
        
        Parameters:
        -----------
        time : array-like
            Time array
        flux : array-like
            Flux array
        period : float
            Period to use for phasing
            
        Returns:
        --------
        dict
            Dictionary with eclipse characteristics
        """
        result = {
            'eclipse_depth': 0.0,
            'eclipse_duration': 0.0
        }
        
        # Check if we have enough points and a valid period
        if len(time) < 20 or period <= 0:
            return result
        
        # Normalize flux
        median_flux = np.median(flux)
        norm_flux = flux / median_flux
        
        # Phase the data
        phase = (time / period) % 1.0
        
        # Sort by phase
        sorted_idx = np.argsort(phase)
        sorted_phase = phase[sorted_idx]
        sorted_flux = norm_flux[sorted_idx]
        
        # Bin the phased light curve
        num_bins = min(30, len(time) // 5)  # Ensure at least 5 points per bin on average
        bins = np.linspace(0, 1, num_bins+1)
        digitized = np.digitize(sorted_phase, bins)
        
        binned_flux = np.zeros(num_bins)
        for i in range(1, num_bins+1):
            if np.sum(digitized == i) > 0:
                binned_flux[i-1] = np.median(sorted_flux[digitized == i])
            else:
                # If no points in bin, use neighboring bins
                nearby = sorted_flux[(digitized == i-1) | (digitized == i+1 if i < num_bins else digitized == 1)]
                binned_flux[i-1] = np.median(nearby) if len(nearby) > 0 else 1.0
        
        # Look for significant dips
        median_flux = np.median(binned_flux)
        min_flux = np.min(binned_flux)
        eclipse_depth = median_flux - min_flux
        
        # Determine if there's a significant eclipse (>10% depth)
        if eclipse_depth > 0.1:
            # Find eclipse duration
            eclipse_threshold = median_flux - 0.5 * eclipse_depth
            below_threshold = binned_flux < eclipse_threshold
            
            if np.any(below_threshold):
                # Find the longest contiguous segment below threshold
                eclipse_segments = []
                current_segment = []
                
                # Handle wraparound by extending the array
                extended_below = np.concatenate([below_threshold, below_threshold])
                
                for i in range(len(extended_below)):
                    if extended_below[i]:
                        current_segment.append(i % num_bins)
                    elif current_segment:
                        eclipse_segments.append(current_segment)
                        current_segment = []
                
                if current_segment:
                    eclipse_segments.append(current_segment)
                
                # Find the longest segment
                if eclipse_segments:
                    longest_segment = max(eclipse_segments, key=len)
                    eclipse_duration = len(longest_segment) / num_bins
                    
                    result['eclipse_depth'] = eclipse_depth
                    result['eclipse_duration'] = eclipse_duration
        
        return result
    
    def detect_superhumps(self, time, flux, orbital_period):
        """
        Detect superhumps (slight period offset from orbital period)
        
        Parameters:
        -----------
        time : array-like
            Time array
        flux : array-like
            Flux array
        orbital_period : float
            Orbital period in days
            
        Returns:
        --------
        dict
            Dictionary with superhump characteristics
        """
        result = {
            'has_superhump': False,
            'superhump_excess': 0.0
        }
        
        # Check if we have enough points and a valid orbital period
        if len(time) < 50 or orbital_period <= 0:
            return result
        
        # Normalize flux
        median_flux = np.median(flux)
        norm_flux = flux / median_flux
        
        # Compute Lomb-Scargle periodogram in a narrow range around the orbital period
        # Superhumps typically have periods 1-7% longer than orbital period
        min_freq = 0.93 / orbital_period
        max_freq = 1.07 / orbital_period
        
        # Use fine frequency grid
        df = 0.0001 / orbital_period
        frequencies = np.arange(min_freq, max_freq, df)
        
        # Compute periodogram
        ls = LombScargle(time, norm_flux)
        power = ls.power(frequencies)
        
        # Find the peak
        peak_idx = np.argmax(power)
        peak_freq = frequencies[peak_idx]
        peak_period = 1.0 / peak_freq
        
        # Calculate superhump excess
        excess = (peak_period - orbital_period) / orbital_period
        
        # Determine if there's a significant superhump
        # Superhump period is typically 1-7% longer than orbital period
        if 0.01 < excess < 0.07 and power[peak_idx] > 0.2:  # Arbitrary power threshold
            result['has_superhump'] = True
            result['superhump_excess'] = excess
        
        return result
    
    def calculate_phase_statistics(self, time, flux, period):
        """
        Calculate statistical properties of the phase-folded light curve
        
        Parameters:
        -----------
        time : array-like
            Time array
        flux : array-like
            Flux array
        period : float
            Period to use for phasing
            
        Returns:
        --------
        dict
            Dictionary with phase statistics
        """
        result = {
            'phase_skewness': 0.0,
            'phase_kurtosis': 0.0
        }
        
        # Check if we have enough points and a valid period
        if len(time) < 20 or period <= 0:
            return result
        
        # Normalize flux
        median_flux = np.median(flux)
        norm_flux = flux / median_flux
        
        # Phase the data
        phase = (time / period) % 1.0
        
        # Sort by phase
        sorted_idx = np.argsort(phase)
        sorted_flux = norm_flux[sorted_idx]
        
        # Calculate skewness
        mean_flux = np.mean(sorted_flux)
        std_flux = np.std(sorted_flux)
        if std_flux > 0:
            skewness = np.mean(((sorted_flux - mean_flux) / std_flux) ** 3)
            kurtosis = np.mean(((sorted_flux - mean_flux) / std_flux) ** 4) - 3  # Excess kurtosis
            
            result['phase_skewness'] = skewness
            result['phase_kurtosis'] = kurtosis
        
        return result
    
    def analyze_light_curve(self, tess_id, primvs_period):
        """
        Analyze a single light curve for CV characteristics
        
        Parameters:
        -----------
        tess_id : int or str
            TESS Input Catalog ID
        primvs_period : float
            Period from PRIMVS in days
            
        Returns:
        --------
        dict
            Dictionary with all CV light curve features
        """
        # Initialize results with default values
        results = {feature: 0.0 for feature in self.cv_lc_features}
        results['tess_id'] = tess_id
        
        # Convert period to hours
        results['orbital_period_hr'] = primvs_period * 24.0
        
        # Fetch the light curve
        lc = self.fetch_lightcurve(tess_id)
        if lc is None:
            # No light curve available
            return results
        
        # Extract time and flux
        time = lc.time.value
        flux = lc.flux.value
        
        # Check if flux_err is available
        flux_err = lc.flux_err.value if hasattr(lc, 'flux_err') else None
        
        # Apply each analysis method
        # 1. Detect outbursts
        outburst_results = self.detect_outbursts(time, flux, flux_err)
        results.update(outburst_results)
        
        # 2. Measure flickering
        flickering_results = self.measure_flickering(time, flux, flux_err)
        results.update(flickering_results)
        
        # 3. Characterize eclipses
        eclipse_results = self.characterize_eclipses(time, flux, primvs_period)
        results.update(eclipse_results)
        
        # 4. Detect superhumps
        superhump_results = self.detect_superhumps(time, flux, primvs_period)
        results.update(superhump_results)
        
        # 5. Calculate phase statistics
        phase_stats = self.calculate_phase_statistics(time, flux, primvs_period)
        results.update(phase_stats)
        
        # Calculate overall CV probability based on light curve features
        # This is a heuristic that could be refined with training data
        cv_score = 0.0
        
        # Score based on period - highest for periods between 1.5-5 hours
        period_hrs = results['orbital_period_hr']
        if 1.5 <= period_hrs <= 5:
            period_score = 1.0
        elif period_hrs < 1.5:
            period_score = period_hrs / 1.5
        else:  # period_hrs > 5
            period_score = max(0, 1.0 - (period_hrs - 5) / 15)  # Linear decrease up to 20 hrs
        
        # Score based on outbursts
        outburst_score = 1.0 if results['has_outburst'] else 0.0
        
        # Score based on flickering - CVs typically show strong flickering
        flickering_score = min(1.0, results['flickering_strength'] / 5.0)
        
        # Score based on eclipses - some CVs show eclipses
        eclipse_score = min(1.0, results['eclipse_depth'] * 3.0)
        
        # Score based on superhumps - strong indicator of CVs, particularly dwarf novae
        superhump_score = 1.0 if results['has_superhump'] else 0.0
        
        # Score based on light curve shape (phase statistics)
        # CVs often have asymmetric light curves
        shape_score = min(1.0, abs(results['phase_skewness']) / 2.0)
        
        # Combine scores with weights
        cv_score = (
            0.3 * period_score +
            0.2 * outburst_score +
            0.2 * flickering_score +
            0.1 * eclipse_score +
            0.1 * superhump_score +
            0.1 * shape_score
        )
        
        results['cv_probability'] = cv_score
        
        return results
    
    def process_candidate(self, candidate):
        """
        Process a single candidate - for parallel execution
        
        Parameters:
        -----------
        candidate : Series
            Candidate data
            
        Returns:
        --------
        dict
            Dictionary with light curve features or None if failed
        """
        try:
            # Get TESS ID and period
            tess_id = candidate['tess_id']
            primvs_id = candidate['primvs_id'] if 'primvs_id' in candidate else None
            primvs_period = candidate['true_period'] if 'true_period' in candidate else None
            
            # Skip if missing essential data
            if tess_id is None or primvs_period is None:
                print(f"Skipping TIC {tess_id}: Missing essential data")
                return None
            
            # Analyze light curve
            lc_features = self.analyze_light_curve(tess_id, primvs_period)
            
            # Add candidate ID
            lc_features['primvs_id'] = primvs_id
            lc_features['tess_id'] = tess_id
            
            return lc_features
            
        except Exception as e:
            print(f"Error processing TIC {candidate.get('tess_id', 'unknown')}: {str(e)}")
            return None
    
    def process_candidates(self, max_candidates=None):
        """
        Process all CV candidates to extract light curve features
        Uses parallel processing for efficiency
        
        Parameters:
        -----------
        max_candidates : int, optional
            Maximum number of candidates to process, useful for testing
            
        Returns:
        --------
        DataFrame
            DataFrame with light curve features for all processed candidates
        """
        if self.candidates is None:
            print("No candidates loaded. Call load_candidates() first.")
            return None
        
        # Determine which candidates to process
        if max_candidates is not None and max_candidates < len(self.candidates):
            process_candidates = self.candidates.iloc[:max_candidates]
            print(f"Processing {max_candidates} of {len(self.candidates)} candidates")
        else:
            process_candidates = self.candidates
            print(f"Processing all {len(self.candidates)} candidates")
        
        # Process in parallel using all available cores
        # We'll limit parallel jobs to a reasonable number to avoid
        # overwhelming the MAST server
        n_jobs = min(N_JOBS, 8)  # Limit to 8 parallel jobs to be nice to MAST
        print(f"Using {n_jobs} parallel workers for light curve processing")
        
        # Convert candidate rows to dictionaries for easier parallel processing
        candidate_dicts = process_candidates.to_dict('records')
        
        # Process candidates in parallel with progress bar
        all_results = []
        with tqdm(total=len(candidate_dicts), desc="Processing light curves") as pbar:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                futures = {executor.submit(self.process_candidate, candidate): i 
                          for i, candidate in enumerate(candidate_dicts)}
                
                # Collect results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                    pbar.update(1)
        
        # Convert to DataFrame
        if all_results:
            self.lc_features = pd.DataFrame(all_results)
            
            # Merge with candidates if possible
            if 'primvs_id' in self.lc_features.columns and 'primvs_id' in self.candidates.columns:
                self.lc_features = pd.merge(
                    self.lc_features,
                    self.candidates,
                    on='primvs_id',
                    how='left',
                    suffixes=('', '_orig')
                )
            
            print(f"Completed light curve analysis for {len(self.lc_features)} candidates")
        else:
            print("No light curve features extracted. Check for errors above.")
            self.lc_features = pd.DataFrame(columns=self.cv_lc_features)
        
        return self.lc_features
    
    def rank_candidates(self):
        """
        Rank CV candidates based on light curve analysis
        
        Returns:
        --------
        DataFrame
            DataFrame with ranked CV candidates
        """
        if self.lc_features is None or len(self.lc_features) == 0:
            print("No light curve features available. Call process_candidates() first.")
            return None
        
        # Create a copy
        ranked = self.lc_features.copy()
        
        # Calculate a weighted score that combines original CV score with light curve features
        if 'cv_score' in ranked.columns:
            # We have the original CV score, use it
            ranked['final_score'] = 0.5 * ranked['cv_score'] + 0.5 * ranked['cv_probability']
        else:
            # Just use the light curve score
            ranked['final_score'] = ranked['cv_probability']
        
        # Sort by final score
        ranked = ranked.sort_values('final_score', ascending=False)
        
        return ranked
    
    def plot_best_candidates(self, n_candidates=10):
        """
        Plot light curves for the best CV candidates
        
        Parameters:
        -----------
        n_candidates : int
            Number of candidates to plot
            
        Returns:
        --------
        bool
            True if plots were generated
        """
        if self.lc_features is None or len(self.lc_features) == 0:
            print("No light curve features available. Call process_candidates() first.")
            return False
        
        # Rank candidates
        ranked = self.rank_candidates()
        
        # Plot at most n_candidates
        n_candidates = min(n_candidates, len(ranked))
        
        for i in range(n_candidates):
            candidate = ranked.iloc[i]
            tess_id = candidate['tess_id']
            primvs_id = candidate['primvs_id'] if 'primvs_id' in candidate else 'unknown'
            primvs_period = candidate['true_period'] if 'true_period' in candidate else None
            
            # Fetch the light curve
            lc = self.fetch_lightcurve(tess_id)
            if lc is None:
                continue
            
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Full light curve
            ax = axes[0]
            lc.plot(ax=ax)
            
            # Add features to plot
            title = (f"TIC {tess_id} / PRIMVS {primvs_id}\n"
                    f"CV Probability: {candidate['cv_probability']:.2f}, "
                    f"Period: {candidate['orbital_period_hr']:.2f} h")
            
            # Add additional features if available
            has_outburst = candidate.get('has_outburst', False)
            if has_outburst:
                title += f", Outburst Amp: {candidate['outburst_amplitude']:.2f}"
            
            has_eclipse = candidate.get('eclipse_depth', 0) > 0.1
            if has_eclipse:
                title += f", Eclipse Depth: {candidate['eclipse_depth']:.2f}"
                
            has_superhump = candidate.get('has_superhump', False)
            if has_superhump:
                title += f", Superhump"
                
            ax.set_title(title)
            
            # Plot 2: Phase-folded light curve
            ax = axes[1]
            if primvs_period is not None and primvs_period > 0:
                # Phase-fold at the PRIMVS period
                lc.fold(period=primvs_period).plot(ax=ax)
                ax.set_title(f"Phase-folded at period = {primvs_period:.6f} days")
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/candidate_{i+1}_TIC{tess_id}.png", dpi=300)
            plt.close()
        
        return True
    
    def save_results(self):
        """Save the light curve analysis results"""
        if self.lc_features is None or len(self.lc_features) == 0:
            print("No light curve features available. Call process_candidates() first.")
            return False
        
        # Rank candidates
        ranked = self.rank_candidates()
        
        # Save as CSV
        csv_path = f"{self.output_dir}/cv_candidates_lc_analysis.csv"
        ranked.to_csv(csv_path, index=False)
        print(f"Saved light curve analysis results to {csv_path}")
        
        # Save as FITS for compatibility with astronomical tools
        fits_path = f"{self.output_dir}/cv_candidates_lc_analysis.fits"
        t = Table.from_pandas(ranked)
        t.write(fits_path, overwrite=True)
        print(f"Saved light curve analysis results to {fits_path}")
        
        return True
    
    def run_pipeline(self, max_candidates=None):
        """
        Run the complete light curve analysis pipeline
        
        Parameters:
        -----------
        max_candidates : int, optional
            Maximum number of candidates to process
            
        Returns:
        --------
        bool
            True if pipeline completed successfully
        """
        # 1. Load candidates
        if not self.load_candidates():
            return False
        
        # 2. Process candidates
        self.process_candidates(max_candidates)
        
        # 3. Generate plots for best candidates
        self.plot_best_candidates()
        
        # 4. Save results
        self.save_results()
        
        print("Light curve analysis pipeline completed successfully!")
        return True


# Example usage (to be customized with your file paths)
if __name__ == "__main__":
    # File paths
    candidates_file = "./cv_results/cv_candidates.csv"
    output_dir = "./cv_results/lc_analysis"
    
    # Create and run the analyzer
    analyzer = LightCurveAnalyzer(candidates_file, output_dir)
    analyzer.run_pipeline(max_candidates=100)  # Limit to 100 for testing