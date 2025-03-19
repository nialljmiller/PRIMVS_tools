#!/usr/bin/env python3
"""
PRIMVS-TESS CV Detection Pipeline - Simplified Version
Uses all available CPU cores and memory automatically
"""

import os
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings
import psutil
import time
from datetime import timedelta
from cv_finder import CVFinder
from lightcurve_analyzer import LightCurveAnalyzer

# Ignore warnings
warnings.filterwarnings('ignore')

# HARDCODED PATHS - MODIFY THESE
PRIMVS_FILE = "/path/to/PRIMVS_P.fits"
TESS_CROSSMATCH_FILE = "/path/to/primvs_tess_crossmatch.fits"
OUTPUT_DIR = "./cv_results"
LC_LIMIT = 500  # Number of light curves to analyze

def setup_parallel_env():
    """Set up environment for optimal parallel processing"""
    # Get number of available cores
    cpu_count = os.cpu_count()
    
    # Configure environment to use all cores
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    
    # Get memory information
    mem = psutil.virtual_memory()
    total_mem_gb = mem.total / (1024 ** 3)
    available_mem_gb = mem.available / (1024 ** 3)
    
    print(f"\nSystem Configuration:")
    print(f"  Using all {cpu_count} CPU cores")
    print(f"  Memory: {available_mem_gb:.1f} GB available of {total_mem_gb:.1f} GB total")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU: None detected")
    except:
        pass
    
    return cpu_count

def create_directories():
    """Create output directories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "lc_analysis"), exist_ok=True)

def generate_summary_report(cv_file, lc_file=None):
    """Generate a summary report of the CV detection results"""
    # Load CV candidates
    try:
        if cv_file.endswith('.fits'):
            cv_data = Table.read(cv_file).to_pandas()
        else:
            cv_data = pd.read_csv(cv_file)
    except Exception as e:
        print(f"Error loading CV data: {str(e)}")
        return False
    
    # Load light curve analysis if available
    lc_data = None
    if lc_file is not None:
        try:
            if lc_file.endswith('.fits'):
                lc_data = Table.read(lc_file).to_pandas()
            else:
                lc_data = pd.read_csv(lc_file)
        except Exception as e:
            print(f"Error loading light curve analysis: {str(e)}")
    
    # Generate report
    report_file = os.path.join(OUTPUT_DIR, "cv_detection_report.txt")
    with open(report_file, 'w') as f:
        # Header
        f.write("======================================\n")
        f.write("PRIMVS-TESS CV DETECTION REPORT\n")
        f.write("======================================\n\n")
        
        # Basic statistics
        f.write(f"Total CV candidates: {len(cv_data)}\n")
        
        # Period distribution
        if 'true_period' in cv_data.columns:
            period_hours = cv_data['true_period'] * 24.0
            f.write("\nPeriod distribution (hours):\n")
            f.write(f"  Min:    {period_hours.min():.2f}\n")
            f.write(f"  1st Q:  {period_hours.quantile(0.25):.2f}\n")
            f.write(f"  Median: {period_hours.median():.2f}\n")
            f.write(f"  3rd Q:  {period_hours.quantile(0.75):.2f}\n")
            f.write(f"  Max:    {period_hours.max():.2f}\n")
            
            # Period breakdown
            p1 = (period_hours < 2).sum()
            p2 = ((period_hours >= 2) & (period_hours < 3)).sum()
            p3 = ((period_hours >= 3) & (period_hours < 4)).sum()
            p4 = ((period_hours >= 4) & (period_hours < 5)).sum()
            p5 = (period_hours >= 5).sum()
            
            f.write("\nPeriod breakdown:\n")
            f.write(f"  < 2 hours:  {p1} ({100*p1/len(cv_data):.1f}%)\n")
            f.write(f"  2-3 hours:  {p2} ({100*p2/len(cv_data):.1f}%)\n")
            f.write(f"  3-4 hours:  {p3} ({100*p3/len(cv_data):.1f}%)\n")
            f.write(f"  4-5 hours:  {p4} ({100*p4/len(cv_data):.1f}%)\n")
            f.write(f"  >= 5 hours: {p5} ({100*p5/len(cv_data):.1f}%)\n")
        
        # Amplitude distribution
        if 'true_amplitude' in cv_data.columns:
            f.write("\nAmplitude distribution (mag):\n")
            f.write(f"  Min:    {cv_data['true_amplitude'].min():.2f}\n")
            f.write(f"  1st Q:  {cv_data['true_amplitude'].quantile(0.25):.2f}\n")
            f.write(f"  Median: {cv_data['true_amplitude'].median():.2f}\n")
            f.write(f"  3rd Q:  {cv_data['true_amplitude'].quantile(0.75):.2f}\n")
            f.write(f"  Max:    {cv_data['true_amplitude'].max():.2f}\n")
        
        # Light curve analysis results if available
        if lc_data is not None:
            high_prob = (lc_data['cv_probability'] > 0.7).sum()
            has_outburst = (lc_data['has_outburst'] == True).sum() if 'has_outburst' in lc_data.columns else 0
            has_eclipse = (lc_data['eclipse_depth'] > 0.1).sum() if 'eclipse_depth' in lc_data.columns else 0
            has_superhump = (lc_data['has_superhump'] == True).sum() if 'has_superhump' in lc_data.columns else 0
            high_flickering = (lc_data['flickering_strength'] > 3).sum() if 'flickering_strength' in lc_data.columns else 0
            
            f.write("\nLight curve analysis results:\n")
            f.write(f"  High CV probability:  {high_prob} ({100*high_prob/len(lc_data):.1f}%)\n")
            f.write(f"  Outbursts detected:   {has_outburst} ({100*has_outburst/len(lc_data):.1f}%)\n")
            f.write(f"  Eclipses detected:    {has_eclipse} ({100*has_eclipse/len(lc_data):.1f}%)\n")
            f.write(f"  Superhumps detected:  {has_superhump} ({100*has_superhump/len(lc_data):.1f}%)\n")
            f.write(f"  High flickering:      {high_flickering} ({100*high_flickering/len(lc_data):.1f}%)\n")
        
        # Top 10 candidates
        f.write("\nTop 10 candidates:\n")
        if lc_data is not None and 'final_score' in lc_data.columns:
            # Use light curve analysis results
            top_candidates = lc_data.sort_values('final_score', ascending=False).head(10)
            for i, cand in top_candidates.iterrows():
                period_hrs = cand['orbital_period_hr'] if 'orbital_period_hr' in cand else cand['true_period'] * 24.0
                f.write(f"  {i+1}. TIC {cand['tess_id']}, PRIMVS {cand['primvs_id']}, "
                       f"Period: {period_hrs:.2f} hrs, "
                       f"CV Score: {cand.get('final_score', cand.get('cv_probability', 0)):.2f}\n")
        else:
            # Use initial CV candidates
            sort_col = 'cv_score' if 'cv_score' in cv_data.columns else 'probability' 
            top_candidates = cv_data.sort_values(sort_col, ascending=False).head(10)
            for i, cand in top_candidates.iterrows():
                period_hrs = cand['true_period'] * 24.0 if 'true_period' in cand else 0
                f.write(f"  {i+1}. TIC {cand.get('tess_id', 'N/A')}, PRIMVS {cand.get('primvs_id', cand.get('sourceid', 'N/A'))}, "
                       f"Period: {period_hrs:.2f} hrs, "
                       f"CV Score: {cand.get(sort_col, 0):.2f}\n")
    
    print(f"Generated summary report: {report_file}")
    return True

def plot_candidate_distributions(cv_file):
    """Generate plots of CV candidate distributions"""
    # Load CV candidates
    try:
        if cv_file.endswith('.fits'):
            cv_data = Table.read(cv_file).to_pandas()
        else:
            cv_data = pd.read_csv(cv_file)
    except Exception as e:
        print(f"Error loading CV data: {str(e)}")
        return False
    
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    
    # 1. Period distribution
    if 'true_period' in cv_data.columns:
        plt.figure(figsize=(10, 6))
        period_hours = cv_data['true_period'] * 24.0
        plt.hist(period_hours, bins=50, alpha=0.7)
        plt.xlabel('Period (hours)')
        plt.ylabel('Number of Candidates')
        plt.title('Period Distribution of CV Candidates')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "period_distribution.png"), dpi=300)
        plt.close()
    
    # 2. Period vs Amplitude (Bailey diagram)
    if 'true_period' in cv_data.columns and 'true_amplitude' in cv_data.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            cv_data['true_period'] * 24.0,  # Convert to hours 
            cv_data['true_amplitude'],
            alpha=0.7, s=5
        )
        plt.xscale('log')
        plt.xlabel('Period (hours)')
        plt.ylabel('Amplitude (mag)')
        plt.title('Bailey Diagram of CV Candidates')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "bailey_diagram.png"), dpi=300)
        plt.close()
    
    # 3. Spatial distribution
    if 'l' in cv_data.columns and 'b' in cv_data.columns:
        plt.figure(figsize=(12, 6))
        plt.scatter(cv_data['l'], cv_data['b'], alpha=0.7, s=5)
        plt.xlabel('Galactic Longitude (l)')
        plt.ylabel('Galactic Latitude (b)')
        plt.title('Spatial Distribution of CV Candidates')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "spatial_distribution.png"), dpi=300)
        plt.close()
    
    # 4. CV Score distribution
    score_col = None
    for col in ['cv_score', 'cv_probability', 'final_score', 'probability']:
        if col in cv_data.columns:
            score_col = col
            break
    
    if score_col is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(cv_data[score_col], bins=50, alpha=0.7)
        plt.xlabel('CV Score')
        plt.ylabel('Number of Candidates')
        plt.title('CV Score Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "cv_score_distribution.png"), dpi=300)
        plt.close()
    
    print(f"Generated distribution plots in {plots_dir}")
    return True

def main():
    """Main pipeline function"""
    # Set up environment for parallel processing
    setup_parallel_env()
    
    # Create output directories
    create_directories()
    
    print("\n" + "="*80)
    print("RUNNING PRIMVS-TESS CV DETECTION PIPELINE")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Step 1: Initial candidate selection and feature-based classification
    print("\nStep 1: Feature-based CV Candidate Selection")
    print("-"*50)
    
    cv_finder = CVFinder(
        primvs_file=PRIMVS_FILE,
        tess_match_file=TESS_CROSSMATCH_FILE,
        output_dir=OUTPUT_DIR
    )
    
    # Run the CV finder pipeline
    step1_start = time.time()
    cv_finder.run_pipeline()  # No known CVs file for simplicity
    step1_time = time.time() - step1_start
    print(f"Step 1 completed in {timedelta(seconds=int(step1_time))}")
    
    # Path to the CV candidates file
    cv_candidates_file = os.path.join(OUTPUT_DIR, "cv_candidates.csv")
    
    # Step 2: Light curve analysis for validation
    print("\nStep 2: Light Curve Analysis for Validation")
    print("-"*50)
    
    lc_analyzer = LightCurveAnalyzer(
        candidates_file=cv_candidates_file,
        output_dir=os.path.join(OUTPUT_DIR, "lc_analysis")
    )
    
    # Run the light curve analysis pipeline
    step2_start = time.time()
    lc_analyzer.run_pipeline(max_candidates=LC_LIMIT)
    step2_time = time.time() - step2_start
    print(f"Step 2 completed in {timedelta(seconds=int(step2_time))}")
    
    # Path to the light curve analysis file
    lc_analysis_file = os.path.join(OUTPUT_DIR, "lc_analysis", "cv_candidates_lc_analysis.csv")
    
    # Step 3: Generate final results
    print("\nStep 3: Generating Final Results")
    print("-"*50)
    
    step3_start = time.time()
    # Generate plots
    plot_candidate_distributions(cv_candidates_file)
    
    # Generate summary report
    generate_summary_report(cv_candidates_file, lc_analysis_file)
    step3_time = time.time() - step3_start
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"PIPELINE COMPLETED in {timedelta(seconds=int(total_time))}")
    print("="*80 + "\n")
    
    print(f"Step 1 (Candidate selection): {timedelta(seconds=int(step1_time))}")
    print(f"Step 2 (Light curve analysis): {timedelta(seconds=int(step2_time))}")
    print(f"Step 3 (Final results): {timedelta(seconds=int(step3_time))}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    
    return 0

if __name__ == "__main__":
    main()