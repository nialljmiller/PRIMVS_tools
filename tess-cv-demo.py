#!/usr/bin/env python3
"""
TESS CV Multi-Cycle Analysis Demo

This script demonstrates the advantages of analyzing cataclysmic variable stars
across multiple TESS observation cycles by processing several well-known CVs.

Author: Claude
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import lightkurve as lk
import pandas as pd
from datetime import datetime

# Import the TESS CV processor
from tess_cv_processor import TESSCVProcessor

# Configure demo parameters
OUTPUT_DIR = "tess_cv_demo_results"
MAX_CYCLES = 4  # Maximum number of TESS cycles to use

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{OUTPUT_DIR}_log.txt", mode='w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

def run_demo():
    """Run the demo with specific CV targets."""
    logger.info("Starting TESS CV Multi-Cycle Analysis Demo")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # List of notable CVs with multiple TESS observations
    # These are selected based on availability of multi-cycle data and showing interesting features
    cv_targets = [
        # V2051 Oph - Eclipsing dwarf nova with extensive TESS coverage
        309953133,
        
        # SS Cyg - Prototype dwarf nova with outbursts
        149253887,
        
        # AM Her - Prototype polar with strong magnetic field
        118327563,
        
        # TW Vir - Dwarf nova with good coverage
        304042981
    ]
    
    # Process targets
    processor = TESSCVProcessor(output_dir=OUTPUT_DIR, max_cycles=MAX_CYCLES, cv_list=cv_targets)
    
    # Run the analysis pipeline
    success = processor.run_pipeline()
    
    if success:
        logger.info(f"Demo completed successfully. Results saved to {OUTPUT_DIR}")
        generate_publication_plots(processor)
    else:
        logger.error("Demo failed")

def generate_publication_plots(processor):
    """Generate publication-quality plots showing the benefits of multi-cycle analysis."""
    logger.info("Generating publication-quality plots")
    
    # Create a directory for the publication plots
    pub_dir = os.path.join(OUTPUT_DIR, "publication_plots")
    os.makedirs(pub_dir, exist_ok=True)
    
    # Create a comparison plot showing period determination improvement
    create_period_comparison_plot(processor, pub_dir)
    
    # Create a comparison of eclipse depths across multiple cycles
    create_eclipse_comparison_plot(processor, pub_dir)

def create_period_comparison_plot(processor, output_dir):
    """
    Create a plot comparing period determination precision with 
    increasing number of TESS cycles.
    """
    # Collect period error data from all processed CVs
    period_data = []
    
    for tic_id in processor.light_curves.keys():
        try:
            # Run or retrieve period analysis
            results = processor.analyze_period_improvement(tic_id)
            
            if results and 'period_errors' in results and results['period_errors']:
                # Convert errors to minutes for better readability
                errors_minutes = np.array(results['period_errors']) * 24 * 60
                
                # Store results
                for i, n_cycles in enumerate(results['n_cycles']):
                    period_data.append({
                        'tic_id': tic_id,
                        'n_cycles': n_cycles,
                        'period_hr': results['periods'][i] * 24,
                        'error_minutes': errors_minutes[i],
                        'points': results['lc_stats'][i]['n_points']
                    })
        except Exception as e:
            logger.error(f"Error processing period data for TIC {tic_id}: {e}")
    
    if not period_data:
        logger.warning("No period data available for comparison plot")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(period_data)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot each CV as a different line
    for tic_id in df['tic_id'].unique():
        cv_data = df[df['tic_id'] == tic_id]
        
        # Line options
        line_style = '-o'
        label = f"TIC {tic_id} (P={cv_data['period_hr'].iloc[-1]:.2f}h)"
        
        # Plot period error vs. number of cycles
        plt.semilogy(cv_data['n_cycles'], cv_data['error_minutes'], 
                   line_style, linewidth=2, markersize=8, label=label)
    
    # Add a reference line showing theoretical sqrt(N) improvement
    x_ref = np.array(range(1, MAX_CYCLES+1))
    y_ref = np.array([1.0, 1.0/np.sqrt(2), 1.0/np.sqrt(3), 1.0/np.sqrt(4)])
    y_ref = y_ref * 10  # Scaling factor
    plt.semilogy(x_ref, y_ref, '--', color='gray', linewidth=2, 
               label=r'$1/\sqrt{N}$ reference')
    
    plt.xlabel('Number of TESS Cycles', fontsize=14)
    plt.ylabel('Period Uncertainty (minutes)', fontsize=14)
    plt.title('CV Period Determination Precision Improvement\nwith Multiple TESS Cycles', 
             fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Customize x-axis ticks
    plt.xticks(range(1, MAX_CYCLES+1))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'period_precision_improvement.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'period_precision_improvement.pdf'))
    plt.close()
    
    logger.info(f"Period comparison plot saved to {output_dir}")

def create_eclipse_comparison_plot(processor, output_dir):
    """
    Create a plot comparing eclipse SNR improvement with 
    increasing number of TESS cycles.
    """
    # Collect eclipse data from all processed CVs
    eclipse_data = []
    
    for tic_id in processor.light_curves.keys():
        try:
            # Run or retrieve eclipse analysis
            results = processor.analyze_eclipse_improvement(tic_id)
            
            if results and 'signal_to_noise' in results and results['signal_to_noise']:
                # Store results
                for i, n_cycles in enumerate(results['n_cycles']):
                    eclipse_data.append({
                        'tic_id': tic_id,
                        'n_cycles': n_cycles,
                        'snr': results['signal_to_noise'][i],
                        'depth': results['eclipse_depths'][i],
                        'error': results['eclipse_errors'][i]
                    })
        except Exception as e:
            logger.error(f"Error processing eclipse data for TIC {tic_id}: {e}")
    
    if not eclipse_data:
        logger.warning("No eclipse data available for comparison plot")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(eclipse_data)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot each CV as a different line
    for tic_id in df['tic_id'].unique():
        cv_data = df[df['tic_id'] == tic_id]
        
        # Line options
        line_style = '-o'
        label = f"TIC {tic_id} (depth={cv_data['depth'].iloc[-1]:.3f})"
        
        # Plot SNR vs. number of cycles
        plt.plot(cv_data['n_cycles'], cv_data['snr'], 
               line_style, linewidth=2, markersize=8, label=label)
    
    # Add a reference line showing theoretical sqrt(N) improvement
    x_ref = np.array(range(1, MAX_CYCLES+1))
    y_ref = np.array([1.0, np.sqrt(2), np.sqrt(3), np.sqrt(4)])
    y_ref = y_ref * df['snr'].min()  # Scale to match data range
    plt.plot(x_ref, y_ref, '--', color='gray', linewidth=2, 
           label=r'$\sqrt{N}$ reference')
    
    plt.xlabel('Number of TESS Cycles', fontsize=14)
    plt.ylabel('Eclipse Signal-to-Noise Ratio', fontsize=14)
    plt.title('CV Eclipse Detection Signal-to-Noise Improvement\nwith Multiple TESS Cycles', 
             fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Customize x-axis ticks
    plt.xticks(range(1, MAX_CYCLES+1))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eclipse_snr_improvement.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'eclipse_snr_improvement.pdf'))
    plt.close()
    
    logger.info(f"Eclipse comparison plot saved to {output_dir}")

if __name__ == "__main__":
    # Display start time for benchmarking
    start_time = datetime.now()
    print(f"Demo started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the demo
    run_demo()
    
    # Display completion time
    end_time = datetime.now()
    runtime = end_time - start_time
    print(f"Demo completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {runtime}")
