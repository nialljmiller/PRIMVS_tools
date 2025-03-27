#!/usr/bin/env python3
"""
PRIMVS-TESS CV Bailey Diagram - Final Version

Plots three independent groups on a Bailey diagram:
1. Candidates (bottom layer)
2. Known CVs (middle layer)
3. Target list (top layer, with 'x' markers)

Amplitude capped at 4 mag, legend on upper left, minor ticks added.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

# File paths
TESS_CROSSMATCH_FILE = "tess_crossmatch_results.csv"  # All candidates
TARGETS_FILE = "targets.csv"  # Target list
KNOWN_CVS_FILE = "../PRIMVS/PRIMVS_CC_CV.fits"  # Known CVs list
OUTPUT_DIR = "plots"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all datasets
print("Loading datasets...")

# Load TESS crossmatch data (candidates)
candidates_df = pd.read_csv(TESS_CROSSMATCH_FILE, low_memory=False)
candidates_df = candidates_df.groupby('sourceid').first().reset_index()
print(f"Loaded {len(candidates_df)} unique candidates")

# Load targets
targets_df = pd.read_csv(TARGETS_FILE, low_memory=False)
targets_df = targets_df.groupby('sourceid').first().reset_index()
print(f"Loaded {len(targets_df)} unique targets")

# Load known CVs
with fits.open(KNOWN_CVS_FILE) as hdul:
    known_cvs_df = Table(hdul[1].data).to_pandas()
print(f"Loaded {len(known_cvs_df)} known CVs")

# Check what period and amplitude columns we have in each dataset
print("\nChecking column names in each dataset:")
print(f"Candidates columns: {[col for col in candidates_df.columns if 'period' in col.lower() or 'amplitude' in col.lower()]}")
print(f"Known CVs columns: {[col for col in known_cvs_df.columns if 'period' in col.lower() or 'amplitude' in col.lower()]}")
print(f"Targets columns: {[col for col in targets_df.columns if 'period' in col.lower() or 'amplitude' in col.lower()]}")

# Determine most appropriate period column for each dataset
def find_best_period_column(df):
    period_priority = ['true_period', 'period', 'ls_period1', 'pdm_period1', 'ce_period1', 'ce_period2']
    for col in period_priority:
        if col in df.columns:
            return col
    return None

# Determine most appropriate amplitude column for each dataset
def find_best_amplitude_column(df):
    amplitude_priority = ['true_amplitude', 'amplitude']
    for col in amplitude_priority:
        if col in df.columns:
            return col
    return None

# Create Bailey diagram with larger figure for better visualization
plt.figure(figsize=(15, 10))

# 1. Plot candidates (bottom layer)
candidate_period_col = find_best_period_column(candidates_df)
candidate_amplitude_col = find_best_amplitude_column(candidates_df)

if candidate_period_col and candidate_amplitude_col:
    # Calculate period in hours
    candidates_df['period_hours'] = candidates_df[candidate_period_col] * 24.0
    
    # Cap amplitude at 4 mag
    candidates_df['capped_amplitude'] = np.minimum(candidates_df[candidate_amplitude_col], 4.0)
    
    plt.scatter(
        candidates_df['period_hours'],
        candidates_df['capped_amplitude'],
        c='lightblue',
        alpha=0.6,
        s=20,
        label=f'All Candidates ({len(candidates_df)})'
    )
    print(f"Plotted {len(candidates_df)} candidates using {candidate_period_col}, {candidate_amplitude_col}")

# 2. Plot known CVs (middle layer)
known_period_col = find_best_period_column(known_cvs_df)
known_amplitude_col = find_best_amplitude_column(known_cvs_df)

if known_period_col and known_amplitude_col:
    # Calculate period in hours
    known_cvs_df['period_hours'] = known_cvs_df[known_period_col] * 24.0
    
    # Cap amplitude at 4 mag
    known_cvs_df['capped_amplitude'] = np.minimum(known_cvs_df[known_amplitude_col], 4.0)
    
    plt.scatter(
        known_cvs_df['period_hours'],
        known_cvs_df['capped_amplitude'],
        c='red',
        marker='*',
        s=120,
        alpha=0.8,
        label=f'Known CVs ({len(known_cvs_df)})'
    )
    print(f"Plotted {len(known_cvs_df)} known CVs using {known_period_col}, {known_amplitude_col}")

# 3. Plot targets (top layer) - with 'x' markers
target_period_col = find_best_period_column(targets_df)
target_amplitude_col = find_best_amplitude_column(targets_df)

if target_period_col and target_amplitude_col:
    # Calculate period in hours
    targets_df['period_hours'] = targets_df[target_period_col] * 24.0
    
    # Cap amplitude at 4 mag
    targets_df['capped_amplitude'] = np.minimum(targets_df[target_amplitude_col], 4.0)
    
    plt.scatter(
        targets_df['period_hours'],
        targets_df['capped_amplitude'],
        c='darkgreen',
        marker='x',  # Changed to 'x' marker
        s=100,
        alpha=0.9,
        linewidths=2.0,  # Thicker lines for better visibility
        label=f'Target List ({len(targets_df)})'
    )
    print(f"Plotted {len(targets_df)} targets using {target_period_col}, {target_amplitude_col}")
else:
    print("ERROR: Could not find period/amplitude columns for targets")

# Highlight the period gap (2-3 hours)
plt.axvspan(2, 3, alpha=0.2, color='red', label='Period Gap (2-3h)')
plt.axvline(x=2, linestyle='--', color='red', alpha=0.7, linewidth=1)
plt.axvline(x=3, linestyle='--', color='red', alpha=0.7, linewidth=1)

# Set axis labels
plt.xlabel('Period (hours)', fontsize=16)
plt.ylabel('Amplitude (mag)', fontsize=16)

# Use log scale for x-axis
plt.xscale('log')

# Set limits - amplitude capped at 4
plt.xlim(0.5, 24)  # 30 min to 24 hours
plt.ylim(0, 4)     # 0 to 4 magnitude amplitude

# Add custom tick marks for hours
x_ticks = [0.5, 1, 2, 3, 4, 6, 12, 24]
plt.xticks(x_ticks, [str(h) for h in x_ticks], fontsize=14)
plt.yticks(fontsize=14)

# Add minor ticks
plt.minorticks_on()
plt.tick_params(which='minor', length=4, width=1)
plt.tick_params(which='major', length=8, width=1.5)

# Add grid
plt.grid(True, alpha=0.3)

# Add legend on upper left
plt.legend(loc='upper left', fontsize=14)

# Save figure
output_file = os.path.join(OUTPUT_DIR, "bailey_diagram.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Bailey diagram saved to {output_file}")

# Print additional analysis about the period gap
for group_name, df, period_col in [
    ("Candidates", candidates_df, candidate_period_col),
    ("Known CVs", known_cvs_df, known_period_col),
    ("Targets", targets_df, target_period_col)
]:
    if period_col and 'period_hours' in df.columns:
        below_gap = (df['period_hours'] < 2).sum()
        in_gap = ((df['period_hours'] >= 2) & (df['period_hours'] <= 3)).sum()
        above_gap = (df['period_hours'] > 3).sum()
        
        print(f"\n{group_name} Period Distribution:")
        print(f"  Total: {len(df)}")
        print(f"  Below gap (<2h): {below_gap} ({100*below_gap/len(df):.1f}%)")
        print(f"  In gap (2-3h): {in_gap} ({100*in_gap/len(df):.1f}%)")
        print(f"  Above gap (>3h): {above_gap} ({100*above_gap/len(df):.1f}%)")

print("\nProcessing complete!")