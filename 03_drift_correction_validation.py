# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 17:11:16 2025

@author: sophiedeboer
"""

import os
import glob
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.collections import LineCollection
import warnings

# Suppress warnings that might arise during curve fitting on noisy data
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from msd_core import msd, power_law, trajectory

# --- Provided MSD Calculation and Fitting Functions ---

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings (useful for individual trajectories)."""
    ax = plt.gca() if ax is None else ax
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    lc.set_array(np.asarray(c))
    ax.add_collection(lc)
    return lc

def powerfit(msd, fps=10, min_lag=20, max_lag=80):
    """Fit a power law to MSD data."""
    y = msd['msd'].values
    x = msd.index.values.astype('float64') / float(fps)
    n_points = len(x)
    fit_start = min(min_lag, n_points - 20)
    fit_end = min(max_lag, n_points)
    
    if fit_end - fit_start < 20:
        return None, 0.0, None

    try:
        pars, cov = curve_fit(
            f=power_law,
            xdata=x[fit_start:fit_end],
            ydata=y[fit_start:fit_end],
            p0=[1, 0.01],
            bounds=(-np.inf, np.inf),
            maxfev=10000
        )
        
        residuals = y[fit_start:fit_end] - power_law(x[fit_start:fit_end], *pars)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y[fit_start:fit_end] - np.mean(y[fit_start:fit_end])) ** 2)
        
        if ss_tot == 0:
            return None, 0.0, None
            
        r_squared = 1.0 - (ss_res / ss_tot)
        perr = np.sqrt(np.diag(cov))
        return pars, r_squared, perr
        
    except Exception as e:
        return None, 0.0, None

def analyze_stuck_bead(msd_df, fps=10, plateau_start=10, plateau_end=50):
    """Measure localization precision from plateau MSD."""
    y = msd_df['msd'].values
    n_points = len(y)
    plateau_start = min(plateau_start, n_points - 10)
    plateau_end = min(plateau_end, n_points)
    
    if plateau_end - plateau_start < 5:
        return None, None, None
        
    plateau_region = y[plateau_start:plateau_end]
    valid_plateau = plateau_region[~np.isnan(plateau_region)]
    
    if len(valid_plateau) < 3:
        return None, None, None
        
    plateau_msd = np.mean(valid_plateau)
    std_msd = np.std(valid_plateau)
    
    if plateau_msd > 0:
        # Based on: MSD(\tau) \approx 4\sigma^2
        localization_error = np.sqrt(plateau_msd / 4.0) 
    else:
        localization_error = None
        
    return plateau_msd, std_msd, localization_error

def imsd(filename, mpp, fps, video, directory, plotindividual=False, max_lagtime=100, r2_threshold=0.3, analyze_stuck=False):
    """
    Process MSD data, calculates ensemble average, and performs power-law fitting.
    Returns: resultsmsd, resultstraj, alpha, ..., total_particles, passed_particles
    """
    statistic = 'msd'
    try:
        traj = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None, None, None, None, None, None, None, None, 0, 0

    pos_columns = ['x', 'y']
    ids = []
    trajec = []
    msds = []
    alpha = []
    msdsuper = []
    msdsub = []
    alphasuper = []
    alphasub = []
    localization_data = [] 
    index = 0
    total_particles = 0
    skip_reasons = { 
        'empty_msd': 0, 'too_short': 0, 'fit_failed': 0, 
        'low_r2': 0, 'invalid_alpha': 0, 'passed': 0 
    }

    for pid, ptraj in traj.groupby('particle'):
        total_particles += 1
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
        msdtemp = msd(pos, max_lagtime)

        if msdtemp is None or msdtemp.empty or 'msd' not in msdtemp.columns or msdtemp['msd'].isna().all() or (msdtemp['msd'] == 0).all():
            skip_reasons['empty_msd'] += 1
            continue

        traj_length = len(msdtemp)
        if traj_length < 80:
            skip_reasons['too_short'] += 1
            continue 

        # Power law fit
        try:
            result = powerfit(msdtemp, fps)
            if result[0] is None:
                skip_reasons['fit_failed'] += 1
                continue
            alphatemp, r_squaredtemp, perr = result
        except Exception:
            skip_reasons['fit_failed'] += 1
            continue

        # Check fit quality
        if alphatemp[0] <= 0:
            skip_reasons['invalid_alpha'] += 1
            continue
        if r_squaredtemp < r2_threshold:
            skip_reasons['low_r2'] += 1
            continue

        # Particle passed!
        skip_reasons['passed'] += 1
        alpha.append(alphatemp)
        msds.append(msdtemp)
        trajec.append(trajectory(pos, pid, max_lagtime, video, pos_columns))
        ids.append(index)
        index += 1

        # Classify as super/sub diffusive
        if alphatemp[0] > 1:
            msdsuper.append(msdtemp)
            alphasuper.append(alphatemp)
        elif alphatemp[0] < 1:
            msdsub.append(msdtemp)
            alphasub.append(alphatemp)
        
        if analyze_stuck:
            plateau_msd, std_msd, loc_error = analyze_stuck_bead(msdtemp, fps)
            if loc_error is not None and 0 < loc_error < 1.0:
                 localization_data.append([plateau_msd, loc_error])
            else:
                 localization_data.append([np.nan, np.nan])

    passed_particles = skip_reasons['passed']
    
    print(f"\n{'─'*60}")
    print(f"File: {os.path.basename(filename)}")
    print(f"{'─'*60}")
    print(f"Total particles: {total_particles}")
    print(f"  Passed (R²≥{r2_threshold}): {passed_particles}")
    print(f"{'─'*60}\n")
    
    if not msds: 
        return None, None, None, None, None, None, None, None, total_particles, 0

    # Concatenate results to form ensemble MSD (resultsmsd)
    resultsmsd = pd.concat(msds, keys=ids)
    resultsmsd = resultsmsd.swaplevel(0, 1)[statistic].unstack()
    lagt = resultsmsd.index.values.astype('float64') / float(fps)
    resultsmsd.set_index(lagt, inplace=True)
    resultsmsd.index.name = 'lag time [s]'
    
    resultstraj = pd.concat(trajec)
    time = resultstraj.index.values.astype('float64') / float(fps)
    resultstraj.set_index(time, inplace=True)
    resultstraj.index.name = 'Time [s]'
    
    alpha = pd.DataFrame(alpha)
    
    return resultsmsd, resultstraj, alpha, None, None, None, None, None, total_particles, passed_particles

# --- Comparison Logic ---

def find_data_file(directory):
    """Find the first .csv file in the given directory."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    return csv_files[0] if csv_files else None

def fit_ensemble_alpha(msd_series, fps, min_lag=20, max_lag=80):
    """
    Fit power law to ensemble MSD to extract alpha.
    Returns: alpha, r_squared
    """
    y = msd_series.values
    x = msd_series.index.values
    n_points = len(x)
    
    fit_start = min(min_lag, n_points - 20)
    fit_end = min(max_lag, n_points)
    
    if fit_end - fit_start < 20:
        return np.nan, np.nan
    
    try:
        pars, cov = curve_fit(
            f=power_law,
            xdata=x[fit_start:fit_end],
            ydata=y[fit_start:fit_end],
            p0=[1, 0.01],
            bounds=(-np.inf, np.inf),
            maxfev=10000
        )
        
        residuals = y[fit_start:fit_end] - power_law(x[fit_start:fit_end], *pars)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y[fit_start:fit_end] - np.mean(y[fit_start:fit_end])) ** 2)
        
        if ss_tot == 0:
            return np.nan, np.nan
            
        r_squared = 1.0 - (ss_res / ss_tot)
        return pars[0], r_squared  # Return alpha (first parameter)
        
    except Exception:
        return np.nan, np.nan

def compare_drift_correction(experiment_path, mpp, fps, r2_threshold, output_dir, summary_data):
    """
    Analyzes and compares MSD data from 'tf' (unfiltered) and 'tm' (filtered) subdirectories.
    """
    tf_dir = os.path.join(experiment_path, 'tf')
    tm_dir = os.path.join(experiment_path, 'tm')
    
    experiment_name = os.path.basename(experiment_path)
    
    if not (os.path.exists(tf_dir) and os.path.exists(tm_dir)):
        print(f"Skipping {experiment_name}: Missing 'tf' or 'tm' directories.")
        return

    tf_file = find_data_file(tf_dir)
    tm_file = find_data_file(tm_dir)
    
    if not (tf_file and tm_file):
        print(f"Skipping {experiment_name}: Could not find CSV data file in 'tf' or 'tm'.")
        return

    print(f"\n--- Analyzing Experiment: {experiment_name} ---")
    
    # 1. Process tf (Raw)
    print("Processing tf (Raw)...")
    results_tf = imsd(tf_file, mpp=mpp, fps=fps, video=experiment_name + "_tf", 
                      directory=tf_dir, r2_threshold=r2_threshold)
    msd_tf, _, alpha_tf, _, _, _, _, _, total_tf, passed_tf = results_tf
    
    # 2. Process tm (Drift Corrected)
    print("Processing tm (Drift Corrected)...")
    results_tm = imsd(tm_file, mpp=mpp, fps=fps, video=experiment_name + "_tm", 
                      directory=tm_dir, r2_threshold=r2_threshold)
    msd_tm, _, alpha_tm, _, _, _, _, _, total_tm, passed_tm = results_tm
    
    if msd_tf is None or msd_tm is None:
        print(f"Could not calculate ensemble MSD for both datasets in {experiment_name}.")
        return

    # Calculate Ensemble Mean MSD for plotting
    msd_mean_tf = msd_tf.mean(axis=1)
    msd_mean_tm = msd_tm.mean(axis=1)
    
    # Calculate ensemble alpha by fitting power law to ensemble MSD
    ensemble_alpha_tf, r2_tf = fit_ensemble_alpha(msd_mean_tf, fps)
    ensemble_alpha_tm, r2_tm = fit_ensemble_alpha(msd_mean_tm, fps)
    
    # Calculate mean alpha from individual trajectories
    mean_alpha_tf = alpha_tf[0].mean() if alpha_tf is not None and len(alpha_tf) > 0 else np.nan
    std_alpha_tf = alpha_tf[0].std() if alpha_tf is not None and len(alpha_tf) > 0 else np.nan
    mean_alpha_tm = alpha_tm[0].mean() if alpha_tm is not None and len(alpha_tm) > 0 else np.nan
    std_alpha_tm = alpha_tm[0].std() if alpha_tm is not None and len(alpha_tm) > 0 else np.nan
    
    # 3. Align and Calculate Difference
    comparison_df = pd.DataFrame({
        'MSD_tf': msd_mean_tf,
        'MSD_tm': msd_mean_tm
    }).dropna()
    
    if comparison_df.empty:
        print(f"No overlapping lag times found for comparison in {experiment_name}.")
        return

    # Difference: Quantifies the magnitude of the measured drift (Raw - Corrected)
    comparison_df['MSD_Difference'] = comparison_df['MSD_tf'] - comparison_df['MSD_tm']
    
    # Calculate statistics for the summary table
    avg_diff = comparison_df['MSD_Difference'].mean()
    std_diff = comparison_df['MSD_Difference'].std()
    max_diff = comparison_df['MSD_Difference'].max()
    
    # Store summary data
    summary_data.append({
        'Experiment': experiment_name,
        'Total_Particles_Raw': total_tf,
        'Passed_Particles_Raw': passed_tf,
        'Total_Particles_Corrected': total_tm,
        'Passed_Particles_Corrected': passed_tm,
        'Mean_ΔMSD (μm²)': avg_diff,
        'Std_ΔMSD (μm²)': std_diff,
        'Max_ΔMSD (μm²)': max_diff,
        'Alpha_Raw (ensemble)': ensemble_alpha_tf,
        'Alpha_Corrected (ensemble)': ensemble_alpha_tm,
        'Alpha_Raw (mean±std)': f'{mean_alpha_tf:.3f}±{std_alpha_tf:.3f}' if not np.isnan(mean_alpha_tf) else 'N/A',
        'Alpha_Corrected (mean±std)': f'{mean_alpha_tm:.3f}±{std_alpha_tm:.3f}' if not np.isnan(mean_alpha_tm) else 'N/A'
    })

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    tau = comparison_df.index.values  # Lag time (s)
    
    # --- Panel 1: Log-Log MSD Comparison ---
    ax1.loglog(tau, comparison_df['MSD_tm'], label='Drift Corrected (tm)', color='tab:orange', linewidth=2)
    ax1.loglog(tau, comparison_df['MSD_tf'], label='Raw (tf)', color='tab:blue', linewidth=2)
    
    # Add alpha values to legend
    alpha_label = f'α_raw={ensemble_alpha_tf:.3f}, α_corr={ensemble_alpha_tm:.3f}'
    ax1.text(0.05, 0.05, alpha_label, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\tau$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
            title=f'MSD Comparison: {experiment_name}')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- Panel 2: Linear Difference Plot ---
    ax2.plot(tau, comparison_df['MSD_Difference'], 'k-', 
             label=r'$\text{MSD}_{\text{raw}} - \text{MSD}_{\text{corrected}}$', linewidth=2)
    
    # Create info text for the plot
    info_text = (f'Mean ΔMSD: {avg_diff:.4f} μm²\n'
                f'Raw: {passed_tf}/{total_tf} particles\n'
                f'Corrected: {passed_tm}/{total_tm} particles\n'
                f'α_raw: {mean_alpha_tf:.3f}±{std_alpha_tf:.3f}\n'
                f'α_corr: {mean_alpha_tm:.3f}±{std_alpha_tm:.3f}')
    
    ax2.set(xlabel=r'$\tau$ (s)', 
            ylabel=r'$\Delta \text{MSD}$ ($\mathrm{\mu}$m$^2$)',
            title=f'Difference in Ensemble MSD')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    # Add text box with statistics
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'MSD_Comparison_{experiment_name}.pdf')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved comparison plot to: {save_path}")
    print("=" * 60)


def process_all_experiments(base_path):
    """
    Iterates through all experiment folders and runs the comparison analysis.
    """
    # --- Configuration Parameters ---
    MICRONS_PER_PIXEL = 0.16  # mpp
    FRAMES_PER_SECOND = 10   # fps
    R2_THRESHOLD = 0.3       # Minimum R^2 for accepting trajectory fits
    OUTPUT_DIR = os.path.join(base_path, 'MSD_Comparison_Results') 
    
    print(f"Starting analysis in base path: {base_path}")
    print(f"Results will be saved in: {OUTPUT_DIR}")

    if not os.path.isdir(base_path):
        print(f"Error: Base path not found: {base_path}")
        return

    # Find all immediate subdirectories
    experiment_dirs = [d for d in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, d))]
    
    # Store summary data for all experiments
    summary_data = []
    
    for experiment_name in experiment_dirs:
        experiment_path = os.path.join(base_path, experiment_name)
        
        compare_drift_correction(experiment_path, 
                                 mpp=MICRONS_PER_PIXEL, 
                                 fps=FRAMES_PER_SECOND, 
                                 r2_threshold=R2_THRESHOLD,
                                 output_dir=OUTPUT_DIR,
                                 summary_data=summary_data)
    
    # Create and save summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(OUTPUT_DIR, 'MSD_Comparison_Summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary table saved to: {summary_csv_path}")
        print(f"{'='*60}\n")
        print(summary_df.to_string(index=False))
        
        # Create a visual table plot
        create_summary_table_plot(summary_df, OUTPUT_DIR)
        

def create_summary_table_plot(summary_df, output_dir):
    """Create a visual table with key metrics."""
    fig, ax = plt.subplots(figsize=(16, max(4, len(summary_df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Select key columns for display
    display_cols = ['Experiment', 'Passed_Particles_Raw', 'Passed_Particles_Corrected', 
                   'Mean_ΔMSD (μm²)', 'Std_ΔMSD (μm²)', 
                   'Alpha_Raw (ensemble)', 'Alpha_Corrected (ensemble)']
    
    display_df = summary_df[display_cols].copy()
    
    # Format numeric columns
    display_df['Mean_ΔMSD (μm²)'] = display_df['Mean_ΔMSD (μm²)'].apply(lambda x: f'{x:.4f}')
    display_df['Std_ΔMSD (μm²)'] = display_df['Std_ΔMSD (μm²)'].apply(lambda x: f'{x:.4f}')
    display_df['Alpha_Raw (ensemble)'] = display_df['Alpha_Raw (ensemble)'].apply(
        lambda x: f'{x:.3f}' if not pd.isna(x) else 'N/A')
    display_df['Alpha_Corrected (ensemble)'] = display_df['Alpha_Corrected (ensemble)'].apply(
        lambda x: f'{x:.3f}' if not pd.isna(x) else 'N/A')
    
    # Rename columns for better display
    display_df.columns = ['Experiment', 'N (Raw)', 'N (Corrected)', 
                         'Mean ΔMSD', 'Std ΔMSD', 'α (Raw)', 'α (Corrected)']
    
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.18, 0.11, 0.14, 0.14, 0.14, 0.13, 0.16])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Drift Correction Analysis Summary', fontsize=14, weight='bold', pad=20)
    
    save_path = os.path.join(output_dir, 'MSD_Summary_Table.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Summary table plot saved to: {save_path}")


# --- Execution Block ---

BASE_DATA_PATH = r"D:\A_Data\A_DNATranslocation\Experiments_2025\DNA Tracking\Corrected_Data\Control Experiments_Plasmid Aggregates on Glass\E_output_data\pEGFP-C1"

if __name__ == '__main__':
    process_all_experiments(BASE_DATA_PATH)