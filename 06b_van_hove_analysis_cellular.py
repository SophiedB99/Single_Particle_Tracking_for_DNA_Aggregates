
# -*- coding: utf-8 -*-
"""
Step 3: Van Hove Analysis for Non-Gaussian Diffusion (Combined 2x2 Panel Plot)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Import necessary Matplotlib components for zoom indicators
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
SIGMA_STUCK_MICRONS = 0.013543
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS = 0.00893
MASS_DIM_REFERENCE = 589.8
USE_MASS_SCALING = True
USE_LOCALIZATION_CORRECTION = True # Enable/disable localization correction
MAX_TRAJ_LENGTHS = [300, 200, 200, 800, 800, 700]
CONDITION_KEYS = ['100bp', '250bp', '500bp', '4713bp', '7614bp', '9553bp']
DIM_FRAGMENTS = ['100bp', '250bp', '500bp']
MPP = 0.102
FPS = 10
# Time lags for the 2x2 panel plot (A, B, C, D)
VAN_HOVE_LAG_SECONDS = [0.1, 1.0, 5.0, 8.0]
BIN_COUNT = 50
MAX_DISPLACEMENT = 0.3
MIN_DISPLACEMENT_FOR_DIFFUSION_PEAK = 0.01 # Critical threshold to ignore sharp peak (in µm)
# File paths (Note: Paths assume environment access to F drive)
Filepath = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/'
FilepathOG = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/'
directories = { 'MDA MB 231': Filepath + 'Aswin Data MDA-MB-231 Corrected for Drift/100 bp/E_output_data/Collected_100bp/tm', 
               '250bp': Filepath + 'Aswin Data CHO Corrected for Drift/250 bp/E_output_data/Collected_250bp/tm', 
               '500bp': Filepath + 'Aswin Data CHO Corrected for Drift/500 bp/E_output_data/Collected_500bp/tm', 
               '4713bp': FilepathOG + 'E_output_data/Collected_pEGFP-C1/tm', 
               '7614bp': FilepathOG + 'E_output_data/Collected_BPK4410/tm', 
               '9553bp': FilepathOG + 'E_output_data/Collected_hCas9D10A/tm'
}

# Initial data labels/colors
colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#DDCC77', '#CC6677']
labels = ['100 bp', '250 bp', '500 bp', '4713 bp', '7614 bp', '9553 bp']
PANEL_LABELS = ['A', 'B', 'C', 'D']
# ============================================================================
# SIGMA CORRECTION FUNCTION
# ============================================================================
def calculate_sigma_for_particle(mass_data, sigma_ref, mass_ref, use_scaling):
    if not use_scaling or mass_ref <= 0 or mass_data.size == 0:
        return sigma_ref
    mean_mass_particle = np.mean(mass_data)
    if mean_mass_particle <= 0:
        return sigma_ref
    mass_ratio = mass_ref / mean_mass_particle
    return sigma_ref * np.sqrt(mass_ratio)
# ============================================================================
# CORE DISPLACEMENT CALCULATION AND CORRECTION
# ============================================================================
def calculate_displacements(pos, time_lags_frames, sigma_particle):
    """
    Calculates displacements and applies the localization correction if sigma_particle > 0.
    """
    displacements = {lt: [] for lt in time_lags_frames}
    # If sigma_particle is 0, noise_term_sq is 0, meaning no correction.
    noise_term_sq = 4 * sigma_particle**2
    for lt in time_lags_frames:
        if lt >= len(pos):
            continue
        diff_vectors = pos[lt:] - pos[:-lt]
        dr_squared_raw = diff_vectors[:, 0]**2 + diff_vectors[:, 1]**2
        # Apply correction: dr_real is corrected displacement magnitude
        # np.maximum(0, ...) ensures displacement magnitude is non-negative
        dr_squared_real = np.maximum(0, dr_squared_raw - noise_term_sq)
        dr_real = np.sqrt(dr_squared_real)
        displacements[lt].extend(dr_real)
    return displacements
def process_file_for_van_hove(filename, mpp, max_length, time_lags_frames, label_key):
    try:
        # Corrected syntax: pd.read_csv
        traj_df = pd.read_csv(filename)
    except Exception:
        return None
    all_displacements = {lt: [] for lt in time_lags_frames}
    for pid, ptraj in traj_df.groupby('particle'):
        # 1. Determine reference sigma and mass
        if label_key in DIM_FRAGMENTS:
            current_sigma_ref = SIGMA_DIM_MICRONS
            current_mass_ref = MASS_DIM_REFERENCE
        else:
            current_sigma_ref = SIGMA_STUCK_MICRONS
            current_mass_ref = MASS_STUCK_REFERENCE
        mass_data = ptraj['mass'].values if 'mass' in ptraj.columns else np.array([])
        
        # 2. Calculate initial sigma (with potential mass scaling)
        sigma_particle_calc = calculate_sigma_for_particle(
            mass_data, current_sigma_ref, current_mass_ref, USE_MASS_SCALING
        )
        
        # 3. Apply global localization correction flag
        if USE_LOCALIZATION_CORRECTION:
            sigma_particle = sigma_particle_calc
        else:
            sigma_particle = 0.0 # Override to 0 if correction is disabled
            
        pos_df = ptraj.set_index('frame')[['x', 'y']] * mpp
        pos = pos_df.reindex(np.arange(pos_df.index[0], 1 + pos_df.index[-1])).to_numpy()
        pos = pos[0:max_length]
        if len(pos) < 50: continue
        
        # 4. Calculate displacements (correction is applied inside this function if sigma_particle > 0)
        particle_displacements = calculate_displacements(pos, time_lags_frames, sigma_particle)
        for lt, dr_list in particle_displacements.items():
            all_displacements[lt].extend(dr_list)
    return all_displacements
def process_directory_for_van_hove(directory, mpp, max_length, time_lags_frames, label_key):
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return None
    combined_displacements = {lt: [] for lt in time_lags_frames}
    total_tracks = 0
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            result = process_file_for_van_hove(
                os.path.join(directory, filename), mpp, max_length, time_lags_frames, label_key
            )
            if result is not None:
                for lt, dr_list in result.items():
                    combined_displacements[lt].extend(dr_list)
            try:
                # Corrected syntax: pd.read_csv
                total_tracks += pd.read_csv(os.path.join(directory, filename))['particle'].nunique()
            except:
                pass
    if not any(combined_displacements.values()):
        print(f"Warning: No valid displacements found in {directory}.")
        return None
    print(f" Total tracks processed: {total_tracks}")
    return combined_displacements
# ============================================================================
# 4. PLOTTING AND DIFFUSION PEAK QUANTIFICATION (COMBINED PANEL PLOT)
# ============================================================================
def plot_van_hove(results_vh, condition_labels, colors, labels, lag_seconds, bin_count, max_disp, fps):
    """ Generates a single 2x2 panel plot (A, B, C, D) of Van Hove distributions with dynamic inset zooms for each panel, sorted by Max Probability Density. """
    min_displacement_threshold = MIN_DISPLACEMENT_FOR_DIFFUSION_PEAK
    # Setup the combined figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for k, lag_time_s in enumerate(lag_seconds):
        lag_time_f = int(lag_time_s * fps)
        ax = axes[k] # Current subplot axis (A, B, C, or D)
        panel_label = PANEL_LABELS[k]
        # --- PASS 1: Calculate Histograms and Find Global Maxima/Minima for this specific Delta t ---
        all_plot_data = {}
        quantified_peaks = []
        max_peak_disp_global = 0.0
        max_density_global = 0.0
        min_peak_Gs_global = np.inf # Track the smallest peak height for dynamic ZOOM_Y_MIN
        
        print("\n" + "="*90)
        print(f"STARTING DATA COLLECTION for Panel {panel_label}: $\\Delta t = {lag_time_s:.1f}$ s")
        
        for i, label_key in enumerate(condition_labels):
            if label_key in results_vh:
                displacements = results_vh[label_key].get(lag_time_f)
                if not displacements:
                    continue
                
                # 1. Calculate histogram (Probability Density Function)
                counts, bins = np.histogram(
                    displacements,
                    bins=bin_count,
                    range=(0, max_disp),
                    density=True
                )
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # 2. Find the index where the bin center exceeds the threshold
                start_index = np.searchsorted(bin_centers, min_displacement_threshold)
                
                # 3. Search for the maximum probability density *only* in the diffusion region
                counts_diffusion = counts[start_index:]
                bin_centers_diffusion = bin_centers[start_index:]
                
                if counts_diffusion.size == 0:
                    # If no significant displacement is found beyond the threshold, skip peak tracking for this fragment
                    # But still plot the raw counts if they exist
                    max_density = 0.0
                    peak_displacement = 0.0
                    n_points = len(displacements)
                else:
                    max_density_index = np.argmax(counts_diffusion)
                    max_density = counts_diffusion[max_density_index]
                    peak_displacement = bin_centers_diffusion[max_density_index]
                    n_points = len(displacements)
                
                # 4. Update Global Maxima and Minima
                max_peak_disp_global = max(max_peak_disp_global, peak_displacement)
                # Use the actual observed density maximum across the entire range for the main plot scale
                max_density_global = max(max_density_global, np.max(counts))
                # Track minimum peak height for tight zoom floor
                if max_density > 0:
                    min_peak_Gs_global = min(min_peak_Gs_global, max_density)
                    
                original_index = CONDITION_KEYS.index(label_key)
                fragment_label = labels[original_index]
                
                # Store data and peak statistics
                all_plot_data[label_key] = {
                    'counts': counts,
                    'bin_centers': bin_centers,
                    'color': colors[original_index],
                    'label': fragment_label,
                    'N': n_points,
                    'peak_disp': peak_displacement,
                    'max_Gs': max_density
                }
                quantified_peaks.append({
                    'Fragment': fragment_label,
                    'N': n_points,
                    'Peak_Disp': peak_displacement,
                    'Max_Gs': max_density
                })
        
        # --- PASS 2: Plotting, Zoom, and Formatting ---
        
        # --- Calculate Dynamic Zoom Limits (Panel-Specific Scaling) ---
        if panel_label == 'A':
            ZOOM_X_MAX = max(0.06, max_peak_disp_global * 1.2) # slightly tighter X zoom
            ZOOM_Y_MAX = max_density_global * 0.2 # smaller Y zoom
        elif panel_label == 'B':
            ZOOM_X_MAX = max(0.06, max_peak_disp_global * 1.3)
            ZOOM_Y_MAX = max_density_global * 0.3
        elif panel_label == 'C':
            ZOOM_X_MAX = max(0.08, max_peak_disp_global * 1.4)
            ZOOM_Y_MAX = max_density_global * 0.4 # a bit lower than D
        elif panel_label == 'D':
            ZOOM_X_MAX = max(0.1, max_peak_disp_global * 1.5) # your current perfect zoom
            ZOOM_Y_MAX = max_density_global * 0.5
        else:
            ZOOM_X_MAX = max(0.1, max_peak_disp_global * 1.3)
            ZOOM_Y_MAX = max_density_global * 0.4
            
        # --- Common lower limit for all ---
        if min_peak_Gs_global == np.inf or min_peak_Gs_global == 0:
            ZOOM_Y_MIN = 0.0
        else:
            ZOOM_Y_MIN = min_peak_Gs_global * 0.7
            
        # --- 1. Create Inset Axes ---
        # Position the inset within the current subplot (ax)
        ax_inset = ax.inset_axes([0.36, 0.35, 0.58, 0.58])
        
        # --- Print Header and Table (Sorted) ---
        print("\n" + "="*90)
        print(f"PANEL {panel_label}: Diffusion Peak Analysis at $\\Delta t = {lag_time_s:.1f}$ s (Sorted by Max Probability Density)")
        print(r"Fragment | N (Displacements) | Diffusion Peak ($\Delta r^{\text{max}}$, $\mu\mathrm{m}$) | Max Probability Density ($G_s^{\text{max}}$)")
        print("-"*90)
        # Sort results by Max_Gs descending
        quantified_peaks.sort(key=lambda x: x['Max_Gs'], reverse=True)
        for peak_data in quantified_peaks:
            print(f"{peak_data['Fragment']:<8} | {peak_data['N']:<17} | {peak_data['Peak_Disp']:.4f} | {peak_data['Max_Gs']:.4f}")
        print("="*90)
        
        # --- 2. Plotting Loop (Main and Inset) ---
        for label_key, data in all_plot_data.items():
            # Plot on MAIN AXES
            ax.plot(
                data['bin_centers'], data['counts'],
                color=data['color'],
                linewidth=2,
                label=f"{data['label']} (N={data['N']})"
            )
            # Highlight the found diffusion peak on MAIN AXES
            if data['max_Gs'] > 0:
                ax.plot(
                    data['peak_disp'], data['max_Gs'],
                    marker='o',
                    color='k',
                    markersize=5,
                    linestyle='',
                    zorder=10
                )
            
            # Plot on INSET AXES
            ax_inset.plot(
                data['bin_centers'], data['counts'],
                color=data['color'],
                linewidth=2
            )
            # Highlight the found diffusion peak on INSET AXES
            if data['max_Gs'] > 0:
                ax_inset.plot(
                    data['peak_disp'], data['max_Gs'],
                    marker='o',
                    color='k',
                    markersize=8,
                    linestyle='',
                    zorder=10
                )
        
        # --- 3. Format Main Plot ---
        ax.set_title(f'$\\Delta t = {lag_time_s:.1f}$ s', fontsize=16)
        ax.set_xlabel(r'$\Delta r$ ($\mu\mathrm{m}$)', fontsize=14)
        # Set Y-label only for the left column plots (A and C)
        if k % 2 == 0:
            ax.set_ylabel(r'$G_s(\Delta r, \Delta t)$', fontsize=14)
        else:
            ax.set_ylabel('')
        # ax.text(0.95, 0.95,
        #         f"N = {data['N']}",
        #         transform=ax.transAxes,
        #         ha = 'right', va = 'top',
        #         fontsize = 12, color = 'k')    
        # Match alpha plot style
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(which="both", axis="both", direction="in", labelsize=10, length=6, width = 1.5)
        ax.set_xlim(0, max_disp)
        ax.grid(False)
        # Add legend only to the top right plot (B) for space efficiency
        if k == 1:
            # Recreate legend using the stored data labels
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [l.split(' (N=')[0] for l in labels], loc='upper right', fontsize=14, frameon=True)
        ax.grid(True, alpha=0.3, linestyle=':')
        # Panel label (A, B, C, D)
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
        
        # --- 4. Format Inset Plot (Dynamic Zoomed Region) ---
        ax_inset.set_xlim(0.0, ZOOM_X_MAX)
        ax_inset.set_ylim(ZOOM_Y_MIN, ZOOM_Y_MAX)
        ax_inset.yaxis.set_ticks_position('both')
        ax_inset.xaxis.set_ticks_position('both')
        for spine in ax_inset.spines.values():
            spine.set_linewidth(1.0)
        ax_inset.tick_params(which="both", axis="both", direction="in", labelsize=8, length=4, width = 1.0)
        ax_inset.grid(False)
        
    plt.tight_layout()
    plt.show()
# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("STEP 3: VAN HOVE CORRELATION ANALYSIS (w/ SIGMA CORRECTION)")
    print(f"Localization Correction: {'Enabled' if USE_LOCALIZATION_CORRECTION else 'Disabled'}")
    print(f"Mass Scaling: {'Enabled' if USE_MASS_SCALING else 'Disabled'}")
    print(f"Diffusion Peak Threshold: {MIN_DISPLACEMENT_FOR_DIFFUSION_PEAK:.2f} μm (to exclude caging/noise)")
    print("="*80)
    
    # Prepare time lags for computation (in frames)
    TIME_LAGS_FRAMES = [int(t * FPS) for t in VAN_HOVE_LAG_SECONDS]
    van_hove_results = {}
    
    # Process each condition directory
    for i, label in enumerate(CONDITION_KEYS):
        if label in directories:
            directory = directories[label]
            max_length = MAX_TRAJ_LENGTHS[i]
            print(f"Processing {label} (Max Length: {max_length})...")
            
            if not os.path.exists(directory):
                print(f"CRITICAL: Directory for {label} not found at {directory}")
                continue
            
            result = process_directory_for_van_hove(
                directory, MPP, max_length, TIME_LAGS_FRAMES, label
            )
            
            if result is not None:
                van_hove_results[label] = result
    
    # Plotting and final quantification
    processed_keys = [k for k in CONDITION_KEYS if k in van_hove_results]
    if not van_hove_results:
        print("\nERROR: No data collected for Van Hove plots.")
    else:
        plot_van_hove(
            van_hove_results,
            processed_keys,
            colors,
            labels,
            VAN_HOVE_LAG_SECONDS,
            BIN_COUNT,
            MAX_DISPLACEMENT,
            FPS
        )
    
    print("\n" + "="*80)
    print("VAN HOVE ANALYSIS COMPLETE")
    print("="*80)