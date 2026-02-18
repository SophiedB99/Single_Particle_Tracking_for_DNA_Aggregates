# -*- coding: utf-8 -*-
"""
Step 4: Trajectory Orientation Analysis (Next Step Angle Bias)
*** REVISED: 1x3 Linear Subplots + New Panel for Directed vs. Confined Ratios ***
@author: sophiedeboer
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import t, chi2
import math

# ============================================================================
# INHERITED CONFIGURATION (UPDATED WITH NEW REFERENCE MASS)
# ============================================================================
MPP = 0.102 # Microns per pixel
FPS = 10 # Frames per second
MAX_TRAJ_LENGTHS = [300, 200, 200, 800, 800, 700]
CONDITION_KEYS = ['100bp', '250bp', '500bp', '4713bp', '7614bp', '9553bp']
DIM_FRAGMENTS = ['100bp', '250bp', '500bp'] # Keys for fragments using dim reference
Filepath = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/'
FilepathOG = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/'
directories = {
    '100bp': Filepath + 'Aswin Data CHO Corrected for Drift/100 bp/E_output_data/Collected_100bp/tm',
    '250bp': Filepath + 'Aswin Data CHO Corrected for Drift/250 bp/E_output_data/Collected_250bp/tm',
    '500bp': Filepath + 'Aswin Data CHO Corrected for Drift/500 bp/E_output_data/Collected_500bp/tm',
    '4713bp': FilepathOG + 'E_output_data/Collected_pEGFP-C1/tm',
    '7614bp': FilepathOG + 'E_output_data/Collected_BPK4410/tm',
    '9553bp': FilepathOG + 'E_output_data/Collected_hCas9D10A/tm'
}
# Colorblind-friendly palette (red-purple theme using ColorBrewer2)
colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#DDCC77', '#CC6677']
labels = ['100 bp', '250 bp', '500 bp', '4713 bp', '7614 bp', '9553 bp']

# --- Reference Values for Localization Error Correction ---
SIGMA_STUCK_MICRONS = 0.013543 # Localization error in microns (um)
MASS_STUCK_REFERENCE = 3381.5 # Reference mass/intensity of the stuck beads
SIGMA_DIM_MICRONS = 0.00893 # um
MASS_DIM_REFERENCE = 589.8 # median mass from YOYO-1 reference
USE_MASS_SCALING = True # Set to True to scale sigma by particle mass/intensity

# ============================================================================
# SIGMA CORRECTION FUNCTIONS
# ============================================================================
def calculate_sigma_for_particle(mass_data, sigma_ref, mass_ref, use_scaling):
    """ Scale sigma based on particle brightness (mass). """
    if not use_scaling or mass_ref <= 0:
        return sigma_ref
    if mass_data.size == 0:
        return sigma_ref
    mean_mass_particle = np.mean(mass_data)
    if mean_mass_particle <= 0:
        return sigma_ref
    mass_ratio = mass_ref / mean_mass_particle
    return sigma_ref * np.sqrt(mass_ratio)

# ============================================================================
# DIRECTIONAL ANALYSIS FUNCTIONS
# ============================================================================
def calculate_next_step_angles(pos, lag_time=1, sigma_particle=0, displacement_multiplier=2.0):
    """Calculates the 'next step angle' with noise filtering."""
    if len(pos) < 2 * lag_time + 1:
        return np.array([])
    
    dr1 = pos[lag_time:] - pos[:-lag_time]
    v1 = dr1[:-lag_time]
    v2 = dr1[lag_time:]
    
    if len(v1) == 0:
        return np.array([])
    
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    
    # Noise filtering based on localization error (sigma)
    noise_threshold = displacement_multiplier * sigma_particle
    non_noise = (mag1 >= noise_threshold) & (mag2 >= noise_threshold)
    
    v1_filtered = v1[non_noise]
    v2_filtered = v2[non_noise]
    mag1_filtered = mag1[non_noise]
    mag2_filtered = mag2[non_noise]
    
    if len(v1_filtered) == 0:
        return np.array([])
    
    dot_product = np.sum(v1_filtered * v2_filtered, axis=1)
    non_zero = (mag1_filtered > 1e-9) & (mag2_filtered > 1e-9)
    dot_product = dot_product[non_zero]
    mag1_filtered = mag1_filtered[non_zero]
    mag2_filtered = mag2_filtered[non_zero]
    
    if len(dot_product) == 0:
        return np.array([])
    
    cosine_theta = dot_product / (mag1_filtered * mag2_filtered)
    cosine_theta = np.clip(cosine_theta, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosine_theta))
    return angles_deg

def directional_analysis_per_file(filename, mpp, max_length, lag_time_steps, label_key):
    """Processes a single trajectory file for angular bias."""
    try:
        traj_df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return {}
        
    all_angles = {lag_time_steps: []}
    pos_columns = ['x', 'y']
    
    for pid, ptraj in traj_df.groupby('particle'):
        if not all(col in ptraj.columns for col in ['frame', 'x', 'y']):
            continue

        if label_key in DIM_FRAGMENTS:
            current_sigma_ref = SIGMA_DIM_MICRONS
            current_mass_ref = MASS_DIM_REFERENCE
        else:
            current_sigma_ref = SIGMA_STUCK_MICRONS
            current_mass_ref = MASS_STUCK_REFERENCE
            
        mass_data = ptraj['mass'].values if 'mass' in ptraj.columns else np.array([])
        sigma_particle = calculate_sigma_for_particle(
            mass_data, current_sigma_ref, current_mass_ref, USE_MASS_SCALING
        )
        
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
        pos = pos[0:max_length]
        
        if len(pos) < 2 * lag_time_steps + 1:
            continue
            
        angles = calculate_next_step_angles(
            pos,
            lag_time=lag_time_steps,
            sigma_particle=sigma_particle,
            displacement_multiplier=2.0
        )
        
        if len(angles) > 0:
            all_angles[lag_time_steps].extend(angles)
            
    return all_angles

def process_directional_directory(directory, mpp, max_length, lag_time_steps, label_key):
    """Processes all CSV files in directory."""
    all_angles_combined = []
    if not os.path.isdir(directory):
        return np.array([])
        
    for filename in os.listdir(directory):
        if not filename.startswith('.') and filename.endswith('.csv'):
            result = directional_analysis_per_file(
                os.path.join(directory, filename),
                mpp, max_length, lag_time_steps, label_key
            )
            
            if lag_time_steps in result and len(result[lag_time_steps]) > 0:
                all_angles_combined.extend(result[lag_time_steps])
                
    return np.array(all_angles_combined)

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================
def chi_square_test_against_uniform(angles, bins):
    """Performs chi-square test against uniform distribution."""
    observed, _ = np.histogram(angles, bins=bins)
    expected = np.full_like(observed, len(angles) / len(observed), dtype=float)
    
    mask = expected > 0
    observed = observed[mask]
    expected = expected[mask]
    
    if len(observed) < 2:
        return np.nan, np.nan
        
    chi_square_stat = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    p_value = 1 - chi2.cdf(chi_square_stat, df)
    return chi_square_stat, p_value

def calculate_directed_fraction(angles):
    """Calculate fraction of angles < 90 degrees with 95% CI (Wilson Score interval)."""
    n = len(angles)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    directed = np.sum(angles < 90)
    fraction = directed / n
    
    z = 1.96 # 95% CI (Wilson Score interval formula implementation)
    denominator = 1 + z**2/n
    centre = (fraction + z**2/(2*n)) / denominator
    margin = z * np.sqrt(fraction*(1-fraction)/n + z**2/(4*n**2)) / denominator
    ci_lower = max(0, centre - margin)
    ci_upper = min(1, centre + margin)
    
    return fraction, ci_lower, ci_upper

# ============================================================================
# NEW PLOTTING FUNCTIONS
# ============================================================================
def plot_linear_nsab_panel(ax, lag_label, distributions, labels, colors, bins, panel_label):
    """Generates a single subplot for the linear histogram plot."""
    present_labels = [label for label in labels if label in distributions]
    max_density = 0

    # First pass to determine max_density for y-limit
    for current_label in present_labels:
        angles_data = distributions[current_label]
        hist, _ = np.histogram(angles_data, bins=bins, density=True)
        if hist.size > 0:
            max_density = max(max_density, np.max(hist))
            
    max_limit = max_density * 1.15
    if max_limit == 0:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
        return

    # Second pass for plotting
    for j, current_label in enumerate(present_labels):
        angles_data = distributions[current_label]
        hist, bin_edges = np.histogram(angles_data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        original_index = labels.index(current_label)

        # Plot the density as a line
        ax.plot(
            bin_centers, hist,
            color=colors[original_index],
            label=current_label,
            linewidth=2.0,
            alpha=0.85
        )

    # --- Add reference lines and shading for motion regimes ---
    # Shading for Directed (0-60), Random (60-120), Confined (120-180)
    # ax.axvspan(0, 60, color='green', alpha=0.05, zorder=0)
    # ax.axvspan(60, 120, color='white', alpha=0.05, zorder=0)
    # ax.axvspan(120, 180, color='red', alpha=0.05, zorder=0)

    # Core reference lines
    ax.axvline(90, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # Formatting
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 0.011)
    # Axis formatting
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width = 1.5)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel(r'$\Delta\Theta^\circ$', fontsize=14)
    ax.set_ylabel(r'Probability Density', fontsize=14)
    ax.set_title(lag_label, fontsize=14)
    ax.grid(False)
    ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    # Add legend only to the first panel (A)
    if panel_label == 'A':
        ax.legend(loc='upper left', frameon=True, fontsize=14)

def plot_ratio_panel(ax, ratio_data, labels, colors, panel_label):    
    def _get_sig_label(p):        
        if p < 0.0001:            
            return "****";        
        elif p < 0.001:            
            return "***";        
        elif p < 0.01:            
            return "**";        
        elif p < 0.05:            
            return "*";        
        else:            
            return "ns";
    # Retrieve lag-labels (keys are used for dict access)
    lag_keys = list(analysis_lags.keys())
    
    # Initialize dictionaries for ratios and p-values
    ratios = {lag: [] for lag in lag_keys}
    pvals = {lag: [] for lag in lag_keys}
    
    for lag in lag_keys:
        for lab in labels:
            entry = ratio_data[lab].get(lag, {})
            ratios[lag].append(entry.get('D/C Ratio', np.nan))
            pvals[lag].append(entry.get('p_value', np.nan))
    
    # Plotting logic (using indices 0, 2, 5 for colors as defined in the global 'colors' list)
    x = np.arange(len(labels))
    width = 0.2
    rects1 = ax.bar(x - width, ratios[lag_keys[0]], width, label=lag_keys[0], color='black', alpha=0.8)
    rects2 = ax.bar(x, ratios[lag_keys[1]], width, label=lag_keys[1], color='dimgray', alpha=0.8)
    rects3 = ax.bar(x + width, ratios[lag_keys[2]], width, label=lag_keys[2], color='lightgray', alpha=0.8)
    
    # Annotation logic: Add significance markers above each bar
    def _annotate(rects, plist):
        for r, p in zip(rects, plist):
            height = r.get_height()
            if np.isnan(p) or height is None or height == 0:
                continue
            
            sig_label = _get_sig_label(p)
            
            ax.text(r.get_x() + r.get_width()/2, height * 1.02, sig_label, 
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
    
    _annotate(rects1, pvals[lag_keys[0]])
    _annotate(rects2, pvals[lag_keys[1]])
    _annotate(rects3, pvals[lag_keys[2]])
    
    # Formatting (unchanged)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.set_ylabel(r'$\Delta\Theta (< 90^\circ$) / $\Delta\Theta (> 90^\circ$)', fontsize=14)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(which='both', axis='both', direction='in', labelsize=12, length=6, width = 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax.grid(False)



# ============================================================================
# MAIN EXECUTION
# ============================================================================

LAG_TIME_STEPS_SHORT = 20 # 1.0 s
LAG_TIME_STEPS_MID = 50 # 5.0 s
LAG_TIME_STEPS_LONG = 80 # 8.0 s

# Define lag times globally for access by plot_ratio_panel
analysis_lags = {
    f'$\Delta t$={LAG_TIME_STEPS_SHORT/FPS:.1f} s': LAG_TIME_STEPS_SHORT,
    f'$\Delta t$={LAG_TIME_STEPS_MID/FPS:.1f} s': LAG_TIME_STEPS_MID,
    f'$\Delta t$={LAG_TIME_STEPS_LONG/FPS:.1f} s': LAG_TIME_STEPS_LONG,
}


def run_step_4_orientation_analysis(directories, condition_keys, max_traj_lengths, mpp, fps, colors, labels):
    """Main function with linear subplots and ratio calculation."""
    print("\n" + "="*80)
    print("STEP 4: NEXT STEP ANGLE BIAS ANALYSIS")
    print("="*80)

    # Use keys for map creation
    lag_keys = list(analysis_lags.keys())
    
    all_angle_distributions = {lag_label: {} for lag_label in lag_keys}
    ratio_analysis_data = {label: {lag: {} for lag in lag_keys} for label in labels}
    
    processed_keys = [k for k in condition_keys if k in directories and os.path.exists(directories[k])]

    if not processed_keys:
        print("\nERROR: No valid directories found for analysis.")
        return

    # --- 1. Process Data & Calculate Ratios ---
    bins = np.linspace(0, 180, 19) # 10-degree bins

    for i, label_key in enumerate(processed_keys):
        directory = directories[label_key]
        idx = condition_keys.index(label_key)
        max_length = max_traj_lengths[idx]
        current_label = labels[idx]

        print(f"-> Analyzing {current_label} (Max Length: {max_length})...")
        
        for lag_label in lag_keys:
            lag_steps = analysis_lags[lag_label]
            angles_array = process_directional_directory(
                directory, mpp, max_length, lag_steps, label_key
            )
            
            if angles_array.size > 0:
                all_angle_distributions[lag_label][current_label] = angles_array
                
                # --- Calculate Ratios and Statistics ---
                N_total = len(angles_array)
                N_directed = np.sum(angles_array < 90)
                N_confined = np.sum(angles_array > 90)
                
                # Chi2 and CI calculation (for console output)
                chi2_stat, p_value = chi_square_test_against_uniform(angles_array, bins)
                directed_frac, ci_low, ci_high = calculate_directed_fraction(angles_array)
                
                # Store Ratio data
                if N_confined > 0:
                    d_c_ratio = N_directed / N_confined
                else:
                    d_c_ratio = N_directed / 1.0 # Avoid division by zero, treat as highly directed
                    
                ratio_analysis_data[current_label][lag_label] = {
                    'N_total': N_total,
                    'N_directed': N_directed,
                    'N_confined': N_confined,
                    'D/C Ratio': d_c_ratio,
                    'Directed Frac': directed_frac,
                    'p_value': p_value
                }

                print(f" - {lag_label}: N={N_total}. D/C Ratio: {d_c_ratio:.2f}. P(Directed): {directed_frac*100:.1f}%. P-value (Chi2): {p_value:.4f}")

            else:
                ratio_analysis_data[current_label][lag_label] = {
                    'N_total': 0, 'N_directed': 0, 'N_confined': 0, 'D/C Ratio': np.nan, 'Directed Frac': np.nan
                }
                print(f" - {lag_label}: No valid angle pairs collected.")

    # ========================================================================
    # 2. GENERATE COMBINED FIGURE (1x3 Linear Plots + 1 Ratio Plot)
    # ========================================================================
    
    # Setup 2x2 grid: Top row is 1x3 linear plots, Bottom row is 1x1 ratio plot
    fig = plt.figure(figsize=(15, 12)) 
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.2)
    
    # Axes for the three linear plots (Panels A, B, C)
    ax_linear_A = fig.add_subplot(gs[0, 0])
    ax_linear_B = fig.add_subplot(gs[0, 1])
    ax_linear_C = fig.add_subplot(gs[0, 2])
    
    linear_axes = {
        lag_keys[0]: (ax_linear_A, 'A'),
        lag_keys[1]: (ax_linear_B, 'B'),
        lag_keys[2]: (ax_linear_C, 'C')
    }
    
    # Axis for the combined ratio plot (Panel D - spans all 3 columns on the second row)
    ax_ratio_D = fig.add_subplot(gs[1, 0:]) 

    # --- Plot Linear Subplots (A, B, C) ---
    for i, (lag_label, (ax, panel_label)) in enumerate(linear_axes.items()):
        distributions_at_lag = all_angle_distributions[lag_label]
        plot_linear_nsab_panel(ax, lag_label, distributions_at_lag, labels, colors, bins, panel_label)
        
        # Ensure only the left-most plot has the y-label
        if panel_label != 'A':
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
 

    # --- Plot Ratio Plot (D) ---
    plot_ratio_panel(ax_ratio_D, ratio_analysis_data, labels, colors, panel_label='D')


    plt.tight_layout()
    
    # Save figure
    filename_out = f"F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Figures/AngularBias.png"
    fig.savefig(filename_out, dpi=600, bbox_inches='tight')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE: Combined Figure Generated")
    print(f"Figure saved to: {filename_out}")
    print("="*80)
    
    plt.show()

# Run the analysis
if __name__ == '__main__':
    run_step_4_orientation_analysis(directories, CONDITION_KEYS, MAX_TRAJ_LENGTHS, MPP, FPS, colors, labels)
