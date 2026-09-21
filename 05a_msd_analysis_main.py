# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:57:55 2025
@author: sophiedeboer
"""
"""
MSD Analysis with Per-Particle Localization Correction
Applies individual sigma corrections based on particle brightness
@author: sophiedeboer (Refactored by Anara)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Shared MSD core (msd, sigma correction, powerfit, imsd, process_directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from msd_core import (
    process_directory,
    check_normality, check_equal_variance, log10_safe,
    run_welch_games_howell, extract_msd_at_lagtime, run_trend_test,
    classify_diffusion_regime, threshold_sensitivity_table, plot_threshold_sensitivity,
    find_stable_threshold_range, find_all_stable_plateaus,
    plot_dapp_alpha_diagnostic,
    msd_lagtime_sensitivity_table, plot_msd_lagtime_sensitivity,
)

NOISE_FLOOR_FRACTION = 0.98  # fraction of lag times a trajectory must be
                              # ABOVE the bead noise floor to be kept
# ============================================================================
# CONFIGURATION - UPDATE THESE FROM STUCK BEAD ANALYSIS
# ============================================================================
# --- Data Configuration ---
# Set the root directory for your experimental data.
BASE_DATA_ROOT = Path('D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data')
# Define the full set of experiments, including paths and parameters
CONDITIONS_MAP = [ { 'key': '100bp', 'size': '100 bp', 'base_folder': 'Aswin Data/Aswin Data CHO Corrected for Drift', 'collected_dir': 'Collected_100bp', 'max_length': 350 }, 
                  { 'key': '250bp', 'size': '250 bp', 'base_folder': 'Aswin Data/Aswin Data CHO Corrected for Drift', 'collected_dir': 'Collected_250bp', 'max_length': 300 }, 
                  { 'key': '500bp', 'size': '500 bp', 'base_folder': 'Aswin Data/Aswin Data CHO Corrected for Drift', 'collected_dir': 'Collected_500bp', 'max_length': 350 }, 
                  { 'key': '4731bp', 'size': '4731 bp', 'base_folder': 'Sophie Corrected for Gaussian Filter', 'collected_dir': 'Collected_pEGFP-C1', 'max_length': 800 }, 
                  { 'key': '7614bp', 'size': '7614 bp', 'base_folder': 'Sophie Corrected for Gaussian Filter', 'collected_dir': 'Collected_BPK4410', 'max_length': 800 }, 
                  { 'key': '9553bp', 'size': '9553 bp', 'base_folder': 'Sophie Corrected for Gaussian Filter', 'collected_dir': 'Collected_hCas9D10A', 'max_length': 800 }, 
                  { 'key': 'beads', 'size': 'beads', 'base_folder': 'Control Experiments_Beads/0.22 um', 'collected_dir': 'Collected_0.22um', 'max_length': 800 }
]
# Create auxiliary lists as requested by the user's iteration logic
CONDITION_KEYS = [c['key'] for c in CONDITIONS_MAP]
MAX_TRAJ_LENGTHS = [c['max_length'] for c in CONDITIONS_MAP]
CONDITION_LABELS = [c['size'] for c in CONDITIONS_MAP]
# Map keys to their condition dictionaries for easy lookup
CONDITION_DICT = {c['key']: c for c in CONDITIONS_MAP}

# --- Reference Values (From Step 1 Stuck Bead Analysis) ---
SIGMA_STUCK_MICRONS = 0.021244 # um # Replace with median sigma_combined from Step 1
MASS_STUCK_REFERENCE = 3381.5 # Replace with median mass from Step 1
SIGMA_DIM_MICRONS = 0.014008 #--> this was calculated previously using 04b python code
MASS_DIM_REFERENCE = 589.8 #median mass from YOYO-1 reference
SIGMA_CY3_MICRONS = 0.013078 # microns; beads-derived Cy3 sigma from 04b mass-scaling validation
MASS_CY3_REFERENCE = 8922.3 # median mass from Cy3 (Bright Ref) dataset; reference point for per-particle mass scaling

# --- Experimental & Calculation Parameters ---
MPP = 0.16 # Microns per pixel
FPS = 10 # Frames per second
MAX_LAGTIME = 100 # Maximum lagtime for MSD calculation
# --- Correction & Filtering Options ---
USE_LOCALIZATION_CORRECTION = True # Enable/disable localization correction
USE_MASS_SCALING = True # Scale sigma based on particle brightness
R_SQUARED_THRESHOLD = 0.8 # Minimum R² for trajectory acceptance
# --- Plotting Parameters (Using requested colors and labels) ---

# --- Truncation Parameters ---
ENABLE_TRUNCATION = False  # Master switch for truncation
ENABLE_MASS_DROP = False  # Include mass drop detection
KEEP_FIRST_SEGMENT = False  # If True: keep segment before break; If False: reject entire particle
MAX_STEP_SIZE = 1  # Maximum allowed spatial jump (µm)
MAX_SINGLE_STEP_MASS_DROP = 0.5  # Maximum allowed fractional mass drop (30%)
MIN_SEGMENT_LENGTH = 30  # Minimum frames required after truncation


# Colors used for the 7 experimental conditions (6 DNA fragment sizes + beads)
PLOT_COLORS = ['#6FADE3','#2874C6','#042C53','#FAC775','#BA7517','#412402','#444441']

LABELS = ['100 bp', '250 bp', '500 bp', '4731 bp', '7614 bp', '9553 bp', 'beads']
OUTPUT_FIGURE_BASE = BASE_DATA_ROOT / 'Figures'

# Numeric DNA fragment size (bp) per condition label, for the trend test
# (run_trend_test) -- treats fragment size as an ordered/continuous
# covariate rather than six unrelated categories. 'beads' has no
# meaningful size, so it's simply absent from this map; run_trend_test
# excludes it via its own exclude_substring filter regardless.
BP_SIZE_MAP = {'100 bp': 100, '250 bp': 250, '500 bp': 500,
                '4731 bp': 4731, '7614 bp': 7614, '9553 bp': 9553}
# ============================================================================
# CORE MSD/FITTING FUNCTIONS + TRAJECTORY TRUNCATION
# Now imported from msd_core.py (single source of truth across all scripts) --
# see the import block at the top of this file. Truncation is opted into via
# TRUNCATION_CFG (built below from this script's ENABLE_TRUNCATION etc. settings)
# and passed to process_directory().
# ============================================================================
TRUNCATION_CFG = {
    'enabled': ENABLE_TRUNCATION,
    'enable_mass_drop': ENABLE_MASS_DROP,
    'keep_first_segment': KEEP_FIRST_SEGMENT,
    'max_step_size': MAX_STEP_SIZE,
    'max_mass_drop': MAX_SINGLE_STEP_MASS_DROP,
    'min_segment_length': MIN_SEGMENT_LENGTH,
}



# ============================================================================
# PLOTTING HELPER FUNCTIONS (Including statistical bar addition)
# ============================================================================

def add_sig_bars_alpha(ax, data_list, actual_labels, alpha_thresh=0.05):
    """
    Plot significance bars (asterisks) on a box‑plot axis.
    Bars are drawn **only** for pair‑wise comparisons whose Games-Howell
    adjusted p‑value is smaller than ``alpha_thresh`` (default 0.05).
    Uses Welch's ANOVA + Games-Howell (msd_core.run_welch_games_howell)
    rather than Tukey HSD, since Levene's test shows unequal variance
    across these conditions -- Tukey's equal-variance assumption doesn't
    hold here, but Games-Howell doesn't require it.
    """
    # ------------------------------------------------------------------
    # 1. Determine the maximum y‑value among the data for scaling.
    # ------------------------------------------------------------------
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())

    # Base height for the first bar (slightly above the data)
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) + 0.1
    spacing = 0.2          # vertical spacing between stacked bars

    # ------------------------------------------------------------------
    # 2. Welch's ANOVA + Games-Howell post-hoc – keep only significant pairs.
    # ------------------------------------------------------------------
    gh = run_welch_games_howell(data_list, actual_labels, exclude_substring='beads', verbose=False)
    p_vals = [(p, i, j) for (i, j), p in gh['pairwise_pvalues'].items() if p < alpha_thresh]

    # Sort by significance (smallest p first) so the most significant bar
    # appears at the bottom.
    p_vals.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # 3. Plot the retained bars.
    # ------------------------------------------------------------------
    for k, (p, i, j) in enumerate(p_vals):
        # Choose the asterisk string according to the p‑value.
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        else:               # 0.01 ≤ p < 0.05
            sig = "*"

        h = base_h + k * spacing
        # Horizontal line connecting the two groups
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False)
        # Text label centred above the line
        ax.text((i + 1 + j + 1)  # centre of the line
                / 2, h + spacing * 0.02,
                sig, ha="center", va="bottom",
                fontsize=10, color="k")

    # ------------------------------------------------------------------
    # 4. Expand the y‑limit only if at least one bar was drawn.
    # ------------------------------------------------------------------
    if p_vals:
        final_h = base_h + len(p_vals) * spacing
        ax.set_ylim(cur_ylim[0], final_h + spacing)

# Define significance bar function for Dapp (log scale)
def add_sig_bars_dapp(ax, data_list, actual_labels, alpha_thresh=0.05):
    """
    Plot significance bars (asterisks) on a violin plot axis with log scale.
    Bars are drawn **only** for pair-wise comparisons whose Games-Howell
    adjusted p-value is smaller than ``alpha_thresh`` (default 0.05).

    IMPORTANT: this function assumes the axis is ALREADY in log scale
    (ax.set_yscale('log') must be called BEFORE this function, not after)
    -- it reads the current y-limits to decide where to start stacking
    bars, and calling it on a still-linear axis produces a badly wrong
    (usually wildly inflated) final range once set_yscale('log') is
    applied afterward.

    Bar heights are stacked with a small, FIXED step in log10 space
    (log_step, default 0.12 decades per bar) rather than by repeatedly
    multiplying by a constant factor. The multiplicative version grows
    exponentially with the number of significant pairs (e.g. 2.3^9 =
    ~1800x for 9 pairs), which can blow up the axis range far beyond the
    data for comparisons with many conditions/pairs; a fixed log-space
    step grows only linearly, so it stays controlled regardless of how
    many pairs are significant.
    """
    # 1. Determine the maximum y-value among the data for scaling.
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())
    
    # Base height for the first bar (slightly above the data), in log10 space.
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) * 1.5
    log_base_h = np.log10(base_h) if base_h > 0 else 0.0
    log_step = 0.2  # decades of vertical space per stacked bar
    
    # 2. Welch's ANOVA + Games-Howell post-hoc - keep only significant pairs.
    # Dapp is strictly positive and typically log-normal, so the
    # statistical test itself runs on log10(Dapp) (matches the console
    # diagnostics), even though the bars are still positioned using the
    # raw data's scale, since that's what's actually plotted here.
    # Games-Howell since Levene's shows unequal variance.
    log_data_list = log10_safe(data_list, verbose=False)
    gh = run_welch_games_howell(log_data_list, actual_labels, exclude_substring='beads', verbose=False)
    p_vals = [(p, i, j) for (i, j), p in gh['pairwise_pvalues'].items() if p < alpha_thresh]
    
    # Sort by significance (ordered by "size distance" so 100 bp v 9553 bp is at the top of the plot)
    p_vals.sort(key=lambda x: x[2] - x[1]) 
    
    # 3. Plot the retained bars.
    for k, (p, i, j) in enumerate(p_vals):
        # Choose the asterisk string according to the p-value.
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        else:
            sig = "*"
        
        h = 10 ** (log_base_h + k * log_step)
        
        # Horizontal line connecting the two groups
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False)
        # Text label centred above the line
        ax.text((i + 1 + j + 1) / 2, h * 10 ** (log_step * 0.02),
                sig, ha="center", va="bottom",
                fontsize=10, color="k")
    
    # 4. Expand the y-limit only if at least one bar was drawn.
    if p_vals:
        final_h = 10 ** (log_base_h + (len(p_vals) - 1) * log_step)
        ax.set_ylim(cur_ylim[0], final_h * 10 ** (log_step * 1.5))

def plot_alpha_boxplot(ax, results_dict, processed_keys, condition_keys, labels_map, plot_colors, jitter=0.05, panel_label=None, add_sig_bars=True): 
    """Plots alpha box plot for a specific subset of keys and adds significance bars."""

    # 1. Gather data and filter labels/colors
    data_list = []
    actual_keys_subset = []
    
    for k in processed_keys:
        if k in results_dict:
            data_list.append(results_dict[k]['alpha2'])
            actual_keys_subset.append(k)

    if not data_list:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center')
        ax.set_ylim(-0.08, 2.5)
        return 

    actual_labels_subset = [labels_map[condition_keys.index(k)] for k in actual_keys_subset]
    x_ticks = list(range(1, len(data_list) + 1))

    # 2. Plot boxplot
    bp = ax.boxplot(
        data_list,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(linewidth=1.5)
    )

    # 3. Plot scatter points and color boxes
    for i, data_arr in enumerate(data_list):
        original_key = actual_keys_subset[i]
        original_index = condition_keys.index(original_key)
        
        # Scatter data points for visualization
        ax.scatter(
            np.full_like(data_arr, x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(data_arr)),
            data_arr,
            color=plot_colors[original_index],
            alpha=1.0,
            marker='o',
            edgecolor='k',
            zorder=10
        )
        
        # Set box colors
        bp['boxes'][i].set_facecolor(plot_colors[original_index])
        bp['boxes'][i].set_alpha(0.6)

    # 4. Add Significance Bars
    if add_sig_bars:
        add_sig_bars_alpha(ax, data_list) 

    # 5. Reference lines
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'Normal diffusion ($\alpha=1$)', zorder=0)
    ax.axhline(0.4, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7, label=r'Threshold for confinement ($\alpha<0.4$)', zorder=0)

    # 6. Formatting (Initial Y-limit is temporarily set here, allowing sig bars to stretch it later)
    ax.set_ylim(-0.08, 2.5) # Set base limit
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(actual_labels_subset, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=14)
    ax.grid(False)
    ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    if panel_label:
        ax.text(-0.05, 1.15, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')


def plot_trend_scatter_panel(ax, data_list, x_values, colors, trend_result,
                              y_log, y_label, panel_label, rng, labels = None):
    """
    Pooled per-particle scatter of a response variable vs. DNA fragment
    size (log x-axis), colored by condition, with the fitted regression
    line (from msd_core.run_trend_test) overlaid, plus the regression
    p-value and R^2 annotated on the panel.

    trend_result : the dict returned by msd_core.run_trend_test(). If the
    response was log10-transformed before being passed to run_trend_test
    (as done for Dapp/MSD, which are log-normal), pass y_log=True and
    data_list should be the RAW (untransformed) values -- the regression
    line is reconstructed in raw units for display on a log-scale axis,
    consistent with how Dapp is already plotted elsewhere in this script
    (raw data + log-scale axis, not pre-transformed values on a linear one).
    """
    for i, (data, x, color) in enumerate(zip(data_list, x_values, colors)):
        if not np.isfinite(x):
            continue
        finite = np.asarray(data, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            continue
        jitter = x * 10 ** rng.uniform(-0.04, 0.04, size=len(finite))
        lbl = labels[i] if labels is not None else None
        ax.scatter(jitter, finite, color=color, alpha=0.6, s=8,
                   edgecolor='none', zorder=2, label=lbl)

    finite_x = [x for x in x_values if np.isfinite(x)]
    if finite_x:
        x_line = np.logspace(np.log10(min(finite_x)), np.log10(max(finite_x)), 100)
        slope = trend_result['regression']['slope']
        intercept = trend_result['regression']['intercept']
        if np.isfinite(slope):
            if y_log:
                y_line = 10 ** (slope * np.log10(x_line) + intercept)
            else:
                y_line = slope * np.log10(x_line) + intercept
            ax.plot(x_line, y_line, color='black', linewidth=2, linestyle='--', zorder=3)

    ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    ax.set_xlabel('DNA fragment size (bp)', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    # legend removed here -- placed once at the figure level by the caller
    ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(False)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left')


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_analysis():
    """
    Figure roadmap (execution order below is intentionally NOT changed by
    figure "type" -- several plots draw jitter from the shared, unseeded
    global np.random stream via np.random.uniform(), so reordering blocks
    would shift which random draws land on which plot, i.e. change dot
    positions in later figures even though nothing about the underlying
    data/statistics would differ. Each figure below is labeled MAIN or
    SUPPLEMENTARY in place instead, so the file can still be navigated
    figure-by-figure (search for "MAIN FIGURE" / "SUPPLEMENTARY FIGURE").

      Figure_Panel_MSD_Diagnostic.png        SUPPLEMENTARY (raw vs corrected MSD, QC)
      Figure_SizeTrend_Alpha_vs_MSD8s.png    SUPPLEMENTARY
      Figure_3.png                           MAIN
      Figure_4.png                           MAIN
      Figure_Supp_ConfinementThresholdSensitivity.png  SUPPLEMENTARY
      Figure_Supp_DappAlphaCoupling.png      SUPPLEMENTARY
      Figure_5.png                           MAIN
    """
    print("\n" + "="*80)
    print("MSD ANALYSIS WITH PER-PARTICLE LOCALIZATION CORRECTION")
    print("="*80)
    print(f"Sigma (beads, fixed): {SIGMA_STUCK_MICRONS:.6f} μm")
    print(f"Sigma (YOYO-1 dim, fixed, 100-500 bp): {SIGMA_DIM_MICRONS:.6f} μm")
    print(f"Sigma (Cy3, mass-scaled reference, 4731-9553 bp): {SIGMA_CY3_MICRONS:.6f} μm (ref mass = {MASS_CY3_REFERENCE:.1f})")
    print(f"Mass reference (beads): {MASS_STUCK_REFERENCE:.1f}")
    print(f"Mass scaling: {'ENABLED' if USE_MASS_SCALING else 'DISABLED'}")
    print(f"Localization correction: {'ENABLED' if USE_LOCALIZATION_CORRECTION else 'DISABLED'}")
    print(f"R² threshold: {R_SQUARED_THRESHOLD}\n")

    results_dict = {}
    directories = {}

    # 1. Calculate all directory paths based on the map
    for condition in CONDITIONS_MAP:
        key = condition['key']
        path = BASE_DATA_ROOT / condition['base_folder'] / 'E_output_data' / condition['collected_dir'] / 'tm'
        directories[key] = path
        
    # ============================================================================
    # PROCESS BEADS FIRST TO ESTABLISH NOISE FLOOR
    # ============================================================================
    noise_floor_msd = None
    if 'beads' in directories:
        bead_directory = directories['beads']
        if bead_directory.is_dir():
            print("Processing beads (noise floor)...")
            # Process beads WITHOUT noise floor filtering (pass None)
            bead_result = process_directory(
                bead_directory, MPP, FPS, MAX_LAGTIME, 0.0, MASS_STUCK_REFERENCE, 
                MAX_TRAJ_LENGTHS[CONDITION_KEYS.index('beads')], False, 
                use_localization_correction=USE_LOCALIZATION_CORRECTION,
                r_squared_threshold=R_SQUARED_THRESHOLD,
                noise_floor_fraction=NOISE_FLOOR_FRACTION,
                truncation_cfg=TRUNCATION_CFG,
                noise_floor_msd=None  # No filtering for beads themselves
            )
            if bead_result is not None:
                results_dict['beads'] = bead_result
                noise_floor_msd = bead_result['ensemble_msd_corrected']
                print(f" Beads processed: {bead_result['n_tracks']} tracks")
                print(f" Noise floor established (mean MSD: {noise_floor_msd.mean():.6e} μm²)")
            else:
                print(" WARNING: No valid bead trajectories found")
        else:
            print(f"WARNING: Bead directory not found at {bead_directory}")
    # ============================================================================

    # 2. Iterate and Process Data (NOW WITH NOISE FLOOR FILTERING)
    for i, label_key in enumerate(CONDITION_KEYS):
        if label_key == 'beads':
            continue  # Skip beads since we already processed them
            
        if label_key in directories:
            directory = directories[label_key]
            max_length = MAX_TRAJ_LENGTHS[i]
            size_label = CONDITION_DICT[label_key]['size']

            # Conditional Sigma Assignment and Scaling Flag
            if label_key in ['100bp', '250bp', '500bp']:
                sigma_to_use = SIGMA_DIM_MICRONS
                mass_to_use = MASS_DIM_REFERENCE
                use_scaling_temp = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION  # now mass-scaled, matching Cy3 treatment
                print(f"Processing {size_label} ({label_key})... using Fixed Sigma ({sigma_to_use:.4f} μm) & MASS SCALING (Dim Dye reference).")
            else:
                sigma_to_use = SIGMA_CY3_MICRONS
                mass_to_use = MASS_CY3_REFERENCE
                use_scaling_temp = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION
                print(f"Processing {size_label} ({label_key})... using Cy3 Sigma ({sigma_to_use:.4f} μm) & MASS SCALING (Cy3 reference).")

            if not directory.is_dir():
                print(f"CRITICAL: Directory for {size_label} not found at {directory}")
                continue

            # ============================================================================
            # MODIFIED: Pass noise_floor_msd to enable filtering
            # ============================================================================
            result = process_directory(
                directory, MPP, FPS, MAX_LAGTIME, sigma_to_use, mass_to_use, 
                max_length, use_scaling_temp, 
                use_localization_correction=USE_LOCALIZATION_CORRECTION,
                r_squared_threshold=R_SQUARED_THRESHOLD,
                noise_floor_fraction=NOISE_FLOOR_FRACTION,
                truncation_cfg=TRUNCATION_CFG,
                noise_floor_msd=noise_floor_msd  # Pass noise floor for filtering
            )
            # ============================================================================
            
            if result is not None:
                results_dict[label_key] = result
                print(f" Tracks accepted: {result['n_tracks']}")
                print(f" Mean mass: {np.mean(result['mass']):.1f}")
                print(f" Mean sigma: {np.mean(result['sigma']):.6f} μm")
            else:
                print(f" WARNING: No valid trajectories for {size_label}")


    # ============================================================================
    # PLOTTING PREPARATION
    # ============================================================================
    processed_keys = [k for k in CONDITION_KEYS if k in results_dict]
    actual_labels = [LABELS[CONDITION_KEYS.index(k)] for k in processed_keys]

    if not results_dict:
        print("\nERROR: results_dict is empty. No data passed filters or directories were invalid.")
        return
    
    # ============================================================================
    # SUPPLEMENTARY FIGURE: 1x2 Panel (Raw MSD, Corrected MSD) -- QC diagnostic
    # ============================================================================
    print("\nSaving 1x2 Panel Figure (Raw MSD, Corrected MSD)...")
    fig_panel, axes_panel = plt.subplots(1, 2, figsize=(18, 6)) #plt.subplots(1, 3, figsize=(12, 3.375))
    ax_raw_panel, ax_corr_panel = axes_panel
    markers = ['o', 'D', 's', '^', 'v', 'P', '*']
    # --- Panel A: Raw MSD ---
    for i, key in enumerate(processed_keys):
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_raw']
        color = PLOT_COLORS[original_index]
        ax_raw_panel.plot(ensemble_msd.index, ensemble_msd.values, 
                   marker=markers[i], 
                   markerfacecolor= color, color = color,
                   linestyle='-', linewidth=2, markersize=6,
                   label=LABELS[original_index])
    
    ax_raw_panel.set_xscale('log')
    ax_raw_panel.set_yscale('log')
    ax_raw_panel.set_xlabel(r'$\Delta t$ (s)', fontsize=16)
    ax_raw_panel.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\mathrm{raw}}$ ($\mu$m$^2$)', fontsize=16)
    ax_raw_panel.set_xlim(0.1, 100)
    ax_raw_panel.set_ylim(1e-4, 1)
    ax_raw_panel.legend(loc='upper left', bbox_to_anchor=(0.05, 0.98),
                     frameon=False, fontsize=10)
    ax_raw_panel.grid(False)
    ax_raw_panel.tick_params(which="both", axis="both", direction="in", labelsize=14, length=6, width=1.5)
    for spine in ax_raw_panel.spines.values():
        spine.set_linewidth(1.5)
    ax_raw_panel.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_raw_panel.axvline(10.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    # shaded gray regions
    ax_raw_panel.axvspan(ax_raw_panel.get_xlim()[0], 1.0, color='lightgray', alpha=0.3, zorder=0)
    ax_raw_panel.axvspan(10.0, ax_raw_panel.get_xlim()[1], color='lightgray', alpha=0.3, zorder=0)
    
    ax_raw_panel.text(-0.15, 1.05, 'A', transform=ax_raw_panel.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    
    # --- Panel B: Corrected MSD ---
    for i, key in enumerate(processed_keys):
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_corrected']
        color = PLOT_COLORS[original_index]
        ax_corr_panel.plot(ensemble_msd.index, ensemble_msd.values,
                    marker=markers[i],
                    markerfacecolor=color, color = color,
                    linestyle='-', linewidth=2, markersize=6,
                    label=LABELS[original_index])
    
    ax_corr_panel.set_xscale('log')
    ax_corr_panel.set_yscale('log')
    ax_corr_panel.set_xlabel(r'$\Delta t$ (s)', fontsize=16)
    ax_corr_panel.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\mathrm{real}}$ ($\mu$m$^2$)', fontsize=16)
    ax_corr_panel.set_xlim(0.1, 100)
    ax_corr_panel.set_ylim(1e-4, 1)
    ax_corr_panel.legend(loc='upper left', bbox_to_anchor=(0.05, 0.98),
                     frameon=False, fontsize=10)
    ax_corr_panel.grid(False)
    ax_corr_panel.tick_params(which="both", axis="both", direction="in", labelsize=14, length=6, width=1.5)
    for spine in ax_corr_panel.spines.values():
        spine.set_linewidth(1.5)
    ax_corr_panel.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_corr_panel.axvline(10.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    # shaded gray regions
    ax_corr_panel.axvspan(ax_corr_panel.get_xlim()[0], 1.0, color='lightgray', alpha=0.3, zorder=0)
    ax_corr_panel.axvspan(10.0, ax_corr_panel.get_xlim()[1], color='lightgray', alpha=0.3, zorder=0)
    ax_corr_panel.text(-0.15, 1.05, 'B', transform=ax_corr_panel.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig_panel.savefig(OUTPUT_FIGURE_BASE / "Figure_Panel_MSD_Diagnostic.png", dpi=600, bbox_inches='tight')
    print(f"Saved: Figure_Panel_MSD_Diagnostic.png")
    

    # Box plot data lists (uses processed_keys list structure for indexing)
    Dapp2_data = [results_dict[k]['Dapp2'] for k in processed_keys] # Long time (2-8s) Dapp
    alpha2_data = [results_dict[k]['alpha2'] for k in processed_keys] # Alpha exponent (2-8s)

    np.random.seed(42)
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    
    
    
    
    
    # ============================================================================
    # STATISTICAL ANALYSIS FOR ALPHA VALUES
    # Welch's ANOVA + Games-Howell post-hoc (family-wise error rate
    # corrected, does NOT assume equal variance), via
    # msd_core.run_welch_games_howell(). Also reports Shapiro-Wilk
    # (normality) and Levene's (equal variance) as diagnostics -- Levene's
    # fails for this dataset (variances differ significantly across DNA
    # sizes), which is why Welch's/Games-Howell is used here instead of
    # standard ANOVA/Tukey HSD.
    # ============================================================================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - ALPHA VALUES (2-8s)")
    print("="*80)

    # x-values (bp size) for the trend test below, aligned to actual_labels/
    # data order. Computed once here, reused for Dapp and MSD@8s sections.
    bp_x_values = [BP_SIZE_MAP.get(label, np.nan) for label in actual_labels]

    check_normality(alpha2_data, actual_labels)
    check_equal_variance(alpha2_data, actual_labels)
    anova_tukey_result = run_welch_games_howell(alpha2_data, actual_labels)

    # Symmetric p-value matrix (Games-Howell-adjusted)
    n_conditions = len(alpha2_data)
    pairwise_pvalues = np.full((n_conditions, n_conditions), np.nan)
    for (i, j), p_adj in anova_tukey_result['pairwise_pvalues'].items():
        pairwise_pvalues[i, j] = p_adj
        pairwise_pvalues[j, i] = p_adj

    # Trend test: does alpha show a systematic trend with DNA fragment
    # size, rather than inferring size-(in)dependence from the pattern of
    # pairwise significance?
    run_trend_test(alpha2_data, bp_x_values, actual_labels)

    print("="*80)

    # ============================================================================
    # STATISTICAL ANALYSIS FOR Dapp VALUES (2-8s)
    # Dapp is strictly positive and typically log-normally distributed
    # (confirmed here: raw Dapp fails Shapiro-Wilk badly in every group,
    # e.g. p ~ 1e-21). Running the diagnostics on log10(Dapp) instead of
    # raw Dapp is standard practice for this kind of quantity and
    # generally satisfies the normality assumption. Even after the log
    # transform, Levene's test still shows unequal variance across DNA
    # sizes, so Welch's ANOVA + Games-Howell is used here rather than
    # standard ANOVA/Tukey HSD.
    # ============================================================================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - Dapp VALUES (2-8s), log10-transformed")
    print("="*80)

    log_dapp2_data = log10_safe(Dapp2_data)
    check_normality(log_dapp2_data, actual_labels)
    check_equal_variance(log_dapp2_data, actual_labels)
    run_welch_games_howell(log_dapp2_data, actual_labels)
    run_trend_test(log_dapp2_data, bp_x_values, actual_labels)
    print("="*80)

    # ============================================================================
    # STATISTICAL ANALYSIS FOR MSD AT FIXED LAG TIME (8s)
    # Dapp's units (um^2/s^alpha) depend on the fitted alpha, so it isn't
    # strictly comparable across particles/conditions whose alpha differs.
    # MSD measured directly at a fixed lag time has units of um^2 for
    # every particle regardless of alpha, avoiding that issue -- extracted
    # straight from the corrected MSD curves (msd_corrected), not refit.
    # Also log10-transformed, since it's derived from the same amplitude-
    # like quantity as Dapp and is expected to be similarly right-skewed.
    # ============================================================================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - MSD AT 8s (directly measured, not fit-derived)")
    print("="*80)
    trend_colors = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in processed_keys]

    msd_at_8s_data = [extract_msd_at_lagtime(results_dict[k]['msd_corrected'], 8.0, FPS)
                       for k in processed_keys]
    log_msd_at_8s_data = log10_safe(msd_at_8s_data)
    check_normality(log_msd_at_8s_data, actual_labels)
    check_equal_variance(log_msd_at_8s_data, actual_labels)
    run_welch_games_howell(log_msd_at_8s_data, actual_labels)
    msd_trend_result = run_trend_test(log_msd_at_8s_data, bp_x_values, actual_labels)
    print("="*80)

    # ============================================================================
    # SUPPLEMENTARY FIGURE (Figure_SizeTrend_Alpha_vs_MSD8s.png)
    # TREND WITH DNA SIZE (alpha vs MSD @ 8s)
    # Pooled per-particle scatter + regression line, side by side, showing
    # alpha's flat/non-significant trend against MSD@8s's small but real
    # declining trend with fragment size.
    # ============================================================================
    print("\nSaving Figure_SizeTrend_Alpha_vs_MSD8s.png...")
    alpha_trend_result = run_trend_test(alpha2_data, bp_x_values, actual_labels, verbose=False)
    
    rng = np.random.default_rng(42)
    
    fig_trend, (ax_trend_alpha, ax_trend_msd) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    plot_trend_scatter_panel(
        ax_trend_alpha, alpha2_data, bp_x_values, trend_colors, alpha_trend_result,
        y_log=False, y_label=r'$\alpha$ (2-8s)', panel_label='A', rng=rng,
        labels=actual_labels
    )
    plot_trend_scatter_panel(
        ax_trend_msd, msd_at_8s_data, bp_x_values, trend_colors, msd_trend_result,
        y_log=True, y_label=r'MSD at $\mathrm{\Delta t}$ = 8s ($\mu$m$^2$)', panel_label='B', rng=rng,
        labels=actual_labels
    )
    
    handles, labels_ = ax_trend_alpha.get_legend_handles_labels()
    fig_trend.legend(handles, labels_, loc='center left',
                      bbox_to_anchor=(1.02, 0.8), bbox_transform=ax_trend_msd.transAxes,
                      frameon=True, fontsize=10)
    
    plt.tight_layout()
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    fig_trend.savefig(OUTPUT_FIGURE_BASE / "Figure_SizeTrend_Alpha_vs_MSD8s.png",
                       dpi=600, bbox_inches='tight')
    print("Figure_SizeTrend_Alpha_vs_MSD8s.png saved.")

    
    # ============================================================================
    # CALCULATED RESULTS SUMMARY TABLE
    # ============================================================================
    print("\n" + "="*80)
    print("CALCULATED RESULTS SUMMARY (Long-time regime 2-8s)")
    print("="*80)
    
    # Filter out 'beads' if present, as it was not in the primary DNA table
    data_keys = [k for k in processed_keys if 'beads' not in k.lower()]
    
    print("\n{:<10} {:>8} {:>10} {:>12} {:>8} {:>8} {:>8}".format(
        "DNA Size", "N", "Med α", "Med Dapp", "Caged %", "Subdiff %", "Superdiff %"))
    print("-" * 74)
    
    for key in data_keys:
        calc = results_dict[key]
        
        # Calculate percentages from alpha2 data (confinement threshold from
        # msd_core.classify_diffusion_regime -- see threshold sensitivity
        # analysis below for justification of this specific value)
        alpha2_arr = calc['alpha2']
        caged_calc, subdiff_calc, superdiff_calc = classify_diffusion_regime(alpha2_arr)
        
        # Get median values (using Dapp2 and alpha2 for long-time regime)
        alpha_calc = np.median(alpha2_arr)
        Dapp_calc = np.median(calc['Dapp2'])
        
        # Get label and track count
        label = LABELS[CONDITION_KEYS.index(key)]
        n_calc = calc['n_tracks']
        
        print("{:<10} {:>8d} {:>10.2f} {:>12.2e} {:>8.0f} {:>8.0f} {:>8.0f}".format(
            label, 
            n_calc, 
            alpha_calc, 
            Dapp_calc, 
            caged_calc, 
            subdiff_calc,
            superdiff_calc
        ))
    print("="*80)

    
    


    # ============================================================================
    # Non-bead subset + derived data lists, reused by several figures below
    # (Figures 3-5). [Previously this section also built an 8-panel combined
    # MSD+Dapp+Alpha figure, but its savefig() call was commented out --
    # the figure was never written to disk, so that dead plotting code has
    # been removed. Only the data it produced that's actually reused
    # downstream is kept here.]
    # ============================================================================
    non_bead_keys = [k for k in CONDITION_KEYS if 'bead' not in k.lower()]
    non_bead_processed = [k for k in processed_keys if 'bead' not in k.lower()]

    non_bead_labels = [LABELS[CONDITION_KEYS.index(k)] for k in non_bead_processed]
    non_bead_colors = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in non_bead_processed]
    non_bead_x_ticks = list(range(1, len(non_bead_processed) + 1))

    dapp_data_to_plot = [results_dict[k]['Dapp2'] for k in non_bead_processed]
    alpha_data_to_plot = [results_dict[k]['alpha2'] for k in non_bead_processed]

    # ============================================================================
    # --- MAIN FIGURE (Figure_3.png): First 3 MSD panels (A-C) ---
    # ============================================================================
    print("\nSaving Figure 3 (First 3 MSD panels A-C)...")
    
    fig_msd_first, axs_msd_first = plt.subplots(1, 3, figsize=(12, 3.375))
    msd_plot_keys_first = [k for k in non_bead_keys[:3] if k in results_dict]
    
    for i, key in enumerate(msd_plot_keys_first):
        original_index = CONDITION_KEYS.index(key)
        current_label = LABELS[original_index]
        data = results_dict[key]
        ax = axs_msd_first[i]
        
        if data['msd_corrected'] is not None and len(data['msd_corrected'].columns) > 0:
            # Plot individual tracks in light gray
            ax.plot(data['msd_corrected'], color='lightgray', alpha=0.3)
            # Plot mean MSD
            ax.plot(data['msd_corrected'].mean(axis=1), marker='o', 
                    markeredgecolor='black', markerfacecolor=PLOT_COLORS[original_index], 
                    linestyle='None', markersize=6)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
            ax.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle$ ($\mu\mathrm{m}^2$)', fontsize=14)
            ax.set_ylim((1e-4, 1))
            ax.grid(False)
            ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            # Panel label
            ax.text(-0.2, 1.08, chr(65 + i), transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
            # Condition label
            ax.text(0.15, 0.9, current_label + '\n$n$ = ' + str(data['n_tracks']), 
                    verticalalignment='top', horizontalalignment='left', 
                    transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    fig_msd_first.savefig(OUTPUT_FIGURE_BASE / "Figure_3.png", dpi=600, bbox_inches='tight')
    print("Figure 3 saved.")
    
    
    # ============================================================================
    # --- MAIN FIGURE (Figure_4.png): Second 3 MSD panels (D-F becomes A-C) ---
    # ============================================================================
    print("\nSaving Figure 4 (Second 3 MSD panels, relabeled A-C)...")
    
    fig_msd_second, axs_msd_second = plt.subplots(1, 3, figsize=(12, 3.375))
    msd_plot_keys_second = [k for k in non_bead_keys[3:6] if k in results_dict]
    
    
    for i, key in enumerate(msd_plot_keys_second):
        original_index = CONDITION_KEYS.index(key)
        current_label = LABELS[original_index]
        data = results_dict[key]
        ax = axs_msd_second[i]
        
        if data['msd_corrected'] is not None and len(data['msd_corrected'].columns) > 0:
            # Plot individual tracks in light gray
            ax.plot(data['msd_corrected'], color='lightgray', alpha=0.3)
            # Plot mean MSD
            ax.plot(data['msd_corrected'].mean(axis=1), marker='o', 
                    markeredgecolor='black', markerfacecolor=PLOT_COLORS[original_index], 
                    linestyle='None', markersize=6)
            
            
            if i == 0:
                bead_mean_msd = results_dict['beads']['msd_corrected'].mean(axis=1)
                
                ax.plot(
                    bead_mean_msd.index,
                    bead_mean_msd.values,
                    marker='o',
                    linestyle='-',
                    color='black',
                    markerfacecolor='black',
                    markeredgecolor='black',
                    markersize=5,
                    linewidth=1.5,
                    label='Bead noise floor',
                    zorder=10
                )
                
                
                x2_start, x2_end = 2, 8
                x2 = np.logspace(np.log10(x2_start), np.log10(x2_end), 100)
            
                y2_anchor = 3e-3
                alpha = 0.8
                y2 = y2_anchor * (x2 / x2_start) ** alpha
            
                ax.plot(x2, y2 * 1.9, linestyle=':', color='black', linewidth=1.5)
                ax.text(5, y2_anchor * 1.9, r'$\Delta t^{\alpha}$', fontsize=14, color='black')
                
                ax.text(0.95, 0.05, 'Bead noise floor' + '\n $n$ = ' + str(results_dict[processed_keys[-1]]['n_tracks']),
                        verticalalignment='bottom', horizontalalignment='right', 
                        transform=ax.transAxes, fontsize=14)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
            ax.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle$ ($\mu\mathrm{m}^2$)', fontsize=14)
            ax.set_ylim((1e-4, 1))
            ax.grid(False)
            ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            # Panel label (A, B, C instead of D, E, F)
            ax.text(-0.2, 1.08, chr(65 + i), transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
            # Condition label
            ax.text(0.15, 0.9, current_label + '\n$n$ = ' + str(data['n_tracks']), 
                    verticalalignment='top', horizontalalignment='left', 
                    transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    fig_msd_second.savefig(OUTPUT_FIGURE_BASE / "Figure_4.png", dpi=600, bbox_inches='tight')
    print("Figure 4 saved.")
    
    
    # ============================================================================
    # --- MAIN FIGURE (Figure_5.png): Alpha (A) + Diffusion Bar Chart (B) + Dapp (C) + MSD@8s (D) ---
    # ============================================================================
    print("\nSaving Figure 5 (Dapp + Alpha + Diffusion bar chart)...")
    
    import matplotlib.gridspec as gridspec
    
    fig_comb = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)
    
    # Trend-test stats (Spearman rho, linear R^2, p) for each of the three
    # violin panels below, annotated directly on each panel. This gives a
    # single, principled headline statistic per panel -- "is there a
    # systematic trend with DNA size" -- rather than relying on the
    # pattern of pairwise significance bars alone, which (per discussion)
    # only cleanly supports the 100 bp vs 7614 bp-type comparisons.
    bp_x_values_fig5 = [BP_SIZE_MAP.get(label, np.nan) for label in non_bead_labels]
    msd8s_data_to_plot = [extract_msd_at_lagtime(results_dict[k]['msd_corrected'], 8.0, FPS)
                           for k in non_bead_processed]

    def _trend_annotation_text(data_list, log_transform):
        data_for_test = log10_safe(data_list, verbose=False) if log_transform else data_list
        result = run_trend_test(data_for_test, bp_x_values_fig5, non_bead_labels, verbose=False)
        p_reg = result['regression']['p_value']
        r2 = result['regression']['r_value'] ** 2 if np.isfinite(result['regression']['r_value']) else np.nan
        slope_reg = result['regression']['slope']
        # p_reg: two-sided p-value testing slope == 0 (t-test, n-2 df) --
        # see msd_core.run_trend_test() docstring for the exact formula.
        return f"slope = {slope_reg:.3g}\n$R^2$={r2:.3f}\n$p$={p_reg:.2e}"

    alpha_trend_text = _trend_annotation_text(alpha_data_to_plot, log_transform=False)
    dapp_trend_text = _trend_annotation_text(dapp_data_to_plot, log_transform=True)
    msd8s_trend_text = _trend_annotation_text(msd8s_data_to_plot, log_transform=True)
    
    
    # ----- Panel G: Alpha violin plot -----
    ax_h = fig_comb.add_subplot(gs[0])
    
    # Create violin plot
    parts_h = ax_h.violinplot(alpha_data_to_plot, positions=non_bead_x_ticks, showmeans=False, showmedians=False, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts_h['bodies']):
        pc.set_facecolor(non_bead_colors[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add median lines manually
    for i, data_arr in enumerate(alpha_data_to_plot):
        median_val = np.median(data_arr)
        ax_h.plot([non_bead_x_ticks[i] - 0.2, non_bead_x_ticks[i] + 0.2], [median_val, median_val], 
                  color='black', linewidth=2, zorder=10)
    
    # Scatter data points
    for i, data_arr in enumerate(alpha_data_to_plot):
        ax_h.scatter(np.full_like(data_arr, non_bead_x_ticks[i]) + np.random.uniform(-0.05, 0.05, size=len(data_arr)),
                     data_arr, color=non_bead_colors[i], alpha=1.0, marker='o', edgecolor='black', s=25, zorder=5)
    
    # Add significance bars
    add_sig_bars_alpha(ax_h, alpha_data_to_plot, non_bead_labels)
    
    ax_h.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Normal diffusion ($\\alpha=1$)', zorder=0)
    ax_h.axhline(0.4, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7, label='Threshold for confinement ($\\alpha<0.4$)', zorder=0)
    
    ax_h.yaxis.set_ticks_position('both')
    ax_h.xaxis.set_ticks_position('both')
    for spine in ax_h.spines.values():
        spine.set_linewidth(1.5)
    ax_h.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    
    if ax_h.get_ylim()[1] < 2.5:
        ax_h.set_ylim(-0.08, 2.5)
            
    ax_h.set_xticks(non_bead_x_ticks)
    ax_h.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_h.set_ylabel(r'$\alpha$')
    ax_h.text(-0.09, 1.12, 'A', transform=ax_h.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    #ax_h.text(0.05, 0.65, alpha_trend_text, transform=ax_h.transAxes, fontsize=11, va='bottom', ha='left')
    ax_h.grid(False)
    
    # ----- Panel H: Diffusion type bar chart (left) + Dapp (right), row 2 -----
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.4)
    ax_i = fig_comb.add_subplot(gs_row1[0])
    ax_g = fig_comb.add_subplot(gs_row1[1])
    
    # Calculate percentages for each diffusion type
    classifications = [classify_diffusion_regime(arr) for arr in alpha_data_to_plot]
    confined  = [c[0] for c in classifications]
    subdiff   = [c[1] for c in classifications]
    superdiff = [c[2] for c in classifications]
    
    x = np.arange(len(non_bead_labels))
    width = 0.25
    
    bar1 = ax_i.bar(x - width, confined,   width, color=non_bead_colors,
                    edgecolor='black', linewidth=1.2, hatch='///', alpha=0.8)
    bar2 = ax_i.bar(x,          subdiff,   width, color=non_bead_colors,
                    edgecolor='black', linewidth=1.2, hatch='',   alpha=0.8)
    bar3 = ax_i.bar(x + width,  superdiff, width, color=non_bead_colors,
                    edgecolor='black', linewidth=1.2, hatch='...', alpha=0.8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#AAAAAA', edgecolor='black', hatch='///',  label=r'Confined ($\leq$0.4)'),
        Patch(facecolor='#AAAAAA', edgecolor='black', hatch='',     label=r'Subdiffusive (0.4<$\alpha$<1)'),
        Patch(facecolor='#AAAAAA', edgecolor='black', hatch='...',  label='Superdiffusive (>1)'),
    ]
    
    ax_i.set_xticks(x)
    ax_i.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_i.set_ylabel('Percentage (%)', fontsize=14)
    ax_i.set_ylim(0, 100)
    ax_i.legend(handles=legend_elements, frameon=False, fontsize=10)
    ax_i.yaxis.set_ticks_position('both')
    ax_i.xaxis.set_ticks_position('both')
    for spine in ax_i.spines.values():
        spine.set_linewidth(1.5)
    ax_i.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_i.grid(False)
    ax_i.text(-0.2, 1.12, 'B', transform=ax_i.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    # ============================================================================
    # SUPPLEMENTARY FIGURE (Figure_Supp_ConfinementThresholdSensitivity.png)
    # CONFINEMENT THRESHOLD SENSITIVITY ANALYSIS
    # Checks whether the 0.4 confinement threshold used above is arbitrary --
    # i.e., whether the relative differences between conditions in the panel
    # above are an artifact of that specific value, or hold across a
    # reasonable range of thresholds used in the literature.
    # ============================================================================
    threshold_sensitivity_table(alpha_data_to_plot, non_bead_labels, exclude_substring=None)

    fig_sens, ax_sens = plt.subplots(figsize=(6, 4.5))
    plot_threshold_sensitivity(ax_sens, alpha_data_to_plot, non_bead_labels, non_bead_colors,
                                exclude_substring=None)
    plt.tight_layout()
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    fig_sens.savefig(OUTPUT_FIGURE_BASE / "Figure_Supp_ConfinementThresholdSensitivity.png",
                      dpi=600, bbox_inches='tight')
    print("Figure_Supp_ConfinementThresholdSensitivity.png saved.")

    # Precise stable-range computation (fine grid, exact crossing boundaries)
    #  Reported both with and without 250 bp, since that
    # condition has a much smaller sample size than the others (n=22) and
    # its exact ordering is expected to be noisy for that reason alone,
    # independent of DNA size.
    find_stable_threshold_range(alpha_data_to_plot, non_bead_labels, target_threshold=0.4)
    find_stable_threshold_range(alpha_data_to_plot, non_bead_labels,
                                 exclude_labels=['250 bp'], target_threshold=0.4)

    # Full-range scan: is there a comparably wide, well-supported plateau
    # somewhere ELSE (e.g. near 0.7) that would tell a different story?
    # This is what actually answers "why 0.4 and not some other value" --
    # the checks above only prove LOCAL robustness around 0.4 itself.
    find_all_stable_plateaus(alpha_data_to_plot, non_bead_labels,
                              exclude_labels=['250 bp'], bounds=(0.1, 0.9))

    # ============================================================================
    # SUPPLEMENTARY FIGURE (Figure_Supp_DappAlphaCoupling.png)
    # Dapp-ALPHA COUPLING DIAGNOSTIC
    # Checks whether the observed Dapp-vs-alpha relationship reflects real
    # physics or is substantially a mechanical artifact of the power-law
    # fit itself (Dapp and alpha are the intercept and slope of the same
    # fit, so they are not independent) -- see msd_core.plot_dapp_alpha_diagnostic
    # docstring for the full reasoning.
    # ============================================================================
    print("\nSaving Figure_Supp_DappAlphaCoupling.png...")
    fig_coupling, (ax_coupling1, ax_coupling2) = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_dapp_alpha_diagnostic(
        ax_coupling1, ax_coupling2, results_dict, non_bead_processed,
        non_bead_labels, non_bead_colors, target_lag_seconds=8.0, fps=FPS
    )
    plt.tight_layout()
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    fig_coupling.savefig(OUTPUT_FIGURE_BASE / "Figure_Supp_DappAlphaCoupling.png",
                          dpi=600, bbox_inches='tight')
    print("Figure_Supp_DappAlphaCoupling.png saved.")
    
    
    # ----- Panel C: Dapp violin plot (row 2, right half; axes already created above) -----
    # Create violin plot
    parts_g = ax_g.violinplot(dapp_data_to_plot, positions=non_bead_x_ticks, showmeans=False, showmedians=False, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts_g['bodies']):
        pc.set_facecolor(non_bead_colors[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add median lines manually
    for i, data_arr in enumerate(dapp_data_to_plot):
        median_val = np.median(data_arr)
        ax_g.plot([non_bead_x_ticks[i] - 0.2, non_bead_x_ticks[i] + 0.2], [median_val, median_val], 
                  color='black', linewidth=2, zorder=10)
    
    # Scatter data points
    for i, data_arr in enumerate(dapp_data_to_plot):
        ax_g.scatter(np.full_like(data_arr, non_bead_x_ticks[i]) + np.random.uniform(-0.05, 0.05, size=len(data_arr)),
                     data_arr, color=non_bead_colors[i], alpha=1.0, marker='o', edgecolor='black', s=25, zorder=5)
    
    # Set log scale BEFORE add_sig_bars_dapp -- that function's multiplicative
    # bar-stacking logic assumes it's operating on an axis already in log
    # scale (see its docstring/comments); calling it on a still-linear axis
    # causes the internal set_ylim() to compute a wildly inflated bound in
    # linear space, which then gets misinterpreted once set_yscale('log')
    # is applied afterward -- this was silently masked here only because
    # of the explicit set_ylim(1e-4, 5) override below, but broke visibly
    # for the MSD@8s panel, which had no such override.
    ax_g.set_yscale('log')

    # Add significance bars
    add_sig_bars_dapp(ax_g, dapp_data_to_plot, non_bead_labels)
    
    ax_g.yaxis.set_ticks_position('both')
    ax_g.xaxis.set_ticks_position('both')
    ax_g.yaxis.set_label_position("left")
    # ax_g.yaxis.tick_right()
    for spine in ax_g.spines.values():
        spine.set_linewidth(1.5)
    ax_g.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_g.set_ylim(1e-4, 0.5)
    ax_g.set_xticks(non_bead_x_ticks)
    ax_g.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_g.set_ylabel(r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)', fontsize=14)
    ax_g.text(-0.09, 1.12, 'C', transform=ax_g.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    #ax_g.text(0.63, 0.65, dapp_trend_text, transform=ax_g.transAxes, fontsize=9, va='bottom', ha='left')
    ax_g.grid(False)

    # ----- Panel D: MSD @ 8s violin plot (row 3, full width) -----
    ax_msd8s = fig_comb.add_subplot(gs[2])

    parts_msd8s = ax_msd8s.violinplot(msd8s_data_to_plot, positions=non_bead_x_ticks, showmeans=False, showmedians=False, showextrema=False)

    for i, pc in enumerate(parts_msd8s['bodies']):
        pc.set_facecolor(non_bead_colors[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    for i, data_arr in enumerate(msd8s_data_to_plot):
        median_val = np.median(data_arr)
        ax_msd8s.plot([non_bead_x_ticks[i] - 0.2, non_bead_x_ticks[i] + 0.2], [median_val, median_val],
                      color='black', linewidth=2, zorder=10)

    for i, data_arr in enumerate(msd8s_data_to_plot):
        ax_msd8s.scatter(np.full_like(data_arr, non_bead_x_ticks[i]) + np.random.uniform(-0.05, 0.05, size=len(data_arr)),
                          data_arr, color=non_bead_colors[i], alpha=1.0, marker='o', edgecolor='black', s=25, zorder=5)

    # Set log scale BEFORE add_sig_bars_dapp -- see comment on the Dapp
    # panel above for why (this panel had no explicit set_ylim() override
    # to mask the bug, so it broke visibly).
    ax_msd8s.set_yscale('log')

    add_sig_bars_dapp(ax_msd8s, msd8s_data_to_plot, non_bead_labels)

    # Now full-width on its own row (no longer paired with Dapp), so the
    # y-axis reverts to the normal left side, matching the alpha panel
    # above -- the right-side convention was only needed when this panel
    # shared a row with another panel to its left.
    ax_msd8s.yaxis.set_ticks_position('both')
    ax_msd8s.xaxis.set_ticks_position('both')
    for spine in ax_msd8s.spines.values():
        spine.set_linewidth(1.5)
    ax_msd8s.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd8s.set_xticks(non_bead_x_ticks)
    ax_msd8s.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_msd8s.set_ylabel(r'MSD at $\mathrm{\Delta t}$ = 8s ($\mathrm{\mu}$m$^2$)', fontsize=14)
    ax_msd8s.text(-0.09, 1.12, 'D', transform=ax_msd8s.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    #ax_msd8s.text(0.97, 0.75, msd8s_trend_text, transform=ax_msd8s.transAxes, fontsize=11, va='bottom', ha='right')
    ax_msd8s.grid(False)
    
    plt.tight_layout()
    fig_comb.savefig(OUTPUT_FIGURE_BASE / "Figure_5.png", dpi=600, bbox_inches='tight')
    print("Figure 5 saved.")
    
    print("\nAll figures generated successfully!")
    



    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {OUTPUT_FIGURE_BASE}")
    print("="*80)

if __name__ == '__main__':
    run_analysis()