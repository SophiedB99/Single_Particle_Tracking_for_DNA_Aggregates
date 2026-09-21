# -*- coding: utf-8 -*-
"""
MSD Analysis with Per-Particle Localization Correction - Cell Types Analysis
Complete integrated version with corrected figure generation
NOW INCLUDES: Noise floor filtering and 8s Dapp calculation
@author: sophiedeboer (Modified)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
import sys
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Shared MSD core (msd, sigma correction, powerfit, imsd, process_directory) 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from msd_core import (
    process_directory,
    check_normality, check_equal_variance, log10_safe,
    run_welch_games_howell, extract_msd_at_lagtime,
    classify_diffusion_regime,
    threshold_sensitivity_table, plot_threshold_sensitivity,
    find_stable_threshold_range, find_all_stable_plateaus,
    msd_lagtime_sensitivity_table, plot_msd_lagtime_sensitivity,
    plot_dapp_alpha_diagnostic,
)

NOISE_FLOOR_FRACTION = 0.98  # fraction of lag times a trajectory must be
                              # ABOVE the bead noise floor to be kept

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DATA_ROOT = Path('D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data')

CONDITIONS_MAP = [
    # Cell types (YOYO-1/dim): localization correction + mass scaling
    # correction proved unreliable for this dye/dataset -- the mass-scaled
    # sigma rejected 54-80% of trajectories, and even the matched dim/bright
    # pair method (04b) couldn't produce a correction that survived the
    # noise-floor filter. Reported MSD/Dapp for these conditions are
    # therefore UNCORRECTED upper-bound estimates -- see methods/limitations.
    { 'key': 'MDA MB 231', 'size': 'MDA MB 231', 'base_folder': 'Aswin Data/Aswin Data MDA-MB-231 Corrected for Drift', 'collected_dir': 'Collected_MDAMB231', 'max_length': 350, 'use_localization_correction': False, 'use_mass_scaling': False },
    { 'key': 'MCF 7', 'size': 'MCF 7', 'base_folder': 'Aswin Data/Aswin Data MCF 7 Corrected for Drift', 'collected_dir': 'Collected_MCF7', 'max_length': 300, 'use_localization_correction': False, 'use_mass_scaling': False },
    { 'key': 'MCF 10A', 'size': 'MCF 10A', 'base_folder': 'Aswin Data/Aswin Data MCF 10A Corrected for Drift', 'collected_dir': 'Collected_MCF10A', 'max_length': 150, 'use_localization_correction': False, 'use_mass_scaling': False },
    # ATP depletion / Control (Cy3, bright): correction validated fine for
    # this dye
    { 'key': 'ATP depleted', 'size': 'LifeActGFP Cy3 ATP depleted', 'base_folder': 'MDA-MB-231/ATP Depletion MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_ATPdepl_MDAMB231', 'max_length': 500, 'use_localization_correction': True, 'use_mass_scaling': True },
    { 'key': 'Control', 'size': 'LifeActGFP Cy3 Control', 'base_folder': 'MDA-MB-231/Cy3 MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_Cy3_MDAMB231', 'max_length': 650, 'use_localization_correction': True, 'use_mass_scaling': True },
    { 'key': 'beads', 'size': 'beads', 'base_folder': 'Control Experiments_Beads/0.22 um', 'collected_dir': 'Collected_0.22um', 'max_length': 800, 'use_localization_correction': False, 'use_mass_scaling': False },
]

CELL_LINE_MARKERS = {
    'MDA MB 231': 'D',
    'MCF 10A': 's',
    'MCF 7': '^',
    'ATP depleted': 'v',
    'Control': 'o', 
    
}

CONDITION_KEYS = [c['key'] for c in CONDITIONS_MAP]
MAX_TRAJ_LENGTHS = [c['max_length'] for c in CONDITIONS_MAP]
CONDITION_LABELS = [c['size'] for c in CONDITIONS_MAP]
CONDITION_DICT = {c['key']: c for c in CONDITIONS_MAP}

SIGMA_STUCK_MICRONS = 0.021244
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS = 0.014008     # YOYO-1
MASS_DIM_REFERENCE = 589.8

SIGMA_CY3_MICRONS = 0.013078     # Cy3
MASS_CY3_REFERENCE = 8922.3      # median mass from Cy3 (Bright Ref) dataset; reference for mass scaling

MPP = 0.16
FPS = 10
MAX_LAGTIME = 100

R_SQUARED_THRESHOLD = 0.8

LABELS = [c['size'] for c in CONDITIONS_MAP]
PLOT_COLORS = ['#533AB7', '#0F6E56', '#EF9F27', '#888780', '#D85A30', '#000000']

PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_FIGURE_BASE = BASE_DATA_ROOT / 'Figures'

# ============================================================================
# CORE MSD FUNCTIONS
# ============================================================================


# ============================================================================
# PLOTTING HELPER FUNCTIONS
# ============================================================================
def add_sig_bars_alpha(ax, data_list, actual_labels, alpha_thresh=0.05):
    """Significance bars using Welch's ANOVA + Games-Howell post-hoc
    (family-wise error rate corrected, does not assume equal variance)
    rather than uncorrected pairwise t-tests or Tukey HSD."""
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) + 0.1
    spacing = 0.12
    gh = run_welch_games_howell(data_list, actual_labels, exclude_substring='beads', verbose=False)
    p_vals = [(p, i, j) for (i, j), p in gh['pairwise_pvalues'].items()]  # keep ALL pairs, not just significant
    p_vals.sort(key=lambda x: x[0])
    for k, (p, i, j) in enumerate(p_vals):
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        h = base_h + k * spacing
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False)
        ax.text((i + 1 + j + 1) / 2, h + spacing * 0.2, sig,
                ha="center", va="bottom", fontsize=10, color="k")
    if p_vals:
        final_h = base_h + len(p_vals) * spacing
        ax.set_ylim(cur_ylim[0], final_h + spacing)


def add_sig_bars_dapp(ax, data_list, actual_labels, alpha_thresh=0.05):
    """Significance bars using Tukey HSD post-hoc (family-wise error rate
    corrected) rather than uncorrected pairwise t-tests."""
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) * 1.5
    spacing_factor = 2.3
    # Dapp is strictly positive and typically log-normal, so the
    # statistical test itself runs on log10(Dapp) (matches the console
    # diagnostics), even though the bars are still positioned using the
    # raw data's scale, since that's what's actually plotted here.
    # Games-Howell since Levene's shows unequal variance.
    log_data_list = log10_safe(data_list, verbose=False)
    gh = run_welch_games_howell(log_data_list, actual_labels, exclude_substring='beads', verbose=False)
    p_vals = [(p, i, j) for (i, j), p in gh['pairwise_pvalues'].items()]  # keep ALL pairs, not just significant
    p_vals.sort(key=lambda x: x[0])
    for k, (p, i, j) in enumerate(p_vals):
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        h = base_h * (spacing_factor ** k)
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False)
        ax.text((i + 1 + j + 1) / 2, h * 1.1, sig,
                ha="center", va="bottom", fontsize=10, color="k")
    if p_vals:
        final_h = base_h * (spacing_factor ** len(p_vals))
        ax.set_ylim(cur_ylim[0], final_h * 1.2)

def log_violin(ax, data, positions, colors):
    # Use RAW data for the violin plot — no log10 transform
    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Color the violin bodies
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Scatter and medians (also raw data)
    for i, raw in enumerate(data):
        med = np.median(raw)
        ax.plot([positions[i] - 0.2, positions[i] + 0.2], [med, med],
                color='black', linewidth=2, zorder=10)
        
        ax.scatter(
            np.full_like(raw, positions[i]) + np.random.uniform(-0.05, 0.05, size=len(raw)),
            raw,
            color=colors[i],
            edgecolor='k',
            s=25,
            alpha=1.0,
            zorder=5
        )
    
    # Apply log scale AFTER plotting raw values
    ax.set_yscale('log')


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_analysis():
    print("\n" + "="*80)
    print("MSD ANALYSIS WITH PER-PARTICLE LOCALIZATION CORRECTION")
    print("="*80)
    print(f"Sigma (beads, fixed): {SIGMA_STUCK_MICRONS:.6f} μm")
    print(f"Sigma (YOYO-1 dim, fixed): {SIGMA_DIM_MICRONS:.6f} μm")
    print(f"Sigma (Cy3, mass-scaled reference): {SIGMA_CY3_MICRONS:.6f} μm (ref mass = {MASS_CY3_REFERENCE:.1f})")
    print(f"Mass reference (beads): {MASS_STUCK_REFERENCE:.1f}")
    print(f"R² threshold: {R_SQUARED_THRESHOLD}")
    print(f"\nLocalization correction / mass scaling, per condition:")
    for c in CONDITIONS_MAP:
        print(f"  {c['size']:<28} correction={'ON ' if c['use_localization_correction'] else 'OFF'}"
              f"   mass_scaling={'ON' if c['use_mass_scaling'] else 'OFF'}")
    print()

    results_dict = {}
    directories = {}

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
            # Process beads WITHOUT noise floor filtering (pass None).
            # sigma_ref=0.0 means the localization-correction toggle is
            # moot here regardless -- kept False to match the beads entry
            # in CONDITIONS_MAP.
            bead_result = process_directory(
                bead_directory, MPP, FPS, MAX_LAGTIME, 0.0, MASS_STUCK_REFERENCE, 
                MAX_TRAJ_LENGTHS[CONDITION_KEYS.index('beads')], False, 
                use_localization_correction=CONDITION_DICT['beads']['use_localization_correction'],
                r_squared_threshold=R_SQUARED_THRESHOLD,
                noise_floor_fraction=NOISE_FLOOR_FRACTION,
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

    for i, label_key in enumerate(CONDITION_KEYS):
        if label_key == 'beads':
            continue  # Skip beads since we already processed them
            
        if label_key in directories:
            directory = directories[label_key]
            max_length = MAX_TRAJ_LENGTHS[i]
            size_label = CONDITION_DICT[label_key]['size']
            use_loc_corr = CONDITION_DICT[label_key]['use_localization_correction']
            use_mass_scale = CONDITION_DICT[label_key]['use_mass_scaling']

            if label_key in ['MDA MB 231', 'MCF 10A', 'MCF 7']:
                sigma_to_use = SIGMA_DIM_MICRONS
                mass_to_use = MASS_DIM_REFERENCE
                use_scaling_temp = use_mass_scale and use_loc_corr
                status = f"correction={'ON' if use_loc_corr else 'OFF'}, mass_scaling={'ON' if use_scaling_temp else 'OFF'}"
                print(f"Processing {size_label} ({label_key})... Fixed Sigma ({sigma_to_use:.4f} μm), {status}.")
            else:
                sigma_to_use = SIGMA_CY3_MICRONS
                mass_to_use = MASS_CY3_REFERENCE
                use_scaling_temp = use_mass_scale and use_loc_corr
                status = f"correction={'ON' if use_loc_corr else 'OFF'}, mass_scaling={'ON' if use_scaling_temp else 'OFF'}"
                print(f"Processing {size_label} ({label_key})... Cy3 Sigma ({sigma_to_use:.4f} μm), {status}.")
            
            if not directory.is_dir():
                print(f"CRITICAL: Directory for {size_label} not found at {directory}")
                continue

            # ============================================================================
            # MODIFIED: Pass noise_floor_msd to enable filtering
            # ============================================================================
            result = process_directory(
                directory, MPP, FPS, MAX_LAGTIME, sigma_to_use, mass_to_use, 
                max_length, use_scaling_temp, 
                use_localization_correction=use_loc_corr,
                r_squared_threshold=R_SQUARED_THRESHOLD,
                noise_floor_fraction=NOISE_FLOOR_FRACTION,
                noise_floor_msd=noise_floor_msd  # Pass noise floor for filtering
            )
            # ============================================================================

            if result is not None:
                results_dict[label_key] = result
                print(f" Tracks accepted: {result['n_tracks']}")
                print(f" Mean mass: {np.mean(result['mass']):.1f}")
                print(f" Mean sigma: {np.mean(result['sigma']):.6f} μm")
                rc = result.get('rejection_counts', {})
                if rc:
                    print(f" Rejection breakdown (of {rc.get('n_total', 0)} particles examined):")
                    print(f"   too short:            {rc.get('n_too_short', 0)}")
                    print(f"   invalid MSD:          {rc.get('n_msd_invalid', 0)}")
                    print(f"   fit invalid:          {rc.get('n_fit_invalid', 0)}")
                    print(f"   failed R²/alpha:      {rc.get('n_failed_r_squared', 0)}")
                    print(f"   accepted pre-noise-floor: {rc.get('n_accepted', 0)}")
                    print(f"   removed by NOISE FLOOR:  {rc.get('n_removed_noise_floor', 0)}")
                    print(f"   -> final accepted:       {result['n_tracks']}")
            else:
                print(f" WARNING: No valid trajectories for {size_label}")

    if not results_dict:
        print("\nERROR: results_dict is empty. No data passed filters or directories were invalid.")
        return

    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    

 
    # ============================================================================
    # MAIN FIGURE (Figure_6.png): Cell Types Analysis
    #   A: MSD ensemble | B: alpha violin | C: Dapp violin |
    #   D: MSD@5s violin | E: MSD@8s violin
    # (Previously panel C was a confined/subdiffusive/superdiffusive
    # percentage bar chart -- removed as redundant with the continuous
    # alpha distribution already shown in panel B; see
    # Figure_Supp_ConfinementThresholdSensitivity_CellTypes.png for the
    # confinement-threshold question it partially addressed.)
    #
    # Figure roadmap for this script:
    #   Figure_Supp_DappAlphaCoupling_CellTypes.png           SUPPLEMENTARY
    #   Figure_Supp_ConfinementThresholdSensitivity_CellTypes.png  SUPPLEMENTARY
    #   Figure_Supp_MSDLagtimeSensitivity_CellTypes.png       SUPPLEMENTARY
    #   Figure_6.png                                          MAIN
    #   Figure_ATP_Depletion.png                              SUPPLEMENTARY
    #   Figure_Supp_ConfinementThresholdSensitivity_ATP.png   SUPPLEMENTARY
    #   Figure_Supp_MSDLagtimeSensitivity_ATP.png             SUPPLEMENTARY
    # ============================================================================
    print("\nGenerating Figure 6 (Cell Types - A-E panel layout)...")
    
    cell_keys = ['MDA MB 231', 'MCF 7', 'MCF 10A']
    cell_labels = [LABELS[CONDITION_KEYS.index(k)] for k in cell_keys]
    cell_colors = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in cell_keys]
    cell_x_ticks = list(range(1, len(cell_keys) + 1))
    
    alpha_cell = [results_dict[k]['alpha2'] for k in cell_keys]
    dapp_cell = [results_dict[k]['Dapp2'] for k in cell_keys]

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - CELL TYPES - ALPHA (2-8s)")
    print("="*80)
    check_normality(alpha_cell, cell_labels)
    check_equal_variance(alpha_cell, cell_labels)
    run_welch_games_howell(alpha_cell, cell_labels)

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - CELL TYPES - Dapp (2-8s), log10-transformed")
    print("="*80)
    log_dapp_cell = log10_safe(dapp_cell)
    check_normality(log_dapp_cell, cell_labels)
    check_equal_variance(log_dapp_cell, cell_labels)
    run_welch_games_howell(log_dapp_cell, cell_labels)

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - CELL TYPES - MSD AT 8s (directly measured, not fit-derived)")
    print("="*80)
    msd_at_8s_cell = [extract_msd_at_lagtime(results_dict[k]['msd_corrected'], 8.0, FPS)
                       for k in cell_keys]
    log_msd_at_8s_cell = log10_safe(msd_at_8s_cell)
    check_normality(log_msd_at_8s_cell, cell_labels)
    check_equal_variance(log_msd_at_8s_cell, cell_labels)
    run_welch_games_howell(log_msd_at_8s_cell, cell_labels)

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - CELL TYPES - MSD AT 2s (directly measured, not fit-derived)")
    print("="*80)
    msd_at_2s_cell = [extract_msd_at_lagtime(results_dict[k]['msd_corrected'], 2.0, FPS)
                       for k in cell_keys]
    log_msd_at_2s_cell = log10_safe(msd_at_2s_cell)
    check_normality(log_msd_at_2s_cell, cell_labels)
    check_equal_variance(log_msd_at_2s_cell, cell_labels)
    run_welch_games_howell(log_msd_at_2s_cell, cell_labels)

    # ============================================================================
    # SUPPLEMENTARY FIGURE (Figure_Supp_DappAlphaCoupling_CellTypes.png)
    # Dapp-ALPHA COUPLING DIAGNOSTIC -- CELL TYPES
    # Cell-types-only counterpart to 05a's Figure_Supp_DappAlphaCoupling.png
    # (same function, restricted to condition_keys=cell_keys rather than
    # pooling across every non-bead condition). Motivation: Dapp and alpha
    # are the intercept and slope of the SAME log-log fit, so a genuine
    # Dapp difference between two conditions with statistically
    # indistinguishable alpha (as tested above) needs to survive this
    # check to be trusted as real mobility difference rather than a
    # by-product of fit geometry -- see
    # msd_core.plot_dapp_alpha_diagnostic docstring for the full reasoning.
    #
    # The Levene's test result on alpha (equal-variance-across-groups,
    # computed above via check_equal_variance) is reported to the console
    # only, not annotated on this figure -- indistinguishable alpha MEAN
    # (Games-Howell) and indistinguishable alpha VARIANCE (Levene's)
    # together make a stronger case that a downstream Dapp/MSD@8s
    # difference isn't secretly being driven by how alpha behaves.
    # ============================================================================
    print("\nSaving Figure_Supp_DappAlphaCoupling_CellTypes.png...")
    fig_coupling_cell, (ax_coupling1_cell, ax_coupling2_cell) = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_dapp_alpha_diagnostic(
        ax_coupling1_cell, ax_coupling2_cell, results_dict, cell_keys,
        cell_labels, cell_colors, target_lag_seconds=8.0, fps=FPS
    )
    plt.tight_layout()
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    fig_coupling_cell.savefig(OUTPUT_FIGURE_BASE / "Figure_Supp_DappAlphaCoupling_CellTypes.png",
                               dpi=600, bbox_inches='tight')
    print("Figure_Supp_DappAlphaCoupling_CellTypes.png saved.")

    # Figure setup: GridSpec instead of plt.subplots(2,2) -- row0 holds
    # A (MSD ensemble) + B (alpha violin) full-width; row1 holds three
    # amplitude/robustness panels: C (Dapp), D (MSD@5s), E (MSD@8s).
    fig6 = plt.figure(figsize=(3.375*3, 3.375*2*2))
    gs6_outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)

    gs6_row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs6_outer[0], wspace=0.4)
    ax_msd = fig6.add_subplot(gs6_row0[0])     # Panel A
    ax_alpha = fig6.add_subplot(gs6_row0[1])   # Panel B

    gs6_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs6_outer[1], wspace=0.5)
    ax_dapp = fig6.add_subplot(gs6_row1[0])    # Panel C
    ax_msd2s = fig6.add_subplot(gs6_row1[1])   # Panel D
    ax_msd8s = fig6.add_subplot(gs6_row1[2])   # Panel E
    
    
    
    # PANEL A: MSD Ensemble
    for key in cell_keys:
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_corrected']
        n_tracks = data.get('n_tracks', 0)
        label_with_n = f"{LABELS[original_index]} \n$n$ = {n_tracks}"
        marker = CELL_LINE_MARKERS.get(key, 'D')
        ax_msd.plot(ensemble_msd.index, ensemble_msd.values, color=PLOT_COLORS[original_index],
            marker=marker, markerfacecolor=PLOT_COLORS[original_index],
            linestyle='-', linewidth=2, markersize=7, label=label_with_n, zorder=3)
    
    ax_msd.set_xscale('log')
    ax_msd.set_yscale('log')
    ax_msd.set_ylim((2e-4, 1e-1))
    ax_msd.set_xlim((1e-1, 1e1))
    ax_msd.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle$ ($\mu\mathrm{m}^2$)', fontsize=14)
    ax_msd.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax_msd.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd.grid(True, alpha=0.3, linestyle='--', zorder=2)
    for spine in ax_msd.spines.values():
        spine.set_linewidth(1.5)
    leg = ax_msd.legend(
        bbox_to_anchor=(0.4, 0.4),
        fontsize=14,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.8,
        borderpad=0,
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(2)
        handle.set_markersize(7)
    
    ax_msd.text(-0.15, 1.05, 'A', transform=ax_msd.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_msd.grid(False)
    # PANEL B: Alpha Violin Plot
    
    # --- Create violin plot ---
    vp = ax_alpha.violinplot(
        alpha_cell,
        positions=cell_x_ticks,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # --- Color violins ---
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(cell_colors[i])
        body.set_alpha(0.4)
        body.set_edgecolor('black')
        body.set_linewidth(1.5)
    
    # --- Add median lines (manual, like your old plot) ---
    for i, data_arr in enumerate(alpha_cell):
        med = np.median(data_arr)
        ax_alpha.plot(
            [cell_x_ticks[i] - 0.2, cell_x_ticks[i] + 0.2],
            [med, med],
            color='black',
            linewidth=2,
            zorder=10
        )
    
    # --- Scatter points on top ---
    jitter = 0.05
    for i, data_arr in enumerate(alpha_cell):
        ax_alpha.scatter(
            np.full_like(data_arr, cell_x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(data_arr)),
            data_arr,
            color=cell_colors[i],
            alpha=1.0,
            marker='o',
            edgecolor='k',
            s=25,
            zorder=10
        )
    
    # --- Significance bars ---
    add_sig_bars_alpha(ax_alpha, alpha_cell, cell_labels)
    
    # --- Reference lines ---
    ax_alpha.axhline(
        1.0, color='gray', linestyle='--',
        linewidth=1.5, alpha=0.7,
        label=r'Normal diffusion ($\alpha=1$)', zorder=0
    )
    ax_alpha.axhline(
        0.4, color='darkgray', linestyle=':',
        linewidth=1.5, alpha=0.7,
        label=r'Threshold for confinement ($\alpha<0.4$)', zorder=0
    )
    
    # --- Axes formatting ---
    ax_alpha.set_ylim(-0.08, 2.5)
    ax_alpha.set_xticks(cell_x_ticks)
    ax_alpha.set_xticklabels(cell_labels, rotation=45, ha='right', fontsize=12)
    ax_alpha.set_ylabel(r'$\alpha$', fontsize=14)
    ax_alpha.grid(False)
    ax_alpha.tick_params(which="both", axis="both", direction="in",
                         labelsize=12, length=6, width=1.5)
    
    for spine in ax_alpha.spines.values():
        spine.set_linewidth(1.5)
    
    # --- Panel label ---
    ax_alpha.text(
        -0.15, 1.05, 'B',
        transform=ax_alpha.transAxes,
        fontsize=16, fontweight='bold',
        va='top', ha='left'
    )
    
    # Non-bead data lists reused by the confinement-threshold-sensitivity
    # supplementary figure below. [Previously this section also built a
    # confined/subdiffusive/superdiffusive percentage bar chart as main
    # Figure 6 Panel C -- removed from the main figure since it's a
    # re-binning of the same alpha distribution already shown continuously
    # in Panel B, and the confinement-threshold question it partially
    # raised (is the 0.4 cutoff doing the work) is already covered by the
    # existing supplementary confinement-threshold-sensitivity figure
    # immediately below.]
    cell_keys_no_beads = [k for k in cell_keys if k != 'beads']
    cell_labels_no_beads = [LABELS[CONDITION_KEYS.index(k)] for k in cell_keys_no_beads]
    alpha_no_beads = [results_dict[k]['alpha2'] for k in cell_keys_no_beads]
    colors_no_beads = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in cell_keys_no_beads]

    # ============================================================================
    # SUPPLEMENTARY FIGURE (Figure_Supp_ConfinementThresholdSensitivity_CellTypes.png)
    # CONFINEMENT THRESHOLD SENSITIVITY ANALYSIS - CELL TYPES
    # ============================================================================
    threshold_sensitivity_table(alpha_no_beads, cell_labels_no_beads, exclude_substring=None)

    fig_sens_cell, ax_sens_cell = plt.subplots(figsize=(6, 4.5))
    plot_threshold_sensitivity(ax_sens_cell, alpha_no_beads, cell_labels_no_beads, colors_no_beads,
                                exclude_substring=None)
    plt.tight_layout()
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    fig_sens_cell.savefig(OUTPUT_FIGURE_BASE / "Figure_Supp_ConfinementThresholdSensitivity_CellTypes.png",
                           dpi=600, bbox_inches='tight')
    print("Figure_Supp_ConfinementThresholdSensitivity_CellTypes.png saved.")
    find_stable_threshold_range(alpha_no_beads, cell_labels_no_beads, target_threshold=0.4)
    find_all_stable_plateaus(alpha_no_beads, cell_labels_no_beads, bounds=(0.1, 0.9))

    # SUPPLEMENTARY FIGURE (Figure_Supp_MSDLagtimeSensitivity_CellTypes.png)
    msd_lagtime_sensitivity_table(results_dict, cell_keys, cell_labels)

    fig_lagsens_cell, ax_lagsens_cell = plt.subplots(figsize=(6, 4.5))
    plot_msd_lagtime_sensitivity(ax_lagsens_cell, results_dict, cell_keys, cell_labels, cell_colors)
    plt.tight_layout()
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    fig_lagsens_cell.savefig(OUTPUT_FIGURE_BASE / "Figure_Supp_MSDLagtimeSensitivity_CellTypes.png",
                              dpi=600, bbox_inches='tight')
    print("Figure_Supp_MSDLagtimeSensitivity_CellTypes.png saved.")

    # --- PANEL C: Dapp Violin Plot | PANEL D: MSD @ 2s Violin Plot | PANEL E: MSD @ 8s Violin Plot ---
    # Three amplitude/robustness panels side by side: Dapp is the headline
    # fit-derived quantity; MSD@5s and MSD@8s are both read directly off
    # the corrected MSD curve (not fit, so immune to the Dapp-alpha
    # fit-coupling -- see Figure_Supp_DappAlphaCoupling_CellTypes.png) at
    # two different extraction times, letting the reader see directly
    # whether the amplitude comparison holds up as the lag time increases
    # (complementing the quantitative version of this same question in
    # Figure_Supp_MSDLagtimeSensitivity_CellTypes.png).

    log_violin(ax_dapp, dapp_cell, cell_x_ticks, cell_colors)
    
    # Add significance bars
    add_sig_bars_dapp(ax_dapp, dapp_cell, cell_labels)
    
    ax_dapp.yaxis.set_ticks_position('both')
    ax_dapp.xaxis.set_ticks_position('both')
    for spine in ax_dapp.spines.values():
        spine.set_linewidth(1.5)
    ax_dapp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_dapp.set_xticks(cell_x_ticks)
    ax_dapp.set_xticklabels(cell_labels, rotation=45, ha='right', fontsize=12)
    ax_dapp.set_ylabel(r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)', fontsize=14)
    ax_dapp.text(-0.15, 1.05, 'C', transform=ax_dapp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_dapp.grid(False)

    # msd_at_5s_cell was already computed above, in the STATISTICAL ANALYSIS
    # - CELL TYPES - MSD AT 5s section -- reused here rather than recomputed.
    log_violin(ax_msd2s, msd_at_2s_cell, cell_x_ticks, cell_colors)

    add_sig_bars_dapp(ax_msd2s, msd_at_2s_cell, cell_labels)

    ax_msd2s.yaxis.set_ticks_position('both')
    ax_msd2s.xaxis.set_ticks_position('both')
    for spine in ax_msd2s.spines.values():
        spine.set_linewidth(1.5)
    ax_msd2s.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd2s.set_xticks(cell_x_ticks)
    ax_msd2s.set_xticklabels(cell_labels, rotation=45, ha='right', fontsize=12)
    ax_msd2s.set_ylabel(r'MSD at $\mathrm{\Delta t}$ = 2s ($\mathrm{\mu}$m$^2$)', fontsize=14)
    ax_msd2s.text(-0.15, 1.05, 'D', transform=ax_msd2s.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_msd2s.grid(False)

    # msd_at_8s_cell was already computed above, in the STATISTICAL ANALYSIS
    # - CELL TYPES - MSD AT 8s section -- reused here rather than recomputed.
    log_violin(ax_msd8s, msd_at_8s_cell, cell_x_ticks, cell_colors)

    add_sig_bars_dapp(ax_msd8s, msd_at_8s_cell, cell_labels)

    ax_msd8s.yaxis.set_ticks_position('both')
    ax_msd8s.xaxis.set_ticks_position('both')
    #ax_msd8s.yaxis.set_label_position("both")
    #ax_msd8s.yaxis.tick_right()
    for spine in ax_msd8s.spines.values():
        spine.set_linewidth(1.5)
    ax_msd8s.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd8s.set_xticks(cell_x_ticks)
    ax_msd8s.set_xticklabels(cell_labels, rotation=45, ha='right', fontsize=12)
    ax_msd8s.set_ylabel(r'MSD MSD at $\mathrm{\Delta t}$ = 8s ($\mathrm{\mu}$m$^2$)', fontsize=14)
    ax_msd8s.text(-0.15, 1.05, 'E', transform=ax_msd8s.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_msd8s.grid(False)

    plt.tight_layout()
    fig6.savefig(OUTPUT_FIGURE_BASE / "Figure_6.png", dpi=600, bbox_inches='tight')
    print(f"Saved: Figure_6.png")
    
    
    # ============================================================================
    # CALCULATED RESULTS SUMMARY TABLE
    # ============================================================================
    print("\n" + "="*80)
    print("CALCULATED RESULTS SUMMARY (Long-time regime 2-8s)")
    print("="*80)
    
    # Filter out 'beads' if present, as it was not in the primary DNA table
    data_keys = [k for k in cell_keys if 'beads' not in k.lower()]
    
    print("\n{:<10} {:>8} {:>10} {:>12} {:>8} {:>8} {:>8}".format(
        "DNA Size", "N", "Med α", "Med Dapp", "Caged %", "Subdiff %", "Superdiff %"))
    print("-" * 74)
    
    for key in data_keys:
        calc = results_dict[key]
        
        # Calculate percentages from alpha2 data (confinement threshold from
        # msd_core.classify_diffusion_regime -- see threshold sensitivity
        # analysis for justification of this specific value)
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
    # SUPPLEMENTARY FIGURE (Figure_ATP_Depletion.png): ATP Depletion Analysis (2x2 Layout, panel D split into D+E)
    # ============================================================================
    print("\nGenerating New ATP Depletion Figure (2x2 Layout)...")
    
    # Define the keys for ATP analysis
    atp_keys = ['ATP depleted', 'Control']
    atp_labels = [LABELS[CONDITION_KEYS.index(k)] for k in atp_keys]
    atp_colors = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in atp_keys]
    atp_x_ticks = list(range(1, len(atp_keys) + 1))
    
    # Extract data
    alpha_atp = [results_dict[k]['alpha2'] for k in atp_keys]
    dapp_atp = [results_dict[k]['Dapp2'] for k in atp_keys]

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - ATP DEPLETION - ALPHA (2-8s)")
    print("="*80)
    check_normality(alpha_atp, atp_labels)
    check_equal_variance(alpha_atp, atp_labels)
    run_welch_games_howell(alpha_atp, atp_labels)

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - ATP DEPLETION - Dapp (2-8s), log10-transformed")
    print("="*80)
    log_dapp_atp = log10_safe(dapp_atp)
    check_normality(log_dapp_atp, atp_labels)
    check_equal_variance(log_dapp_atp, atp_labels)
    run_welch_games_howell(log_dapp_atp, atp_labels)

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - ATP DEPLETION - MSD AT 8s (directly measured, not fit-derived)")
    print("="*80)
    msd_at_8s_atp = [extract_msd_at_lagtime(results_dict[k]['msd_corrected'], 8.0, FPS)
                      for k in atp_keys]
    log_msd_at_8s_atp = log10_safe(msd_at_8s_atp)
    check_normality(log_msd_at_8s_atp, atp_labels)
    check_equal_variance(log_msd_at_8s_atp, atp_labels)
    run_welch_games_howell(log_msd_at_8s_atp, atp_labels)
    
    # Create figure -- GridSpec instead of plt.subplots(2,2), same reasoning
    # as Figure 6 above, so panel D can be split into Dapp + MSD@8s.
    fig_atp = plt.figure(figsize=(3.375*2*2, 3.375*2*2))
    gs_atp = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # --- PANEL A: MSD Ensemble ---
    ax_msd_atp = fig_atp.add_subplot(gs_atp[0, 0])
    for key in atp_keys:
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_corrected']
        n_tracks = data.get('n_tracks', 0)
        label_with_n = f"{LABELS[original_index]} \n$n$ = {n_tracks}"
        marker = CELL_LINE_MARKERS.get(key, 'NO MARKER')
        ax_msd_atp.plot(
            ensemble_msd.index,
            ensemble_msd.values,
            color=PLOT_COLORS[original_index],
            marker=marker,
            markerfacecolor=PLOT_COLORS[original_index],
            linestyle='-',
            linewidth=2,
            markersize=7,
            label=label_with_n,
            zorder=3
        )
    
    ax_msd_atp.set_xscale('log')
    ax_msd_atp.set_yscale('log')
    ax_msd_atp.set_ylim((2e-4, 1e-1))
    ax_msd_atp.set_xlim((1e-1, 1e1))
    ax_msd_atp.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle$ ($\mu\mathrm{m}^2$)', fontsize=14)
    ax_msd_atp.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax_msd_atp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd_atp.grid(True, alpha=0.3, linestyle='--', zorder=2)
    for spine in ax_msd_atp.spines.values():
        spine.set_linewidth(1.5)
    leg = ax_msd_atp.legend(
        bbox_to_anchor=(0.8, 0.35),
        fontsize=14,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.8,
        borderpad=0,
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(2)
        handle.set_markersize(7)
    ax_msd_atp.text(-0.15, 1.05, 'A', transform=ax_msd_atp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_msd_atp.grid(False)
    # --- PANEL B: Alpha Violin Plot (ATP) ---
    ax_alpha_atp = fig_atp.add_subplot(gs_atp[0, 1])
    
    # --- Create violin plot ---
    vp_atp = ax_alpha_atp.violinplot(
        alpha_atp,
        positions=atp_x_ticks,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # --- Color violin bodies ---
    for i, body in enumerate(vp_atp['bodies']):
        body.set_facecolor(atp_colors[i])
        body.set_alpha(0.4)
        body.set_edgecolor('black')
        body.set_linewidth(1.5)
    
    # --- Add median lines manually ---
    for i, data_arr in enumerate(alpha_atp):
        med = np.median(data_arr)
        ax_alpha_atp.plot(
            [atp_x_ticks[i] - 0.2, atp_x_ticks[i] + 0.2],
            [med, med],
            color='black',
            linewidth=2,
            zorder=10
        )
    
    # --- Scatter data points ---
    jitter = 0.05
    for i, data_arr in enumerate(alpha_atp):
        ax_alpha_atp.scatter(
            np.full_like(data_arr, atp_x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(data_arr)),
            data_arr,
            color=atp_colors[i],
            alpha=1.0,
            marker='o',
            edgecolor='k',
            s=25,
            zorder=10
        )
    
    # --- Significance bars ---
    add_sig_bars_alpha(ax_alpha_atp, alpha_atp, atp_labels)
    
    # --- Reference lines ---
    ax_alpha_atp.axhline(
        1.0, color='gray', linestyle='--',
        linewidth=1.5, alpha=0.7,
        label=r'Normal diffusion ($\alpha=1$)', zorder=0
    )
    ax_alpha_atp.axhline(
        0.4, color='darkgray', linestyle=':',
        linewidth=1.5, alpha=0.7,
        label=r'Threshold for confinement ($\alpha<0.4$)', zorder=0
    )
    
    # --- Axis formatting ---
    ax_alpha_atp.set_ylim(-0.08, 2.5)
    ax_alpha_atp.set_xticks(atp_x_ticks)
    ax_alpha_atp.set_xticklabels(atp_keys, rotation=45, ha='right', fontsize=12)
    ax_alpha_atp.set_ylabel(r'$\alpha$', fontsize=14)
    ax_alpha_atp.grid(False)
    ax_alpha_atp.tick_params(which="both", axis="both", direction="in",
                             labelsize=12, length=6, width=1.5)
    
    for spine in ax_alpha_atp.spines.values():
        spine.set_linewidth(1.5)
    
    # --- Panel label ---
    ax_alpha_atp.text(
        -0.15, 1.05, 'B',
        transform=ax_alpha_atp.transAxes,
        fontsize=16, fontweight='bold',
        va='top', ha='left'
    )

    # --- PANEL C: Percentage Bar Chart ---
    ax_pct_atp = fig_atp.add_subplot(gs_atp[1, 0])
    
    # Calculate percentages (excluding beads)
    atp_keys_no_beads = ['ATP depleted', 'Control']
    atp_labels_no_beads = [LABELS[CONDITION_KEYS.index(k)] for k in atp_keys_no_beads]
    alpha_atp_no_beads = [results_dict[k]['alpha2'] for k in atp_keys_no_beads]
    
    confined_atp, subdiff_atp, superdiff_atp = [], [], []
    for arr in alpha_atp_no_beads:
        c, s, sd = classify_diffusion_regime(arr)
        confined_atp.append(c)
        subdiff_atp.append(s)
        superdiff_atp.append(sd)

    x_atp = np.arange(len(atp_labels_no_beads))
    width = 0.25

    colors_no_beads = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in atp_keys_no_beads]

    bar1_atp = ax_pct_atp.bar(x_atp - width, confined_atp,   width, color=colors_no_beads, 
                    edgecolor='black', linewidth=1.2, hatch='///', alpha=0.8)
    bar2_atp = ax_pct_atp.bar(x_atp,          subdiff_atp,   width, color=colors_no_beads,
                    edgecolor='black', linewidth=1.2, hatch='',   alpha=0.8)
    bar3_atp = ax_pct_atp.bar(x_atp + width,  superdiff_atp, width, color= colors_no_beads,
                    edgecolor='black', linewidth=1.2, hatch='...', alpha=0.8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFFFFF', edgecolor='black', hatch='///',  label=r'Confined ($\leq$0.4)'),
        Patch(facecolor='#FFFFFF', edgecolor='black', hatch='',     label=r'Subdiffusive (0.4<$\alpha$<1)'),
        Patch(facecolor='#FFFFFF', edgecolor='black', hatch='...',  label='Superdiffusive (>1)'),
    ]
    
    
    ax_pct_atp.set_xticks(x_atp)
    ax_pct_atp.set_xticklabels(atp_keys_no_beads, rotation=45, ha='right', fontsize=12)
    ax_pct_atp.set_ylabel('Percentage (%)', fontsize=14)
    ax_pct_atp.set_ylim(0, 100)
    ax_pct_atp.legend(handles=legend_elements, frameon=False, fontsize=14)
    ax_pct_atp.yaxis.set_ticks_position('both')
    ax_pct_atp.xaxis.set_ticks_position('both')
    for spine in ax_pct_atp.spines.values():
        spine.set_linewidth(1.5)
    ax_pct_atp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_pct_atp.grid(False)
    ax_pct_atp.text(-0.15, 1.05, 'C', transform=ax_pct_atp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

    # ============================================================================
    # SUPPLEMENTARY FIGURE (Figure_Supp_ConfinementThresholdSensitivity_ATP.png)
    # CONFINEMENT THRESHOLD SENSITIVITY ANALYSIS - ATP DEPLETION
    # ============================================================================
    threshold_sensitivity_table(alpha_atp_no_beads, atp_labels_no_beads, exclude_substring=None)

    fig_sens_atp, ax_sens_atp = plt.subplots(figsize=(6, 4.5))
    plot_threshold_sensitivity(ax_sens_atp, alpha_atp_no_beads, atp_labels_no_beads, colors_no_beads,
                                exclude_substring=None)
    plt.tight_layout()
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    fig_sens_atp.savefig(OUTPUT_FIGURE_BASE / "Figure_Supp_ConfinementThresholdSensitivity_ATP.png",
                          dpi=600, bbox_inches='tight')
    print("Figure_Supp_ConfinementThresholdSensitivity_ATP.png saved.")


    # --- PANEL D: Dapp Violin Plot + PANEL E: MSD @ 8s Violin Plot ---
    gs_atp_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_atp[1, 1], wspace=0.6)
    ax_dapp_atp = fig_atp.add_subplot(gs_atp_d[0])
    ax_msd8s_atp = fig_atp.add_subplot(gs_atp_d[1])
    
    log_violin(ax_dapp_atp, dapp_atp, atp_x_ticks, atp_colors)
    
    add_sig_bars_dapp(ax_dapp_atp, dapp_atp, atp_labels)
    
    ax_dapp_atp.yaxis.set_ticks_position('both')
    ax_dapp_atp.xaxis.set_ticks_position('both')
    for spine in ax_dapp_atp.spines.values():
        spine.set_linewidth(1.5)
    ax_dapp_atp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_dapp_atp.set_xticks(atp_x_ticks)
    ax_dapp_atp.set_xticklabels(atp_keys_no_beads, rotation=45, ha='right', fontsize=12)
    ax_dapp_atp.set_ylabel(r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)', fontsize=14)
    ax_dapp_atp.text(-0.15, 1.05, 'D', transform=ax_dapp_atp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_dapp_atp.grid(False)

    # msd_at_8s_atp was already computed above, in the STATISTICAL ANALYSIS -
    # ATP DEPLETION - MSD AT 8s section -- reused here rather than recomputed.
    log_violin(ax_msd8s_atp, msd_at_8s_atp, atp_x_ticks, atp_colors)

    add_sig_bars_dapp(ax_msd8s_atp, msd_at_8s_atp, atp_labels)

    ax_msd8s_atp.yaxis.set_ticks_position('both')
    ax_msd8s_atp.xaxis.set_ticks_position('both')
    # ax_msd8s_atp.yaxis.set_label_position("right")
    # ax_msd8s_atp.yaxis.tick_right()
    for spine in ax_msd8s_atp.spines.values():
        spine.set_linewidth(1.5)
    ax_msd8s_atp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd8s_atp.set_xticks(atp_x_ticks)
    ax_msd8s_atp.set_xticklabels(atp_keys_no_beads, rotation=45, ha='right', fontsize=12)
    ax_msd8s_atp.set_ylabel(r'MSD at $\mathrm{\Delta t}$ = 8s ($\mathrm{\mu}$m$^2$)', fontsize=14)
    ax_msd8s_atp.text(-0.15, 1.05, 'E', transform=ax_msd8s_atp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_msd8s_atp.grid(False)
    
    plt.tight_layout()
    fig_atp.savefig(OUTPUT_FIGURE_BASE / "Figure_ATP_Depletion.png", dpi=600, bbox_inches='tight')
    print(f"Saved: Figure_ATP_Depletion.png")
    
    print("\n" + "="*80)
    print("UPDATED FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    
    # ============================================================================
    # CALCULATED RESULTS SUMMARY TABLE
    # ============================================================================
    print("\n" + "="*80)
    print("CALCULATED RESULTS SUMMARY (Long-time regime 2-8s)")
    print("="*80)
    
    # Filter out 'beads' if present, as it was not in the primary DNA table
    data_keys = [k for k in atp_keys if 'beads' not in k.lower()]
    
    print("\n{:<10} {:>8} {:>10} {:>12} {:>8} {:>8} {:>8}".format(
        "DNA Size", "N", "Med α", "Med Dapp", "Caged %", "Subdiff %", "Superdiff %"))
    print("-" * 74)
    
    for key in data_keys:
        calc = results_dict[key]
        
        # Calculate percentages from alpha2 data (confinement threshold from
        # msd_core.classify_diffusion_regime -- see threshold sensitivity
        # analysis for justification of this specific value)
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

if __name__ == '__main__':
    run_analysis()