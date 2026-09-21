"""
MSD Analysis with Per-Particle Localization Correction
Applies individual sigma corrections based on particle brightness
Additions: noise floor filtering (bead ensemble MSD) + violin plots
@author: sophiedeboer (Refactored by Anara)
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
)

# ============================================================================
# CONFIGURATION 
# ============================================================================
BASE_DATA_ROOT = Path('D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data')

CONDITIONS_MAP = [
    # YOYO-1 (dim): localization correction + mass scaling off - see
    # 05b_msd_analysis_cellular.py. Correction proved unreliable for
    # this dye.
    { 'key': 'MDA MB 231', 'size': 'MDA MB 231', 'base_folder': 'Aswin Data/Aswin Data MDA-MB-231 Corrected for Drift', 'collected_dir': 'Collected_MDAMB231', 'max_length': 350, 'use_localization_correction': False, 'use_mass_scaling': False },
    { 'key': 'MCF 7', 'size': 'MCF 7', 'base_folder': 'Aswin Data/Aswin Data MCF 7 Corrected for Drift', 'collected_dir': 'Collected_MCF7', 'max_length': 300, 'use_localization_correction': False, 'use_mass_scaling': False },
    { 'key': 'MCF 10A', 'size': 'MCF 10A', 'base_folder': 'Aswin Data/Aswin Data MCF 10A Corrected for Drift', 'collected_dir': 'Collected_MCF10A', 'max_length': 150, 'use_localization_correction': False, 'use_mass_scaling': False },
    # Cy3 (bright): correction validated fine.
    { 'key': 'ATP depleted', 'size': 'LifeActGFP Cy3 ATP depleted', 'base_folder': 'MDA-MB-231/ATP Depletion MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_ATPdepl_MDAMB231', 'max_length': 750, 'use_localization_correction': True, 'use_mass_scaling': True },
    { 'key': 'Control', 'size': 'LifeActGFP Cy3 Control', 'base_folder': 'MDA-MB-231/Cy3 MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_Cy3_MDAMB231', 'max_length': 650, 'use_localization_correction': True, 'use_mass_scaling': True },
    { 'key': 'beads', 'size': 'beads', 'base_folder': 'Control Experiments_Beads/0.22 um', 'collected_dir': 'Collected_0.22um', 'max_length': 800, 'use_localization_correction': False, 'use_mass_scaling': False },
    { 'key': 'MDA MB 231 WT', 'size': 'MDA MB 231 WT', 'base_folder': 'MDA-MB-231/Cy3 MDA-MB-231 WT/21.11.2025', 'collected_dir': 'Collected_MDAMB231_WT', 'max_length': 750, 'use_localization_correction': True, 'use_mass_scaling': True }
]

CELL_LINE_MARKERS = {
    'MDA MB 231': 'D',
    'MCF 10A': 's',
    'MCF 7': '^',
    'ATP depleted': 'v',
    'Control': 'o',
    'MDA MB 231 WT': '*' ,
    
}

CONDITION_KEYS   = [c['key'] for c in CONDITIONS_MAP]
MAX_TRAJ_LENGTHS = [c['max_length'] for c in CONDITIONS_MAP]

CONDITION_DICT = {c['key']: c for c in CONDITIONS_MAP}

SIGMA_STUCK_MICRONS  = 0.021244
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS    = 0.014008   # YOYO-1
MASS_DIM_REFERENCE   = 589.8

SIGMA_CY3_MICRONS    = 0.013078   # Cy3
MASS_CY3_REFERENCE   = 8922.3     # median mass from Cy3 (Bright Ref) dataset; reference for mass scaling

MPP         = 0.16
FPS         = 10
MAX_LAGTIME = 100

# Localization correction / mass scaling are now set PER CONDITION in
# CONDITIONS_MAP above, not globally: OFF for the YOYO-1 conditions,
# ON for the Cy3 conditions.
R_SQUARED_THRESHOLD         = 0.8

# Fraction of lag times a trajectory must be ABOVE the noise floor to be kept
NOISE_FLOOR_FRACTION = 0.98

LABELS       = [c['size'] for c in CONDITIONS_MAP]
PLOT_COLORS = ['#533AB7', '#0F6E56', '#EF9F27', '#888780', '#D85A30', '#000000', '#C2456A']
OUTPUT_FIGURE_BASE = BASE_DATA_ROOT / 'Figures'

# ============================================================================
# CORE MSD AND FITTING FUNCTIONS
# Now imported from msd_core.py (single source of truth across all scripts) --
# see the import block at the top of this file.
# ============================================================================

# ============================================================================
# PLOTTING HELPERS
# ============================================================================
def plot_msd_ensemble(ax, results_dict, keys_to_plot, condition_keys, plot_colors, labels_list, mkr, panel_label): 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim((2e-3, 1e-1))
    ax.set_xlim((1e-1, 1e1))
    handles, labels_output = [], []
    for i, key in enumerate(keys_to_plot):
        if key in results_dict:
            try:
                original_index = condition_keys.index(key) 
            except ValueError:
                print(f"Error: Key '{key}' not found in condition_keys.")
                continue
            data  = results_dict[key]
            emsd  = data['ensemble_msd_corrected']
            label = f"{labels_list[original_index]} \n$n$ = {data.get('n_tracks', 0)}"
            line, = ax.plot(
                emsd.index, emsd.values,
                color=plot_colors[original_index],
                marker=CELL_LINE_MARKERS.get(key, 'D'),
                markerfacecolor=plot_colors[original_index],
                linestyle='-', linewidth=2, markersize=7, label=label, zorder=3
            )
            handles.append(line)
            labels_output.append(label)
    ax.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle$ ($\mu\mathrm{m}^2$)', fontsize=14)
    ax.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=2)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax.legend(handles, labels_output, loc='upper left', frameon=False, fontsize=13, ncol=1) 
    ax.grid(False)


def add_sig_bars_alpha(ax, data_list, labels=None):
    """Significance bars for alpha violin plots. Skips beads if labels provided.
    Uses Welch's ANOVA + Games-Howell post-hoc (family-wise error rate
    corrected, does not assume equal variance) rather than uncorrected
    pairwise t-tests or Tukey HSD."""
    y_max = 0
    for data in data_list:
        if len(data) > 0:
            y_max = max(y_max, np.max(data))
    current_y_max = max(ax.get_ylim()[1], y_max)
    initial_offset = current_y_max + 0.1
    stack_spacing  = 0.1
    # run_welch_games_howell needs labels to know which group to exclude
    # ('beads'); if none were given, fall back to placeholders so nothing
    # gets excluded (matches this function's original behavior when
    # labels=None).
    labels_for_gh = labels if labels is not None else [str(i) for i in range(len(data_list))]
    gh = run_welch_games_howell(data_list, labels_for_gh, exclude_substring='beads', verbose=False)
    p_values = [(p, i, j) for (i, j), p in gh['pairwise_pvalues'].items()]
    p_values.sort(key=lambda x: x[0], reverse=True)
    for k, (p_val, i, j) in enumerate(p_values):
        sig = 'ns'
        if p_val < 0.0001: sig = '****'
        elif p_val < 0.001: sig = '***'
        elif p_val < 0.01:  sig = '**'
        elif p_val < 0.05:  sig = '*'
        h = initial_offset + k * stack_spacing
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color='k', clip_on=False)
        ax.text((i + 1 + j + 1) / 2, h + stack_spacing * 0.1, sig,
                ha='center', va='bottom', fontsize=10, color='k')
    if p_values:
        ax.set_ylim(ax.get_ylim()[0], initial_offset + len(p_values) * stack_spacing + 0.1)


def plot_alpha_violin(ax, results_dict, keys_to_plot, condition_keys,
                      labels_map, plot_colors, jitter=0.05, panel_label=None):
    """Violin plot of alpha — drop-in replacement for plot_alpha_boxplot."""
    data_list, actual_labels, actual_indices = [], [], []
    for k in keys_to_plot:
        if k in results_dict:
            try:
                idx = condition_keys.index(k)
            except ValueError:
                continue
            data_list.append(results_dict[k]['alpha2'])
            actual_labels.append(labels_map[idx])
            actual_indices.append(idx)

    if not data_list:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center')
        ax.set_ylim(-0.08, 2.5)
        return

    x_ticks = list(range(1, len(data_list) + 1))

    # Violin bodies
    parts = ax.violinplot(data_list, positions=x_ticks,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plot_colors[actual_indices[i]])
        pc.set_alpha(0.4)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Median lines + scatter
    for i, data_arr in enumerate(data_list):
        med = np.median(data_arr)
        ax.plot([x_ticks[i] - 0.2, x_ticks[i] + 0.2], [med, med],
                color='black', linewidth=2, zorder=10)
        ax.scatter(
            np.full_like(data_arr, x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(data_arr)),
            data_arr,
            color=plot_colors[actual_indices[i]], alpha=1.0,
            marker='o', edgecolor='k', s=25, zorder=5
        )

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - DYE COMPARISON - ALPHA (2-8s)")
    print("="*80)
    check_normality(data_list, actual_labels)
    check_equal_variance(data_list, actual_labels)
    run_welch_games_howell(data_list, actual_labels)

    add_sig_bars_alpha(ax, data_list, labels=actual_labels)

    ax.axhline(1.0, color='gray',     linestyle='--', linewidth=1.5, alpha=0.7,
               label=r'Normal diffusion ($\alpha=1$)',         zorder=0)
    ax.axhline(0.4, color='darkgray', linestyle=':',  linewidth=1.5, alpha=0.7,
               label=r'Threshold for confinement ($\alpha<0.4$)', zorder=0)
    ax.set_ylim(-0.08, 2.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(actual_labels, fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=14)
    ax.grid(False)
    ax.tick_params(which='both', axis='both', direction='in', labelsize=12, length=6, width=1.5)
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    if panel_label:
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')


def add_sig_bars_dapp(ax, data_list, actual_labels, alpha_thresh=0.05):
    """Significance bars for Dapp / MSD@8s violin plots (ported from
    06b_msd_analysis_cellular.py so the ATP-style 5-panel figure layout can
    be reused here for the dye comparison). Uses Welch's ANOVA +
    Games-Howell post-hoc (family-wise error rate corrected, does not
    assume equal variance) rather than uncorrected pairwise t-tests or
    Tukey HSD."""
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) * 1.5
    spacing_factor = 2.3
    # Dapp is strictly positive and typically log-normal, so the statistical
    # test itself runs on log10(Dapp) (matches the console diagnostics),
    # even though the bars are still positioned using the raw data's scale,
    # since that's what's actually plotted here.
    log_data_list = log10_safe(data_list, verbose=False)
    gh = run_welch_games_howell(log_data_list, actual_labels, exclude_substring='beads', verbose=False)
    p_vals = [(p, i, j) for (i, j), p in gh['pairwise_pvalues'].items()]
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
    """Violin plot on a log y-axis for Dapp / MSD@8s (ported from
    05b_msd_analysis_cellular.py)."""
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
    print(f"Sigma (from stuck beads): {SIGMA_STUCK_MICRONS:.6f} μm")
    print(f"R² threshold:             {R_SQUARED_THRESHOLD}")
    print(f"Noise floor fraction:     {NOISE_FLOOR_FRACTION}")
    print(f"\nLocalization correction / mass scaling, per condition:")
    for c in CONDITIONS_MAP:
        print(f"  {c['size']:<24} correction={'ON ' if c['use_localization_correction'] else 'OFF'}"
              f"   mass_scaling={'ON' if c['use_mass_scaling'] else 'OFF'}")
    print()
    
    results_dict = {}
    directories  = {}
    
    for condition in CONDITIONS_MAP:
        key  = condition['key']
        path = BASE_DATA_ROOT / condition['base_folder'] / 'E_output_data' / condition['collected_dir'] / 'tm'
        print(f"Path for {key}: {path}")
        directories[key] = path

    # -------------------------------------------------------------------------
    # STEP 1: Process beads first to establish noise floor
    # -------------------------------------------------------------------------
    noise_floor_msd = None
    if 'beads' in directories and directories['beads'].is_dir():
        print("\nProcessing beads (noise floor)...")
        bead_result = process_directory(
            directories['beads'], MPP, FPS, MAX_LAGTIME,
            sigma_ref=0.0, mass_ref=MASS_STUCK_REFERENCE,
            max_length=MAX_TRAJ_LENGTHS[CONDITION_KEYS.index('beads')],
            use_scaling_flag=False,
            use_localization_correction=CONDITION_DICT['beads']['use_localization_correction'],
            r_squared_threshold=R_SQUARED_THRESHOLD,
            noise_floor_fraction=NOISE_FLOOR_FRACTION,
            noise_floor_msd=None,   # no filtering for beads themselves
        )
        if bead_result is not None:
            results_dict['beads'] = bead_result
            noise_floor_msd       = bead_result['ensemble_msd_corrected']
            print(f"  Beads: {bead_result['n_tracks']} tracks | "
                  f"mean noise floor = {noise_floor_msd.mean():.4e} μm²")
        else:
            print("  WARNING: no valid bead trajectories — noise floor filtering disabled.")
    else:
        print("WARNING: bead directory not found — noise floor filtering disabled.")

    # -------------------------------------------------------------------------
    # STEP 2: Process all other conditions with noise floor filtering
    # -------------------------------------------------------------------------
    for i, label_key in enumerate(CONDITION_KEYS):
        if label_key == 'beads':
            continue
        if label_key not in directories:
            continue

        directory  = directories[label_key]
        max_length = MAX_TRAJ_LENGTHS[i]
        use_loc_corr   = CONDITION_DICT[label_key]['use_localization_correction']
        use_mass_scale = CONDITION_DICT[label_key]['use_mass_scaling']

        if label_key in ['MDA MB 231', 'MCF 10A', 'MCF 7']: 
            sigma_to_use  = SIGMA_DIM_MICRONS
            mass_to_use   = MASS_DIM_REFERENCE
            use_scaling   = use_mass_scale and use_loc_corr
        else: 
            sigma_to_use  = SIGMA_CY3_MICRONS
            mass_to_use   = MASS_CY3_REFERENCE
            use_scaling   = use_mass_scale and use_loc_corr

        if not directory.is_dir():
            print(f"CRITICAL: directory not found at {directory}")
            continue

        print(f"\nProcessing {label_key}... correction={'ON' if use_loc_corr else 'OFF'}, "
              f"mass_scaling={'ON' if use_scaling else 'OFF'}")
        result = process_directory(
            directory, MPP, FPS, MAX_LAGTIME,
            sigma_to_use, mass_to_use, max_length, use_scaling,
            use_localization_correction=use_loc_corr,
            r_squared_threshold=R_SQUARED_THRESHOLD,
            noise_floor_fraction=NOISE_FLOOR_FRACTION,
            noise_floor_msd=noise_floor_msd,
        )
        if result is not None:
            results_dict[label_key] = result
            print(f"  Tracks accepted: {result['n_tracks']}")
        else:
            print(f"  WARNING: no valid trajectories for {label_key}")

    # -------------------------------------------------------------------------
    # MAIN FIGURE (Figure8_MDA231_vs_Control_2Panel.png): MDA MB 231 vs Control (before aggregation)
    # -------------------------------------------------------------------------
    keys_figure_8          = ['MDA MB 231', 'Control', 'MDA MB 231 WT']
    original_condition_keys = [c['key']  for c in CONDITIONS_MAP]
    
    if all(key in results_dict for key in keys_figure_8):
        fig8_indices = [original_condition_keys.index(k) for k in keys_figure_8]
        fig8_colors  = [PLOT_COLORS[i] for i in fig8_indices]
        labels_fig8  = ['MDAMB231 WT YOYO-1', 'MDAMB231 LifeActGFP Cy3', 'MDAMB231 WT Cy3']
        fig8_mkrs    = ['o'] * len(keys_figure_8)

        fig8, axes8 = plt.subplots(1, 2, figsize=(12, 5)) 
        print("\n" + "="*80)
        print("Saving Figure 8 (MDA MB 231 vs Control)...")

        plot_msd_ensemble(
            ax=axes8[0], results_dict=results_dict,
            keys_to_plot=keys_figure_8, condition_keys=keys_figure_8,
            plot_colors=fig8_colors, labels_list=labels_fig8,
            mkr=fig8_mkrs, panel_label='A'
        )
        plot_alpha_violin(
            ax=axes8[1], results_dict=results_dict,
            keys_to_plot=keys_figure_8, condition_keys=keys_figure_8,
            labels_map=labels_fig8, plot_colors=fig8_colors,
            panel_label='B'
        )
        plt.tight_layout()
        fig8.savefig(OUTPUT_FIGURE_BASE / "Figure8_MDA231_vs_Control_2Panel.png", dpi=600, bbox_inches='tight')
        print("Figure 8 saved.")

    # -------------------------------------------------------------------------
    # DYE COMPARISON FIGURE (Figure_DyeComparison_YOYO1_vs_LifeActGFP.png)
    # Same 5-panel layout (2x2 GridSpec, panel D split into D+E) as the ATP
    # depletion figure in 05b_msd_analysis_cellular.py, but comparing dyes
    # (YOYO-1 vs LifeActGFP Cy3) rather than ATP depletion vs Control, and
    # excluding the 'MDA MB 231 WT' condition.
    # -------------------------------------------------------------------------
    dye_keys   = ['MDA MB 231', 'Control']   # 'MDA MB 231' = YOYO-1, 'Control' = LifeActGFP Cy3
    dye_labels = ['YOYO-1', 'Cy3']
    dye_colors = [PLOT_COLORS[original_condition_keys.index(k)] for k in dye_keys]
    dye_x_ticks = list(range(1, len(dye_keys) + 1))

    if all(key in results_dict for key in dye_keys):
        print("\nGenerating Dye Comparison Figure (2x2 Layout: YOYO-1 vs LifeActGFP)...")

        alpha_dye = [results_dict[k]['alpha2'] for k in dye_keys]
        dapp_dye  = [results_dict[k]['Dapp2']  for k in dye_keys]

        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS - DYE COMPARISON - ALPHA (2-8s)")
        print("="*80)
        check_normality(alpha_dye, dye_labels)
        check_equal_variance(alpha_dye, dye_labels)
        run_welch_games_howell(alpha_dye, dye_labels)

        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS - DYE COMPARISON - Dapp (2-8s), log10-transformed")
        print("="*80)
        log_dapp_dye = log10_safe(dapp_dye)
        check_normality(log_dapp_dye, dye_labels)
        check_equal_variance(log_dapp_dye, dye_labels)
        run_welch_games_howell(log_dapp_dye, dye_labels)

        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS - DYE COMPARISON - MSD AT 8s (directly measured, not fit-derived)")
        print("="*80)
        msd_at_8s_dye = [extract_msd_at_lagtime(results_dict[k]['msd_corrected'], 8.0, FPS)
                          for k in dye_keys]
        log_msd_at_8s_dye = log10_safe(msd_at_8s_dye)
        check_normality(log_msd_at_8s_dye, dye_labels)
        check_equal_variance(log_msd_at_8s_dye, dye_labels)
        run_welch_games_howell(log_msd_at_8s_dye, dye_labels)

        fig_dye = plt.figure(figsize=(3.375*2*2, 3.375*2*2))
        gs_dye = gridspec.GridSpec(2, 2, hspace=0.2, wspace=0.3)

        # --- PANEL A: MSD Ensemble ---
        ax_msd_dye = fig_dye.add_subplot(gs_dye[0, 0])
        plot_msd_ensemble(
            ax=ax_msd_dye, results_dict=results_dict,
            keys_to_plot=dye_keys, condition_keys=dye_keys,
            plot_colors=dye_colors, labels_list=dye_labels,
            mkr=['o'] * len(dye_keys), panel_label='A'
        )
        ax_msd_dye.set_ylim((1e-3, 1e-1))

        # --- PANEL B: Alpha Violin Plot ---
        ax_alpha_dye = fig_dye.add_subplot(gs_dye[0, 1])
        plot_alpha_violin(
            ax=ax_alpha_dye, results_dict=results_dict,
            keys_to_plot=dye_keys, condition_keys=dye_keys,
            labels_map=dye_labels, plot_colors=dye_colors,
            panel_label='B'
        )

        # --- PANEL C: Percentage Bar Chart ---
        ax_pct_dye = fig_dye.add_subplot(gs_dye[1, 0])

        confined_dye, subdiff_dye, superdiff_dye = [], [], []
        for arr in alpha_dye:
            c, s, sd = classify_diffusion_regime(arr)
            confined_dye.append(c)
            subdiff_dye.append(s)
            superdiff_dye.append(sd)

        x_dye = np.arange(len(dye_labels))
        width = 0.25

        ax_pct_dye.bar(x_dye - width, confined_dye,   width, color=dye_colors,
                        edgecolor='black', linewidth=1.2, hatch='///', alpha=0.8)
        ax_pct_dye.bar(x_dye,          subdiff_dye,   width, color=dye_colors,
                        edgecolor='black', linewidth=1.2, hatch='',   alpha=0.8)
        ax_pct_dye.bar(x_dye + width,  superdiff_dye, width, color=dye_colors,
                        edgecolor='black', linewidth=1.2, hatch='...', alpha=0.8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='black', hatch='///',  label=r'Confined ($\leq$0.4)'),
            Patch(facecolor='#FFFFFF', edgecolor='black', hatch='',     label=r'Subdiffusive (0.4<$\alpha$<1)'),
            Patch(facecolor='#FFFFFF', edgecolor='black', hatch='...',  label='Superdiffusive (>1)'),
        ]

        ax_pct_dye.set_xticks(x_dye)
        ax_pct_dye.set_xticklabels(dye_labels, fontsize=12)
        ax_pct_dye.set_ylabel('Percentage (%)', fontsize=14)
        ax_pct_dye.set_ylim(0, 100)
        ax_pct_dye.legend(handles=legend_elements, frameon=False, fontsize=14)
        ax_pct_dye.yaxis.set_ticks_position('both')
        ax_pct_dye.xaxis.set_ticks_position('both')
        for spine in ax_pct_dye.spines.values():
            spine.set_linewidth(1.5)
        ax_pct_dye.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
        ax_pct_dye.grid(False)
        ax_pct_dye.text(-0.15, 1.05, 'C', transform=ax_pct_dye.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

        # --- PANEL D: Dapp Violin Plot + PANEL E: MSD @ 8s Violin Plot ---
        gs_dye_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_dye[1, 1], wspace=0.6)
        ax_dapp_dye  = fig_dye.add_subplot(gs_dye_d[0])
        ax_msd8s_dye = fig_dye.add_subplot(gs_dye_d[1])

        log_violin(ax_dapp_dye, dapp_dye, dye_x_ticks, dye_colors)
        add_sig_bars_dapp(ax_dapp_dye, dapp_dye, dye_labels)

        ax_dapp_dye.yaxis.set_ticks_position('both')
        ax_dapp_dye.xaxis.set_ticks_position('both')
        for spine in ax_dapp_dye.spines.values():
            spine.set_linewidth(1.5)
        ax_dapp_dye.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
        ax_dapp_dye.set_xticks(dye_x_ticks)
        ax_dapp_dye.set_xticklabels(dye_labels, fontsize=12)
        ax_dapp_dye.set_ylabel(r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)', fontsize=14)
        ax_dapp_dye.text(-0.15, 1.05, 'D', transform=ax_dapp_dye.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
        ax_dapp_dye.grid(False)

        log_violin(ax_msd8s_dye, msd_at_8s_dye, dye_x_ticks, dye_colors)
        add_sig_bars_dapp(ax_msd8s_dye, msd_at_8s_dye, dye_labels)

        ax_msd8s_dye.yaxis.set_ticks_position('both')
        ax_msd8s_dye.xaxis.set_ticks_position('both')
        # ax_msd8s_dye.yaxis.set_label_position("right")
        # ax_msd8s_dye.yaxis.tick_right()
        for spine in ax_msd8s_dye.spines.values():
            spine.set_linewidth(1.5)
        ax_msd8s_dye.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
        ax_msd8s_dye.set_xticks(dye_x_ticks)
        ax_msd8s_dye.set_xticklabels(dye_labels, fontsize=12)
        ax_msd8s_dye.set_ylabel(r'MSD at $\mathrm{\Delta t}$ = 8s ($\mathrm{\mu}$m$^2$)', fontsize=14)
        ax_msd8s_dye.text(-0.15, 1.05, 'E', transform=ax_msd8s_dye.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
        ax_msd8s_dye.grid(False)

        plt.tight_layout()
        fig_dye.savefig(OUTPUT_FIGURE_BASE / "Figure_DyeComparison_YOYO1_vs_LifeActGFP.png", dpi=600, bbox_inches='tight')
        print("Figure_DyeComparison_YOYO1_vs_LifeActGFP.png saved.")
    else:
        missing = [k for k in dye_keys if k not in results_dict]
        print(f"WARNING: skipping dye comparison figure -- missing results for {missing}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {OUTPUT_FIGURE_BASE}")
    print("="*80)

if __name__ == '__main__': 
    run_analysis()