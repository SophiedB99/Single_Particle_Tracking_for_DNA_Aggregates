# -*- coding: utf-8 -*-
"""
Van Hove Analysis for Cell Types (MDA MB 231, MCF 7, MCF 10A)
Adapted from base-pair Van Hove script to use cell-type conditions and
sigma corrections matching the cell types MSD analysis script.

Noise floor filtering: beads are processed first to establish an ensemble
corrected MSD noise floor. Any particle trajectory whose corrected MSD does
not exceed the bead floor for at least NOISE_FLOOR_FRACTION of shared lag
times is excluded before its displacements are collected — consistent with
the MSD analysis script.

MSD-acceptance consistency (APPLY_MSD_ACCEPTANCE_FILTER, default True):
a trajectory is also excluded here unless it would ALSO have passed
05b_msd_analysis_cellular.py's long-time (2-8s) power-law fit acceptance
test (R^2 >= R_SQUARED_THRESHOLD and alpha > 0), via
msd_core.check_long_time_fit_acceptance(). Before this was added, the Van
Hove/kurtosis population and the Dapp/alpha/MSD population were two
different sets of trajectories from the same directories -- this analysis
included trajectories the headline MSD analysis had already decided
weren't trustworthy enough to fit. Set APPLY_MSD_ACCEPTANCE_FILTER = False
to recover the old (inconsistent) behavior, e.g. for a direct before/after
comparison of how much this matters.

@author: sophiedeboer (adapted)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Shared MSD core -- see msd_core.py. Guarantees this script's corrected-MSD
# and noise-floor logic are IDENTICAL to 05a/05b/05c/06a/07a/07b.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from msd_core import (
    calculate_sigma_for_particle_from_mass_array as calculate_sigma_for_particle,
    compute_bead_noise_floor,
    trajectory_above_noise_floor,
    check_long_time_fit_acceptance,
    NOISE_FLOOR_FRACTION,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
SIGMA_DIM_MICRONS   = 0.014008    # Dim dye sigma, YOYO-1
MASS_DIM_REFERENCE  = 589.8
SIGMA_BEAD_MICRONS  = 0.0         # Beads: no localization correction
MASS_BEAD_REFERENCE = 3381.5

USE_MASS_SCALING            = False
USE_LOCALIZATION_CORRECTION = False

# --- MSD-acceptance consistency  ---
# These MUST match 05b_msd_analysis_cellular.py's values for the exclusion
# to actually restrict this script to 05b's accepted population.
APPLY_MSD_ACCEPTANCE_FILTER = True
R_SQUARED_THRESHOLD = 0.8

MAX_TRAJ_LENGTHS = [350, 300, 150]   # MDA MB 231, MCF 7, MCF 10A
BEAD_MAX_LENGTH  = 800

CONDITION_KEYS = ['MDA MB 231', 'MCF 7', 'MCF 10A']

MPP         = 0.16
FPS         = 10
MAX_LAGTIME = 100   # max lag time in frames for MSD (used in noise floor check)

# Time lags for the 2x2 panel plot (panels A, B, C, D)
VAN_HOVE_LAG_SECONDS = [0.1, 1.0, 5.0, 8.0]
KURTOSIS_LAG_SECONDS = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0]


BIN_COUNT                          = 50
MAX_DISPLACEMENT                   = 0.3
MIN_DISPLACEMENT_FOR_DIFFUSION_PEAK = 0.01   # µm — threshold to exclude caging/noise peak

BASE_PATH = 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data'

directories = {
    'MDA MB 231': BASE_PATH + '/Aswin Data/Aswin Data MDA-MB-231 Corrected for Drift/E_output_data/Collected_MDAMB231/tm',
    'MCF 7':      BASE_PATH + '/Aswin Data/Aswin Data MCF 7 Corrected for Drift/E_output_data/Collected_MCF7/tm',
    'MCF 10A':    BASE_PATH + '/Aswin Data/Aswin Data MCF 10A Corrected for Drift/E_output_data/Collected_MCF10A/tm',
    'beads':      BASE_PATH + '/Control Experiments_Beads/0.22 um/E_output_data/Collected_0.22um/tm',
}

colors = ['#533AB7', '#0F6E56', '#EF9F27']
labels = ['MDA MB 231', 'MCF 7', 'MCF 10A']

PANEL_LABELS = ['A', 'B', 'C', 'D']
KURTOSIS_PANEL_INDEX = 3   # zero-based index of panel D in the flattened 2×2 grid

# ============================================================================
# SIGMA CORRECTION + NOISE FLOOR
# Now imported from msd_core.py (single source of truth across all scripts)
# ============================================================================


# ============================================================================
# CORE DISPLACEMENT CALCULATION AND CORRECTION
# ============================================================================
def calculate_displacements(pos, time_lags_frames, sigma_particle):
    displacements = {lt: [] for lt in time_lags_frames}
    noise_term_sq = 4 * sigma_particle**2
    for lt in time_lags_frames:
        if lt >= len(pos):
            continue
        diff_vectors    = pos[lt:] - pos[:-lt]
        dr_squared_raw  = diff_vectors[:, 0]**2 + diff_vectors[:, 1]**2
        dr_squared_real = np.maximum(0, dr_squared_raw - noise_term_sq)
        dr_real         = np.sqrt(dr_squared_real)
        displacements[lt].extend(dr_real)
    return displacements


# ============================================================================
# PER-FILE PROCESSING  (with noise floor filtering)
# ============================================================================
def process_file_for_van_hove(filename, mpp, max_length, time_lags_frames,
                               label_key, noise_floor_msd, excluded_particles=None):
    """
    excluded_particles : optional set of 'basename::particle_id' strings
        (matching the key convention used by
        mislinking_density_check_cellular.py's flagged-trajectory output)
        to skip entirely -- used for the mislinking-exclusion sensitivity
        check, i.e. does excluding specific candidate-mislink trajectories
        change the resulting kurtosis curve. None/empty set excludes
        nothing (normal behavior, unchanged from before).

    Returns (all_displacements, n_total, n_rejected, n_excluded, n_failed_msd_acceptance).
    """
    try:
        traj_df = pd.read_csv(filename)
    except Exception:
        return None, 0, 0, 0, 0

    basename = os.path.basename(filename)
    excluded_particles = excluded_particles or set()

    all_displacements = {lt: [] for lt in time_lags_frames}
    n_total    = 0
    n_rejected = 0
    n_excluded = 0
    n_failed_msd_acceptance = 0

    for pid, ptraj in traj_df.groupby('particle'):
        if f"{basename}::{pid}" in excluded_particles:
            n_excluded += 1
            continue

        current_sigma_ref = SIGMA_DIM_MICRONS
        current_mass_ref  = MASS_DIM_REFERENCE

        mass_data = ptraj['mass'].values if 'mass' in ptraj.columns else np.array([])
        sigma_particle_calc = calculate_sigma_for_particle(
            mass_data, current_sigma_ref, current_mass_ref, USE_MASS_SCALING  # now mass-scaled, matching Cy3 treatment
        )
        sigma_particle = sigma_particle_calc if USE_LOCALIZATION_CORRECTION else 0.0

        pos_df = ptraj.set_index('frame')[['x', 'y']] * mpp
        pos    = pos_df.reindex(
            np.arange(pos_df.index[0], 1 + pos_df.index[-1])
        ).to_numpy()
        pos = pos[:max_length]

        if len(pos) < 50:
            continue

        n_total += 1

        if noise_floor_msd is not None:
            if not trajectory_above_noise_floor(
                pos, sigma_particle, FPS, MAX_LAGTIME, noise_floor_msd
            ):
                n_rejected += 1
                continue

        if APPLY_MSD_ACCEPTANCE_FILTER:
            accepted, _alpha2, _r2 = check_long_time_fit_acceptance(
                pos, MAX_LAGTIME, sigma_particle, FPS,
                R_SQUARED_THRESHOLD, USE_LOCALIZATION_CORRECTION,
            )
            if not accepted:
                n_failed_msd_acceptance += 1
                continue

        particle_displacements = calculate_displacements(
            pos, time_lags_frames, sigma_particle
        )
        for lt, dr_list in particle_displacements.items():
            all_displacements[lt].extend(dr_list)

    return all_displacements, n_total, n_rejected, n_excluded, n_failed_msd_acceptance


# ============================================================================
# PER-DIRECTORY PROCESSING
# ============================================================================
def process_directory_for_van_hove(directory, mpp, max_length,
                                    time_lags_frames, label_key,
                                    noise_floor_msd, excluded_particles=None):
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return None

    combined_displacements = {lt: [] for lt in time_lags_frames}
    total_tracks   = 0
    total_rejected = 0
    total_excluded = 0
    total_failed_msd_acceptance = 0

    for filename in os.listdir(directory):
        if not filename.endswith('.csv') or filename.startswith('.'):
            continue
        result, n_total, n_rejected, n_excluded, n_failed_msd_acceptance = process_file_for_van_hove(
            os.path.join(directory, filename),
            mpp, max_length, time_lags_frames, label_key, noise_floor_msd,
            excluded_particles=excluded_particles,
        )
        if result is not None:
            for lt, dr_list in result.items():
                combined_displacements[lt].extend(dr_list)
        total_tracks   += n_total
        total_rejected += n_rejected
        total_excluded += n_excluded
        total_failed_msd_acceptance += n_failed_msd_acceptance

    if not any(combined_displacements.values()):
        print(f"  Warning: No valid displacements found in {directory}.")
        return None

    n_kept = total_tracks - total_rejected - total_failed_msd_acceptance
    excl_note = f" | {total_excluded} excluded (mislinking screen)" if excluded_particles else ""
    msd_note = (f" | {total_failed_msd_acceptance} failed MSD acceptance (R^2/alpha, "
                f"consistency with 06b)" if APPLY_MSD_ACCEPTANCE_FILTER else "")
    print(f"  Trajectories: {total_tracks} total | "
          f"{total_rejected} rejected (noise floor){msd_note} | {n_kept} kept{excl_note}")
    return combined_displacements


# ============================================================================
# EXCESS KURTOSIS CALCULATION  (single lag time, from displacement list)
# ============================================================================
def kurtosis_from_displacements(dr_list, min_count=20):
    """
    Returns the excess (Fisher) kurtosis of the underlying 1D displacement
    components (e.g. Δx, Δy), computed from the radial displacement dr =
    sqrt(Δx^2 + Δy^2) via the exact isotropic-2D relationship

        gamma = 2 * <dr^4> / <dr^2>^2 - 4

    Derivation: for isotropic 2D diffusion, Δx and Δy are i.i.d. and
    zero-mean, so dr^2 = Δx^2 + Δy^2 gives
        <dr^2>  = 2<Δx^2>
        <dr^4>  = 2<Δx^4> + 2<Δx^2>^2
    Combining these with the definition of excess kurtosis of Δx,
    gamma = <Δx^4>/<Δx^2>^2 - 3, gives the formula above exactly (no
    approximation) -- equivalently gamma = 4 * NGP, where
    NGP = <dr^4>/(2<dr^2>^2) - 1 is the standard 2D non-Gaussian
    parameter. Returns 0 for an exactly Gaussian distribution, matching
    the standard excess-kurtosis convention. Returns NaN if insufficient
    data.
    """
    if not dr_list or len(dr_list) < min_count:
        return np.nan
    dr = np.asarray(dr_list, dtype=float)
    dr = dr[np.isfinite(dr) & (dr >= 0)]
    if len(dr) < min_count:
        return np.nan
    dr2      = dr ** 2
    mean_dr2 = np.mean(dr2)
    if mean_dr2 == 0:
        return np.nan
    mean_dr4 = np.mean(dr ** 4)
    return mean_dr4 / (mean_dr2 ** 2) - 3


# ============================================================================
# COMPUTE EXCESS KURTOSIS OVER TIME for all conditions
# Returns: dict  {label_key: (lag_times_array, kurtosis_array)}
# ============================================================================
def compute_kurtosis_time_series(van_hove_results, condition_keys, fps):
    """
    Evaluates excess kurtosis at every lag time stored in van_hove_results.
    Only lags that are explicitly present as keys in the displacement dict
    are used (they were computed during the main processing pass).
    """
    kurtosis_series = {}
    for label_key in condition_keys:
        if label_key not in van_hove_results:
            continue
        displacements = van_hove_results[label_key]   # {lag_frames: [dr, ...]}
        lag_frames_sorted = sorted(displacements.keys())

        lag_t_list  = []
        kurt_list   = []
        for lf in lag_frames_sorted:
            g = kurtosis_from_displacements(displacements[lf])
            if not np.isnan(g):
                lag_t_list.append(lf / fps)
                kurt_list.append(g)

        if lag_t_list:
            kurtosis_series[label_key] = (np.array(lag_t_list), np.array(kurt_list))

    return kurtosis_series


# ============================================================================
# PRINT EXCESS KURTOSIS TABLE  (unchanged structure, formula/labels updated)
# ============================================================================
def compute_and_print_kurtosis(results_vh, condition_labels, labels, lag_seconds, fps):
    print("\n" + "="*90)
    print("EXCESS KURTOSIS ANALYSIS")
    print("gamma = 2 * <dr^4> / <dr^2>^2 - 4   |   Gaussian = 0, heavy tails > 0")
    print("(exact isotropic-2D rescaling of the radial displacement dr; "
          "equivalent to the standard excess kurtosis of the underlying "
          "1D displacement components)")
    print("="*90)

    for lag_time_s in lag_seconds:
        lag_time_f = int(lag_time_s * fps)
        print(f"\nΔt = {lag_time_s:.1f} s")
        print(f"  {'Condition':<12} {'N':>8} {'<dr^2>':>12} {'<dr^4>':>12} {'Kurtosis':>10}")
        print(f"  {'-'*56}")

        for label_key in condition_labels:
            if label_key not in results_vh:
                continue
            dr_list = results_vh[label_key].get(lag_time_f)
            if not dr_list or len(dr_list) < 10:
                continue
            dr = np.array(dr_list, dtype=float)
            dr = dr[np.isfinite(dr) & (dr >= 0)]
            if len(dr) < 10:
                continue
            dr2      = dr ** 2
            mean_dr2 = np.mean(dr2)
            mean_dr4 = np.mean(dr ** 4)
            if mean_dr2 == 0:
                continue
            kurt = 2 * mean_dr4 / (mean_dr2 ** 2) - 4
            original_index = CONDITION_KEYS.index(label_key)
            fragment_label = labels[original_index]
            print(f"  {fragment_label:<12} {len(dr):>8} {mean_dr2:>12.4e} {mean_dr4:>12.4e} {kurt:>10.4f}")

    print("="*90)


# ============================================================================
# PLOTTING: combined 2×2 panel
#   A, B  → Van Hove at Δt = 0.1 s, 2.0 s
#   C     → Excess kurtosis vs. lag time
#   D     → Van Hove at Δt = 8.0 s
# ============================================================================
def plot_van_hove_with_kurtosis(results_vh, condition_labels, colors, labels,
                                 lag_seconds, bin_count, max_disp, fps,
                                 kurtosis_series, save_path=None):
    """
    2×2 figure where panels A, B, D are Van Hove distributions
    and panel C is the excess-kurtosis-vs-Δt plot.
    """
    min_displacement_threshold = MIN_DISPLACEMENT_FOR_DIFFUSION_PEAK

    fig, axes = plt.subplots(2, 2, figsize=(3.375*2*2, 3.375*2*2))
    axes_flat = axes.flatten()   # A=0, B=1, C=2, D=3

    # ------------------------------------------------------------------
    # Determine which axes are Van Hove panels and their Δt values
    # panel indices 0,1,3  →  lag_seconds[0], lag_seconds[1], lag_seconds[2]
    # ------------------------------------------------------------------
    vh_panel_slots = [0, 1, 2]   # axes_flat indices for Van Hove
    vh_lag_map     = {slot: lag_seconds[i] for i, slot in enumerate(vh_panel_slots)}

    # ================================================================
    # PANELS A, B, C  — Van Hove distributions
    # ================================================================
    for slot, lag_time_s in vh_lag_map.items():
        lag_time_f  = int(lag_time_s * fps)
        ax          = axes_flat[slot]
        panel_label = PANEL_LABELS[slot]

        all_plot_data        = {}
        quantified_peaks     = []
        max_peak_disp_global = 0.0
        max_density_global   = 0.0
        min_peak_Gs_global   = np.inf

        for i, label_key in enumerate(condition_labels):
            if label_key not in results_vh:
                continue
            displacements = results_vh[label_key].get(lag_time_f)
            if not displacements:
                continue

            counts, bins = np.histogram(
                displacements, bins=bin_count,
                range=(0, max_disp), density=True
            )
            bin_centers = (bins[:-1] + bins[1:]) / 2

            start_index           = np.searchsorted(bin_centers, min_displacement_threshold)
            counts_diffusion      = counts[start_index:]
            bin_centers_diffusion = bin_centers[start_index:]

            if counts_diffusion.size == 0:
                max_density       = 0.0
                peak_displacement = 0.0
            else:
                max_density_index = np.argmax(counts_diffusion)
                max_density       = counts_diffusion[max_density_index]
                peak_displacement = bin_centers_diffusion[max_density_index]

            n_points = len(displacements)
            max_peak_disp_global = max(max_peak_disp_global, peak_displacement)
            max_density_global   = max(max_density_global, np.max(counts))
            if max_density > 0:
                min_peak_Gs_global = min(min_peak_Gs_global, max_density)

            original_index = CONDITION_KEYS.index(label_key)
            fragment_label = labels[original_index]

            all_plot_data[label_key] = {
                'counts':      counts,
                'bin_centers': bin_centers,
                'color':       colors[original_index],
                'label':       fragment_label,
                'N':           n_points,
                'peak_disp':   peak_displacement,
                'max_Gs':      max_density,
            }
            quantified_peaks.append({
                'Fragment':  fragment_label,
                'N':         n_points,
                'Peak_Disp': peak_displacement,
                'Max_Gs':    max_density,
            })

        
        # Print summary table
        print(f"\nPANEL {panel_label}: Δt = {lag_time_s:.1f} s")
        quantified_peaks.sort(key=lambda x: x['Max_Gs'], reverse=True)
        for peak_data in quantified_peaks:
            print(f"  {peak_data['Fragment']:<8} | N={peak_data['N']:>8} | "
                  f"Peak={peak_data['Peak_Disp']:.4f} µm | Max Gs={peak_data['Max_Gs']:.4f}")

        # Plot on main and inset axes
        for label_key, data in all_plot_data.items():
            ax.plot(data['bin_centers'], data['counts'],
                    color=data['color'], linewidth=2,
                    label=f"{data['label']} (N={data['N']})")
            
        # Format main axes
        ax.set_title(f'$\\Delta t = {lag_time_s:.1f}$ s', fontsize=16)
        ax.set_xlabel(r'$\Delta r$ ($\mu\mathrm{m}$)', fontsize=14)
        ax.set_ylabel(r'$G_s(\Delta r, \Delta t)$', fontsize=14)

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(which="both", axis="both", direction="in",
                       labelsize=10, length=6, width=1.5)
        ax.set_xlim(0, max_disp)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

        # Legend only on panel B
        if panel_label == 'B':
            handles, leg_labels = ax.get_legend_handles_labels()
            ax.legend(handles, [l.split(' (N=')[0] for l in leg_labels],
                      loc='upper right', fontsize=13, frameon=False)

       
        ax.grid(False)

    # ================================================================
    # PANEL D  - Excess kurtosis vs. lag time
    # ================================================================
    ax_kurt     = axes_flat[KURTOSIS_PANEL_INDEX]
    panel_label = PANEL_LABELS[KURTOSIS_PANEL_INDEX]   # 'D'

    print(f"\nPANEL {panel_label}: Excess kurtosis vs. lag time")
    for label_key in condition_labels:
        if label_key not in kurtosis_series:
            continue
        lag_t, kurt_vals = kurtosis_series[label_key]
        original_index   = CONDITION_KEYS.index(label_key)
        frag_label       = labels[original_index]
        color            = colors[original_index]

        ax_kurt.plot(lag_t, kurt_vals,
                     color=color, linewidth=2,
                     marker='o', markersize=4,
                     label=frag_label)
        print(f"  {frag_label}: {len(lag_t)} points, "
              f"kurtosis range [{kurt_vals.min():.3f}, {kurt_vals.max():.3f}]")

    # Gaussian reference line
    ax_kurt.axhline(0, color='k', linewidth=1.0, linestyle='--', alpha=0.6,
                     label='Gaussian (Excess Kurtosis = 0)')

    ax_kurt.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax_kurt.set_ylabel(r'Excess Kurtosis ($\gamma$)', fontsize=14)
    ax_kurt.set_title('Excess Kurtosis', fontsize=16)

    ax_kurt.yaxis.set_ticks_position('both')
    ax_kurt.xaxis.set_ticks_position('both')
    for spine in ax_kurt.spines.values():
        spine.set_linewidth(1.5)
    ax_kurt.tick_params(which="both", axis="both", direction="in",
                        labelsize=10, length=6, width=1.5)
    ax_kurt.grid(True, alpha=0.3, linestyle=':')
    ax_kurt.text(-0.15, 1.05, panel_label, transform=ax_kurt.transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='left')

    # Kurtosis legend inside the panel
    ax_kurt.legend(loc='upper right', fontsize=13, frameon=False)
    ax_kurt.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    plt.show()


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("STEP 3: VAN HOVE + EXCESS KURTOSIS PANEL ANALYSIS (w/ SIGMA CORRECTION + NOISE FLOOR)")
    print(f"Localization Correction  : {'Enabled' if USE_LOCALIZATION_CORRECTION else 'Disabled'}")
    print(f"Mass Scaling             : {'Enabled' if USE_MASS_SCALING else 'Disabled'}")
    print(f"Noise floor fraction     : {NOISE_FLOOR_FRACTION}")
    print(f"Diffusion Peak Threshold : {MIN_DISPLACEMENT_FOR_DIFFUSION_PEAK:.2f} µm")
    print(f"Van Hove panels          : Δt = {VAN_HOVE_LAG_SECONDS} s  (panel D → Excess Kurtosis)")
    print("="*80)

    # Combine all lag times needed: Van Hove lags + kurtosis lags (deduplicated)
    kurtosis_lag_frames = [int(t * FPS) for t in KURTOSIS_LAG_SECONDS]
    vh_lag_frames        = [int(t * FPS) for t in VAN_HOVE_LAG_SECONDS]
    all_lag_frames        = sorted(set(vh_lag_frames + kurtosis_lag_frames))

    # ------------------------------------------------------------------
    # STEP 1: Process beads to establish noise floor
    # ------------------------------------------------------------------
    noise_floor_msd = None
    bead_dir = directories.get('beads')
    if bead_dir and os.path.isdir(bead_dir):
        print("\nProcessing beads (noise floor)...")
        noise_floor_msd = compute_bead_noise_floor(
            bead_dir, MPP, FPS, MAX_LAGTIME, BEAD_MAX_LENGTH
        )
        if noise_floor_msd is None:
            print("  WARNING: No valid bead trajectories — noise floor filtering disabled.")
    else:
        print("WARNING: Bead directory not found — noise floor filtering disabled.")

    # ------------------------------------------------------------------
    # STEP 2: Process each condition using the combined lag frame set
    # ------------------------------------------------------------------
    van_hove_results = {}

    for i, label in enumerate(CONDITION_KEYS):
        if label not in directories:
            continue
        directory  = directories[label]
        max_length = MAX_TRAJ_LENGTHS[i]
        print(f"\nProcessing {label} (max_length={max_length})...")

        if not os.path.exists(directory):
            print(f"  CRITICAL: Directory for {label} not found at {directory}")
            continue

        result = process_directory_for_van_hove(
            directory, MPP, max_length, all_lag_frames, label, noise_floor_msd
        )
        if result is not None:
            van_hove_results[label] = result

    # ------------------------------------------------------------------
    # STEP 2.5: Compute excess kurtosis — console table (at Van Hove lag times)
    # ------------------------------------------------------------------
    processed_keys = [k for k in CONDITION_KEYS if k in van_hove_results]

    compute_and_print_kurtosis(
        van_hove_results, processed_keys,
        labels, VAN_HOVE_LAG_SECONDS + [5.0],   # keep 5 s in table for reference
        FPS
    )

    # ------------------------------------------------------------------
    # STEP 2.6: Compute excess kurtosis time series for panel D
    # ------------------------------------------------------------------
    kurtosis_series = compute_kurtosis_time_series(van_hove_results, processed_keys, FPS)

    # ------------------------------------------------------------------
    # STEP 3: Plot
    # ------------------------------------------------------------------
    processed_keys = [k for k in CONDITION_KEYS if k in van_hove_results]

    if not van_hove_results:
        print("\nERROR: No data collected for Van Hove plots.")
    else:
        plot_van_hove_with_kurtosis(
            van_hove_results, processed_keys,
            colors, labels,
            VAN_HOVE_LAG_SECONDS, BIN_COUNT, MAX_DISPLACEMENT, FPS,
            kurtosis_series,
            save_path=BASE_PATH + '/Figures/vanHove_cellular.png'
        )

    print("\n" + "="*80)
    print("VAN HOVE ANALYSIS COMPLETE")
    print("="*80)