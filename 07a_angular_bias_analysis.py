# -*- coding: utf-8 -*-
"""
Step 3: Van Hove Analysis for Non-Gaussian Diffusion (1x3 Linear Panel Plot)
INCLUDES: Noise floor filtering using bead ensemble corrected MSD,
consistent with the cell types Van Hove and MSD analysis scripts.

ALSO: a summary table (fraction of next-step
turning angles > 90 deg, i.e. anti-persistent/confinement-consistent
turns) for each DNA-size condition, with the bead/noise-floor condition
included as an unfiltered reference row.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2

# Shared MSD core -- see msd_core.py. Guarantees this script's corrected-MSD
# and noise-floor logic are IDENTICAL to 05a/05b/05c/06a/06b/07b.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from msd_core import (
    calculate_sigma_for_particle_from_mass_array as calculate_sigma_for_particle,
    compute_bead_noise_floor,
    trajectory_above_noise_floor,
    check_long_time_fit_acceptance,
    NOISE_FLOOR_FRACTION,
)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
SIGMA_STUCK_MICRONS  = 0.021244
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS    = 0.014008    # YOYO-1
MASS_DIM_REFERENCE   = 589.8

SIGMA_CY3_MICRONS    = 0.013078    # Cy3
MASS_CY3_REFERENCE   = 8922.3      # median mass from Cy3 (Bright Ref) dataset; reference for mass scaling

USE_MASS_SCALING            = True
USE_LOCALIZATION_CORRECTION = True

R_SQUARED_THRESHOLD = 0.8
# MSD-acceptance filter, for consistency with 05a/05b/05c/06a/06b/07b: applies
# the same check_long_time_fit_acceptance() test (R^2 >= R_SQUARED_THRESHOLD
# and alpha > 0) used to decide inclusion in the Dapp/alpha/MSD and Van Hove
# analyses. Set to False to recover the previous behavior (noise-floor
# filtering only). Never applied to the bead/reference condition --
# see is_reference_condition in directional_analysis_per_file.
APPLY_MSD_ACCEPTANCE_FILTER = True

MAX_TRAJ_LENGTHS = [350, 300, 350, 800, 800, 800]
BEAD_MAX_LENGTH  = 800

CONDITION_KEYS = ['100bp', '250bp', '500bp', '4713bp', '7614bp', '9553bp']
DIM_FRAGMENTS  = ['100bp', '250bp', '500bp']
BEAD_LABEL     = 'Beads (noise floor, unfiltered)'

MPP         = 0.16
FPS         = 10
MAX_LAGTIME = 100

Filepath   = 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/'
FilepathOG = 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/'
BASE_PATH  = 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data'

directories = {
    '100bp':  Filepath   + 'Aswin Data CHO Corrected for Drift/100 bp/E_output_data/Collected_100bp/tm',
    '250bp':  Filepath   + 'Aswin Data CHO Corrected for Drift/250 bp/E_output_data/Collected_250bp/tm',
    '500bp':  Filepath   + 'Aswin Data CHO Corrected for Drift/500 bp/E_output_data/Collected_500bp/tm',
    '4713bp': FilepathOG + 'E_output_data/Collected_pEGFP-C1/tm',
    '7614bp': FilepathOG + 'E_output_data/Collected_BPK4410/tm',
    '9553bp': FilepathOG + 'E_output_data/Collected_hCas9D10A/tm',
    'beads':  BASE_PATH  + '/Control Experiments_Beads/0.22 um/E_output_data/Collected_0.22um/tm',
}

colors = ['#6FADE3','#2874C6','#042C53','#FAC775','#BA7517','#412402']
labels = ['100 bp', '250 bp', '500 bp', '4713 bp', '7614 bp', '9553 bp']

# ============================================================================
# SIGMA CORRECTION + NOISE FLOOR
# Now imported from msd_core.py (single source of truth across all scripts) --
# see the import block at the top of this file.
# ============================================================================

# ============================================================================
# ANGLE CALCULATION FUNCTIONS
# ============================================================================
def calculate_next_step_angles(pos, lag_time=1, sigma_particle=0,
                                displacement_multiplier=2.0):
    if len(pos) < 2 * lag_time + 1:
        return np.array([])
    dr1 = pos[lag_time:] - pos[:-lag_time]
    v1  = dr1[:-lag_time]
    v2  = dr1[lag_time:]
    if len(v1) == 0:
        return np.array([])
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    noise_threshold = displacement_multiplier * sigma_particle
    non_noise       = (mag1 >= noise_threshold) & (mag2 >= noise_threshold)
    v1_f, v2_f     = v1[non_noise], v2[non_noise]
    mag1_f, mag2_f = mag1[non_noise], mag2[non_noise]
    if len(v1_f) == 0:
        return np.array([])
    dot_product = np.sum(v1_f * v2_f, axis=1)
    non_zero    = (mag1_f > 1e-9) & (mag2_f > 1e-9)
    dot_product = dot_product[non_zero]
    mag1_f      = mag1_f[non_zero]
    mag2_f      = mag2_f[non_zero]
    if len(dot_product) == 0:
        return np.array([])
    cosine_theta = np.clip(dot_product / (mag1_f * mag2_f), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_theta))


def directional_analysis_per_file(filename, mpp, max_length, lag_time_steps,
                                   label_key, noise_floor_msd,
                                   is_reference_condition=False):
    """
    is_reference_condition : bool
        True for the bead/noise-floor condition itself. Beads use
        sigma=0.0 (no localization correction, matching
        compute_bead_noise_floor()'s convention elsewhere in the
        pipeline) and are exempt from BOTH the noise-floor filter (which
        would otherwise compare the beads' own MSD against a floor built
        from themselves -- circular) and the MSD R^2/alpha acceptance
        filter (beads are not expected to show directed diffusion, so
        this filter would just remove most/all of them). Their turning-
        angle distribution is therefore raw, unfiltered localization
        noise -- this is what makes it a valid "no real correlated
        motion" baseline to compare the DNA conditions against, rather
        than assuming 50% above 90 deg as the null by fiat.
    """
    try:
        traj_df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return {}, 0, 0, 0

    all_angles = {lag_time_steps: []}
    n_total    = 0
    n_rejected = 0
    n_failed_msd_acceptance = 0

    for pid, ptraj in traj_df.groupby('particle'):
        if not all(col in ptraj.columns for col in ['frame', 'x', 'y']):
            continue

        if is_reference_condition:
            sigma_particle = 0.0
        else:
            if label_key in DIM_FRAGMENTS:
                current_sigma_ref = SIGMA_DIM_MICRONS
                current_mass_ref  = MASS_DIM_REFERENCE
                use_scaling_this  = USE_MASS_SCALING   # mass-scaled, matching Cy3 treatment
            else:
                current_sigma_ref = SIGMA_CY3_MICRONS
                current_mass_ref  = MASS_CY3_REFERENCE
                use_scaling_this  = USE_MASS_SCALING   # Cy3, mass-scaled per particle

            mass_data      = ptraj['mass'].values if 'mass' in ptraj.columns else np.array([])
            sigma_particle = calculate_sigma_for_particle(
                mass_data, current_sigma_ref, current_mass_ref, use_scaling_this
            )

        pos_df = ptraj.set_index('frame')[['x', 'y']] * mpp
        pos    = pos_df.reindex(
            np.arange(pos_df.index[0], 1 + pos_df.index[-1])
        ).to_numpy()
        pos = pos[:max_length]

        if len(pos) < 2 * lag_time_steps + 1:
            continue

        n_total += 1

        if not is_reference_condition and noise_floor_msd is not None:
            if not trajectory_above_noise_floor(
                pos, sigma_particle, FPS, MAX_LAGTIME, noise_floor_msd
            ):
                n_rejected += 1
                continue

        if not is_reference_condition and APPLY_MSD_ACCEPTANCE_FILTER:
            accepted, _alpha2, _r2 = check_long_time_fit_acceptance(
                pos, MAX_LAGTIME, sigma_particle, FPS,
                R_SQUARED_THRESHOLD, USE_LOCALIZATION_CORRECTION,
            )
            if not accepted:
                n_failed_msd_acceptance += 1
                continue

        angles = calculate_next_step_angles(
            pos, lag_time=lag_time_steps,
            sigma_particle=sigma_particle,
            displacement_multiplier=2.0
        )
        if len(angles) > 0:
            all_angles[lag_time_steps].extend(angles)

    return all_angles, n_total, n_rejected, n_failed_msd_acceptance


def process_directional_directory(directory, mpp, max_length, lag_time_steps,
                                   label_key, noise_floor_msd,
                                   is_reference_condition=False):
    all_angles_combined = []
    total_tracks        = 0
    total_rejected      = 0
    total_failed_msd_acceptance = 0

    if not os.path.isdir(directory):
        return np.array([])

    for filename in os.listdir(directory):
        if filename.startswith('.') or not filename.endswith('.csv'):
            continue
        result, n_total, n_rejected, n_failed_msd_acceptance = directional_analysis_per_file(
            os.path.join(directory, filename),
            mpp, max_length, lag_time_steps, label_key, noise_floor_msd,
            is_reference_condition=is_reference_condition,
        )
        if lag_time_steps in result and len(result[lag_time_steps]) > 0:
            all_angles_combined.extend(result[lag_time_steps])
        total_tracks   += n_total
        total_rejected += n_rejected
        total_failed_msd_acceptance += n_failed_msd_acceptance

    n_kept = total_tracks - total_rejected - total_failed_msd_acceptance
    print(f"    Trajectories: {total_tracks} total | "
          f"{total_rejected} rejected (noise floor) | "
          f"{total_failed_msd_acceptance} rejected (R^2 < {R_SQUARED_THRESHOLD} or alpha <= 0) | "
          f"{n_kept} kept")
    return np.array(all_angles_combined)

# ============================================================================
# BACKWARD-BIAS SUMMARY (fraction of turning angles > threshold)
# ============================================================================
def summarize_backward_bias(all_angle_distributions, labels, lag_keys,
                             threshold=90.0, reference_label=None):
    """
    For each condition and lag time, compute the fraction of next-step
    turning angles exceeding `threshold` degrees (default 90 -- the
    isotropic/uncorrelated-diffusion expectation, NOT itself a
    confinement cutoff: a perfectly flat/uncorrelated angle distribution
    already puts ~50% of angles above 90 deg by symmetry). A fraction
    meaningfully above the reference condition's fraction indicates
    anti-persistent, confinement-consistent turning; a fraction close to
    the reference indicates the DNA condition's apparent turning bias is
    not distinguishable from what raw localization noise alone produces
    in this pipeline.

    reference_label : optional str
        If provided and present in all_angle_distributions, this
        condition (e.g. BEAD_LABEL) is printed as a separate row below
        the DNA-size conditions, set off with a divider, since it is the
        "no real correlated motion" null baseline rather than a
        condition being compared to the others on equal footing.

    Returns a nested dict: {lag_label: {condition_label: pct_above_threshold}}
    and also prints a formatted table.
    """
    results = {lag_label: {} for lag_label in lag_keys}
    ordered_labels = [l for l in labels if l != reference_label]

    col_width = 20
    print(f"\nFraction of turning angles > {threshold:.0f} deg (backward-biased / "
          f"confinement-consistent), by condition and lag time:")
    header = "  Condition".ljust(28) + "".join(f"{lag}".rjust(col_width) for lag in lag_keys)
    print(header)

    def _print_row(label):
        row = f"  {label}".ljust(28)
        for lag_label in lag_keys:
            dist = all_angle_distributions[lag_label].get(label)
            if dist is not None and len(dist) > 0:
                pct = float(np.mean(dist > threshold) * 100)
                n = len(dist)
                results[lag_label][label] = pct
                row += f"{pct:.1f}% (n={n})".rjust(col_width)
            else:
                results[lag_label][label] = np.nan
                row += f"{'n/a':>{col_width}}"
        print(row)

    for label in ordered_labels:
        _print_row(label)

    has_reference = reference_label and any(
        reference_label in all_angle_distributions[lag] for lag in lag_keys
    )
    if has_reference:
        print("  " + "-" * (28 + col_width * len(lag_keys) - 28))
        _print_row(reference_label)
        print("  (Reference row = unfiltered bead/noise-floor condition, i.e. the "
              "null expectation for 'no real correlated motion' given this "
              "pipeline's localization noise -- not a DNA condition on equal footing.)")

    return results

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================
def plot_linear_nsab_panel(ax, lag_label, distributions, labels, colors, bins, panel_label):
    present_labels = [label for label in labels if label in distributions]
    max_density    = 0

    for current_label in present_labels:
        hist, _ = np.histogram(distributions[current_label], bins=bins, density=True)
        if hist.size > 0:
            max_density = max(max_density, np.max(hist))

    if max_density == 0:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
        return

    for current_label in present_labels:
        angles_data     = distributions[current_label]
        hist, bin_edges = np.histogram(angles_data, bins=bins, density=True)
        bin_centers     = (bin_edges[:-1] + bin_edges[1:]) / 2
        original_index  = labels.index(current_label)
        ax.plot(bin_centers, hist, color=colors[original_index],
                label=current_label, linewidth=2.0, alpha=0.85)

    ax.axvline(90, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 0.0155)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(which="both", axis="both", direction="in",
                   labelsize=12, length=6, width=1.5)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel(r'$\theta [^\circ]$', fontsize=14)
    ax.set_ylabel(r'Probability Density', fontsize=14)
    ax.set_title(lag_label, fontsize=14)
    ax.grid(False)
    ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')
    if panel_label == 'C':
        ax.legend(loc='upper left', frameon=False, fontsize=14)

# ============================================================================
# LAG TIMES
# ============================================================================
LAG_TIME_STEPS_SHORT = 1   # 2.0 s
LAG_TIME_STEPS_MID   = 20   # 5.0 s
LAG_TIME_STEPS_LONG  = 80   # 8.0 s

analysis_lags = {
    f'$\\Delta t$={LAG_TIME_STEPS_SHORT/FPS:.1f} s': LAG_TIME_STEPS_SHORT,
    f'$\\Delta t$={LAG_TIME_STEPS_MID/FPS:.1f} s':   LAG_TIME_STEPS_MID,
    f'$\\Delta t$={LAG_TIME_STEPS_LONG/FPS:.1f} s':  LAG_TIME_STEPS_LONG,
}

# ============================================================================
# MAIN
# ============================================================================
def run_step_4_orientation_analysis(directories, condition_keys, max_traj_lengths,
                                     mpp, fps, colors, labels):
    print("\n" + "="*80)
    print("STEP 4: NEXT STEP ANGLE BIAS ANALYSIS — CELL TYPES")
    print(f"Conditions           : {condition_keys}")
    print(f"Noise floor fraction : {NOISE_FLOOR_FRACTION}")
    print(f"MSD acceptance filter: {'Enabled' if APPLY_MSD_ACCEPTANCE_FILTER else 'Disabled'} "
          f"(R^2 >= {R_SQUARED_THRESHOLD}, alpha > 0)")
    print("="*80)

    lag_keys = list(analysis_lags.keys())

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
    # STEP 2: Process each condition
    # ------------------------------------------------------------------
    all_angle_distributions = {lag_label: {} for lag_label in lag_keys}
    bins = np.linspace(0, 180, 19)

    processed_keys = [k for k in condition_keys
                      if k in directories and os.path.exists(directories[k])]
    if not processed_keys:
        print("\nERROR: No valid directories found for analysis.")
        return

    for i, label_key in enumerate(processed_keys):
        directory     = directories[label_key]
        idx           = condition_keys.index(label_key)
        max_length    = max_traj_lengths[idx]
        current_label = labels[idx]

        print(f"\n-> Analyzing {current_label} (max_length={max_length})...")

        for lag_label in lag_keys:
            lag_steps    = analysis_lags[lag_label]
            angles_array = process_directional_directory(
                directory, mpp, max_length, lag_steps, label_key, noise_floor_msd
            )

            if angles_array.size > 0:
                all_angle_distributions[lag_label][current_label] = angles_array
                print(f"  {lag_label}: N={len(angles_array)}")
            else:
                print(f"  {lag_label}: No valid angle pairs collected.")

    # ------------------------------------------------------------------
    # STEP 2b: Process beads themselves as a reference/null distribution
    # (raw localization noise, unfiltered -- see is_reference_condition
    # in directional_analysis_per_file). Gives a baseline turning-angle
    # distribution against which the DNA conditions' backward bias can
    # actually be judged, since ~50% above 90 deg is roughly the null
    # expectation for uncorrelated motion on its own.
    # ------------------------------------------------------------------
    if bead_dir and os.path.isdir(bead_dir):
        print(f"\n-> Analyzing {BEAD_LABEL} (max_length={BEAD_MAX_LENGTH})...")
        for lag_label in lag_keys:
            lag_steps    = analysis_lags[lag_label]
            angles_array = process_directional_directory(
                bead_dir, mpp, BEAD_MAX_LENGTH, lag_steps, 'beads', None,
                is_reference_condition=True,
            )
            if angles_array.size > 0:
                all_angle_distributions[lag_label][BEAD_LABEL] = angles_array
                print(f"  {lag_label}: N={len(angles_array)}")
            else:
                print(f"  {lag_label}: No valid angle pairs collected.")

    # ------------------------------------------------------------------
    # STEP 2c: Backward-bias summary table (with bead reference row)
    # ------------------------------------------------------------------
    backward_bias_results = summarize_backward_bias(
        all_angle_distributions, labels + [BEAD_LABEL], lag_keys,
        reference_label=BEAD_LABEL,
    )

    # ------------------------------------------------------------------
    # STEP 3: Generate 1x3 linear panel figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    linear_axes = {
        lag_keys[0]: (axes[0], 'C'),
        lag_keys[1]: (axes[1], 'D'),
        lag_keys[2]: (axes[2], 'E'),
    }

    for lag_label, (ax, panel_label) in linear_axes.items():
        plot_linear_nsab_panel(
            ax, lag_label, all_angle_distributions[lag_label],
            labels, colors, bins, panel_label
        )
        if panel_label != 'C':
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

    plt.tight_layout()

    filename_out = (
        "D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/"
        "Corrected_Data/Figures/AngularBias_DNA_size.png"
    )
    fig.savefig(filename_out, dpi=600, bbox_inches='tight')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE: Combined Figure Generated")
    print(f"Figure saved to: {filename_out}")
    print("="*80)

    plt.show()

    return backward_bias_results


if __name__ == '__main__':
    run_step_4_orientation_analysis(
        directories, CONDITION_KEYS, MAX_TRAJ_LENGTHS, MPP, FPS, colors, labels
    )