# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:13:44 2025
@author: sophiedeboer
Modified to include per-particle mass scaling and localization correction
Modified again to apply the same noise-floor + R^2/alpha acceptance
filtering used in 06a/06b/06c/07a/07b/08a/08b (via msd_core.py), so the
population of trajectories feeding the Rg analysis matches the population
used everywhere else in the pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import string
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib.ticker import LogLocator, NullFormatter

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Shared MSD core -- see msd_core.py. Guarantees this script's noise-floor
# and R^2/alpha acceptance logic are IDENTICAL to 05a/05b/05c/06a/06b/07a/07b.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from msd_core import (
    calculate_sigma_for_particle_from_mass_array as calculate_sigma_for_particle,
    compute_bead_noise_floor,
    trajectory_above_noise_floor,
    check_long_time_fit_acceptance,
    NOISE_FLOOR_FRACTION,
)

# ============================================================================
# EXPLANATION
# The script takes the first 800 frames of a particle's trajectory.
# It computes the center of mass using all these frames.
# It calculates the deviation of all 800 positions from the center.
# It computes the root mean squared deviation in x and y.
# Finally, it uses the Pythagorean theorem to find the radius of gyration.
# PER-PARTICLE LOCALIZATION CORRECTION: Each particle gets individual sigma based on brightness
# MASS SCALING: Brighter particles have smaller localization errors
# NOISE FLOOR + R^2/ALPHA ACCEPTANCE FILTERING: trajectories are excluded on
# the same two criteria used in 05a/05b/05c/06a/06b/07a/07b, so the Rg
# analysis population matches the rest of the pipeline.
# ============================================================================

def trajectory(pos, pid, max_lagtime, video, pos_columns):
    max_lagtime = len(pos)
    lagtimes = np.arange(0, max_lagtime)
    result = pd.DataFrame(pos - pos[0], columns=pos_columns, index=lagtimes)
    result['particle'] = pid
    result['video'] = video
    return result

# ============================================================================
# CONFIGURATION
# ============================================================================
filepaths = [
    'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/Aswin Data CHO Corrected for Drift',
    'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Sophie Corrected for Gaussian Filter'
]

bps = [
    '/Collected_100bp', '/Collected_250bp', '/Collected_500bp',
    '/Collected_pEGFP-C1', '/Collected_BPK4410', '/Collected_hCas9D10A'
]
bpcs = [
    '100 bp', '250 bp', '500 bp',
    '4713 bp', '7614 bp', '9553 bp'
]
plasmid = [
    '/100 bp', '/250 bp', '/500 bp',
    '/pEGFP-C1', '/BPK4410', '/hCas9D10A'
]

# --- Reference Values ---
SIGMA_STUCK_MICRONS  = 0.021244
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS    = 0.014008    # YOYO-1, 
MASS_DIM_REFERENCE   = 589.8

SIGMA_CY3_MICRONS    = 0.013078    # Cy3, 
MASS_CY3_REFERENCE   = 8922.3      # median mass from Cy3 (Bright Ref) dataset; reference for mass scaling

# --- Experimental Parameters ---
mpp = 0.16
fps = 10

# --- Correction Options ---
USE_LOCALIZATION_CORRECTION = False
USE_MASS_SCALING = False

# --- Filtering Options ---
BASE_DATA_ROOT  = 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data'
BEAD_DIRECTORY  = BASE_DATA_ROOT + '/Control Experiments_Beads/0.22 um/E_output_data/Collected_0.22um/tm'
BEAD_MAX_LENGTH = 800
MAX_LAGTIME_FILTER = 100   # matches 05a/06a/07a's noise-floor/fit window; independent of this
                            # script's own `max_lagtime` (=50), which is unrelated (used only
                            # inside trajectory() and effectively overridden there anyway).

R_SQUARED_THRESHOLD = 0.8
APPLY_MSD_ACCEPTANCE_FILTER = True

PLOT_COLORS = ['#6FADE3', '#2874C6', '#042C53', '#FAC775', '#BA7517', '#412402']

mkr = ['*', 'D', 'x', '^', 'o', 's']

print("\n" + "="*80)
print("RADIUS OF GYRATION ANALYSIS WITH PER-PARTICLE LOCALIZATION CORRECTION")
print("="*80)
print(f"Sigma (from stuck beads): {SIGMA_STUCK_MICRONS:.6f} μm")
print(f"Sigma (Dim Dye, 100-500 bp): {SIGMA_DIM_MICRONS:.6f} μm")
print(f"Sigma (Cy3, 4713-9553 bp): {SIGMA_CY3_MICRONS:.6f} μm")
print(f"Mass reference (stuck beads): {MASS_STUCK_REFERENCE:.1f}")
print(f"Mass reference (dim dye): {MASS_DIM_REFERENCE:.1f}")
print(f"Mass reference (Cy3): {MASS_CY3_REFERENCE:.1f}")
print(f"Mass scaling: {'ENABLED' if USE_MASS_SCALING else 'DISABLED'}")
print(f"Localization correction: {'ENABLED' if USE_LOCALIZATION_CORRECTION else 'DISABLED'}")
print(f"Noise floor fraction: {NOISE_FLOOR_FRACTION}")
print(f"MSD acceptance filter: {'ENABLED' if APPLY_MSD_ACCEPTANCE_FILTER else 'DISABLED'} "
      f"(R^2 >= {R_SQUARED_THRESHOLD}, alpha > 0)")
print("="*80 + "\n")

# ============================================================================
# ESTABLISH NOISE FLOOR FROM BEAD TRAJECTORIES 
# ============================================================================
print("Processing beads (noise floor)...")
noise_floor_msd = None
if os.path.isdir(BEAD_DIRECTORY):
    noise_floor_msd = compute_bead_noise_floor(
        BEAD_DIRECTORY, mpp, fps, MAX_LAGTIME_FILTER, BEAD_MAX_LENGTH
    )
    if noise_floor_msd is None:
        print("  WARNING: No valid bead trajectories -- noise floor filtering disabled.")
else:
    print(f"  WARNING: Bead directory not found at {BEAD_DIRECTORY} -- "
          "noise floor filtering disabled. Check BASE_DATA_ROOT/BEAD_DIRECTORY.")
print()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
max_lagtime = 50
pos_columns = ['x', 'y']

fig, axs_grid = plt.subplots(2, 3, figsize=(3.375*2, 3.375*2*1.6/3))
axs = axs_grid.flat

all_rg_raw = []
all_rg_corrected = []
all_mass = []
all_sigma = []
video = 1
axnum = 0

for i, bp in enumerate(bps):
    base_path = filepaths[0] if i < 3 else filepaths[1]

    trajec = []
    resultstraj = None

    if i < 3:
        sigma_ref = SIGMA_DIM_MICRONS
        mass_ref = MASS_DIM_REFERENCE
        use_scaling_flag = USE_MASS_SCALING
        print(f"Processing {bpcs[i]}... using Dim Dye Sigma ({sigma_ref:.6f} μm) for dim dye.")
    else:
        sigma_ref = SIGMA_CY3_MICRONS
        mass_ref = MASS_CY3_REFERENCE
        use_scaling_flag = USE_MASS_SCALING
        print(f"Processing {bpcs[i]}... using Cy3 Sigma ({sigma_ref:.6f} μm) with MASS SCALING.")

    for dirs in ['/MSDcollected', '/Trajcollected', '/MSDindividual', '/Figures']:
        os.makedirs(base_path + '/E_output_data/' + str(bp) + dirs, exist_ok=True)

    directory  = base_path + '/E_output_data/' + str(bp) + '/tm'
    directory2 = base_path + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = base_path + '/E_output_data/' + str(bp) + '/Figures'

    rg_raw = []
    rg_corrected = []
    mass_values = []
    sigma_values = []

    # Cache sigma per (particle, video) so the Rg loop below can reuse
    # the value already computed here instead of recomputing from scratch.
    sigma_by_id = {}
    n_total_particles = 0
    n_rejected_noise_floor = 0
    n_rejected_msd_acceptance = 0

    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            traj = pd.read_csv(os.path.join(directory, filename))
            for pid, ptraj in traj.groupby('particle'):
                pos = ptraj.set_index('frame')[pos_columns] * mpp
                pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()

                mass_data = ptraj['mass'].values if 'mass' in ptraj.columns else np.array([])
                if USE_LOCALIZATION_CORRECTION:
                    sigma_particle = calculate_sigma_for_particle(
                        mass_data, sigma_ref, mass_ref, use_scaling_flag
                    )
                else:
                    sigma_particle = 0.0

                n_total_particles += 1

                # ---- NOISE FLOOR FILTER ----
                if noise_floor_msd is not None:
                    if not trajectory_above_noise_floor(
                        pos, sigma_particle, fps, MAX_LAGTIME_FILTER, noise_floor_msd
                    ):
                        n_rejected_noise_floor += 1
                        continue
                # -----------------------------------

                # ---- MSD ACCEPTANCE FILTER (R^2 >= threshold, alpha > 0) ----
                if APPLY_MSD_ACCEPTANCE_FILTER:
                    accepted, _alpha2, _r2 = check_long_time_fit_acceptance(
                        pos, MAX_LAGTIME_FILTER, sigma_particle, fps,
                        R_SQUARED_THRESHOLD, USE_LOCALIZATION_CORRECTION,
                    )
                    if not accepted:
                        n_rejected_msd_acceptance += 1
                        continue
                # ---------------------------------------------------------------

                trajec.append(trajectory(pos, pid, max_lagtime, video, pos_columns))
                sigma_by_id[(pid, video)] = sigma_particle

            if trajec:
                resultstraj = pd.concat(trajec)
                time = resultstraj.index.values.astype('float64') / float(fps)
                resultstraj.set_index(time, inplace=True)
                resultstraj.index.name = 'Time [s]'
            video += 1

    n_kept = n_total_particles - n_rejected_noise_floor - n_rejected_msd_acceptance
    print(f"  Trajectories: {n_total_particles} total | "
          f"{n_rejected_noise_floor} rejected (noise floor) | "
          f"{n_rejected_msd_acceptance} rejected (R^2 < {R_SQUARED_THRESHOLD} or alpha <= 0) | "
          f"{n_kept} kept")

    if resultstraj is None:
        print(f"  WARNING: no trajectories survived filtering for {bpcs[i]} -- skipping.\n")
        all_rg_raw.append([])
        all_rg_corrected.append([])
        all_mass.append([])
        all_sigma.append([])
        axnum += 1
        continue

    for id, traj in resultstraj.groupby(['particle', 'video']):
        x = traj['x'].head(800)
        y = traj['y'].head(800)

        sigma_particle = sigma_by_id.get(id, 0.0)
        mass_particle = np.nan  # per-particle mean mass not retained past filtering step

        xcom = np.nanmean(x)
        ycom = np.nanmean(y)

        xsqdis = [(x1 - xcom)**2 for x1 in x]
        ysqdis = [(y1 - ycom)**2 for y1 in y]

        xrg = np.sqrt(np.nanmean(xsqdis))
        yrg = np.sqrt(np.nanmean(ysqdis))

        rg_raw_value = np.sqrt(xrg**2 + yrg**2)
        rg_raw.append(rg_raw_value)

        if USE_LOCALIZATION_CORRECTION and np.isfinite(sigma_particle):
            rg_squared_corrected = rg_raw_value**2 - 2 * sigma_particle**2
            if rg_squared_corrected > 0:
                rg_corrected.append(np.sqrt(rg_squared_corrected))
            else:
                rg_corrected.append(np.nan)
        else:
            rg_corrected.append(rg_raw_value)

        mass_values.append(mass_particle)
        sigma_values.append(sigma_particle)

    rg_corrected_filtered = [r for r in rg_corrected if not np.isnan(r)]

    all_rg_raw.append(rg_raw.copy())
    all_rg_corrected.append(rg_corrected_filtered)
    all_mass.append(mass_values)
    all_sigma.append(sigma_values)

    print(f"  Tracks processed: {len(rg_corrected_filtered)}")
    print(f"  Mean Rg (corrected): {np.mean(rg_corrected_filtered):.4f} μm")
    print(f"  Mean sigma: {np.nanmean(sigma_values):.6f} μm\n")

    # Histogram
    axs[axnum].hist(rg_corrected_filtered, bins=np.arange(0, 1.01, 0.05),
                    edgecolor='black', color=PLOT_COLORS[axnum], alpha=0.6,
                    density=True, rwidth=1)
    axs[axnum].yaxis.set_ticks_position('both')
    axs[axnum].set_yticks([10, 8, 6, 4, 2, 0])
    axs[axnum].set_xticks([1.0, 0.8, 0.6, 0.4, 0.2, 0])
    axs[axnum].xaxis.set_ticks_position('both')
    axs[axnum].tick_params(which='both', axis="both", direction="in")
    axs[axnum].set_xlim((-0.05, 1))
    axs[axnum].text(0.95, 0.95, bpcs[axnum], verticalalignment='top',
                    horizontalalignment='right', multialignment="left",
                    transform=axs[axnum].transAxes, color='black')
    axs[axnum].set(xlabel=r'$ \left |\mathbf{R}_{\mathrm{g}}\right |$' + ' ' + r'($\mu$ m)',
                   ylabel=r'$P(\left |\mathbf{R}_{\mathrm{g}}\right |)$' + ' ' + r'($\mu$ m$^{-1}$)')
    axs[axnum].set_ylim((0, 10))
    axnum += 1

# ============================================================================
# Violin Plot
# ============================================================================
fig2, ax_box = plt.subplots(figsize=(4, 4.5))

x_ticks = list(range(1, len(all_rg_corrected) + 1))
jitter = 0.05
np.random.seed(42)

vp = ax_box.violinplot(all_rg_corrected, positions=x_ticks, showextrema=False, widths=0.8)

# Violin fill: alpha=0.2, matching main analysis figure style
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(PLOT_COLORS[i])
    pc.set_alpha(0.2)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Scatter: alpha=1.0, s=25, black edge — fully opaque dots pop against transparent violin
for i, rg_vals in enumerate(all_rg_corrected):
    x_jitter = np.full_like(rg_vals, x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(rg_vals))
    ax_box.scatter(x_jitter, rg_vals,
                   color=PLOT_COLORS[i], alpha=1.0,
                   marker='o', edgecolor='black', linewidths=0.5,
                   s=25, zorder=10)

    # Median line
    median_val = np.median(rg_vals)
    ax_box.plot([x_ticks[i] - 0.2, x_ticks[i] + 0.2], [median_val, median_val],
                color='black', linewidth=2.0, zorder=11)

ax_box.yaxis.set_ticks_position('both')
ax_box.xaxis.set_ticks_position('both')
for spine in ax_box.spines.values():
    spine.set_linewidth(1.5)
ax_box.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
ax_box.set_ylim(0, 0.8)
ax_box.set_xticks(x_ticks)
ax_box.set_xticklabels(bpcs, rotation=45, ha='right', fontsize=12)
ax_box.set_ylabel(r'$ \left |\mathbf{R}_{\mathrm{g}}\right |$' + ' ' + r'($\mu$ m)', fontsize=14)
ax_box.grid(False)

# ============================================================================
# SAVE FIGURES
# ============================================================================
Filepath = 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Figures'

fig.tight_layout()
fig.savefig(Filepath + '/Rg_histograms_corrected.png', dpi=600)
fig2.tight_layout()
fig2.savefig(Filepath + '/Rg_violinplot.png', dpi=600)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS - CORRECTED Rg VALUES")
print("="*80)
print(f"{'Base Pairs':>12} | {'Median (μm)':>12} | {'SEM (μm)':>12} | {'n':>5}")
print("-" * 80)

for bp, rg_vals in zip(bpcs, all_rg_corrected):
    if len(rg_vals) > 0:
        median_rg = np.median(rg_vals)
        sem_rg = stats.sem(rg_vals)
        count = len(rg_vals)
    else:
        median_rg = np.nan
        sem_rg = np.nan
        count = 0
    print(f"{bp:>12} | {median_rg:12.4f} | {sem_rg:12.4f} | {count:>5}")


def calculate_radius_of_gyration_linear(bp, rise_per_bp=0.34, persistence_length=50):
    if bp <= 0:
        raise ValueError("Base pair count must be positive.")
    L = bp * rise_per_bp
    return math.sqrt((L**2) / 12)


def calculate_radius_of_gyration_coil(bp, persistence_length=50, rise_per_bp=0.34):
    if bp <= 0:
        raise ValueError("Base pair count must be positive.")
    lp = persistence_length
    L = bp * rise_per_bp
    exponent_term = np.exp(-L / lp)
    Rg_squared = (2 * lp * L) - (2 * (lp**2) * (1 - exponent_term))
    if Rg_squared > 0:
        return math.sqrt(Rg_squared)
    else:
        return np.nan

print("\n" + "="*80)
print("PERCENTAGE OF TRAJECTORIES WITH Rg < 200 nm")
print("="*80)
print(f"{'Base Pairs':>12} | {'% below 200 nm':>18} | {'Count':>10} | {'Total':>10}")
print("-" * 80)

threshold = 0.2  # 200 nm in microns

for bp, rg_vals in zip(bpcs, all_rg_corrected):
    rg_array = np.array(rg_vals)
    total = len(rg_array)

    if total > 0:
        below_threshold = np.sum(rg_array < threshold)
        percentage = (below_threshold / total) * 100
    else:
        below_threshold = 0
        percentage = np.nan

    print(f"{bp:>12} | {percentage:18.2f} | {below_threshold:10} | {total:10}")


print("\n" + "="*80)
print("THEORETICAL Radius of Gyration")
print("="*80)
print(f"{'Base Pairs':>10} | {'Linear Rg (nm)':>15} | {'WLC Coil Rg (nm)':>17}")
print("-" * 80)
bp_sizes = [100, 250, 500, 4713, 7614, 9553]
for bp in bp_sizes:
    linear_rg = calculate_radius_of_gyration_linear(bp)
    coil_rg = calculate_radius_of_gyration_coil(bp)
    print(f"{bp:>10} | {linear_rg:15.3g} | {coil_rg:17.3g}")
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)

plt.show()