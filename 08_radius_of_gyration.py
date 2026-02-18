
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:13:44 2025
@author: sophiedeboer
Modified to include per-particle mass scaling and localization correction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import string
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib.ticker import LogLocator, NullFormatter

'Explanation'
####### EXPLANATION #####################
# The script takes the first 800 frames of a particle's trajectory.
# It computes the center of mass using all these frames.
# It calculates the deviation of all 800 positions from the center.
# It computes the root mean squared deviation in x and y
# Finally, it uses the Pythagorean theorem to find the radius of gyration.
# PER-PARTICLE LOCALIZATION CORRECTION: Each particle gets individual sigma based on brightness
# MASS SCALING: Brighter particles have smaller localization errors
'done'

def trajectory(pos, pid, max_lagtime, video, pos_columns):
    """Shift the starting point of the trajectory to (0,0).
    Parameters
    ----------
    pos : Positions x, and y for individual particles 
    max_lagtime : intervals of frames out to which MSD is computed
    Returns
    -------
    DataFrame([<x>, <y>, pid], index=t)
    Notes
    -----
    Input units are microns and frames. Output units are microns and frames."""
    max_lagtime = len(pos)
    lagtimes = np.arange(0, max_lagtime)
    result = pd.DataFrame(pos - pos[0], columns=pos_columns, index=lagtimes)
    result['particle'] = pid
    result['video'] = video
    return result

def calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_mass_scaling_flag):
    """Scale sigma based on particle brightness.
    Brighter particles have smaller sigma.
    """
    if not use_mass_scaling_flag or mass_particle <= 0 or mass_ref <= 0:
        return sigma_ref
    mass_ratio = mass_ref / mass_particle
    return sigma_ref * np.sqrt(mass_ratio)

# ============================================================================
# CONFIGURATION
# ============================================================================
filepaths = [
    'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/Aswin Data CHO Corrected for Drift',
    'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Sophie Corrected for Gaussian Filter'
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

# --- Reference Values (From Stuck Bead Analysis) ---
SIGMA_STUCK_MICRONS = 0.013543  # um - Replace with your stuck bead sigma
MASS_STUCK_REFERENCE = 3381.5   # Replace with your stuck bead mass

# NEW CONSTANT FOR DIM FRAGMENTS (100-500 bp)
SIGMA_DIM_MICRONS = 0.015280    # um
MASS_DIM_REFERENCE = 589.8      # median mass from YOYO-1 reference

# --- Experimental Parameters ---
mpp = 0.16  # microns per pixel (note: MSD code uses 0.102, verify which is correct)
fps = 10    # frames per second

# --- Correction Options ---
USE_LOCALIZATION_CORRECTION = False  # Enable/disable localization correction
USE_MASS_SCALING = False             # Scale sigma based on particle brightness

print("\n" + "="*80)
print("RADIUS OF GYRATION ANALYSIS WITH PER-PARTICLE LOCALIZATION CORRECTION")
print("="*80)
print(f"Sigma (from stuck beads): {SIGMA_STUCK_MICRONS:.6f} μm")
print(f"Sigma (Dim Dye, 100-500 bp): {SIGMA_DIM_MICRONS:.6f} μm")
print(f"Mass reference (stuck beads): {MASS_STUCK_REFERENCE:.1f}")
print(f"Mass reference (dim dye): {MASS_DIM_REFERENCE:.1f}")
print(f"Mass scaling: {'ENABLED' if USE_MASS_SCALING else 'DISABLED'}")
print(f"Localization correction: {'ENABLED' if USE_LOCALIZATION_CORRECTION else 'DISABLED'}")
print("="*80 + "\n")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
i = 0
max_lagtime = 50
pos_columns = ['x', 'y']
trajec = []

# Create a figure for histograms (matching your original style)
fig, axs_grid = plt.subplots(2, 3, figsize=(3.375*2, 3.375*2*1.6/3))
axs = axs_grid.flat

# Colors matching the MSD analysis code
PLOT_COLORS = ['#332288', '#88CCEE', '#44AA99', '#117733', '#DDCC77', '#CC6677']
mkr = ['*', 'D', 'x', '^', 'o', 's']

all_rg_raw = []
all_rg_corrected = []
all_mass = []
all_sigma = []
video = 1
axnum = 0

for i, bp in enumerate(bps):
    # Select correct base path: first three from filepaths[0], rest from filepaths[1]
    base_path = filepaths[0] if i < 3 else filepaths[1]
    
    # Conditional Sigma Assignment (matching MSD code logic)
    if i < 3:  # 100bp, 250bp, 500bp
        sigma_ref = SIGMA_DIM_MICRONS
        mass_ref = MASS_DIM_REFERENCE
        use_scaling_flag = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION
        print(f"Processing {bpcs[i]}... using Fixed Sigma ({sigma_ref:.6f} μm) for dim dye.")
    else:  # 4713bp, 7614bp, 9553bp
        sigma_ref = SIGMA_STUCK_MICRONS
        mass_ref = MASS_STUCK_REFERENCE
        use_scaling_flag = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION
        print(f"Processing {bpcs[i]}... using Stuck Bead Sigma ({sigma_ref:.6f} μm) with MASS SCALING.")
    
    # Create output directories
    for dirs in ['/MSDcollected', '/Trajcollected', '/MSDindividual', '/Figures']:
        if not os.path.exists(base_path + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(base_path + '/E_output_data/' + str(bp) + dirs)
    
    directory = base_path + '/E_output_data/' + str(bp) + '/tm'
    directory2 = base_path + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = base_path + '/E_output_data/' + str(bp) + '/Figures'
    
    rg_raw = []
    rg_corrected = []
    mass_values = []
    sigma_values = []
    
    # Process all files in directory
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            traj = pd.read_csv(os.path.join(directory, filename))
            for pid, ptraj in traj.groupby('particle'):
                pos = ptraj.set_index('frame')[pos_columns] * mpp
                pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
                trajec.append(trajectory(pos, pid, max_lagtime, video, pos_columns))
                resultstraj = pd.concat(trajec)
            
            time = resultstraj.index.values.astype('float64') / float(fps)
            resultstraj.set_index(time, inplace=True)
            resultstraj.index.name = 'Time [s]' 
            video += 1
    
    # Calculate Rg for each particle with per-particle correction
    for id, traj in resultstraj.groupby(['particle', 'video']):
        x = traj['x'].head(800)
        y = traj['y'].head(800)
        
        # Get particle mass for this trajectory
        # Note: We need to trace back to original data to get mass
        # For now, using placeholder - you may need to add mass tracking to trajectory function
        mass_particle = np.nan  # This should be retrieved from original data
        
        # Calculate per-particle sigma
        if USE_LOCALIZATION_CORRECTION:
            sigma_particle = calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_scaling_flag)
        else:
            sigma_particle = 0.0
        
        xcom = np.nanmean(x)
        ycom = np.nanmean(y)
        
        xsqdis = [(x1 - xcom)**2 for x1 in x]
        ysqdis = [(y1 - ycom)**2 for y1 in y]
        
        xrg = np.sqrt(np.nanmean(xsqdis))
        yrg = np.sqrt(np.nanmean(ysqdis))
        
        # Raw (uncorrected) Rg
        rg_raw_value = np.sqrt(xrg**2 + yrg**2)
        rg_raw.append(rg_raw_value)
        
        # Corrected Rg: Rg_corrected^2 = Rg_measured^2 - 2*sigma^2
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
    
    # Filter out NaN values
    rg_corrected_filtered = [r for r in rg_corrected if not np.isnan(r)]
    
    all_rg_raw.append(rg_raw.copy())
    all_rg_corrected.append(rg_corrected_filtered)
    all_mass.append(mass_values)
    all_sigma.append(sigma_values)
    
    print(f"  Tracks processed: {len(rg_corrected_filtered)}")
    print(f"  Mean Rg (corrected): {np.mean(rg_corrected_filtered):.4f} μm")
    print(f"  Mean sigma: {np.nanmean(sigma_values):.6f} μm\n")
    
    # Plot histogram (corrected values)
    axs[axnum].hist(rg_corrected_filtered, bins=np.arange(0, 1.01, 0.05), 
                    edgecolor='black', color='white', density=True, rwidth=1)
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
                  ylabel=r'$P(\left |\mathbf{R}_{\mathrm{g}}\right |)$' + ' ' + '($\mu$ m$^{-1}$)')
    axs[axnum].set_ylim((0, 10))
    axnum += 1

# ============================================================================
# Violin Plot
# ============================================================================
fig2, ax_box = plt.subplots(figsize=(4, 4.5))

x_ticks = list(range(1, len(all_rg_corrected) + 1))
jitter = 0.05
np.random.seed(42)

# Create violin plot
vp = ax_box.violinplot(all_rg_corrected, positions=x_ticks, showextrema=False, widths=0.8)
# bp = ax_box.boxplot(all_rg_corrected, showfliers=False, patch_artist=True,
#                     medianprops=dict(color='black', linewidth=1.5),
#                     boxprops=dict(linewidth=1.5))

# Style violins (similar to boxes)
for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(PLOT_COLORS[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Scatter points and color boxes
for i, rg_vals in enumerate(all_rg_corrected):
    # Scatter data points
    x_jitter = np.full_like(rg_vals, x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(rg_vals))
    ax_box.scatter(x_jitter, rg_vals, color=PLOT_COLORS[i], alpha=0.8,
                  marker='o', edgecolor='k', s=20, zorder=10)
    
    # # Set box colors
    # bp['boxes'][i].set_facecolor(PLOT_COLORS[i])
    # bp['boxes'][i].set_alpha(0.6)

# Axis formatting (matching alpha plot style)
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
Filepath = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Figures'

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
# Updated header to show Median and SEM
print(f"{'Base Pairs':>12} | {'Median (μm)':>12} | {'SEM (μm)':>12} | {'n':>5}")
print("-" * 80)

for bp, rg_vals in zip(bpcs, all_rg_corrected):
    # Ensure list is not empty before calculating
    if len(rg_vals) > 0:
        median_rg = np.median(rg_vals)
        # Calculate Standard Error of the Mean (SEM)
        sem_rg = stats.sem(rg_vals)
        count = len(rg_vals)
    else:
        median_rg = np.nan
        sem_rg = np.nan
        count = 0
    
    # Print Median and SEM
    print(f"{bp:>12} | {median_rg:12.4f} | {sem_rg:12.4f} | {count:>5}")


def calculate_radius_of_gyration_linear(bp, rise_per_bp=0.34, persistence_length=50):
    """Calculate the radius of gyration for linear (rod-like) DNA (WLC approximation)."""
    # Assuming persistence_length (lp) and rise_per_bp are provided in nanometers (nm)
    if bp <= 0:
        raise ValueError("Base pair count must be positive.")
    
    L = bp * rise_per_bp # Contour length L in nm
    #lp = persistence_length # Persistence length in nm
    
    # Rg approximation for L >> lp regime: Rg^2 = L*lp / 3
    # Note: The calculation returns nm if inputs are nm.
    return math.sqrt((L **2) / 12)

def calculate_radius_of_gyration_coil(bp, persistence_length=50, rise_per_bp=0.34):
    """Calculate the Radius of Gyration for a linear polymer coil (WLC model) 
    using the full form of the mean-square distance of a linear chain.
    Rg^2 = 2*lp*L - 2*lp^2 * [1 - exp(-L/lp)]
    """
    if bp <= 0:
        raise ValueError("Base pair count must be positive.")
    
    lp = persistence_length # 50 nm
    L = bp * rise_per_bp # Contour length L in nm

    # The formula provided by the user is R_g^2 = 2 l_p L - 2 l_p^2 [1 - exp(-L/l_p)]
    # Note: This expression is typically associated with the mean square end-to-end distance <R^2> 
    # for a WLC, but is implemented here as requested for R_g^2.
    
    exponent_term = np.exp(-L / lp)
    
    # R_g^2 = 2 * lp * L - 2 * lp^2 * (1 - exp(-L / lp))
    Rg_squared = (2 * lp * L) - (2 * (lp**2) * (1 - exponent_term))
    
    if Rg_squared > 0:
        return math.sqrt(Rg_squared) # R_g in nm
    else:
        return np.nan

# ============================================================================
# THEORETICAL RADIUS OF GYRATION
# ============================================================================


print("\n" + "="*80)
print("THEORETICAL Radius of Gyration")
print("="*80)
# Updated header to reflect R_g,coil calculation
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