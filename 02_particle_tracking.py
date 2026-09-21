# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function

import os
import pims
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

# =============================================================================
# CONFIGURATION
# =============================================================================

FILEPATH = r'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/MDA-MB-231/Cy3 MDA-MB-231 WT/21.11.2025'
EXPERIMENT = 'Experiment3.7'
DATA_DIR   = r'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/MDA-MB-231/Cy3 MDA-MB-231 WT/21.11.2025/B_ProcessedData'

# Locate parameters
DIAMETER      = 11    # Expected feature diameter in pixels (must be odd). Err on the large side — too small causes subpixel bias.
MINMASS       = 7000  # Minimum integrated brightness of a valid feature. Raise to reject dim spurious spots; lower to capture faint particles.
MAXSIZE       = 4     # Maximum radius-of-gyration of brightness. Rejects features that are too spatially spread out (e.g. blobs, aggregates).
NOISE_SIZE    = 1     # Width of the Gaussian blur kernel (px) used to suppress camera noise before feature detection.
SMOOTHING     = 17    # Size of the uniform smoothing kernel (px) used to remove background. Should stay consistent across experiments.
N_FRAMES      = 800   # Number of frames to process from the image sequence.

# Link parameters
MAX_DISPLACEMENT = 2  # Maximum distance (px) a particle can travel between frames. Raise if particles move fast; lower to avoid false links.
MEMORY           = 3  # Number of frames a particle can disappear and still be linked to the same trajectory. Raise to bridge blinking events.

# Filter parameters
MIN_FRAMES = 150      # Minimum track length (frames) to keep. Removes short-lived spurious detections.
MIN_MASS   = 10       # Per-frame minimum mass. Removes frames where a particle is nearly invisible (e.g. mid-blink).
MAX_SIZE   = 3.6      # Per-frame maximum size (pixels). Removes frames where a particle appears bloated or overlapping with another.
MAX_ECC    = 0.3      # Per-frame maximum eccentricity (0 = circle, 1 = line). Removes non-circular features such as aggregates or debris.
EXCLUDE_PARTICLES = [215, 355, 309]  # Particle IDs to manually exclude (e.g. confirmed overlaps or artefacts).

# MSD parameters
MICRONS_PER_PIXEL = 0.16  # Spatial calibration factor. Converts pixel distances to microns.
FPS               = 10        # Frames per second. Converts frame lag to time lag in MSD plots.

# =============================================================================
# SETUP
# =============================================================================

exp_dir    = f'/E_output_data/{EXPERIMENT}'
output_dir = FILEPATH + exp_dir
data_dir   = DATA_DIR + f'/{EXPERIMENT}'

for subdir in ['/f', '/t', '/tf', '/im_plot', '/em_plot', '/trajectories', '/tm', '/hist_subpix']:
    os.makedirs(output_dir + subdir, exist_ok=True)

frames = pims.ImageSequence(data_dir)
tp.quiet()

# =============================================================================
# STEP 1 — LOCATE FEATURES IN FRAME 0 (preview)
# =============================================================================

f0 = tp.locate(frames[0], DIAMETER, minmass=MINMASS)

tp.annotate(f0, frames[0])

fig, ax = plt.subplots()
ax.hist(f0['mass'], bins=10)
ax.set(xlabel='mass', ylabel='count')
tp.subpx_bias(f0)
plt.savefig(output_dir + '/hist_subpix/hist_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 2 — LOCATE FEATURES IN ALL FRAMES
# =============================================================================

f = tp.batch(
    frames[0:N_FRAMES],
    DIAMETER,
    minmass=MINMASS,
    maxsize=MAXSIZE,
    noise_size=NOISE_SIZE,
    smoothing_size=SMOOTHING,
    processes=1,
    threshold=None,
)

tp.subpx_bias(f)

x_frac, y_frac = f['x'] % 1, f['y'] % 1
print("KS test X:", kstest(x_frac, 'uniform', args=(0, 1)))
print("KS test Y:", kstest(y_frac, 'uniform', args=(0, 1)))

# =============================================================================
# STEP 3 — LINK INTO TRAJECTORIES
# =============================================================================

t = tp.link(f, MAX_DISPLACEMENT, memory=MEMORY)

# =============================================================================
# STEP 4 — FILTER SPURIOUS TRAJECTORIES
# =============================================================================

# 4a. Remove short-lived tracks
t1 = tp.filter_stubs(t, MIN_FRAMES)
print(f"Particles — before stub filter: {t['particle'].nunique()}, after: {t1['particle'].nunique()}")

# 4b. Remove by mass / size / eccentricity
tp.mass_size(t1.groupby('particle').mean())

mask = (
    (t1['mass'] > MIN_MASS) &
    (t1['size'] < MAX_SIZE) &
    (t1['ecc']  < MAX_ECC)  &
    (~t1['particle'].isin(EXCLUDE_PARTICLES))
)
t2 = t1[mask]
print(f"Particles after mass/size/ecc filter: {t2['particle'].nunique()}")
print(f"Average localisation error: {t2['ep'].mean():.4f} px")

tp.annotate(t2[t2['frame'] == 0], frames[0])

# =============================================================================
# STEP 5 — PLOT TRAJECTORIES & COMPUTE DRIFT
# =============================================================================

def save_traj_plot(data, path, title=None):
    fig, ax = plt.subplots()
    tp.plot_traj(data, ax=ax)
    if title:
        ax.set_title(title)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

save_traj_plot(t2, output_dir + '/trajectories/tf_trajectory_plot.png', 'Filtered trajectories')

d = tp.compute_drift(t2)
d.plot()
plt.savefig(output_dir + '/trajectories/tf_drift_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 6 — SUBTRACT DRIFT
# =============================================================================

tm = tp.subtract_drift(t2.copy(), d).reset_index(drop=True)

d_post = tp.compute_drift(tm)
d_post.plot()

save_traj_plot(tm, output_dir + '/trajectories/tm_trajectory_plot.png', 'Drift-corrected trajectories')

counts = tm.groupby('particle').size()
counts = counts[counts > MIN_FRAMES]
min_particle = counts.idxmin()
print(f"Particle with fewest coordinates: {min_particle} ({counts.min()} points)")

# =============================================================================
# STEP 7 — SAVE RAW DATA
# =============================================================================

filename = f'beads_movie_{EXPERIMENT}'

f.to_csv(output_dir  + f'/f/f_{filename}.csv')
t.to_csv(output_dir  + f'/t/t_{filename}.csv')
t2.to_csv(output_dir + f'/tf/tf_{filename}.csv')
tm.to_csv(output_dir + f'/tm/tm_{filename}.csv')

# =============================================================================
# STEP 8 — MSD ANALYSIS
# =============================================================================

im = tp.imsd(tm, MICRONS_PER_PIXEL, FPS)
em = tp.emsd(tm, MICRONS_PER_PIXEL, FPS)

# Individual MSDs
fig, ax = plt.subplots()
ax.plot(im.index.to_numpy(), im.to_numpy(), 'k-', alpha=0.1)
ax.set(xlabel='lag time $t$', ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xscale='log', yscale='log')
plt.savefig(output_dir + '/im_plot/im_plot.png', dpi=300, bbox_inches='tight')

# Ensemble MSD overlaid on individual MSDs
fig, ax = plt.subplots()
ax.plot(im.index.to_numpy(), im.to_numpy(), 'k-', alpha=0.1, label='individual')
ax.plot(em.index, em, 'o', label='ensemble')
ax.set(xlabel='lag time $t$', ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xscale='log', yscale='log', ylim=(1e-4, 10))
ax.legend()
plt.savefig(output_dir + '/em_plot/em_plot.png', dpi=300, bbox_inches='tight')
plt.show()