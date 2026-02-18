# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:44:32 2025

@author: sophiedeboer
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np

# --- Load the background image (composite BF + LSM) ---
image_path = "F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Figures/Figure 1/Chamber2_BF_Experiment2-16bit.tif"  # or .png/.jpg
img = cv2.imread(image_path)

# Optional: convert BGR to RGB for matplotlib
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Load trajectory data (CSV with X, Y, Frame) ---
csv_path = "F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Sophie Corrected for Gaussian Filter/BPK4410/25.4.24/E_output_data/Experiment2.2/tm/tm_25.4.2024_movie_Experiment2.2.csv"
df = pd.read_csv(csv_path)

# --- Get unique track IDs ---
track_ids = df['particle'].unique()
num_tracks = len(track_ids)
mpp = 0.166 #micron per pixel

# --- Assign a unique color to each track using a colormap ---
cmap = plt.get_cmap("tab20")  # Use tab20, tab10, or any matplotlib colormap
#colors = {track_id: cmap(i % 7) for i, track_id in enumerate(track_ids)}

# --- Plot image and color-coded trajectories ---
plt.figure(figsize=(10, 10))
plt.imshow(img)

# Plot each track with its corresponding color
for track_id in track_ids:
    track = df[df['particle'] == track_id]
    plt.plot(track['x'], track['y'], color='red', linewidth=3, label=f'Track {track_id}')

# --- Add the scale bar ---
# Define the scale bar properties
scale_bar_width = 120.48  # in pixels
scale_bar_height = 20  # in pixels
color = "white"
font_size = 12

# Position of the scale bar (bottom-right corner, with a 10-pixel margin)
x_pos = img.shape[1] - scale_bar_width - 10  # 10 pixels margin from the right
y_pos = img.shape[0] - scale_bar_height - 10  # 10 pixels margin from the bottom

# Add the scale bar as a rectangle
rect = Rectangle((x_pos, y_pos), scale_bar_width, scale_bar_height, linewidth=0, edgecolor='none', facecolor=color)
plt.gca().add_patch(rect)


# --- Final Adjustments ---
plt.axis('off')
# Uncomment below to show legend if you want
# plt.legend(loc='upper right', fontsize=6)

plt.tight_layout()

# Save the image with the scale bar
#plt.savefig("O:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Figures/Figure 1/Scalebar_BF_Trajectories_Overlay_BPK4410_Ex2.2.tif", dpi=600)
plt.show()


###############################################################################
# Filter for particle 4
###############################################################################
track1 = df[df['particle'] == 4]

# Extract X and Y
x = track1['x'].values * mpp
y = track1['y'].values * mpp

# Create segments for LineCollection
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Normalize time (e.g., by index)
time = np.linspace(0, 80, len(segments))  # From 0 (start) to 1 (end)
norm = mcolors.Normalize(vmin=0, vmax=80)
cmap = cm.get_cmap('plasma')  # Choose a colormap: plasma, viridis, inferno, etc.

# Create colorized line
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(time)
lc.set_linewidth(1.5)

# Plot
plt.figure(figsize=(6, 6))
ax = plt.gca()
line = ax.add_collection(lc)

plt.xlabel("")
plt.ylabel("")

# Remove ticks completely
ax.set_xticks([])
ax.set_yticks([])

plt.ylim(59.7,60.3)#delta = 0.6
plt.xlim(43.8, 44.4)#delta = 0.6
plt.gca().invert_yaxis()

# Style the axis spines
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# Ticks in both directions
plt.tick_params(
    which="both", axis="both", direction="in",
    labelsize=14, length=6, width=1.5
)
plt.gca().xaxis.set_ticks_position('both')
plt.gca().yaxis.set_ticks_position('both')


# Colorbar formatting
cbar = plt.colorbar(line)
cbar.set_label('Time [s]', fontsize=16)
cbar.ax.tick_params(labelsize=14, width=1.5)
cbar.outline.set_linewidth(2)

# --- Add Scale Bar ---
scalebar = AnchoredSizeBar(
    transform=plt.gca().transData,   # use data coordinates
    size=0.2,                        # length of scalebar in data units
    label='',                    # text label 200 nm
    loc='lower right',               # position in the plot
    pad=0.5,
    color='black',
    frameon=False,
    size_vertical=0.03,              # thickness of bar
    fontproperties=fm.FontProperties(size=14)
)
plt.gca().add_artist(scalebar)


plt.grid(False)
plt.tight_layout()
#plt.savefig("O:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Figures/Figure 1/Particle4_Trajectory over time.png", dpi=600)
plt.show()


###############################################################################
# Filter for particle 4
# Plot simple displacements (delta x, delta y, delta r) over time
###############################################################################
# Calculate displacement components relative to the starting position (t=0)
# These arrays (delta_x, delta_y, delta_r_magnitude) all have length N (e.g., 800)
# This time array must match the length of x, y, delta_x, and delta_y (e.g., 800)
TOTAL_DURATION = 80 # s
time_full = np.linspace(0, TOTAL_DURATION, len(x)) # Now length N (e.g., 800)

delta_x = x - x[0]
delta_y = y - y[0]
delta_r_magnitude = np.sqrt(delta_x**2 + delta_y**2)

# Define the relevant time window for the visualization
TARGET_TIME_LIMIT = 14.0 # seconds
# Determine which indices fall within the target time limit (0-14s)
# Note: Since the plotting function needs arrays of the same length, we truncate all of them.
truncate_index = np.where(time_full >= TARGET_TIME_LIMIT)[0]
if len(truncate_index) > 0:
    end_idx = truncate_index[0]
else:
    end_idx = len(time_full) # Use all data if total time is < 14s

# Truncate all necessary arrays for the 0-14s plot
time_truncated = time_full[:end_idx]
delta_x_truncated = delta_x[:end_idx]
delta_y_truncated = delta_y[:end_idx]
delta_r_magnitude_truncated = delta_r_magnitude[:end_idx]


# --- 2. Create the Multi-Panel Plot ---
fig, (ax_comp, ax_mag) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'hspace': 0.1})

#Panel labels
ax_comp.text(-0.15, 1, "A", transform=ax_comp.transAxes,             
             fontsize=12, fontweight="bold", va= 'top')

ax_mag.text(-0.15, 1, "B", transform=ax_mag.transAxes,
            fontsize=12, fontweight='bold', va='top')


# --- Plot A: Displacement Components (Δx, Δy) ---
# Note: Corrected title to reference Particle 4, as filtered in the code.
#ax_comp.set_title("Displacement Analysis for Particle 4", fontsize=12)
ax_comp.plot(time_truncated, delta_x_truncated, color='red', linewidth=1.5, label='$\Delta x$')
ax_comp.plot(time_truncated, delta_y_truncated, color='blue', linewidth=1.5, label='$\Delta y$')

# Axes limits matching the example image
ax_comp.set_ylim(-0.05, 0.14)
ax_comp.set_ylabel(r"$\Delta x$, $\Delta y$ [$\mu $m]", fontsize=12)
ax_comp.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax_comp.legend(loc='upper right', fontsize=10)
ax_comp.grid(True, linestyle=':', alpha=0.5)

# --- Plot B: Displacement Magnitude (|Δr|) ---
ax_mag.plot(time_truncated, delta_r_magnitude_truncated, color='black', linewidth=1.5)

# Axes limits matching the example image
ax_mag.set_ylim(0.00, 0.15)
ax_mag.set_ylabel(r"$|\Delta r|$ [$\mu $m]", fontsize=12)

# X-axis setup
ax_mag.set_xlabel("$t$ (s)", fontsize=12)
ax_mag.set_xlim(0, TARGET_TIME_LIMIT)
ax_mag.grid(True, linestyle=':', alpha=0.5)

# Final adjustments
plt.tight_layout()
plt.savefig("F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Figures/Particle4_Displacement_vs_Time_14s.png", dpi=300)
plt.show()



###############################################################################
# Filter for particle 213
###############################################################################
track2 = df[df['particle'] == 213]

# Extract X and Y
x = track2['x'].values * mpp
y = track2['y'].values * mpp

# Create segments for LineCollection
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Normalize time (e.g., by index)
time = np.linspace(0, 80, len(segments))  # From 0 (start) to 1 (end)
norm = mcolors.Normalize(vmin=0, vmax=80)
cmap = cm.get_cmap('plasma')  # Choose a colormap: plasma, viridis, inferno, etc.

# Create colorized line
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(time)
lc.set_linewidth(1.5)

# Plot
plt.figure(figsize=(6, 6))
ax = plt.gca()
line = ax.add_collection(lc)

plt.xlabel("")
plt.ylabel("")

plt.ylim(9.65, 10.25) #delta = 0.6
plt.xlim(3.35, 3.95)
plt.gca().invert_yaxis()

# Remove tick labels but keep ticks
ax.set_xticklabels([])
ax.set_yticklabels([])

# Style the axis spines
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# Ticks in both directions
plt.tick_params(
    which="both", axis="both", direction="in",
    labelsize=14, length=6, width=1.5
)
plt.gca().xaxis.set_ticks_position('both')
plt.gca().yaxis.set_ticks_position('both')


# Colorbar formatting
cbar = plt.colorbar(line)
cbar.set_label('Time [s]', fontsize=16)
cbar.ax.tick_params(labelsize=14, width=1.5)
cbar.outline.set_linewidth(2)
plt.grid(False)
plt.tight_layout()
#plt.savefig("O:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Figures/Figure 1/Particle213_Trajectory over time.png", dpi=600)
plt.show()