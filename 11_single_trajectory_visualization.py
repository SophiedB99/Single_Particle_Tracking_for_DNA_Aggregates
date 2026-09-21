# -*- coding: utf-8 -*-
"""
Trajectory visualization: BF overlay, per-particle colored trajectory plots,
and displacement-vs-time panels.
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

# =============================================================================
# CONFIGURATION 
# =============================================================================

IMAGE_PATH = (
    "D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/"
    "Figures/Figure 1/Chamber2_BF_Experiment2-16bit.tif"
)
CSV_PATH = (
    "D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/"
    "Sophie Corrected for Gaussian Filter/BPK4410/25.4.24/E_output_data/"
    "Experiment2.2/tm/tm_25.4.2024_movie_Experiment2.2.csv"
)
SAVE_DIR = (
    "D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/"
    "Corrected_Data/Figures/"
)

MPP            = 0.16   # microns per pixel
FPS            = 10        # frames per second
TOTAL_DURATION = 80        # s — full trajectory duration for colorbar

# Scale bar: width in pixels corresponding to a known physical length
SCALE_BAR_PX     = 120.48  # pixels
SCALE_BAR_HEIGHT = 20      # pixels

# Per-particle settings
# window_half: half-width of the zoom window in microns (0.3 → ±0.3 µm)
# target_time: how many seconds to show in the displacement plot
# disp_ylim_comp: (ymin, ymax) for the Δx/Δy panel
# disp_ylim_mag:  (ymin, ymax) for the |Δr| panel
PARTICLE_CONFIGS = [
    {
        "particle_id":    4,
        "window_half":    0.3,
        "target_time":    14.0,
        "disp_ylim_comp": (-0.05, 0.14),
        "disp_ylim_mag":  (0.00,  0.15),
        "save_traj":      SAVE_DIR + "Particle4_Trajectory.png",
        "save_disp":      SAVE_DIR + "Particle4_Displacement_vs_Time_14s.png",
    },
    {
        "particle_id":    213,
        "window_half":    0.3,
        "target_time":    TOTAL_DURATION,
        "disp_ylim_comp": (-0.05, 0.14),
        "disp_ylim_mag":  (0.00,  0.15),
        "save_traj":      SAVE_DIR + "Particle213_Trajectory.png",
        "save_disp":      None,
    },
]

# =============================================================================
# LOAD DATA
# =============================================================================

img = cv2.imread(IMAGE_PATH)
# Always convert BGR → RGB so colours render correctly in matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
df = pd.read_csv(CSV_PATH)

# =============================================================================
# PLOT 1 — all trajectories overlaid on brightfield image
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_rgb)

for track_id in df["particle"].unique():
    track = df[df["particle"] == track_id]
    ax.plot(track["x"], track["y"], color="red", linewidth=3)

# Scale bar (pixel-space rectangle in bottom-right corner)
h, w = img_rgb.shape[:2]
x_pos = w - SCALE_BAR_PX - 10
y_pos = h - SCALE_BAR_HEIGHT - 10
ax.add_patch(Rectangle(
    (x_pos, y_pos), SCALE_BAR_PX, SCALE_BAR_HEIGHT,
    linewidth=0, edgecolor="none", facecolor="white"
))

ax.axis("off")
plt.tight_layout()
# plt.savefig(SAVE_DIR + "BF_Trajectories_Overlay.tif", dpi=600, bbox_inches="tight")
plt.show()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def plot_colored_trajectory(df, particle_id, mpp, fps, total_duration,
                             window_half, save_path=None):
    """
    Plot a single particle trajectory colored by time (blue=start, red=end).
    Overlays: center of mass dot, Rg circle, and a few example displacement vectors.
    """
    track = df[df["particle"] == particle_id]
    x = track["x"].values * mpp
    y = track["y"].values * mpp

    # --- Compute Rg (matching your analysis script exactly) ---
    x_head = x[:800]
    y_head = y[:800]

    xcom = np.nanmean(x_head)
    ycom = np.nanmean(y_head)

    xsqdis = (x_head - xcom) ** 2
    ysqdis = (y_head - ycom) ** 2

    xrg = np.sqrt(np.nanmean(xsqdis))
    yrg = np.sqrt(np.nanmean(ysqdis))
    rg  = np.sqrt(xrg**2 + yrg**2)
    print("The Rg value is", rg)

    # Auto-center zoom window on track
    x_center = np.median(x)
    y_center = np.median(y)

    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    time = np.linspace(0, total_duration, len(segments))
    norm = mcolors.Normalize(vmin=0, vmax=total_duration)
    cmap = cm.get_cmap("coolwarm_r")

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(time)
    lc.set_linewidth(1.5)

    fig, ax = plt.subplots(figsize=(6, 6))
    line = ax.add_collection(lc)

    # --- Rg circle centered on CoM ---
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(xcom + rg * np.cos(theta),
            ycom + rg * np.sin(theta),
            color='black', linewidth=1.5, linestyle='--',
            zorder=4, label=f'$R_g$ = {rg*1000:.1f} nm')

    # --- Center of mass dot ---
    ax.scatter([xcom], [ycom],
               color='#8B0000', s=80, zorder=7,
               edgecolors='black', linewidths=0.8,
               label=r'$r_\mathrm{CoM}$')

    # --- Rg label on the circle (at the 9 o'clock position) ---
    ax.annotate(
        '', 
        xy=(xcom - rg, ycom),
        xytext=(xcom, ycom),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
    )
    ax.text(xcom - rg - 0.005, ycom,
            r'$\mathbf{R}_g$',
            fontsize=13, color='black',
            ha='right', va='center', zorder=6)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_center - window_half, x_center + window_half)
    ax.set_ylim(y_center - window_half, y_center + window_half)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(which="both", axis="both", direction="in", labelsize=14, length=6, width=1.5)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Time [s]", fontsize=16)
    cbar.ax.tick_params(labelsize=14, width=1.5)
    cbar.outline.set_linewidth(2)

    scalebar = AnchoredSizeBar(
        transform=ax.transData,
        size=0.2,
        label="",
        loc="lower right",
        pad=0.5,
        color="black",
        frameon=False,
        size_vertical=0.02,
        fontproperties=fm.FontProperties(size=14),
    )
    ax.add_artist(scalebar)

    ax.grid(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


def plot_displacement(df, particle_id, mpp, fps, total_duration,
                      target_time, ylim_comp, ylim_mag, save_path=None):
    """
    Two-panel displacement plot:
      A) Δx and Δy vs time
      B) |Δr| vs time
    Truncated to target_time seconds.
    """
    track  = df[df["particle"] == particle_id]
    x      = track["x"].values * mpp
    y      = track["y"].values * mpp
    t_full = np.linspace(0, total_duration, len(x))

    delta_x = x - x[0]
    delta_y = y - y[0]
    delta_r = np.sqrt(delta_x**2 + delta_y**2)

    # Truncate to target_time
    truncate_index = np.where(t_full >= target_time)[0]
    end_idx = truncate_index[0] if len(truncate_index) > 0 else len(t_full)

    t      = t_full[:end_idx]
    dxT    = delta_x[:end_idx]
    dyT    = delta_y[:end_idx]
    drT    = delta_r[:end_idx]

    fig, (ax_comp, ax_mag) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"hspace": 0.1}
    )

    ax_comp.text(-0.15, 1, "A", transform=ax_comp.transAxes,
                 fontsize=12, fontweight="bold", va="top")
    ax_mag.text( -0.15, 1, "B", transform=ax_mag.transAxes,
                 fontsize=12, fontweight="bold", va="top")

    ax_comp.plot(t, dxT, color="red",  linewidth=1.5, label=r"$\Delta x$")
    ax_comp.plot(t, dyT, color="blue", linewidth=1.5, label=r"$\Delta y$")
    ax_comp.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_comp.set_ylim(ylim_comp)
    ax_comp.set_ylabel(r"$\Delta x$, $\Delta y$ [$\mu$m]", fontsize=12)
    ax_comp.legend(loc="upper right", fontsize=10)
    ax_comp.grid(False)

    ax_mag.plot(t, drT, color="black", linewidth=1.5)
    ax_mag.set_ylim(ylim_mag)
    ax_mag.set_ylabel(r"$|\Delta r|$ [$\mu$m]", fontsize=12)
    ax_mag.set_xlabel("$t$ (s)", fontsize=12)
    ax_mag.set_xlim(0, target_time)
    ax_mag.grid(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# MAIN — per-particle plots
# =============================================================================

for cfg in PARTICLE_CONFIGS:
    pid = cfg["particle_id"]
    print(f"Plotting particle {pid}...")

    plot_colored_trajectory(
        df, pid, MPP, FPS,
        total_duration=TOTAL_DURATION,
        window_half=cfg["window_half"],
        save_path=cfg["save_traj"],
    )

    plot_displacement(
        df, pid, MPP, FPS,
        total_duration=TOTAL_DURATION,
        target_time=cfg["target_time"],
        ylim_comp=cfg["disp_ylim_comp"],
        ylim_mag=cfg["disp_ylim_mag"],
        save_path=cfg["save_disp"],
    )