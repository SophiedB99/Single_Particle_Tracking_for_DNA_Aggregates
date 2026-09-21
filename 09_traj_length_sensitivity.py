import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning, message=r'The behavior of obj\[\] with a float-dtype index is deprecated.')

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ============================================================================
# CORRECTION PARAMETERS
# ============================================================================
SIGMA_STUCK_MICRONS  = 0.021244
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS    = 0.014008    # YOYO-1
MASS_DIM_REFERENCE   = 589.8

SIGMA_CY3_MICRONS    = 0.013078    # Cy3
MASS_CY3_REFERENCE   = 8922.3      # median mass from Cy3 (Bright Ref) dataset; reference for mass scaling

NOISE_FLOOR_THRESHOLD_DNA   = 0.98
NOISE_FLOOR_THRESHOLD_CELLS = 0.98

# Datasets using YOYO-1 dim dye reference (sigma is now mass-scaled per
# particle here too, same as Cy3 datasets -- see imsd() below)
DIM_DATASETS = {'Aswin DNA (Ladders)', 'Aswin DNA (Cell Types)'}
CELL_TYPE_DATASETS = {'Aswin DNA (Cell Types)', 'Sophie MDAMB231'}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _msd_iter(pos, lagtimes):
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff_no_nan = np.nan_to_num(diff, nan=0)
        if not np.all(diff_no_nan == 0):
            yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))
        else:
            yield np.full(pos.shape[1] * 2, np.nan)


def trajectory(pos, pid, max_lagtime, video, pos_columns):
    max_lagtime = len(pos)
    lagtimes = np.arange(0, max_lagtime)
    result = pd.DataFrame(pos - pos[0], columns=pos_columns, index=lagtimes)
    result['particle'] = pid
    result['video'] = video
    return result


def msd(pos, max_lagtime, sigma=0.0):
    max_lagtime = min(max_lagtime, len(pos) - 1)
    lagtimes = np.arange(1, max_lagtime + 1)
    pos_columns = ['x', 'y']
    result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                     ['<{}^2>'.format(p) for p in pos_columns]
    resultmsd = list(_msd_iter(pos, lagtimes))
    result = pd.DataFrame(resultmsd, columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(axis=1, skipna=False)
    if sigma > 0:
        result['msd'] = (result['msd'] - 4.0 * sigma ** 2).clip(lower=0.0)
    return result


def power_law(x, alpha, A):
    return A * np.power(x, alpha)


def powerfit(msd_df, a, b, fps=10):
    y = msd_df['msd']
    x = msd_df.index.values.astype('float64') / float(fps)
    y_data = y.iloc[a:b]
    x_data = x[a:b]
    if len(x_data) < 2:
        raise ValueError("Insufficient data points for fitting.")
    pars, _ = curve_fit(f=power_law, xdata=x_data, ydata=y_data,
                        p0=[0, 0], bounds=(-np.inf, np.inf))
    residuals = y_data - power_law(x_data, *pars)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 0.0 if (ss_tot == 0 or np.isnan(ss_tot)) else 1.0 - (ss_res / ss_tot)
    return pars, r_squared


def imsd(filename, mpp, fps, video, directory, max_lagtime=100,
         sigma_ref=0.0, mass_ref=0.0, is_dim=False,
         noise_floor_msd=None, noise_threshold=0.98):
    statistic = 'msd'
    try:
        traj = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return [], [], [], [], []

    pos_columns = ['x', 'y']
    ids, trajec, msds, alpha, alpha2, length = [], [], [], [], [], []
    index = 0

    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()

        # Both dim (YOYO-1) and non-dim (Cy3) conditions are mass-scaled per
        # particle from their respective reference (FIXED: dim used to be
        # excluded here, inconsistent with the rest of the pipeline).
        sigma_particle = sigma_ref
        if sigma_ref > 0 and mass_ref > 0 and 'mass' in ptraj.columns:
            mean_mass = ptraj['mass'].mean()
            if mean_mass > 0:
                sigma_particle = sigma_ref * np.sqrt(mass_ref / mean_mass)

        msdtemp = msd(pos[:800], max_lagtime, sigma=sigma_particle)
        if np.any(msdtemp.values == 0):
            msdtemp = msdtemp.replace(0, np.nan)
        msdtemp = msdtemp.dropna(axis='rows', how='all')

        if msdtemp.empty or 'msd' not in msdtemp.columns or len(msdtemp) < 20:
            continue

        if noise_floor_msd is not None:
            track_msd = msdtemp['msd']
            common_idx = track_msd.index.intersection(noise_floor_msd.index)
            if len(common_idx) > 0:
                frac_above = np.mean(track_msd.loc[common_idx].values >
                                     noise_floor_msd.loc[common_idx].values)
                if frac_above <= noise_threshold:
                    continue

        try:
            alphatemp,  r_sq1 = powerfit(msdtemp, 0,  8,  fps=fps)   # short window, 0.1-0.8s -- diagnostic only
            alphatemp2, r_sq2 = powerfit(msdtemp, 19, 80, fps=fps)   # long window, 2.0-8.0s -- matches msd_core
        except (RuntimeError, ValueError):
            continue

        # Acceptance is gated on the LONG window only (matches msd_core /
        # 05a/05b/05c): the short window is contaminated by localization
        # error and shouldn't decide which trajectories are trusted. The
        # short-window fit is still computed above for potential
        # diagnostic use, but no longer part of the accept/reject decision.
        if r_sq2 >= 0.9 and alphatemp2[0] > 0:
            alpha.append(alphatemp)
            alpha2.append(alphatemp2)
            msds.append(msdtemp)
            trajec.append(trajectory(pos, pid, max_lagtime, video, pos_columns))
            ids.append(index)
            index += 1
            length.append(len(pos))

    if msds:
        resultsmsd = pd.concat(msds, keys=ids)
        resultsmsd = resultsmsd.swaplevel(0, 1)[statistic].unstack()
        lagt = resultsmsd.index.values.astype('float64') / float(fps)
        resultsmsd.set_index(lagt, inplace=True)
        resultsmsd.index.name = 'lag time [s]'
        resultstraj = pd.concat(trajec)
        time = resultstraj.index.values.astype('float64') / float(fps)
        resultstraj.set_index(time, inplace=True)
        resultstraj.index.name = 'Time [s]'
        return resultsmsd, resultstraj, pd.DataFrame(alpha), pd.DataFrame(alpha2), pd.DataFrame(length)
    else:
        return [], [], [], [], []


# ============================================================================
# BEAD NOISE FLOOR
# ============================================================================

def compute_bead_noise_floor(bead_dir, mpp, fps, max_lagtime=100):
    print("  Computing bead noise floor...")
    bead_msds = []
    if not os.path.exists(bead_dir):
        print(f"  WARNING: Bead directory not found: {bead_dir}")
        return None
    for filename in os.listdir(bead_dir):
        if filename.startswith('.'):
            continue
        result = imsd(os.path.join(bead_dir, filename), mpp, fps,
                      video=0, directory=bead_dir, max_lagtime=max_lagtime,
                      sigma_ref=0.0, mass_ref=0.0, noise_floor_msd=None)
        if isinstance(result[0], pd.DataFrame) and not result[0].empty:
            bead_msds.append(result[0])
    if not bead_msds:
        print("  WARNING: No bead data found. Noise floor filter will be skipped.")
        return None
    bead_all = pd.concat(bead_msds, axis=1)
    noise_floor = bead_all.mean(axis=1).dropna()
    print(f"  Bead noise floor computed from {bead_all.shape[1]} tracks.")
    return noise_floor


# ============================================================================
# ALPHA CONVERGENCE ANALYSIS
# ============================================================================

def analyze_length_convergence(length_df, msd_df, fps, powerfit_func,
                                max_trajectory_length_frames):
    length_sec         = length_df.to_numpy().flatten() / fps
    trajectory_indices = msd_df.columns.to_list()

    thresholds = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    thresholds = thresholds[thresholds <= max_trajectory_length_frames / fps]

    results = []
    print("\n--- Analyzing Alpha Convergence ---")
    for min_len_sec in thresholds:
        min_len_frames = int(min_len_sec * fps)
        valid_indices  = [idx for idx, l in enumerate(length_df.to_numpy().flatten())
                          if l >= min_len_frames]
        cols_to_use    = [col for col in trajectory_indices if col in valid_indices]

        if len(cols_to_use) < 1:
            continue
        if len(cols_to_use) < 3:
            print(f"  Only {len(cols_to_use)} trajectory/ies at "
                  f"min_length={min_len_sec}s — treat alpha estimate with caution.")

        mean_msd = msd_df[cols_to_use].mean(axis=1).dropna()
        if mean_msd.empty or len(mean_msd) < 8:
            continue

        try:
            pars, r_sq = powerfit_func(mean_msd.to_frame(name='msd'), 0, 8, fps)
            results.append({
                'min_length_sec': min_len_sec,
                'N_trajectories': len(cols_to_use),
                'alpha':          pars[0],
                'D_eff':          pars[1] / 4.0,
                'R_squared':      r_sq,
            })
        except (RuntimeError, ValueError):
            continue

    return pd.DataFrame(results)


# ============================================================================
# PATH RESOLVER
# ============================================================================

def resolve_paths(data_set, i, pl, bp):
    filepath_base = data_set['filepath_base']
    pl_clean = pl.strip('/').strip('\\')
    bp_clean = bp.strip('/').strip('\\')
    name     = data_set['name']

    if name in ('Sophie DNA (Plasmids)', 'Aswin DNA (Cell Types)'):
        base_dir = os.path.join(filepath_base, pl_clean, 'E_output_data', bp_clean)
    elif name == 'Sophie MDAMB231':
        # filepath_base / plasmid / subfolder / E_output_data / collected  (E_output_data is in bp)
        base_dir = os.path.join(filepath_base, pl_clean, bp_clean)
    elif name == 'Aswin DNA (Ladders)':
        base_dir = os.path.join(filepath_base, 'E_output_data', bp_clean)
    else:
        base_dir = os.path.join(filepath_base, bp_clean)

    return (base_dir,
            os.path.join(base_dir, 'tm'),
            os.path.join(base_dir, 'MSDindividual'),
            os.path.join(base_dir, 'Figures'))


# ============================================================================
# MAIN ANALYSIS RUNNER
# ============================================================================

COLORS_BY_DATASET = {
    'Sophie DNA (Plasmids)':  ['#FAC775', '#BA7517', '#412402'],
    'Sophie MDAMB231':        ['#888780', '#D85A30', '#C2456A'],
    'Aswin DNA (Ladders)':    ['#B5D4F4', '#378ADD', '#042C53'],
    'Aswin DNA (Cell Types)': ['#EF9F27', '#0F6E56', '#533AB7'],
}


def run_analysis(data_set, global_fig_axes, noise_floor_msd):
    print(f"\n{'#'*50}")
    print(f"STARTING DATASET: {data_set['name']}")
    print(f"{'#'*50}")

    fps         = 10
    max_lagtime = 100
    mpp         = 0.16     # FIXED: was 0.10091, inconsistent with rest of pipeline (all other scripts use 0.16)

    colors = COLORS_BY_DATASET.get(data_set['name'], ['#B5D4F4', '#378ADD', '#042C53'])
    mkr    = ['^', 'o', 's']

    is_dim    = data_set['name'] in DIM_DATASETS
    sigma_ref = SIGMA_DIM_MICRONS  if is_dim else SIGMA_CY3_MICRONS
    mass_ref  = MASS_DIM_REFERENCE if is_dim else MASS_CY3_REFERENCE
    noise_threshold = (NOISE_FLOOR_THRESHOLD_CELLS
                       if data_set['name'] in CELL_TYPE_DATASETS
                       else NOISE_FLOOR_THRESHOLD_DNA)

    hist_axes = global_fig_axes[data_set['row']]

    for i, (bp, pl, sz) in enumerate(zip(
            data_set['bpc_folders'],
            data_set.get('plasmid_folders', [''] * len(data_set['bpc_folders'])),
            data_set['size_labels'])):

        base_dir, directory, directory2, directory3 = resolve_paths(data_set, i, pl, bp)
        print(f"\nProcessing Condition {i+1}: {sz}")

        for d in ['MSDcollected', 'Trajcollected', 'MSDindividual', 'Figures']:
            os.makedirs(os.path.join(base_dir, d), exist_ok=True)

        if not os.path.exists(directory):
            print(f"Input directory not found: {directory}. Skipping.")
            continue

        currentlength = 0
        video = 0
        Trajcollected, alphacollected, alphacollected2, lengthcollected, msd_list = \
            [], [], [], [], []

        for filename in os.listdir(directory):
            if filename.startswith('.'):
                continue
            meansquaredis, traj, alpha, alpha2, length = imsd(
                os.path.join(directory, filename),
                mpp, fps, video, directory2, max_lagtime,
                sigma_ref=sigma_ref, mass_ref=mass_ref, is_dim=is_dim,
                noise_floor_msd=noise_floor_msd, noise_threshold=noise_threshold,
            )
            if isinstance(meansquaredis, pd.DataFrame) and not meansquaredis.empty:
                num_tracks = len(meansquaredis.columns)
                meansquaredis.columns = [j + currentlength for j in range(num_tracks)]
                msd_list.append(meansquaredis)
                Trajcollected.append(traj)
                alphacollected.append(alpha)
                alphacollected2.append(alpha2)
                video += 1
                currentlength += num_tracks
                lengthcollected.append(length)

        if not msd_list:
            print(f"No valid MSD data collected for {sz}.")
            continue

        MSDcollected_df    = pd.concat(msd_list, axis=1).dropna(axis='columns', how='all')
        Alphacollected_df  = pd.concat(alphacollected)
        Alphacollected2_df = pd.concat(alphacollected2)
        lengthcollected_df = pd.concat(lengthcollected, ignore_index=True)

        totaltracks    = len(MSDcollected_df.columns)
        max_len_frames = lengthcollected_df.to_numpy().max()
        print(f"Total Tracks: {totaltracks}  |  sigma={sigma_ref*1000:.2f} nm  "
              f"|  dim={is_dim}  |  noise threshold={noise_threshold*100:.0f}%")

        # --- Alpha convergence ---
        convergence_df = analyze_length_convergence(
            lengthcollected_df, MSDcollected_df, fps, powerfit, max_len_frames)

        print(f"\nAlpha Convergence Results for {sz}:")
        if not convergence_df.empty:
            print(convergence_df.to_string(index=False))
            if len(convergence_df) > 1:
                alphas   = convergence_df['alpha'].values
                d_alphas = np.abs(np.diff(alphas) / alphas[:-1])
                optimal_length = None
                for j in range(len(d_alphas) - 1, -1, -1):
                    if d_alphas[j] <= 0.05:
                        optimal_length = convergence_df['min_length_sec'].iloc[j + 1]
                        break
                if optimal_length:
                    print(f"  → Optimal length (alpha stable <5%): {optimal_length:.1f} s")
                else:
                    print("  → Alpha did not stabilize within tested range.")
        else:
            print("  No convergence data.")

        # --- MSD vs length plot ---
        length_sec        = lengthcollected_df.to_numpy().flatten() / fps
        length_bins_list  = data_set['msd_bin_params']['bins']
        bin_labels        = data_set['msd_bin_params']['labels']
        last_edge         = max(length_bins_list[-1] + 1, np.max(length_sec) + 1)
        final_bins        = length_bins_list + [last_edge]
        length_assignment = pd.cut(length_sec[:totaltracks], bins=final_bins,
                                   labels=bin_labels, right=False)

        fig_msd, ax_msd = plt.subplots(figsize=(5, 4))
        for label in bin_labels:
            bin_cols = [col for col, bl in zip(MSDcollected_df.columns, length_assignment)
                        if bl == label]
            if bin_cols:
                ax_msd.plot(MSDcollected_df[bin_cols].mean(axis=1), marker='o',
                            linestyle='-', label=f'{label} (n={len(bin_cols)})')
        ax_msd.plot(MSDcollected_df.mean(axis=1),
                    marker=mkr[i % len(mkr)], markeredgecolor='k',
                    markerfacecolor=colors[i % len(colors)], linestyle='None',
                    label=f"All (N={totaltracks})")
        ax_msd.set_xscale('log'); ax_msd.set_yscale('log')
        ax_msd.set_xlabel(r'$\Delta t$ (s)')
        ax_msd.set_ylabel(r'$\langle \overline{\Delta r^2} \rangle$ ($\mu$m$^2$)')
        ax_msd.set_title(f"MSD vs trajectory length: {sz}")
        ax_msd.legend(frameon=False, fontsize=14)
        ax_msd.tick_params(direction='in', which='both')
        ax_msd.grid(False)
        fig_msd.savefig(os.path.join(directory3, f'MSD_Length_Bias_{sz.replace(" ", "_")}.png'))
        plt.close(fig_msd)

        # --- Trajectory length histogram ---
        ax = hist_axes[i]
        lengths_hist = lengthcollected_df.to_numpy().flatten() / fps

        N_BINS        = 15
        MIN_BIN_WIDTH = 2.0
        data_min   = np.floor(lengths_hist.min())
        data_max   = np.ceil(lengths_hist.max())
        data_range = max(data_max - data_min, MIN_BIN_WIDTH * N_BINS)
        bin_width  = max(data_range / N_BINS, MIN_BIN_WIDTH)
        hist_bins  = np.arange(data_min, data_max + bin_width, bin_width)

        ax.hist(lengths_hist, bins=hist_bins, edgecolor='black',
                color=colors[i % len(colors)], density=True, rwidth=0.9)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(which='both', axis='both', direction='in', labelsize=7)
        ax.set_xlabel('Length (s)', fontsize=12)
        ax.set_xlim(0, data_max + 0.1 * (data_max - data_min + 1))
        ax.set_ylim(0, None)
        ax.grid(False)
        ax.set_title(sz, fontsize=14, fontweight='bold', pad=4)
        if i == 0:
            ax.set_ylabel(r'$P(\mathrm{Length})$', fontsize=12)

        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

        # --- Summary statistics ---
        print('\n--- Summary Statistics ---')
        print(f'  n={len(lengths_hist)}  median={np.median(lengths_hist):.1f}s  '
              f'mean={np.mean(lengths_hist):.1f}s  '
              f'IQR={np.percentile(lengths_hist,25):.1f}–{np.percentile(lengths_hist,75):.1f}s  '
              f'range={lengths_hist.min():.1f}–{lengths_hist.max():.1f}s')
        if not Alphacollected_df.empty:
            print(f'  mean alpha (regime 1) = {Alphacollected_df[0].mean():.3f} '
                  f'+/- {Alphacollected_df[0].sem():.3f}')
            print(f'  mean alpha (regime 2) = {Alphacollected2_df[0].mean():.3f} '
                  f'+/- {Alphacollected2_df[0].sem():.3f}')


# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

BEAD_DIR = (
    'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data'
    '/Control Experiments_Beads/0.22 um/E_output_data/Collected_0.22um/tm'
)

DATASETS = [
    {
        'name': 'Aswin DNA (Ladders)',
        'filepath_base': 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/Aswin Data CHO Corrected for Drift',
        'bpc_folders':    ['/Collected_100bp', '/Collected_250bp', '/Collected_500bp'],
        'plasmid_folders':[''] * 3,
        'size_labels':    ['100 bp', '250 bp', '500 bp'],
        'row': 0,
        'msd_bin_params': {'bins': [0, 10, 20], 'labels': ['0-10s', '10-20s', '20s+']},
    },
    {
        'name': 'Sophie DNA (Plasmids)',
        'filepath_base': 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Sophie Corrected for Gaussian Filter',
        'bpc_folders':    ['/Collected_pEGFP-C1', '/Collected_BPK4410', '/Collected_hCas9D10A'],
        'plasmid_folders':['/pEGFP-C1', '/BPK4410', '/hCas9D10A'],
        'size_labels':    ['4713 bp', '7614 bp', '9553 bp'],
        'row': 1,
        'msd_bin_params': {'bins': [0, 10, 20, 40], 'labels': ['0-10s', '10-20s', '20-40s', '40s+']},
    },
    {
        'name': 'Aswin DNA (Cell Types)',
        'filepath_base': 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data',
        'bpc_folders': ['/Collected_MCF10A', '/Collected_MCF7', '/Collected_MDAMB231'],
        'plasmid_folders': [
            '/Aswin Data MCF 10A Corrected for Drift',
            '/Aswin Data MCF 7 Corrected for Drift',
            '/Aswin Data MDA-MB-231 Corrected for Drift',
        ],
        'size_labels': ['MCF 10A', 'MCF 7', 'MDA-MB-231'],
        'row': 2,
        'msd_bin_params': {'bins': [0, 10, 20], 'labels': ['0-10s', '10-20s', '20s+']},
    },
    {
        'name': 'Sophie MDAMB231',
        'filepath_base': 'D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/MDA-MB-231',
        'bpc_folders': [
            '/500 bp_most_recent/E_output_data/Collected_ATPdepl_MDAMB231',
            '/500 bp_most_recent/E_output_data/Collected_Cy3_MDAMB231',
            '/21.11.2025/E_output_data/Collected_MDAMB231_WT',
        ],
        'plasmid_folders': ['/ATP Depletion MDA-MB-231', '/Cy3 MDA-MB-231', '/Cy3 MDA-MB-231 WT'],
        'size_labels':     ['ATP Depl.', 'LifeActGFP', 'WT'],
        'row': 3,
        'msd_bin_params': {'bins': [0, 10, 20, 40], 'labels': ['0-10s', '10-20s', '20-40s', '40s+']},
    },
]

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':

    fps         = 10
    max_lagtime = 100
    mpp         = 0.16     # FIXED: was 0.10091, inconsistent with rest of pipeline (all other scripts use 0.16)
    save_base   = ('D:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking'
                   '/Corrected_Data/Figures')

    # Step 1 — bead noise floor
    noise_floor_msd = compute_bead_noise_floor(BEAD_DIR, mpp, fps, max_lagtime)

    # Step 2 — single 4-row × 3-col supplementary figure
    # Row order matches DATASETS: Plasmids / Sophie MDAMB231 / Ladders / Cell Types
    # Row heights are equal; width sized for a typical journal supplement page.
    N_ROWS, N_COLS = 4, 3
    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(3.375 * 2.8, 3.375 * 2.8 * N_ROWS / N_COLS),
    )
    fig.subplots_adjust(
        wspace=0.38, hspace=0.3,
        left=0.09, right=0.97,
        top=0.96, bottom=0.06,
    )

    # Panel labels A–L across the grid (row-major)
    import string
    panel_letters = list(string.ascii_uppercase[:N_ROWS * N_COLS])
    for idx, ax in enumerate(axes.flat):
        ax.text(-0.18, 1.08, panel_letters[idx],
                transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    # Map each dataset's hist_ax key to the corresponding row of axes
    global_fig_axes = {
        0: axes[0],   # Aswin DNA (Ladders)     — 100/250/500 bp
        1: axes[1],   # Sophie DNA (Plasmids)   — 4713/7614/9553 bp
        2: axes[2],   # Aswin DNA (Cell Types)  — MCF10A/MCF7/MDA-MB-231
        3: axes[3],   # Sophie MDAMB231          — ATP Depl/LifeActGFP/WT
    }

    # Step 3 — run
    for dataset in DATASETS:
        run_analysis(dataset, global_fig_axes, noise_floor_msd)

    # Step 4 — save combined supplementary figure
    os.makedirs(save_base, exist_ok=True)
    out_path = os.path.join(save_base, 'TrajLength_Supplementary.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Supplementary figure saved: {out_path}")

    plt.show()