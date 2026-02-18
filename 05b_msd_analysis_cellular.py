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
from scipy.optimize import curve_fit
from scipy import stats
from pathlib import Path
import warnings
import os
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DATA_ROOT = Path('F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data')

CONDITIONS_MAP = [
    { 'key': 'MDA MB 231', 'size': 'MDA MB 231', 'base_folder': 'Aswin Data/Aswin Data MDA-MB-231 Corrected for Drift', 'collected_dir': 'Collected_MDAMB231', 'max_length': 300 },
    { 'key': 'MCF 7', 'size': 'MCF 7', 'base_folder': 'Aswin Data/Aswin Data MCF 7 Corrected for Drift', 'collected_dir': 'Collected_MCF7', 'max_length': 300 },
    { 'key': 'MCF 10A', 'size': 'MCF 10A', 'base_folder': 'Aswin Data/Aswin Data MCF 10A Corrected for Drift', 'collected_dir': 'Collected_MCF10A', 'max_length': 300 },
    { 'key': 'ATP depleted', 'size': 'ATP depleted', 'base_folder': 'MDA-MB-231/ATP Depletion MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_ATPdepl_MDAMB231', 'max_length': 800 },
    { 'key': 'Control', 'size': 'Control', 'base_folder': 'MDA-MB-231/Cy3 MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_Cy3_MDAMB231', 'max_length': 800 },
    { 'key': 'beads', 'size': 'beads', 'base_folder': 'Control Experiments_Beads/0.22 um', 'collected_dir': 'Collected_0.22um', 'max_length': 800 }
]

CONDITION_KEYS = [c['key'] for c in CONDITIONS_MAP]
MAX_TRAJ_LENGTHS = [c['max_length'] for c in CONDITIONS_MAP]
CONDITION_LABELS = [c['size'] for c in CONDITIONS_MAP]
CONDITION_DICT = {c['key']: c for c in CONDITIONS_MAP}

SIGMA_STUCK_MICRONS = 0.013543
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS = 0.00893
MASS_DIM_REFERENCE = 589.8

MPP = 0.102
FPS = 10
MAX_LAGTIME = 100

USE_LOCALIZATION_CORRECTION = True
USE_MASS_SCALING = True
R_SQUARED_THRESHOLD = 0.8

LABELS = [c['size'] for c in CONDITIONS_MAP]
PLOT_COLORS = ['#009ACD', '#009ACD', '#009ACD', '#FF00FF', '#C71585', '#000000']

PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_FIGURE_BASE = BASE_DATA_ROOT / 'Figures'

# ============================================================================
# CORE MSD FUNCTIONS (keeping your existing implementations)
# ============================================================================
def _msd_iter(pos, lagtimes):
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff_no_nan = np.nan_to_num(diff, nan=0)
        if not np.all(diff_no_nan == 0):
            yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))

def msd(pos, max_lagtime, sigma=None):
    max_lagtime = min(int(max_lagtime), len(pos) - 1)
    lagtimes = np.arange(1, max_lagtime + 1)
    pos_columns = ['x', 'y']
    result_columns = [f'<{p}>' for p in pos_columns] + [f'<{p}^2>' for p in pos_columns]
    resultmsd = list(_msd_iter(pos, lagtimes))
    lagtimes = lagtimes[:len(resultmsd)]
    result = pd.DataFrame(resultmsd, columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(1)
    if sigma is not None and np.isfinite(sigma):
        correction = 4 * (sigma ** 2)
        result['msd_corrected'] = result['msd'] - correction
        result['msd_corrected'] = result['msd_corrected'].clip(lower=0)
    else:
        result['msd_corrected'] = result['msd']
    return result

def calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_mass_scaling_flag):
    if not use_mass_scaling_flag or mass_particle <= 0 or mass_ref <= 0:
        return sigma_ref
    mass_ratio = mass_ref / mass_particle
    return sigma_ref * np.sqrt(mass_ratio)

def power_law(x, alpha, A):
    return A * np.power(x, alpha)

def powerfit(msd_df, start_index, end_index, fps=10, use_corrected=False):
    """ 
    Fit power law to MSD data slice using linear regression on log-transformed data. 
    Returns: [alpha, A], R² (R² is calculated on the log-transformed fit)
    """
    if not isinstance(msd_df, pd.DataFrame):
        return [np.nan, np.nan], 0.0

    # Determine which column to use for fitting
    if use_corrected and 'msd_corrected' in msd_df.columns:
        y_raw = msd_df['msd_corrected'].values
    elif 'msd' in msd_df.columns:
        y_raw = msd_df['msd'].values
    else:
        return [np.nan, np.nan], 0.0

    x_raw = msd_df.index.values.astype('float64')

    # Define the slice indices
    a = start_index
    b = end_index
    y_slice = y_raw[a:b]
    x_slice = x_raw[a:b]

    # Filter for valid, positive data points (CRITICAL for log transform)
    valid = (np.isfinite(y_slice)) & (y_slice > 0) & (x_slice > 0)
    y_fit = y_slice[valid]
    x_fit = x_slice[valid]

    if len(y_fit) < 2:
        return [np.nan, np.nan], 0.0

    try:
        # 1. Log-transform the data
        log_x = np.log10(x_fit)
        log_y = np.log10(y_fit)

        # 2. Perform linear regression: log(MSD) = alpha * log(Dt) + log(A)
        coeffs, ss_res, _, _, _ = np.polyfit(log_x, log_y, 1, full=True)
        
        # Extract parameters
        alpha = coeffs[0] # Slope
        log_A = coeffs[1] # Intercept
        A = 10**log_A      # Convert intercept back to prefactor A

        # Calculate R-squared of the linear fit (on the log scale)
        log_y_pred = alpha * log_x + log_A
        ss_res = np.sum((log_y - log_y_pred)**2)
        ss_tot = np.sum((log_y - np.mean(log_y))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

        return [alpha, A], r_squared
    except:
        return [np.nan, np.nan], 0.0

def imsd(file_path, mpp, fps, max_lagtime, sigma_ref, mass_ref, max_length, use_scaling_flag):
    try:
        traj = pd.read_csv(file_path)
    except Exception as e:
        return None, None, None, None, None, None, None
    pos_columns = ['x', 'y']
    ids = []
    msds_raw = []
    msds_corrected = []
    alpha = []
    alpha2 = []
    length = []
    mass_values = []
    sigma_values = []
    index = 0
    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
        if max_length is not None:
            pos = pos[0:max_length]
        if len(pos) < 10:
            continue
        mass_particle = ptraj['mass'].mean() if 'mass' in ptraj.columns else np.nan
        if USE_LOCALIZATION_CORRECTION:
            sigma_particle = calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_scaling_flag)
        else:
            sigma_particle = None
        msdtemp = msd(pos, max_lagtime, sigma=sigma_particle)
        if np.any(msdtemp.values == 0):
            msdtemp = msdtemp.replace(0, np.nan)
        msdtemp = msdtemp.dropna(axis='columns', how='all')
        if len(msdtemp.columns) <= 1:
            continue
        use_corrected_col = 'msd_corrected' if USE_LOCALIZATION_CORRECTION and 'msd_corrected' in msdtemp.columns else 'msd'
        msd_for_fit = pd.DataFrame({'msd': msdtemp[use_corrected_col]}, index=msdtemp.index)
        msd_for_fit.index = msd_for_fit.index.values.astype('float64') / float(fps)
        alphatemp, r_squaredtemp = powerfit(msd_for_fit, 0, 8, fps, use_corrected=False)
        alphatemp2, r_squaredtemp2 = powerfit(msd_for_fit, 19, 79, fps, use_corrected=False)
        
        
        # target_time_s   = 8.0
        # target_idx      = int(target_time_s * FPS)          # → 80
        # # Use a small window of ±0.1 s (±1 frame) for a robust fit
        # window_frames   = 2                                   # total width = 2 frames
        # half_window     = window_frames // 2                  # 1 frame on each side
        # start_idx       = max(target_idx - half_window, 0)    # → 79
        # end_idx         = min(target_idx + half_window, len(msd_for_fit) - 1)  # → 81
        
        # alphatemp2, r_squaredtemp2 = powerfit(
        #         msd_for_fit,
        #         start_idx,
        #         end_idx,
        #         fps,
        #         use_corrected=False
        # )
        
        if not (np.isfinite(alphatemp[0]) and np.isfinite(r_squaredtemp)):
            continue
        if r_squaredtemp >= R_SQUARED_THRESHOLD and alphatemp[0] > 0 and alphatemp2[0] > 0:
            alpha.append(alphatemp)
            alpha2.append(alphatemp2)
            msds_raw.append(msdtemp['msd'])
            msds_corrected.append(msdtemp.get('msd_corrected', msdtemp['msd']))
            mass_values.append(mass_particle)
            sigma_values.append(sigma_particle if sigma_particle is not None else np.nan)
            ids.append(index)
            index += 1
            length.append(len(pos))
    if not msds_raw:
        return None, None, None, None, None, None, None
    resultsmsd_raw = pd.concat(msds_raw, axis=1, keys=ids)
    resultsmsd_corrected = pd.concat(msds_corrected, axis=1, keys=ids)
    lagt = resultsmsd_raw.index.values.astype('float64') / float(fps)
    resultsmsd_raw.index = lagt
    resultsmsd_corrected.index = lagt
    alpha_df = pd.DataFrame(alpha)
    alpha2_df = pd.DataFrame(alpha2)
    length_df = pd.DataFrame(length)
    mass_df = pd.DataFrame(mass_values)
    sigma_df = pd.DataFrame(sigma_values)
    return resultsmsd_raw, resultsmsd_corrected, alpha_df, alpha2_df, length_df, mass_df, sigma_df

def process_directory(directory, mpp, fps, max_lagtime, sigma_ref, mass_ref, max_length, use_scaling_flag, noise_floor_msd=None):
    """ MODIFIED: Added noise_floor_msd parameter for filtering trajectories below noise floor """ 
    MSDcollected_raw = []
    MSDcollected_corrected = []
    alphacollected = []
    alphacollected2 = []
    lengthcollected = []
    masscollected = []
    sigmacollected = []
    # 1. Iterate through files and collect all track data
    for file_path in directory.glob('*.csv'):
        if not file_path.name.startswith('.'):
            result = imsd(file_path, mpp, fps, max_lagtime, sigma_ref, mass_ref, max_length, use_scaling_flag)
            msd_raw, msd_corr, alpha, alpha2, length, mass, sigma = result
            if msd_raw is not None:
                MSDcollected_raw.append(msd_raw)
                MSDcollected_corrected.append(msd_corr)
                alphacollected.append(alpha)
                alphacollected2.append(alpha2)
                lengthcollected.append(length)
                masscollected.append(mass)
                sigmacollected.append(sigma)
    if not MSDcollected_raw:
        return None
    
    # 2. Concatenate results
    MSDcollected_raw_df = pd.concat(MSDcollected_raw, axis=1)
    MSDcollected_corrected_df = pd.concat(MSDcollected_corrected, axis=1)

    # --- FIX: Ensure unique column indexing to avoid broadcasting error ---
    num_tracks = len(MSDcollected_corrected_df.columns)
    new_columns = list(range(num_tracks))
    MSDcollected_raw_df.columns = new_columns
    MSDcollected_corrected_df.columns = new_columns
    # --------------------------------------------------------------------
    
    Alphacollected_df = pd.concat(alphacollected, ignore_index=True)
    Alphacollected2_df = pd.concat(alphacollected2, ignore_index=True)
    if masscollected and sigmacollected:
        masscollected_df = pd.concat(masscollected, ignore_index=True)
        sigmacollected_df = pd.concat(sigmacollected, ignore_index=True)
        mass_data = masscollected_df[0]
        sigma_data = sigmacollected_df[0]
    else:
        mass_data = np.array([])
        sigma_data = np.array([])
    # ============================================================================
    # NEW: NOISE FLOOR FILTERING (LAG-TIME-WISE COMPARISON)
    # ============================================================================
    if noise_floor_msd is not None:
        valid_trajectories_list = []
        for col in MSDcollected_corrected_df.columns:
            trajectory_msd = MSDcollected_corrected_df[col]
            # Align indices before comparison
            common_index = trajectory_msd.index.intersection(noise_floor_msd.index)
            if len(common_index) == 0:
                valid_trajectories_list.append(False)
                continue
            # Get values at common indices and convert to numpy arrays for comparison
            trajectory_values = trajectory_msd.loc[common_index].values
            noise_values = noise_floor_msd.loc[common_index].values
            # Element-wise comparison at each lag time
            above_noise = trajectory_values > noise_values
            # Keep trajectory if it's above noise floor for at least 99% of lag times
            fraction_above = above_noise.sum() / len(above_noise) if len(above_noise) > 0 else 0
            valid_trajectories_list.append(fraction_above > 0.95) # 99% threshold
        # Create boolean series for filtering
        valid_trajectories = pd.Series(valid_trajectories_list, index=MSDcollected_corrected_df.columns)
        # Filter MSD data
        MSDcollected_raw_df = MSDcollected_raw_df.loc[:, valid_trajectories]
        MSDcollected_corrected_df = MSDcollected_corrected_df.loc[:, valid_trajectories]
        # Get valid indices for metadata
        valid_indices = [i for i, v in enumerate(valid_trajectories) if v]
        # Filter metadata
        Alphacollected_df = Alphacollected_df.iloc[valid_indices]
        Alphacollected2_df = Alphacollected2_df.iloc[valid_indices]
        if len(mass_data) > 0:
            mass_data = mass_data.iloc[valid_indices]
        if len(sigma_data) > 0:
            sigma_data = sigma_data.iloc[valid_indices]
        print(f" Filtered: {len(valid_trajectories_list) - sum(valid_trajectories_list)} trajectories below noise floor")
    # ============================================================================
    totaltracks = len(MSDcollected_raw_df.columns)
    # 3. Calculate Dapp and ensemble means
    Dapp_data = Alphacollected_df[1].values / 4.0
    Dapp2_data = Alphacollected2_df[1].values / 4.0
    alpha2_data = Alphacollected2_df[0].values
    ensemble_msd_raw = MSDcollected_raw_df.mean(axis=1)
    ensemble_msd_corrected = MSDcollected_corrected_df.mean(axis=1)
    # 4. Return complete results dictionary
    return {
        'msd_raw': MSDcollected_raw_df,
        'msd_corrected': MSDcollected_corrected_df,
        'ensemble_msd_raw': ensemble_msd_raw,
        'ensemble_msd_corrected': ensemble_msd_corrected,
        'alpha': Alphacollected_df[0].values,
        'alpha2': alpha2_data,
        'Dapp': Dapp_data,
        'Dapp2': Dapp2_data,
        'mass': mass_data,
        'sigma': sigma_data,
        'n_tracks': totaltracks
    }

# ============================================================================
# PLOTTING HELPER FUNCTIONS
# ============================================================================
def add_sig_bars_alpha(ax, data_list, actual_labels, alpha_thresh=0.05):
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) + 0.1
    spacing = 0.12
    p_vals = []
    n_groups = len(data_list)
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if 'beads' in actual_labels[i].lower() or 'beads' in actual_labels[j].lower():
                continue
            try:
                t_stat, p = stats.ttest_ind(data_list[i], data_list[j], nan_policy='omit')
                if p < alpha_thresh:
                    p_vals.append((p, i, j))
            except ValueError:
                continue
    p_vals.sort(key=lambda x: x[0])
    for k, (p, i, j) in enumerate(p_vals):
        if p < 0.0001:
            sig = "****"
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        else:
            sig = "*"
        h = base_h + k * spacing
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False)
        ax.text((i + 1 + j + 1) / 2, h + spacing * 0.2, sig,
                ha="center", va="bottom", fontsize=10, color="k")
    if p_vals:
        final_h = base_h + len(p_vals) * spacing
        ax.set_ylim(cur_ylim[0], final_h + spacing)

def add_sig_bars_dapp(ax, data_list, actual_labels, alpha_thresh=0.05):
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) * 1.5
    spacing_factor = 2.3
    p_vals = []
    n_groups = len(data_list)
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if 'beads' in actual_labels[i].lower() or 'beads' in actual_labels[j].lower():
                continue
            try:
                t_stat, p = stats.ttest_ind(data_list[i], data_list[j], nan_policy='omit')
                if p < alpha_thresh:
                    p_vals.append((p, i, j))
            except ValueError:
                continue
    p_vals.sort(key=lambda x: x[0])
    for k, (p, i, j) in enumerate(p_vals):
        if p < 0.0001:
            sig = "****"
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        else:
            sig = "*"
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
        pc.set_alpha(0.6)
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
            s=20,
            alpha=0.5,
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
    print(f"Sigma (Dim Dye, 100-500 bp): {SIGMA_DIM_MICRONS:.6f} μm")
    print(f"Mass reference: {MASS_STUCK_REFERENCE:.1f}")
    print(f"Mass scaling: {'ENABLED' if USE_MASS_SCALING else 'DISABLED'}")
    print(f"Localization correction: {'ENABLED' if USE_LOCALIZATION_CORRECTION else 'DISABLED'}")
    print(f"R² threshold: {R_SQUARED_THRESHOLD}\n")

    results_dict = {}
    directories = {}

    for condition in CONDITIONS_MAP:
        key = condition['key']
        path = BASE_DATA_ROOT / condition['base_folder'] / 'E_output_data' / condition['collected_dir'] / 'tm'
        directories[key] = path

    # ============================================================================
    # NEW: PROCESS BEADS FIRST TO ESTABLISH NOISE FLOOR
    # ============================================================================
    noise_floor_msd = None
    if 'beads' in directories:
        bead_directory = directories['beads']
        if bead_directory.is_dir():
            print("Processing beads (noise floor)...")
            # Process beads WITHOUT noise floor filtering (pass None)
            bead_result = process_directory(
                bead_directory, MPP, FPS, MAX_LAGTIME, 0.0, MASS_STUCK_REFERENCE, 
                MAX_TRAJ_LENGTHS[CONDITION_KEYS.index('beads')], False, 
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

            if label_key in ['MDA MB 231', 'MCF 10A', 'MCF 7']:
                sigma_to_use = SIGMA_DIM_MICRONS
                mass_to_use = MASS_DIM_REFERENCE
                use_scaling_temp = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION
                print(f"Processing {size_label} ({label_key})... using Fixed Sigma ({sigma_to_use:.4f} μm) & NO MASS SCALING (Dim Dye Correction).")
            else:
                sigma_to_use = SIGMA_STUCK_MICRONS
                mass_to_use = MASS_STUCK_REFERENCE
                use_scaling_temp = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION
                print(f"Processing {size_label} ({label_key})... using Stuck Bead Sigma ({sigma_to_use:.4f} μm) & MASS SCALING.")
            
            if not directory.is_dir():
                print(f"CRITICAL: Directory for {size_label} not found at {directory}")
                continue

            # ============================================================================
            # MODIFIED: Pass noise_floor_msd to enable filtering
            # ============================================================================
            result = process_directory(
                directory, MPP, FPS, MAX_LAGTIME, sigma_to_use, mass_to_use, 
                max_length, use_scaling_temp, 
                noise_floor_msd=noise_floor_msd  # Pass noise floor for filtering
            )
            # ============================================================================

            if result is not None:
                results_dict[label_key] = result
                print(f" Tracks accepted: {result['n_tracks']}")
                print(f" Mean mass: {np.mean(result['mass']):.1f}")
                print(f" Mean sigma: {np.mean(result['sigma']):.6f} μm")
            else:
                print(f" WARNING: No valid trajectories for {size_label}")

    if not results_dict:
        print("\nERROR: results_dict is empty. No data passed filters or directories were invalid.")
        return

    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    

 
    # ============================================================================
    # FIGURE 6: Cell Types Analysis (2x2 Layout)
    # ============================================================================
    print("\nGenerating Figure 6 (Cell Types - 2x2 Layout)...")
    
    cell_keys = ['MDA MB 231', 'MCF 7', 'MCF 10A']
    cell_labels = [LABELS[CONDITION_KEYS.index(k)] for k in cell_keys]
    cell_colors = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in cell_keys]
    cell_x_ticks = list(range(1, len(cell_keys) + 1))
    
    alpha_cell = [results_dict[k]['alpha2'] for k in cell_keys]
    dapp_cell = [results_dict[k]['Dapp2'] for k in cell_keys]
    
    fig6, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    # PANEL A: MSD Ensemble
    ax_msd = axs[0]
    for key in cell_keys:
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_corrected']
        n_tracks = data.get('n_tracks', 0)
        label_with_n = f"{LABELS[original_index]} ($n$ = {n_tracks})"
        
        ax_msd.plot(ensemble_msd.index, ensemble_msd.values, color=PLOT_COLORS[original_index],
            marker='o', markeredgecolor='black', markerfacecolor=PLOT_COLORS[original_index],
            linestyle='-', linewidth=2, markersize=7, label=label_with_n, zorder=3)
    
    # ax_msd.axvspan(1e-1, 1.0, alpha=0.15, color='gray', zorder=0)
    # ax_msd.axvspan(10.0, 1e2, alpha=0.15, color='gray', zorder=0)
    # ax_msd.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    # ax_msd.axvline(x=10.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax_msd.set_xscale('log')
    ax_msd.set_yscale('log')
    ax_msd.set_ylim((2e-4, 1e-1))
    ax_msd.set_xlim((1e-1, 1e1))
    ax_msd.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\text{real}}$ ($\mu\mathrm{m}^2$)', fontsize=14)
    ax_msd.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax_msd.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd.grid(True, alpha=0.3, linestyle='--', zorder=2)
    for spine in ax_msd.spines.values():
        spine.set_linewidth(1.5)
    ax_msd.legend(loc='upper left', frameon=True, fontsize=10, ncol=1)
    ax_msd.text(-0.15, 1.05, 'A', transform=ax_msd.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    # PANEL B: Alpha Violin Plot
    ax_alpha = axs[1]
    
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
        body.set_alpha(0.6)
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
            alpha=0.8,
            marker='o',
            edgecolor='k',
            s=20,
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
    
    # PANEL C: Percentage Bar Chart
    ax_pct = axs[2]
    cell_keys_no_beads = [k for k in cell_keys if k != 'beads']
    cell_labels_no_beads = [LABELS[CONDITION_KEYS.index(k)] for k in cell_keys_no_beads]
    alpha_no_beads = [results_dict[k]['alpha2'] for k in cell_keys_no_beads]
    confined = [np.mean(arr < 0.4) * 100 for arr in alpha_no_beads]
    subdiff = [np.mean((arr >= 0.4) & (arr < 1.0)) * 100 for arr in alpha_no_beads]
    superdiff = [np.mean(arr >= 1.0) * 100 for arr in alpha_no_beads]
    x = np.arange(len(cell_labels_no_beads))
    width = 0.25 

    
   
    bar1_atp = ax_pct.bar(x - width, confined, width, label='Confined (<0.4)', 
                              color='#332288', edgecolor='black', linewidth=1.5)
    bar2_atp = ax_pct.bar(x, subdiff, width, label='Sub-diffusive (0.4-1)', 
                              color='#88CCEE', edgecolor='black', linewidth=1.5)
    bar3_atp = ax_pct.bar(x + width, superdiff, width, label='Super-diffusive (>1)', 
                              color='#44AA99', edgecolor='black', linewidth=1.5)
    
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(cell_labels_no_beads, rotation=45, ha='right', fontsize=12)
    ax_pct.set_ylabel('Percentage (%)', fontsize=14)
    ax_pct.set_ylim(0, 100)
    ax_pct.legend(frameon=False, fontsize=10)
    ax_pct.yaxis.set_ticks_position('both')
    ax_pct.xaxis.set_ticks_position('both')
    for spine in ax_pct.spines.values():
        spine.set_linewidth(1.5)
    ax_pct.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_pct.grid(False)
    ax_pct.text(-0.15, 1.05, 'C', transform=ax_pct.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    # --- PANEL D: Dapp Violin Plot ---
    ax_dapp = axs[3]
    
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
    ax_dapp.text(-0.15, 1.05, 'D', transform=ax_dapp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_dapp.grid(False)

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
        
        # Calculate percentages from alpha2 data (where 0.4 and 1.0 are boundaries)
        alpha2_arr = calc['alpha2']
        caged_calc = np.mean(alpha2_arr < 0.4) * 100
        subdiff_calc = np.mean((alpha2_arr >= 0.4) & (alpha2_arr < 1.0)) * 100
        superdiff_calc = np.mean(alpha2_arr >= 1.0) * 100
        
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
    # NEW FIGURE: ATP Depletion Analysis (2x2 Layout)
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
    
    # Create figure
    fig_atp, axs_atp = plt.subplots(2, 2, figsize=(12, 10))
    axs_atp = axs_atp.flatten()
    
    # --- PANEL A: MSD Ensemble ---
    ax_msd_atp = axs_atp[0]
    for key in atp_keys:
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_corrected']
        n_tracks = data.get('n_tracks', 0)
        label_with_n = f"{LABELS[original_index]} ($n$ = {n_tracks})"
        
        ax_msd_atp.plot(
            ensemble_msd.index,
            ensemble_msd.values,
            color=PLOT_COLORS[original_index],
            marker='o',
            markeredgecolor='black',
            markerfacecolor=PLOT_COLORS[original_index],
            linestyle='-',
            linewidth=2,
            markersize=7,
            label=label_with_n,
            zorder=3
        )
    
    # ax_msd_atp.axvspan(1e-1, 1.0, alpha=0.15, color='gray', zorder=0)
    # ax_msd_atp.axvspan(10.0, 1e2, alpha=0.15, color='gray', zorder=0)
    # ax_msd_atp.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    # ax_msd_atp.axvline(x=10.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    
    ax_msd_atp.set_xscale('log')
    ax_msd_atp.set_yscale('log')
    ax_msd_atp.set_ylim((2e-4, 1e-1))
    ax_msd_atp.set_xlim((1e-1, 1e1))
    ax_msd_atp.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\text{real}}$ ($\mu\mathrm{m}^2$)', fontsize=14)
    ax_msd_atp.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax_msd_atp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_msd_atp.grid(True, alpha=0.3, linestyle='--', zorder=2)
    for spine in ax_msd_atp.spines.values():
        spine.set_linewidth(1.5)
    ax_msd_atp.legend(loc='upper left', frameon=True, fontsize=10, ncol=1)
    ax_msd_atp.text(-0.15, 1.05, 'A', transform=ax_msd_atp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    # --- PANEL B: Alpha Violin Plot (ATP) ---
    ax_alpha_atp = axs_atp[1]
    
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
        body.set_alpha(0.6)
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
            alpha=0.8,
            marker='o',
            edgecolor='k',
            s=20,
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
    ax_alpha_atp.set_xticklabels(atp_labels, rotation=45, ha='right', fontsize=12)
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
    ax_pct_atp = axs_atp[2]
    
    # Calculate percentages (excluding beads)
    atp_keys_no_beads = ['ATP depleted', 'Control']
    atp_labels_no_beads = [LABELS[CONDITION_KEYS.index(k)] for k in atp_keys_no_beads]
    alpha_atp_no_beads = [results_dict[k]['alpha2'] for k in atp_keys_no_beads]
    
    confined_atp = [np.mean(arr < 0.4) * 100 for arr in alpha_atp_no_beads]
    subdiff_atp = [np.mean((arr >= 0.4) & (arr < 1.0)) * 100 for arr in alpha_atp_no_beads]
    superdiff_atp = [np.mean(arr >= 1.0) * 100 for arr in alpha_atp_no_beads]
    
    x_atp = np.arange(len(atp_labels_no_beads))
    
    bar1_atp = ax_pct_atp.bar(x_atp - width, confined_atp, width, label='Confined (<0.4)', 
                              color='#332288', edgecolor='black', linewidth=1.5)
    bar2_atp = ax_pct_atp.bar(x_atp, subdiff_atp, width, label='Sub-diffusive (0.4-1)', 
                              color='#88CCEE', edgecolor='black', linewidth=1.5)
    bar3_atp = ax_pct_atp.bar(x_atp + width, superdiff_atp, width, label='Super-diffusive (>1)', 
                              color='#44AA99', edgecolor='black', linewidth=1.5)
    
    ax_pct_atp.set_xticks(x_atp)
    ax_pct_atp.set_xticklabels(atp_labels_no_beads, rotation=45, ha='right', fontsize=12)
    ax_pct_atp.set_ylabel('Percentage (%)', fontsize=14)
    ax_pct_atp.set_ylim(0, 100)
    ax_pct_atp.legend(frameon=False, fontsize=10)
    ax_pct_atp.yaxis.set_ticks_position('both')
    ax_pct_atp.xaxis.set_ticks_position('both')
    for spine in ax_pct_atp.spines.values():
        spine.set_linewidth(1.5)
    ax_pct_atp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_pct_atp.grid(False)
    ax_pct_atp.text(-0.15, 1.05, 'C', transform=ax_pct_atp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    # --- PANEL D: Dapp Violin Plot ---
    ax_dapp_atp = axs_atp[3]
    
    log_violin(ax_dapp_atp, dapp_atp, atp_x_ticks, atp_colors)
    
    add_sig_bars_dapp(ax_dapp_atp, dapp_atp, atp_labels)
    
    ax_dapp_atp.yaxis.set_ticks_position('both')
    ax_dapp_atp.xaxis.set_ticks_position('both')
    for spine in ax_dapp_atp.spines.values():
        spine.set_linewidth(1.5)
    ax_dapp_atp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_dapp_atp.set_xticks(atp_x_ticks)
    ax_dapp_atp.set_xticklabels(atp_labels, rotation=45, ha='right', fontsize=12)
    ax_dapp_atp.set_ylabel(r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)', fontsize=14)
    ax_dapp_atp.text(-0.15, 1.05, 'D', transform=ax_dapp_atp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_dapp_atp.grid(False)
    
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
        
        # Calculate percentages from alpha2 data (where 0.4 and 1.0 are boundaries)
        alpha2_arr = calc['alpha2']
        caged_calc = np.mean(alpha2_arr < 0.4) * 100
        subdiff_calc = np.mean((alpha2_arr >= 0.4) & (alpha2_arr < 1.0)) * 100
        superdiff_calc = np.mean(alpha2_arr >= 1.0) * 100
        
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