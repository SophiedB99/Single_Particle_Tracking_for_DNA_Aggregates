"""
MSD Analysis with Per-Particle Localization Correction
Applies individual sigma corrections based on particle brightness
@author: sophiedeboer (Refactored by Anara)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import itertools
from matplotlib.ticker import LogLocator, NullFormatter
from pathlib import Path
import warnings
import os
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ============================================================================
# CONFIGURATION 
# ============================================================================
# --- Data Configuration ---
BASE_DATA_ROOT = Path('F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data')
# Define the full set of experiments, including paths and parameters
CONDITIONS_MAP = [ 
    { 'key': 'MDA MB 231', 'size': 'MDA MB 231', 'base_folder': 'Aswin Data/Aswin Data MDA-MB-231 Corrected for Drift', 'collected_dir': 'Collected_MDAMB231', 'max_length': 300 }, 
    { 'key': 'MCF 7', 'size': 'MCF 7', 'base_folder': 'Aswin Data/Aswin Data MCF 7 Corrected for Drift', 'collected_dir': 'Collected_MCF7', 'max_length': 300 }, 
    { 'key': 'MCF 10A', 'size': 'MCF 10A', 'base_folder': 'Aswin Data/Aswin Data MCF 10A Corrected for Drift', 'collected_dir': 'Collected_MCF10A', 'max_length': 300 }, 
    { 'key': 'ATP depleted', 'size': 'ATP depleted', 'base_folder': 'MDA-MB-231/ATP Depletion MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_ATPdepl_MDAMB231', 'max_length': 800 }, 
    { 'key': 'Control', 'size': 'Control', 'base_folder': 'MDA-MB-231/Cy3 MDA-MB-231/500 bp_most_recent', 'collected_dir': 'Collected_Cy3_MDAMB231', 'max_length': 800 }, 
    { 'key': 'beads', 'size': 'beads', 'base_folder': 'Control Experiments_Beads/0.22 um', 'collected_dir': 'Collected_0.22um', 'max_length': 800 }
]
# Create auxiliary lists 
CONDITION_KEYS = [c['key'] for c in CONDITIONS_MAP]
MAX_TRAJ_LENGTHS = [c['max_length'] for c in CONDITIONS_MAP]

# --- Aggregation Configuration ---
COMBINED_KEY = "MDA MB 231 + Control"
COMBINED_LABEL = "MDA MB 231"
CONDITION_DICT = {c['key']: c for c in CONDITIONS_MAP}

# --- Reference Values ---
SIGMA_STUCK_MICRONS = 0.013543 # um 
MASS_STUCK_REFERENCE = 3381.5 
SIGMA_DIM_MICRONS = 0.00893 
MASS_DIM_REFERENCE = 589.8 

# --- Experimental & Calculation Parameters ---
MPP = 0.102 # Microns per pixel
FPS = 10 # Frames per second
MAX_LAGTIME = 100 # Maximum lagtime for MSD calculation

# --- Correction & Filtering Options ---
USE_LOCALIZATION_CORRECTION = True 
USE_MASS_SCALING = True 
R_SQUARED_THRESHOLD = 0.8 # Minimum R² for trajectory acceptance

# --- Plotting Parameters (Mutable Global Lists) ---
LABELS = [c['size'] for c in CONDITIONS_MAP]
PLOT_COLORS = ['#00FFFF', '#00BFFF', '#009ACD', '#FF00FF', '#C71585', '#000000'] 
PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_FIGURE_BASE = BASE_DATA_ROOT / 'Figures'

# ============================================================================
# CORE MSD AND FITTING FUNCTIONS
# ============================================================================
def _msd_iter(pos, lagtimes): 
    """Iterator for MSD calculation."""
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff_no_nan = np.nan_to_num(diff, nan=0)
        if not np.all(diff_no_nan == 0):
            yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))

def msd(pos, max_lagtime, sigma=None): 
    """ Compute MSD with optional localization correction. """
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
    """ Scale sigma based on particle brightness. """
    if not use_mass_scaling_flag or mass_particle <= 0 or mass_ref <= 0:
        return sigma_ref
    mass_ratio = mass_ref / mass_particle
    return sigma_ref * np.sqrt(mass_ratio)

def power_law(x, alpha, A): 
    """Power law function for MSD fitting"""
    return A * np.power(x, alpha)

# CRITICAL MODIFICATION: Implement Linear Regression in Log-Space
# def powerfit(msd_df, start_index, end_index, fps=10, use_corrected=False):
#     """ Fit power law to MSD data slice using linear regression on log-transformed data. 
#     Returns: [alpha, A], R² (R² is calculated on the log-transformed fit) 
#     """
#     if not isinstance(msd_df, pd.DataFrame):
#         return [np.nan, np.nan], 0.0

#     # Determine which column to use for fitting
#     if use_corrected and 'msd_corrected' in msd_df.columns:
#         y_raw = msd_df['msd_corrected'].values
#     elif 'msd' in msd_df.columns:
#         y_raw = msd_df['msd'].values
#     else:
#         return [np.nan, np.nan], 0.0

#     # --- FIX: Ensure y_raw is float type before calling isfinite ---
#     try:
#         y_raw = y_raw.astype('float64')
#     except ValueError:
#         # Handle cases where conversion fails, perhaps logging or returning NaN
#         return [np.nan, np.nan], 0.0
    
#     x_raw = msd_df.index.values.astype('float64')

#     # Define the slice indices
#     a = start_index
#     b = end_index
#     y_slice = y_raw[a:b]
#     x_slice = x_raw[a:b]

#     # Filter for valid, positive data points (CRITICAL for log transform)
#     valid = (np.isfinite(y_slice)) & (y_slice > 0) & (x_slice > 0) # Line 119 in your original traceback
#     y_fit = y_slice[valid]
#     x_fit = x_slice[valid]

#     if len(y_fit) < 2:
#         return [np.nan, np.nan], 0.0

#     try:
#         # 1. Log-transform the data
#         log_x = np.log10(x_fit)
#         log_y = np.log10(y_fit)

#         # 2. Perform linear regression: log(MSD) = alpha * log(Dt) + log(A)
#         # polyfit returns [slope (alpha), intercept (log(A))]
#         coeffs, ss_res_list, _, _, _ = np.polyfit(log_x, log_y, 1, full=True)

#         # Extract parameters
#         alpha = coeffs[0] # Slope
#         log_A = coeffs[1] # Intercept
#         A = 10**log_A # Convert intercept back to prefactor A

#         # Calculate R-squared of the linear fit (on the log scale)
#         log_y_pred = alpha * log_x + log_A
#         ss_res = np.sum((log_y - log_y_pred)**2)
#         ss_tot = np.sum((log_y - np.mean(log_y))**2)
#         r_squared = 1.0 - (ss_res / ss_tot) if ss_tot!= 0 else 1.0

#         return [alpha, A], r_squared
#     except:
#         return [np.nan, np.nan], 0.0


def powerfit(msd_df, start_index, end_index, fps=10, use_corrected=False):
    """ Fit power law to MSD data slice. Returns: [alpha, A], R² """
    if not isinstance(msd_df, pd.DataFrame):
        return [np.nan, np.nan], 0.0

    # Determine which column to use for fitting
    if use_corrected and 'msd_corrected' in msd_df.columns:
        y = msd_df['msd_corrected'].values
    elif 'msd' in msd_df.columns:
        y = msd_df['msd'].values
    else:
        return [np.nan, np.nan], 0.0

    x = msd_df.index.values.astype('float64')

    # Define the slice indices
    a = start_index
    b = end_index
    
    y_slice = y[a:b]
    x_slice = x[a:b]
    x_slice = np.array(x_slice, dtype=float)
    y_slice = np.array(y_slice, dtype=float)

    # Filter for valid, positive data points
    valid = (np.isfinite(y_slice)) & (y_slice > 0) & (x_slice > 0)
    y_fit = y_slice[valid]
    x_fit = x_slice[valid]

    if len(y_fit) < 2:
        return [np.nan, np.nan], 0.0

    try:
        # Initial guess: [subdiffusive (0.5), small prefactor (0.01)]
        pars, cov = curve_fit(f=power_law, xdata=x_fit, ydata=y_fit, 
                              p0=[0.5, 0.01], bounds=([0, 0], [2, np.inf]))
        
        residuals = y_fit - power_law(x_fit, *pars)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot!= 0 else 1.0
        
        return pars, r_squared
    except:
        return [np.nan, np.nan], 0.0


def imsd(file_path, mpp, fps, max_lagtime, sigma_ref, mass_ref, max_length, use_scaling_flag): 
    """ Process individual file with per-particle MSD correction. """
    try: 
        traj = pd.read_csv(file_path)
    except Exception as e:
        return None, None, None, None, None, None, None
    
    pos_columns = ['x', 'y'] 
    ids, msds_raw, msds_corrected, alpha, alpha2, length, mass_values, sigma_values = [], [], [], [], [], [], [], []
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
        
        if not (np.isfinite(alphatemp[0]) and np.isfinite(r_squaredtemp)): continue
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

def process_directory(directory, mpp, fps, max_lagtime, sigma_ref, mass_ref, max_length, use_scaling_flag): 
    MSDcollected_raw, MSDcollected_corrected, alphacollected, alphacollected2, lengthcollected, masscollected, sigmacollected = [], [], [], [], [], [], []
    
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
        
    MSDcollected_raw_df = pd.concat(MSDcollected_raw, axis=1)
    MSDcollected_corrected_df = pd.concat(MSDcollected_corrected, axis=1)
    Alphacollected_df = pd.concat(alphacollected, ignore_index=True)
    Alphacollected2_df = pd.concat(alphacollected2, ignore_index=True)
    
    if masscollected and sigmacollected:
        masscollected_df = pd.concat(masscollected, ignore_index=True)
        sigmacollected_df = pd.concat(sigmacollected, ignore_index=True)
        mass_data = masscollected_df[0]
        sigma_data = sigmacollected_df[0]
    else:
        mass_data, sigma_data = np.array([]), np.array([])
        
    totaltracks = len(MSDcollected_raw_df.columns)
    
    Dapp_data = Alphacollected_df[1].values / 4.0 
    Dapp2_data = Alphacollected2_df[1].values / 4.0 
    alpha2_data = Alphacollected2_df[0].values
    ensemble_msd_raw = MSDcollected_raw_df.mean(axis=1)
    ensemble_msd_corrected = MSDcollected_corrected_df.mean(axis=1)
    
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
# NEW PLOTTING HELPER FUNCTIONS 
# ============================================================================
# MODIFIED: Accepts labels_list explicitly 
def plot_msd_ensemble(ax, results_dict, keys_to_plot, condition_keys, plot_colors, labels_list, mkr, panel_label): 
    """Plots corrected MSD ensemble curves for a specific subset of keys."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim((2e-4, 1e-1))
    ax.set_xlim((1e-1, 1e1))
    
    ax.axvspan(1e-1, 1.0, alpha=0.15, color='gray', zorder=0)
    ax.axvspan(10.0, 1e2, alpha=0.15, color='gray', zorder=0)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax.axvline(x=10.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    
    handles = []
    labels_output = [] 

    for i, key in enumerate(keys_to_plot):
        if key in results_dict:
            try:
                # Find index within the lists passed to this function
                original_index = condition_keys.index(key) 
            except ValueError:
                print(f"Error: Key '{key}' not found in current CONDITION_KEYS list for plotting.")
                continue
            
            data = results_dict[key]
            ensemble_msd_corrected = data['ensemble_msd_corrected']
            current_label = labels_list[original_index] 
            n_tracks = data.get('n_tracks', 0)
            label_with_n = f"{current_label} ($n$ = {n_tracks})"
            
            line, = ax.plot(
                ensemble_msd_corrected.index,
                ensemble_msd_corrected.values,
                color=plot_colors[original_index],
                marker=mkr[original_index],
                markeredgecolor='black',
                markerfacecolor=plot_colors[original_index],
                linestyle='-',
                linewidth=2,
                markersize=7,
                label=label_with_n,
                zorder=3
            )
            handles.append(line)
            labels_output.append(label_with_n)

    ax.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\text{real}}$ ($\mu\mathrm{m}^2$)', fontsize=14)
    ax.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=2)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax.legend(handles, labels_output, loc='upper left', frameon=True, fontsize=10, ncol=1) 

def add_sig_bars_alpha(ax, data_list):
    y_max = 0
    for data in data_list:
        if len(data) > 0:
            y_max = max(y_max, np.max(data))
    current_y_max = max(ax.get_ylim()[1], y_max)
    n_groups = len(data_list)
    initial_offset = current_y_max + 0.1 
    stack_spacing = 0.1 
    
    p_values = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            try:
                t_stat, p_val = stats.ttest_ind(data_list[i], data_list[j])
                p_values.append((p_val, i, j))
            except ValueError:
                continue
    p_values.sort(key=lambda x: x[0], reverse=True) 
    
    for k, (p_val, i, j) in enumerate(p_values):
        sig = "ns"
        if p_val < 0.0001: sig = "****"
        elif p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        
        h = initial_offset + k * stack_spacing 
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False) 
        ax.text((i + 1 + j + 1) / 2, h + stack_spacing * 0.1, f'{sig}', ha="center", va="bottom", fontsize=10, color="k")

    if p_values:
        final_y_max = initial_offset + len(p_values) * stack_spacing
        ax.set_ylim(ax.get_ylim()[0], final_y_max + 0.1)

def plot_alpha_boxplot(ax, results_dict, keys_to_plot, condition_keys, labels_map, plot_colors, jitter=0.05, panel_label=None): 
    """Plots alpha box plot for a specific subset of keys."""
    data_list, actual_keys_subset, actual_indices_subset = [], [], []
    for k in keys_to_plot:
        if k in results_dict:
            data_list.append(results_dict[k]['alpha2'])
            actual_keys_subset.append(k)
            try:
                actual_indices_subset.append(condition_keys.index(k))
            except ValueError:
                continue
    
    if not data_list:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center')
        ax.set_ylim(-0.08, 2.5)
        return

    # Map labels using the determined indices
    actual_labels_subset = [labels_map[condition_keys.index(k)] for k in actual_keys_subset]
    x_ticks = list(range(1, len(data_list) + 1)) 

    bp = ax.boxplot(
        data_list,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(linewidth=1.5)
    )

    for i, data_arr in enumerate(data_list):
        original_index = actual_indices_subset[i]
        ax.scatter(
            np.full_like(data_arr, x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(data_arr)),
            data_arr,
            color=plot_colors[original_index],
            alpha=0.8,
            marker='o',
            edgecolor='k',
            zorder=10
        )
        bp['boxes'][i].set_facecolor(plot_colors[original_index])
        bp['boxes'][i].set_alpha(0.6)
        
    add_sig_bars_alpha(ax, data_list) 

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'Normal diffusion ($\alpha=1$)', zorder=0)
    ax.axhline(0.4, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7, label=r'Threshold for confinement ($\alpha<0.4$)', zorder=0)
    
    ax.set_ylim(-0.08, 2.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(actual_labels_subset, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=14)
    ax.grid(False)
    ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    if panel_label:
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_analysis():
    global CONDITION_KEYS, LABELS, PLOT_COLORS
    
    print("\n" + "="*80)
    print("MSD ANALYSIS WITH PER-PARTICLE LOCALIZATION CORRECTION")
    print("="*80)
    print(f"Sigma (from stuck beads): {SIGMA_STUCK_MICRONS:.6f} μm")
    print(f"R² threshold: {R_SQUARED_THRESHOLD}\n")
    
    results_dict = {}
    directories = {}
    
    # --- 1. Calculate all directory paths based on the map ---
    for condition in CONDITIONS_MAP:
        key = condition['key']
        path = BASE_DATA_ROOT / condition['base_folder'] / 'E_output_data' / condition['collected_dir'] / 'tm'
        print(f"Path for {key}: {path}")
        directories[key] = path

    # --- 2. Iterate and Process Data ---
    for i, label_key in enumerate(CONDITION_KEYS):
        if label_key in directories:
            directory = directories[label_key]
            max_length = MAX_TRAJ_LENGTHS[i]
            size_label = CONDITION_DICT[label_key]['size']
            
            # Conditional Sigma Assignment and Scaling Flag
            if label_key in ['MDA MB 231', 'MCF 10A', 'MCF 7']: 
                sigma_to_use = SIGMA_DIM_MICRONS
                mass_to_use = MASS_DIM_REFERENCE
                use_scaling_temp = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION
            elif label_key == 'beads': 
                sigma_to_use = 0.0
                use_scaling_temp = False
            else: 
                sigma_to_use = SIGMA_STUCK_MICRONS
                mass_to_use = MASS_STUCK_REFERENCE
                use_scaling_temp = USE_MASS_SCALING and USE_LOCALIZATION_CORRECTION
            
            if directory.is_dir():
                result = process_directory(directory, MPP, FPS, MAX_LAGTIME, sigma_to_use, mass_to_use, max_length, use_scaling_temp)
                if result is not None:
                    results_dict[label_key] = result
    
    # --- FIGURE 8: MDA MB 231 vs Control (PLOT BEFORE AGGREGATION) ---
    keys_figure_8 = ['MDA MB 231', 'Control']
    original_condition_keys = [c['key'] for c in CONDITIONS_MAP]
    original_labels = [c['size'] for c in CONDITIONS_MAP]
    
    if all(key in results_dict for key in keys_figure_8):
        # Determine specific colors/indices for the plot using the immutable map
        fig8_indices = [original_condition_keys.index(k) for k in keys_figure_8]
        fig8_colors = [PLOT_COLORS[i] for i in fig8_indices]
        fig8_labels = [original_labels[i] for i in fig8_indices]
        labels_fig8 = ['YOYO-1', 'Cy3']
        fig8_mkrs = ['o'] * len(keys_figure_8)

        fig8, axes8 = plt.subplots(1, 2, figsize=(12, 5)) 
        print("\n" + "="*80)
        print(f"Saving Figure 8 (Individual MDA MB 231 vs Control MSD)...")
        
        # Panel A: MSD Ensemble
        plot_msd_ensemble(
            ax=axes8[0], 
            results_dict=results_dict, 
            keys_to_plot=keys_figure_8, 
            condition_keys=keys_figure_8, # Pass only the plotting keys as the reference
            plot_colors=fig8_colors,
            labels_list=labels_fig8,
            mkr=fig8_mkrs, 
            panel_label='A'
        )
        
        # Panel B: Alpha Boxplot
        plot_alpha_boxplot(
            ax=axes8[1], 
            results_dict=results_dict, 
            keys_to_plot=keys_figure_8, 
            condition_keys=keys_figure_8, 
            labels_map=labels_fig8, # Pass the correct labels map for boxplot
            plot_colors=fig8_colors, 
            panel_label='B'
        )
        #axes8[1].set_title(r'$\alpha$(2-8 s) MDA MB 231 500 bp, YOYO-1 vs Cy3 ', fontsize=12)
        plt.tight_layout()
        fig8.savefig(OUTPUT_FIGURE_BASE / "Figure8_MDA231_vs_Control_2Panel.png", dpi=600, bbox_inches='tight')
        print("Figure 8 (2-panel) generated successfully.")
    
    # -------------------------------------------------------------------------
    # === START AGGREGATION LOGIC (Run AFTER Figure 8) ===
    if 'MDA MB 231' in results_dict and 'Control' in results_dict:
        data_231 = results_dict['MDA MB 231']
        data_control = results_dict['Control']
        
        alpha_combined = np.concatenate([data_231['alpha'], data_control['alpha']])
        alpha2_combined = np.concatenate([data_231['alpha2'], data_control['alpha2']])
        Dapp_combined = np.concatenate([data_231['Dapp'], data_control['Dapp']])
        Dapp2_combined = np.concatenate([data_231['Dapp2'], data_control['Dapp2']])
        mass_combined = np.concatenate([data_231['mass'], data_control['mass']])
        sigma_combined = np.concatenate([data_231['sigma'], data_control['sigma']])
        n_tracks_combined = data_231['n_tracks'] + data_control['n_tracks']
        
        ensemble_msd_corrected_combined = (
            data_231['ensemble_msd_corrected'] * data_231['n_tracks'] + 
            data_control['ensemble_msd_corrected'] * data_control['n_tracks'] 
        ) / n_tracks_combined
        
        results_dict[COMBINED_KEY] = {
            'ensemble_msd_corrected': ensemble_msd_corrected_combined,
            'alpha': alpha_combined,
            'alpha2': alpha2_combined,
            'Dapp': Dapp_combined,
            'Dapp2': Dapp2_combined,
            'mass': mass_combined,
            'sigma': sigma_combined,
            'n_tracks': n_tracks_combined
        }
        
        # Remove redundant individual keys
        del results_dict['MDA MB 231']
        del results_dict['Control']
        
        print(f"Aggregated MDA MB 231 ({data_231['n_tracks']} tracks) and Control ({data_control['n_tracks']} tracks) into {COMBINED_KEY} ({n_tracks_combined} total tracks).")

        # --- Update Global Lists for Plotting Indexing (Affects Figure 7) ---
        indices_to_remove = sorted([original_condition_keys.index(k) for k in keys_figure_8], reverse=True)
        for idx in indices_to_remove:
            CONDITION_KEYS.pop(idx)
            LABELS.pop(idx)
            PLOT_COLORS.pop(idx)
        
        idx_231_original = original_condition_keys.index('MDA MB 231') 
        CONDITION_KEYS.insert(idx_231_original, COMBINED_KEY)
        LABELS.insert(idx_231_original, COMBINED_LABEL)
        PLOT_COLORS.insert(idx_231_original, PLOT_COLORS[idx_231_original]) 

    # === END AGGREGATION LOGIC ===

    # ============================================================================
    # PLOTTING PREPARATION (for Figure 7 using aggregated data)
    # ============================================================================
    processed_keys = [k for k in CONDITION_KEYS if k in results_dict]
    if not results_dict: 
        print("\nERROR: results_dict is empty. No data passed filters or directories were invalid.")
        return 

    # --- FIGURE 7: Targeted 2x2 MSD/Alpha Comparison (A-D) ---
    keys_row1_plot = [COMBINED_KEY, 'MCF 7', 'MCF 10A', 'beads'] 
    keys_row2_plot = ['ATP depleted', COMBINED_KEY, 'beads']
    
    print("\n" + "="*80)
    print("Saving Figure 7 (Targeted 2x2 MSD/Alpha Comparison)...")
    fig7, axes7 = plt.subplots(2, 2, figsize=(10, 10)) 
    axes7 = axes7.flatten() 
    
    # 1. Panel A (Top Left): MSD Ensemble - Cell Lines + Beads
    plot_msd_ensemble( 
        ax=axes7[0], 
        results_dict=results_dict, 
        keys_to_plot=keys_row1_plot, 
        condition_keys=CONDITION_KEYS, 
        plot_colors=PLOT_COLORS,
        labels_list=LABELS,
        mkr=['o'] * len(CONDITION_KEYS),
        panel_label='A' 
    )
    axes7[0].set_xlabel(r'') 
    
    # 2. Panel B (Top Right): Alpha Box Plot - Cell Lines + Beads
    plot_alpha_boxplot( 
        ax=axes7[1], 
        results_dict=results_dict, 
        keys_to_plot=keys_row1_plot, 
        condition_keys=CONDITION_KEYS, 
        labels_map=LABELS, 
        plot_colors=PLOT_COLORS, 
        panel_label='B' 
    )
    axes7[1].set_xlabel(r'') 
    
    # 3. Panel C (Bottom Left): MSD Ensemble - ATP Depletion vs Combined Control
    plot_msd_ensemble( 
        ax=axes7[2], 
        results_dict=results_dict, 
        keys_to_plot=keys_row2_plot, 
        condition_keys=CONDITION_KEYS, 
        plot_colors=PLOT_COLORS,
        labels_list=LABELS, 
        mkr=['o'] * len(CONDITION_KEYS),
        panel_label='C' 
    ) 
    
    # 4. Panel D (Bottom Right): Alpha Box Plot - ATP Depletion vs Combined Control
    plot_alpha_boxplot( 
        ax=axes7[3], 
        results_dict=results_dict, 
        keys_to_plot=keys_row2_plot, 
        condition_keys=CONDITION_KEYS, 
        labels_map=LABELS, 
        plot_colors=PLOT_COLORS, 
        panel_label='D' 
    ) 
    
    plt.tight_layout()
#    fig7.savefig(OUTPUT_FIGURE_BASE / "Figure_5.png", dpi=600, bbox_inches='tight')
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE: FIGURE 7 GENERATED")
    print(f"Figures saved to: {OUTPUT_FIGURE_BASE}")
    print("="*80)

if __name__ == '__main__': 
    run_analysis()
