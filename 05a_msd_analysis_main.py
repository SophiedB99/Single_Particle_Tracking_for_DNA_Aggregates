# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:57:55 2025
@author: sophiedeboer
"""
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
from matplotlib.ticker import LogLocator, NullFormatter
from pathlib import Path
import warnings
import os # Keep os imported for os.path.exists check in pseudocode
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# ============================================================================
# CONFIGURATION - UPDATE THESE FROM STUCK BEAD ANALYSIS
# ============================================================================
# --- Data Configuration ---
# Set the root directory for your experimental data.
BASE_DATA_ROOT = Path('F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data')
# Define the full set of experiments, including paths and parameters
CONDITIONS_MAP = [ { 'key': '100bp', 'size': '100 bp', 'base_folder': 'Aswin Data/Aswin Data CHO Corrected for Drift', 'collected_dir': 'Collected_100bp', 'max_length': 300 }, { 'key': '250bp', 'size': '250 bp', 'base_folder': 'Aswin Data/Aswin Data CHO Corrected for Drift', 'collected_dir': 'Collected_250bp', 'max_length': 200 }, { 'key': '500bp', 'size': '500 bp', 'base_folder': 'Aswin Data/Aswin Data CHO Corrected for Drift', 'collected_dir': 'Collected_500bp', 'max_length': 200 }, { 'key': '4713bp', 'size': '4713 bp', 'base_folder': 'Sophie Corrected for Gaussian Filter', 'collected_dir': 'Collected_pEGFP-C1', 'max_length': 800 }, { 'key': '7614bp', 'size': '7614 bp', 'base_folder': 'Sophie Corrected for Gaussian Filter', 'collected_dir': 'Collected_BPK4410', 'max_length': 800 }, { 'key': '9553bp', 'size': '9553 bp', 'base_folder': 'Sophie Corrected for Gaussian Filter', 'collected_dir': 'Collected_hCas9D10A', 'max_length': 700 }, { 'key': 'beads', 'size': 'beads', 'base_folder': 'Control Experiments_Beads/0.22 um', 'collected_dir': 'Collected_0.22um', 'max_length': 800 }
]
# Create auxiliary lists as requested by the user's iteration logic
CONDITION_KEYS = [c['key'] for c in CONDITIONS_MAP]
MAX_TRAJ_LENGTHS = [c['max_length'] for c in CONDITIONS_MAP]
CONDITION_LABELS = [c['size'] for c in CONDITIONS_MAP]
# Map keys to their condition dictionaries for easy lookup
CONDITION_DICT = {c['key']: c for c in CONDITIONS_MAP}
# --- Reference Values (From Step 1 Stuck Bead Analysis) ---
SIGMA_STUCK_MICRONS = 0.013543 # um # Replace with median sigma_combined from Step 1
MASS_STUCK_REFERENCE = 3381.5 # Replace with median mass from Step 1
# NEW CONSTANT FOR DIM FRAGMENTS (100-500 bp) - User-specified fixed sigma
SIGMA_DIM_MICRONS = 0.00893 #0.015280 #um 0.024897 --> this was calculated previously using the subtraction method
MASS_DIM_REFERENCE = 589.8 #median mass from YOYO-1 reference
# --- Experimental & Calculation Parameters ---
MPP = 0.102 # Microns per pixel
FPS = 10 # Frames per second
MAX_LAGTIME = 100 # Maximum lagtime for MSD calculation
# --- Correction & Filtering Options ---
USE_LOCALIZATION_CORRECTION = True # Enable/disable localization correction
USE_MASS_SCALING = True # Scale sigma based on particle brightness
R_SQUARED_THRESHOLD = 0.8 # Minimum R² for trajectory acceptance
# --- Plotting Parameters (Using requested colors and labels) ---

# --- NEW: Truncation Parameters ---
ENABLE_TRUNCATION = False  # Master switch for truncation
ENABLE_MASS_DROP = False  # Include mass drop detection
KEEP_FIRST_SEGMENT = False  # If True: keep segment before break; If False: reject entire particle
MAX_STEP_SIZE = 1  # Maximum allowed spatial jump (µm)
MAX_SINGLE_STEP_MASS_DROP = 0.5  # Maximum allowed fractional mass drop (30%)
MIN_SEGMENT_LENGTH = 30  # Minimum frames required after truncation


# New colors requested by user (7 items)
#PLOT_COLORS = ['#332288', '#88CCEE', '#44AA99', '#117733', '#DDCC77', '#CC6677', 'black']
PLOT_COLORS = ['#332288', '#332288', '#332288', '#CC6677', '#CC6677', '#CC6677', 'black']

LABELS = ['100 bp', '250 bp', '500 bp', '4713 bp', '7614 bp', '9553 bp', 'beads']
PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
OUTPUT_FIGURE_BASE = BASE_DATA_ROOT / 'Figures'
# ============================================================================
# FUNCTIONS (Definitions remain consistent with the previous cleaned script)
# ============================================================================

# ============================================================================
# NEW: TRAJECTORY TRUNCATION FUNCTIONS
# ============================================================================
def find_first_mass_discontinuity(mass_arr, max_drop_frac, ignore_initial=3, avg_window=3):
    """
    Detect first frame with abrupt mass drop.
    
    Parameters:
    -----------
    mass_arr : array-like
        Mass values for each frame
    max_drop_frac : float
        Maximum allowed fractional drop (e.g., 0.3 for 30%)
    ignore_initial : int
        Number of initial frames to ignore
    avg_window : int
        Window size for calculating baseline mass
    
    Returns:
    --------
    int : Index of first discontinuity, or len(mass_arr) if none found
    """
    mass_arr = np.asarray(mass_arr, dtype=float)
    if len(mass_arr) < 2 or np.isnan(mass_arr).all():
        return len(mass_arr)
    
    for i in range(ignore_initial, len(mass_arr) - 1):
        # Calculate baseline mass using preceding window
        win_start = max(0, i - avg_window + 1)
        baseline = np.nanmean(mass_arr[win_start:i + 1])
        nxt = mass_arr[i + 1]
        
        if np.isnan(baseline) or np.isnan(nxt):
            return i
        
        drop = baseline - nxt
        if baseline > 0 and drop > max_drop_frac * baseline:
            return i
            
    return len(mass_arr)

def find_first_spatial_discontinuity(pos, max_step_size):
    """
    Find the index of the first step larger than max_step_size or a missing frame (NaN).
    
    Parameters:
    -----------
    pos : ndarray
        Position array (N x 2) with x, y coordinates
    max_step_size : float
        Maximum allowed step size in microns
    
    Returns:
    --------
    int : Index of first discontinuity, or len(pos) if none found
    """
    original_len = len(pos)
    for i in range(original_len - 1):
        # Check for missing frame (NaN)
        if np.isnan(pos[i]).any() or np.isnan(pos[i+1]).any():
            return i
        
        # Check for large spatial jump
        step = np.linalg.norm(pos[i+1] - pos[i])
        if step > max_step_size:
            return i
            
    return original_len

def handle_truncation(pos, mass_series, max_step_size, max_mass_drop, keep_first_segment):
    """
    Unified truncation logic based on spatial jumps and mass drops.
    
    Parameters:
    -----------
    pos : ndarray
        Position array (N x 2)
    mass_series : pd.Series or None
        Mass values for each frame
    max_step_size : float
        Maximum allowed spatial step (µm)
    max_mass_drop : float
        Maximum allowed fractional mass drop
    keep_first_segment : bool
        If True, keep segment before break; if False, reject particle
    
    Returns:
    --------
    tuple : (processed_pos, segment_info) or (None, None) if rejected
    """
    original_len = len(pos)
    
    # 1. Find all discontinuity indices
    spatial_break = find_first_spatial_discontinuity(pos, max_step_size)
    mass_break = original_len
    
    if ENABLE_MASS_DROP and mass_series is not None:
        mass_arr = mass_series.values if hasattr(mass_series, 'values') else mass_series
        mass_break = find_first_mass_discontinuity(mass_arr, max_mass_drop)
    
    # 2. Determine earliest break
    final_break_index = min(spatial_break, mass_break)
    has_break = final_break_index < original_len
    
    # Initialize segment info
    segment_info = {
        'is_continuous': not has_break,
        'original_length': original_len,
        'first_break_at': final_break_index if has_break else None,
        'break_type': 'spatial' if spatial_break <= mass_break else 'mass',
        'used_length': original_len,
        'frames_discarded': 0
    }
    
    if not ENABLE_TRUNCATION:
        # Truncation disabled - keep everything
        return pos, segment_info
    
    if has_break:
        if keep_first_segment:
            # Keep the first clean segment
            new_pos = pos[:final_break_index]
            segment_info['used_length'] = len(new_pos)
            segment_info['frames_discarded'] = original_len - len(new_pos)
            return new_pos, segment_info
        else:
            # Reject the particle entirely
            return None, segment_info
    
    # No breaks - keep everything
    return pos, segment_info




def _msd_iter(pos, lagtimes):
    """Iterator for MSD calculation. Computes mean displacement (<dr>) and mean squared displacement (<dr^2>) components."""
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
    result_columns = [f'<{p}>' for p in pos_columns] + \
                     [f'<{p}^2>' for p in pos_columns]
    resultmsd = list(_msd_iter(pos, lagtimes))
    
    # resultmsd = resultmsd[:len(lagtimes)]
    lagtimes = lagtimes[:len(resultmsd)]
    result = pd.DataFrame(resultmsd, columns=result_columns, index=lagtimes)

    # Raw MSD: Sum of squared displacement components
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(1)

    # Corrected MSD
    if sigma is not None and np.isfinite(sigma):
        # Localization error is related to 4 * sigma^2 correction
        correction = 4 * (sigma ** 2)
        result['msd_corrected'] = result['msd'] - correction
        # Displacement cannot be negative
        result['msd_corrected'] = result['msd_corrected'].clip(lower=0)
    else:
        result['msd_corrected'] = result['msd']
        
    return result

def calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_mass_scaling_flag):
    """ Scale sigma based on particle brightness. Brighter particles have smaller sigma. """
    if not use_mass_scaling_flag or mass_particle <= 0 or mass_ref <= 0:
        return sigma_ref
    mass_ratio = mass_ref / mass_particle
    return sigma_ref * np.sqrt(mass_ratio)

def power_law(x, alpha, A):
    """Power law function for MSD fitting"""
    return A * np.power(x, alpha)

# def powerfit(msd_df, start_index, end_index, fps=10, use_corrected=False):
#     """ Fit power law to MSD data slice. Returns: [alpha, A], R² """
#     if not isinstance(msd_df, pd.DataFrame):
#         return [np.nan, np.nan], 0.0

#     # Determine which column to use for fitting
#     if use_corrected and 'msd_corrected' in msd_df.columns:
#         y = msd_df['msd_corrected'].values
#     elif 'msd' in msd_df.columns:
#         y = msd_df['msd'].values
#     else:
#         return [np.nan, np.nan], 0.0

#     x = msd_df.index.values.astype('float64')

#     # Define the slice indices
#     a = start_index
#     b = end_index
    
#     y_slice = y[a:b]
#     x_slice = x[a:b]

#     # Filter for valid, positive data points
#     valid = (np.isfinite(y_slice)) & (y_slice > 0) & (x_slice > 0)
#     y_fit = y_slice[valid]
#     x_fit = x_slice[valid]

#     if len(y_fit) < 2:
#         return [np.nan, np.nan], 0.0

#     try:
#         # Initial guess: [subdiffusive (0.5), small prefactor (0.01)]
#         pars, cov = curve_fit(f=power_law, xdata=x_fit, ydata=y_fit, 
#                               p0=[0.5, 0.01], bounds=([0, 0], [2, np.inf]))
        
#         residuals = y_fit - power_law(x_fit, *pars)
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
#         r_squared = 1.0 - (ss_res / ss_tot) if ss_tot!= 0 else 1.0
        
#         return pars, r_squared
#     except:
#         return [np.nan, np.nan], 0.0

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
        # 1. Log-transform the data (using natural log, np.log, or log10 is equivalent for slope)
        log_x = np.log10(x_fit)
        log_y = np.log10(y_fit)

        # 2. Perform linear regression: log(MSD) = alpha * log(Dt) + log(A)
        # polyfit returns [slope (alpha), intercept (log(A))]
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
    """
    Process individual file with per-particle MSD correction and truncation.
    
    MODIFIED: Now includes trajectory truncation logic
    """
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
        # 1. Prepare Trajectory Data
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
        
        # Get mass series for truncation
        mass_series = ptraj['mass'] if 'mass' in ptraj.columns else None
        
        # Apply trajectory length limit
        if max_length is not None:
            pos = pos[0:max_length]
            if mass_series is not None:
                mass_series = mass_series.iloc[0:max_length]
        
        # 2. NEW: Apply Truncation
        pos, segment_info = handle_truncation(
            pos, mass_series, MAX_STEP_SIZE, MAX_SINGLE_STEP_MASS_DROP, KEEP_FIRST_SEGMENT
        )
        
        if pos is None:
            continue
        
        if len(pos) < MIN_SEGMENT_LENGTH:
            continue
        
        # 3. Get Mass and Calculate Sigma Correction
        mass_particle = ptraj['mass'].mean() if 'mass' in ptraj.columns else np.nan

        if USE_LOCALIZATION_CORRECTION:
            sigma_particle = calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_scaling_flag)
        else:
            sigma_particle = None

        # 4. Calculate MSD
        msdtemp = msd(pos, max_lagtime, sigma=sigma_particle)
        
        if np.any(msdtemp.values == 0):
            msdtemp = msdtemp.replace(0, np.nan)
        msdtemp = msdtemp.dropna(axis='columns', how='all')
        if len(msdtemp.columns) <= 1:
            continue
        
        # 5. Fit power laws
        use_corrected_col = 'msd_corrected' if USE_LOCALIZATION_CORRECTION and 'msd_corrected' in msdtemp.columns else 'msd'
        msd_for_fit = pd.DataFrame({'msd': msdtemp[use_corrected_col]}, index=msdtemp.index)
        msd_for_fit.index = msd_for_fit.index.values.astype('float64') / float(fps)

        alphatemp, r_squaredtemp = powerfit(msd_for_fit, 0, 8, fps, use_corrected=False)
        alphatemp2, r_squaredtemp2 = powerfit(msd_for_fit, 19, 79, fps, use_corrected=False)
        
        # ----------------------------------------------------------------------
        # ORIGINAL (2‑8 s window)
        # alphatemp2, r_squaredtemp2 = powerfit(msd_for_fit, 19, 79, fps,
        #                                      use_corrected=False)
        # ----------------------------------------------------------------------
        # NEW – fit only around the exact 8‑s lag time
        # --------------------------------------------------------------
        # Convert the desired time (8 s) to frame indices
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

        # 6. Filter and Store Results
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

# Wrapping function to process an entire directory and calculate metrics
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
        # After the fix, 'col' is now a unique integer ID for each trajectory (0, 1, 2, ...)
        for col in MSDcollected_corrected_df.columns: 
            trajectory_msd = MSDcollected_corrected_df[col] # Now correctly returns a Series
            # Align indices before comparison 
            common_index = trajectory_msd.index.intersection(noise_floor_msd.index) 
            if len(common_index)== 0: 
                valid_trajectories_list.append(False) 
                continue 
            # Get values at common indices and convert to numpy arrays for comparison 
            trajectory_values = trajectory_msd.loc[common_index].values # Now shape (100,)
            noise_values = noise_floor_msd.loc[common_index].values 
            # Element-wise comparison at each lag time 
            above_noise = trajectory_values > noise_values 
            # Keep trajectory if it's above noise floor for at least 99% of lag times 
            fraction_above = above_noise.sum() / len(above_noise) if len(above_noise) > 0 else 0 
            valid_trajectories_list.append(fraction_above > 0.99) # 99% threshold 
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
# NEW PLOTTING HELPER FUNCTIONS (Including statistical bar addition)
# ============================================================================

def log_violin(ax, data, positions, colors):
    """
    Plot a violin whose KDE is computed on log10(data) and then shown on a log‑scaled y‑axis, 
    with extrema bars suppressed.
    """
    import numpy as np
    
    # 1) Transform each dataset to log10 (remove zeros)
    log_data = [np.log10(d[d > 0]) for d in data] # remove zeros
    
    # 2) Build the violin on the transformed data
    parts = ax.violinplot(
        log_data, 
        positions=positions, 
        showmeans=False, 
        showmedians=False, 
        showextrema=False # <-- THIS IS THE CRITICAL ADDITION
    ) 
    
    # 3) Colour the bodies
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
        
    # 4) Plot median lines and raw points on the *original* scale
    for i, raw in enumerate(data):
        current_color = colors[i]
        
        # Plot Median Line
        med = np.median(raw)
        ax.plot([positions[i] - 0.2, positions[i] + 0.2], [med, med], color='black', linewidth=2, zorder=10)
        
        # Plot Scatter Points
        ax.scatter(
            np.full_like(raw, positions[i]) + np.random.uniform(-0.05, 0.05, size=len(raw)),
            raw, 
            color=current_color,
            edgecolor='k', 
            s=20, 
            alpha=0.5, 
            zorder=5
        )
        
    # 5) Keep the y‑axis logarithmic
    ax.set_yscale('log')




def add_sig_bars_alpha(ax, data_list, actual_labels, alpha_thresh=0.05):
    """
    Plot significance bars (asterisks) on a box‑plot axis.
    Bars are drawn **only** for pair‑wise comparisons whose p‑value
    is smaller than ``alpha_thresh`` (default 0.05).
    """
    # ------------------------------------------------------------------
    # 1. Determine the maximum y‑value among the data for scaling.
    # ------------------------------------------------------------------
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())

    # Base height for the first bar (slightly above the data)
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) + 0.1
    spacing = 0.12          # vertical spacing between stacked bars

    # ------------------------------------------------------------------
    # 2. Pair‑wise t‑tests – keep only those that are significant.
    # ------------------------------------------------------------------
    p_vals = []
    n_groups = len(data_list)
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if 'beads' in actual_labels[i] or 'beads' in actual_labels[j]:
                continue
            try:
                t_stat, p = stats.ttest_ind(
                    data_list[i], data_list[j], nan_policy='omit')
                if p < alpha_thresh:          # keep only significant pairs
                    p_vals.append((p, i, j))
            except ValueError:
                continue

    # Sort by significance (smallest p first) so the most significant bar
    # appears at the bottom.
    p_vals.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # 3. Plot the retained bars.
    # ------------------------------------------------------------------
    for k, (p, i, j) in enumerate(p_vals):
        # Choose the asterisk string according to the p‑value.
        if p < 0.0001:
            sig = "****"
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        else:               # 0.01 ≤ p < 0.05
            sig = "*"

        h = base_h + k * spacing
        # Horizontal line connecting the two groups
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False)
        # Text label centred above the line
        ax.text((i + 1 + j + 1)  # centre of the line
                / 2, h + spacing * 0.2,
                sig, ha="center", va="bottom",
                fontsize=10, color="k")

    # ------------------------------------------------------------------
    # 4. Expand the y‑limit only if at least one bar was drawn.
    # ------------------------------------------------------------------
    if p_vals:
        final_h = base_h + len(p_vals) * spacing
        ax.set_ylim(cur_ylim[0], final_h + spacing)

# Define significance bar function for Dapp (log scale)
def add_sig_bars_dapp(ax, data_list, actual_labels, alpha_thresh=0.05):
    """
    Plot significance bars (asterisks) on a violin plot axis with log scale.
    Bars are drawn **only** for pair-wise comparisons whose p-value
    is smaller than ``alpha_thresh`` (default 0.05).
    """
    # 1. Determine the maximum y-value among the data for scaling.
    y_max = 0
    for data in data_list:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            y_max = max(y_max, finite.max())
    
    # Base height for the first bar (slightly above the data)
    # For log scale, we multiply instead of add
    cur_ylim = ax.get_ylim()
    base_h = max(cur_ylim[1], y_max) * 1.5
    spacing_factor = 2.3  # multiplicative spacing for log scale
    
    # 2. Pair-wise t-tests – keep only those that are significant.
    p_vals = []
    n_groups = len(data_list)
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if 'beads' in actual_labels[i].lower() or 'beads' in actual_labels[j].lower():
                continue
            try:
                t_stat, p = stats.ttest_ind(
                    data_list[i], data_list[j], nan_policy='omit')
                if p < alpha_thresh:
                    p_vals.append((p, i, j))
            except ValueError:
                continue
    
    # Sort by significance (smallest p first)
    p_vals.sort(key=lambda x: x[0])
    
    # 3. Plot the retained bars.
    for k, (p, i, j) in enumerate(p_vals):
        # Choose the asterisk string according to the p-value.
        if p < 0.0001:
            sig = "****"
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        else:
            sig = "*"
        
        h = base_h * (spacing_factor ** k)
        
        # Horizontal line connecting the two groups
        ax.plot([i + 1, j + 1], [h, h], lw=1.0, color="k", clip_on=False)
        # Text label centred above the line
        ax.text((i + 1 + j + 1) / 2, h * 1.1,
                sig, ha="center", va="bottom",
                fontsize=10, color="k")
    
    # 4. Expand the y-limit only if at least one bar was drawn.
    if p_vals:
        final_h = base_h * (spacing_factor ** len(p_vals))
        ax.set_ylim(cur_ylim[0], final_h * 1.2)

def plot_alpha_boxplot(ax, results_dict, processed_keys, condition_keys, labels_map, plot_colors, jitter=0.05, panel_label=None, add_sig_bars=True): 
    """Plots alpha box plot for a specific subset of keys and adds significance bars."""

    # 1. Gather data and filter labels/colors
    data_list = []
    actual_keys_subset = []
    
    for k in processed_keys:
        if k in results_dict:
            data_list.append(results_dict[k]['alpha2'])
            actual_keys_subset.append(k)

    if not data_list:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha='center')
        ax.set_ylim(-0.08, 2.5)
        return 

    actual_labels_subset = [labels_map[condition_keys.index(k)] for k in actual_keys_subset]
    x_ticks = list(range(1, len(data_list) + 1))

    # 2. Plot boxplot
    bp = ax.boxplot(
        data_list,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(linewidth=1.5)
    )

    # 3. Plot scatter points and color boxes
    for i, data_arr in enumerate(data_list):
        original_key = actual_keys_subset[i]
        original_index = condition_keys.index(original_key)
        
        # Scatter data points for visualization
        ax.scatter(
            np.full_like(data_arr, x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(data_arr)),
            data_arr,
            color=plot_colors[original_index],
            alpha=0.8,
            marker='o',
            edgecolor='k',
            zorder=10
        )
        
        # Set box colors
        bp['boxes'][i].set_facecolor(plot_colors[original_index])
        bp['boxes'][i].set_alpha(0.6)

    # 4. Add Significance Bars
    if add_sig_bars:
        add_sig_bars_alpha(ax, data_list) 

    # 5. Reference lines
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r'Normal diffusion ($\alpha=1$)', zorder=0)
    ax.axhline(0.4, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7, label=r'Threshold for confinement ($\alpha<0.4$)', zorder=0)

    # 6. Formatting (Initial Y-limit is temporarily set here, allowing sig bars to stretch it later)
    ax.set_ylim(-0.08, 2.5) # Set base limit
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(actual_labels_subset, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=14)
    ax.grid(False)
    ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    if panel_label:
        ax.text(-0.05, 1.15, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')


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

    # 1. Calculate all directory paths based on the map
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

    # 2. Iterate and Process Data (NOW WITH NOISE FLOOR FILTERING)
    for i, label_key in enumerate(CONDITION_KEYS):
        if label_key == 'beads':
            continue  # Skip beads since we already processed them
            
        if label_key in directories:
            directory = directories[label_key]
            max_length = MAX_TRAJ_LENGTHS[i]
            size_label = CONDITION_DICT[label_key]['size']

            # Conditional Sigma Assignment and Scaling Flag
            if label_key in ['100bp', '250bp', '500bp']:
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


    # ============================================================================
    # PLOTTING PREPARATION
    # ============================================================================
    processed_keys = [k for k in CONDITION_KEYS if k in results_dict]
    actual_labels = [LABELS[CONDITION_KEYS.index(k)] for k in processed_keys]

    if not results_dict:
        print("\nERROR: results_dict is empty. No data passed filters or directories were invalid.")
        return
    
    # ============================================================================
    # --- NEW FIGURE: 1x3 Panel (Raw MSD, Corrected MSD, Mass-Sigma Diagnostic) ---
    # ============================================================================
    print("\nSaving 1x3 Panel Figure (Raw MSD, Corrected MSD, Mass-Sigma)...")
    fig_panel, axes_panel = plt.subplots(1, 3, figsize=(18, 6))
    ax_raw_panel, ax_corr_panel, ax_diagnostic = axes_panel
    
    # --- Panel A: Raw MSD ---
    for key in processed_keys:
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_raw']
        color = PLOT_COLORS[original_index]
        ax_raw_panel.plot(ensemble_msd.index, ensemble_msd.values, 
                   marker='o', markeredgecolor='black', 
                   markerfacecolor= color, color = color,
                   linestyle='-', linewidth=2, markersize=6,
                   label=LABELS[original_index])
    
    ax_raw_panel.set_xscale('log')
    ax_raw_panel.set_yscale('log')
    ax_raw_panel.set_xlabel(r'$\Delta t$ (s)', fontsize=16)
    ax_raw_panel.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\mathrm{raw}}$ ($\mu$m$^2$)', fontsize=16)
    ax_raw_panel.set_xlim(0.1, 100)
    ax_raw_panel.set_ylim(1e-4, 1)
    ax_raw_panel.legend(loc='upper left', fontsize=10)
    ax_raw_panel.grid(False)
    ax_raw_panel.tick_params(which="both", axis="both", direction="in", labelsize=14, length=6, width=1.5)
    for spine in ax_raw_panel.spines.values():
        spine.set_linewidth(1.5)
    ax_raw_panel.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_raw_panel.axvline(10.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    # shaded gray regions
    ax_raw_panel.axvspan(ax_raw_panel.get_xlim()[0], 1.0, color='lightgray', alpha=0.3, zorder=0)
    ax_raw_panel.axvspan(10.0, ax_raw_panel.get_xlim()[1], color='lightgray', alpha=0.3, zorder=0)
    
    ax_raw_panel.text(-0.15, 1.05, 'A', transform=ax_raw_panel.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    
    # --- Panel B: Corrected MSD ---
    for key in processed_keys:
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_corrected']
        color = PLOT_COLORS[original_index]
        ax_corr_panel.plot(ensemble_msd.index, ensemble_msd.values,
                    marker='o', markeredgecolor='black',
                    markerfacecolor=color, color = color,
                    linestyle='-', linewidth=2, markersize=6,
                    label=LABELS[original_index])
    
    ax_corr_panel.set_xscale('log')
    ax_corr_panel.set_yscale('log')
    ax_corr_panel.set_xlabel(r'$\Delta t$ (s)', fontsize=16)
    ax_corr_panel.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\mathrm{real}}$ ($\mu$m$^2$)', fontsize=16)
    ax_corr_panel.set_xlim(0.1, 100)
    ax_corr_panel.set_ylim(1e-4, 1)
    ax_corr_panel.legend(loc='upper left', fontsize=10)
    ax_corr_panel.grid(False)
    ax_corr_panel.tick_params(which="both", axis="both", direction="in", labelsize=14, length=6, width=1.5)
    for spine in ax_corr_panel.spines.values():
        spine.set_linewidth(1.5)
    ax_corr_panel.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_corr_panel.axvline(10.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    # shaded gray regions
    ax_corr_panel.axvspan(ax_corr_panel.get_xlim()[0], 1.0, color='lightgray', alpha=0.3, zorder=0)
    ax_corr_panel.axvspan(10.0, ax_corr_panel.get_xlim()[1], color='lightgray', alpha=0.3, zorder=0)
    ax_corr_panel.text(-0.15, 1.05, 'B', transform=ax_corr_panel.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    
    # --- Panel C: Mass-Sigma Diagnostic ---
    for key in processed_keys:
        if key == 'beads':
            continue  # Skip beads for diagnostic plot
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        mass = data['mass']
        sigma = data['sigma'] * 1000  # Convert to nm
        
        ax_diagnostic.scatter(mass, sigma, 
                            color=PLOT_COLORS[original_index],
                            alpha=0.6, s=50, edgecolor='black', linewidth=0.5,
                            label=LABELS[original_index])
    
    # Add reference line for stuck beads
    ax_diagnostic.axhline(SIGMA_STUCK_MICRONS * 1000, color='red', linestyle='--', 
                         linewidth=2, label=f'Stuck beads: {SIGMA_STUCK_MICRONS*1000:.1f} nm')
    
    ax_diagnostic.set_xlabel('Particle Mass (brightness)', fontsize=16)
    ax_diagnostic.set_ylabel(' $\sigma$ (nm)', fontsize=16)
    ax_diagnostic.set_xlim(0, 160000)
    ax_diagnostic.set_xticks([20000, 40000, 80000, 160000])
    ax_diagnostic.set_xticklabels(['2', '4', '8', '16'])
    ax_diagnostic.set_xlabel('Particle Mass (brightness) (×10⁴)', fontsize=16)
    ax_diagnostic.set_ylim(0, 25)
    ax_diagnostic.legend(loc='upper right', fontsize=10)
    ax_diagnostic.grid(False)
    ax_diagnostic.tick_params(which="both", axis="both", direction="in", labelsize=14, length=6, width=1.5)
    for spine in ax_diagnostic.spines.values():
        spine.set_linewidth(1.5)
    ax_diagnostic.text(-0.15, 1.05, 'C', transform=ax_diagnostic.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig_panel.savefig(OUTPUT_FIGURE_BASE / "Figure_Panel_MSD_Diagnostic.png", dpi=600, bbox_inches='tight')
    print(f"Saved: Figure_Panel_MSD_Diagnostic.png")
    

    # Box plot data lists (uses processed_keys list structure for indexing)
    Dapp_data = [results_dict[k]['Dapp'] for k in processed_keys] # Short time Dapp
    Dapp2_data = [results_dict[k]['Dapp2'] for k in processed_keys] # Long time (2-8s) Dapp
    alpha2_data = [results_dict[k]['alpha2'] for k in processed_keys] # Alpha exponent (2-8s)
    
    x_ticks = list(range(1, len(Dapp2_data) + 1))
    jitter = 0.05
    np.random.seed(42)
    mkr = ['o'] * len(CONDITION_KEYS)
    OUTPUT_FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    
    
    # ============================================================================
    # --- NEW FIGURE 2: MSD corrected + noisefloor ---
    # ============================================================================
    # Identify the key conditions (first and last in the processed list)
    # Note: Ensure 'processed_keys' and 'results_dict' are populated before this step.
    key_subset = [processed_keys[3], processed_keys[-1]]
    
    # Define the figure with a single subplot (1 row, 1 column)
    fig2, ax_corr_subset = plt.subplots(1, 1, figsize=(8, 6)) # Using standard figsize for 1-column
    
    msd_4713_individual = results_dict[processed_keys[3]]['msd_corrected']
    ax_corr_subset.plot(msd_4713_individual, color='lightgray', alpha=0.5, linewidth=1)
    
    # --- Panel (Corrected MSD Subset) ---
    for key in key_subset:
        original_index = CONDITION_KEYS.index(key)
        data = results_dict[key]
        ensemble_msd = data['ensemble_msd_corrected']
        color = PLOT_COLORS[original_index]
        ax_corr_subset.plot(
            ensemble_msd.index, 
            ensemble_msd.values, 
            marker='o', 
            markeredgecolor='black', 
            markerfacecolor=color, 
            color=color, 
            linestyle='-', 
            linewidth=2, 
            markersize=6, 
            label=LABELS[original_index]
        )
    

    x2_start, x2_end = 2, 8
    x2 = np.logspace(np.log10(x2_start), np.log10(x2_end), 100)
    y2_anchor = 3e-3
    y2 = y2_anchor * (x2 / x2_start) ** 0.8
    ax_corr_subset.plot(x2, y2, linestyle=':', color='black')
    ax_corr_subset.text(5, y2_anchor * 1, r'$\Delta t^{\alpha}$', fontsize=14, color='black')
    
    # Formatting 
    ax_corr_subset.set_xscale('log')
    ax_corr_subset.set_yscale('log')
    ax_corr_subset.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
    ax_corr_subset.set_ylabel(r'$\langle \overline{\Delta \mathbf{r}^2(\Delta t)}\rangle_{\mathrm{real}}$ ($\mu\mathrm{m}^2$)', fontsize=14)
    ax_corr_subset.set_xlim(0.1, 10) # Adjusted xlim for better focus
    ax_corr_subset.set_ylim(1e-4, 1e-1) # Adjusted ylim for better focus
    ax_corr_subset.legend(loc='upper left', fontsize=10)
    ax_corr_subset.grid(False)
    ax_corr_subset.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    for spine in ax_corr_subset.spines.values():
        spine.set_linewidth(1.5)
    
    # Highlight short-time and long-time regions 
    ax_corr_subset.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_corr_subset.axvline(10.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Shaded regions (adjusting limits based on new xlim)
    ax_corr_subset.axvspan(ax_corr_subset.get_xlim()[0], 1.0, color='lightgray', alpha=0.15, zorder=0)
    ax_corr_subset.axvspan(10.0, ax_corr_subset.get_xlim()[1], color='lightgray', alpha=0.15, zorder=0)
    
    plt.tight_layout()
    fig2.savefig(OUTPUT_FIGURE_BASE / "Figure_2.png", dpi=600, bbox_inches='tight')
    print(f"Saved: Figure_2.png")





    # ============================================================================
    # STATISTICAL ANALYSIS FOR ALPHA VALUES
    # (This section remains for console output, using the original calculation logic)
    # ============================================================================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - ALPHA VALUES (2-8s)")
    print("="*80)
    
    # 1. One-way ANOVA
    f_stat, p_anova = stats.f_oneway(*alpha2_data)
    print(f"\nOne-way ANOVA:")
    print(f" F-statistic: {f_stat:.4f}")
    print(f" P-value: {p_anova:.6e}")
    if p_anova < 0.0001:
        print(f" Significance: **** (P < 0.0001)")
    elif p_anova < 0.001:
        print(f" Significance: *** (P < 0.001)")
    elif p_anova < 0.01:
        print(f" Significance: ** (P < 0.01)")
    elif p_anova < 0.05:
        print(f" Significance: * (P < 0.05)")
    else:
        print(f" Significance: ns (P >= 0.05)")

    # 2. Pairwise t-tests (two-sided)
    print(f"\nPairwise two-sided t-tests:")
    n_conditions = len(alpha2_data)
    pairwise_pvalues = np.zeros((n_conditions, n_conditions))
    for i in range(n_conditions):
        for j in range(i+1, n_conditions):
            t_stat, p_val = stats.ttest_ind(alpha2_data[i], alpha2_data[j])
            pairwise_pvalues[i, j] = p_val
            pairwise_pvalues[j, i] = p_val
            sig_label = ""
            if p_val < 0.0001: sig_label = "****"
            elif p_val < 0.001: sig_label = "***"
            elif p_val < 0.01: sig_label = "**"
            elif p_val < 0.05: sig_label = "*"
            else: sig_label = "ns"
            print(f" {actual_labels[i]} vs {actual_labels[j]}: P = {p_val:.6e} ({sig_label})")

    # Calculate means and standard errors for error bars
    alpha2_means = [np.mean(data) for data in alpha2_data]
    alpha2_sem = [stats.sem(data) for data in alpha2_data]
    print("="*80)
    
    # ============================================================================
    # CALCULATED RESULTS SUMMARY TABLE
    # ============================================================================
    print("\n" + "="*80)
    print("CALCULATED RESULTS SUMMARY (Long-time regime 2-8s)")
    print("="*80)
    
    # Filter out 'beads' if present, as it was not in the primary DNA table
    data_keys = [k for k in processed_keys if 'beads' not in k.lower()]
    
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
    # --- FIGURE 2: Combined MSD (A-F) + Dapp (G) + Alpha (H) ---
    # ============================================================================
    print("\nSaving Figure 2 (Combined 8-panel figure with MSD + Dapp + Alpha)...")
    
    # Filter out bead data
    non_bead_keys = [k for k in CONDITION_KEYS if 'bead' not in k.lower()]
    non_bead_processed = [k for k in processed_keys if 'bead' not in k.lower()]
    
    # Update labels and colors for non-bead data
    non_bead_labels = [LABELS[CONDITION_KEYS.index(k)] for k in non_bead_processed]
    non_bead_colors = [PLOT_COLORS[CONDITION_KEYS.index(k)] for k in non_bead_processed]
    non_bead_x_ticks = list(range(1, len(non_bead_processed) + 1))
    
    # 1. Setup the figure using GridSpec 
    fig_combined = plt.figure(figsize=(3.375 * 3, 3.375 * 3))
    gs = fig_combined.add_gridspec(3, 6, hspace=0.4, wspace=0.5)
    
    # Axes list for A-F (MSD plots)
    msd_axes = []
    msd_positions = [(0, 0), (0, 2), (0, 4), (1, 0), (1, 2), (1, 4)]
    for r, c in msd_positions:
        ax = fig_combined.add_subplot(gs[r, c:c+2])
        msd_axes.append(ax)
    
    # Axis G (Dapp Box Plot - spans first 3 columns in row 2)
    ax_dapp = fig_combined.add_subplot(gs[2, 0:3])
    
    # Axis H (Alpha Box Plot - spans next 3 columns in row 2)
    ax_alpha = fig_combined.add_subplot(gs[2, 3:6])
    
    # 2. Plot MSD panels (A-F) - exclude bead data
    msd_plot_keys = [k for k in non_bead_keys if k in results_dict][:6]
    for i, key in enumerate(msd_plot_keys):
        original_index = CONDITION_KEYS.index(key)
        current_label = LABELS[original_index]
        data = results_dict[key]
        MSDcollected_real = data['msd_corrected']
        totaltracks = data['n_tracks']
        ax = msd_axes[i]
        
        if MSDcollected_real is not None and len(MSDcollected_real.columns) > 0:
            ax.plot(MSDcollected_real, color='lightgray', alpha=0.3)
            ax.plot(MSDcollected_real.mean(axis=1), marker='o', markeredgecolor='black', 
                    markerfacecolor=PLOT_COLORS[original_index], linestyle='None', markersize=6)
    
            ax.set_xscale('log')
            ax.set_yscale('log')
    
            if i == 0:
                x2_start, x2_end = 1, 10
                x2 = np.logspace(np.log10(x2_start), np.log10(x2_end), 100)
                y2_anchor = 1e-3
                y2 = y2_anchor * (x2 / x2_start) ** 0.8
                ax.plot(x2, y2, linestyle=':', color='black')
                ax.text(2.5, y2_anchor * 0.5, r'$\Delta t^{\alpha}$', fontsize=14, color='black')
    
            if i in [0, 3]:
                ax.set_ylabel(r'$\langle \overline {\Delta \mathbf{r}^2(\Delta t)}\rangle$ ($\mathrm{\mu}$m$^2$)', fontsize=14)
            else:
                ax.set_ylabel('')
                
            ax.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
            ax.set_ylim((1e-4, 10))
            ax.grid(False)
            ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            ax.set_aspect(0.5, adjustable='box')
            ax.text(-0.25, 1.2, PANEL_LABELS[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
            ax.text(0.15, 0.9, current_label + '\n$n$ = ' + str(totaltracks), 
                    verticalalignment='top', horizontalalignment='left', 
                    multialignment="left", transform=ax.transAxes, fontsize=16)
    
    # 3. Plot Dapp Violin Plot (G) 
    dapp_data_to_plot = [results_dict[k]['Dapp2'] for k in non_bead_processed] 
    log_violin(ax_dapp, dapp_data_to_plot, non_bead_x_ticks, non_bead_colors) 
    
    # Add significance bars (ensure this is called after log_violin and before any manual ylim reset)
    add_sig_bars_dapp(ax_dapp, dapp_data_to_plot, non_bead_labels) 
    
    ax_dapp.yaxis.set_ticks_position('both') 
    ax_dapp.xaxis.set_ticks_position('both') 
    for spine in ax_dapp.spines.values(): 
        spine.set_linewidth(1.5) 
    ax_dapp.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5) 
     
    ax_dapp.set_xticks(non_bead_x_ticks) 
    ax_dapp.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12) 
    ax_dapp.set_ylabel(r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)', fontsize=14) 
    ax_dapp.text(-0.09, 1.15, 'G', transform=ax_dapp.transAxes, fontsize=16, fontweight='bold', va='top', ha='left') 
    ax_dapp.grid(False)

    
    # 4. Plot Alpha Violin Plot (H)
    alpha_data_to_plot = [results_dict[k]['alpha2'] for k in non_bead_processed]
    
    # Create violin plot
    parts = ax_alpha.violinplot(alpha_data_to_plot, positions=non_bead_x_ticks, showmeans=False, showmedians=False)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(non_bead_colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add median lines manually
    for i, data_arr in enumerate(alpha_data_to_plot):
        median_val = np.median(data_arr)
        ax_alpha.plot([non_bead_x_ticks[i] - 0.2, non_bead_x_ticks[i] + 0.2], [median_val, median_val], 
                      color='black', linewidth=2, zorder=10)
    
    # Scatter data points for visualization
    for i, data_arr in enumerate(alpha_data_to_plot):
        ax_alpha.scatter(np.full_like(data_arr, non_bead_x_ticks[i]) + np.random.uniform(-0.05, 0.05, size=len(data_arr)),
                         data_arr, color=non_bead_colors[i], alpha=0.5, marker='o', edgecolor='k', s=20, zorder=5)
        
    add_sig_bars_alpha(ax_alpha, alpha_data_to_plot, non_bead_labels)
    
    ax_alpha.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Normal diffusion ($\\alpha=1$)', zorder=0)
    ax_alpha.axhline(0.4, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7, label='Threshold for confinement ($\\alpha<0.4$)', zorder=0)
    
    ax_alpha.yaxis.set_ticks_position('both')
    ax_alpha.xaxis.set_ticks_position('both')
    ax_alpha.yaxis.set_label_position("right")
    ax_alpha.yaxis.tick_right()
    for spine in ax_alpha.spines.values():
        spine.set_linewidth(1.5)
    ax_alpha.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    
    if ax_alpha.get_ylim()[1] < 2.5: 
        ax_alpha.set_ylim(-0.08, 2.5) 
        
    ax_alpha.set_xticks(non_bead_x_ticks)
    ax_alpha.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_alpha.set_ylabel(r'$\alpha$', fontsize=14, labelpad=25, rotation=-90)
    ax_alpha.text(-0.05, 1.15, 'H', transform=ax_alpha.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_alpha.grid(False)
    
    plt.tight_layout()
    fig_combined.savefig(OUTPUT_FIGURE_BASE / "Figure_2_original.png", dpi=600, bbox_inches='tight')
    print("Figure 2 saved.")
    
    
    # ============================================================================
    # --- FIGURE 3: First 3 MSD panels (A-C) ---
    # ============================================================================
    print("\nSaving Figure 3 (First 3 MSD panels A-C)...")
    
    fig_msd_first, axs_msd_first = plt.subplots(1, 3, figsize=(12, 4))
    msd_plot_keys_first = [k for k in non_bead_keys[:3] if k in results_dict]
    
    for i, key in enumerate(msd_plot_keys_first):
        original_index = CONDITION_KEYS.index(key)
        current_label = LABELS[original_index]
        data = results_dict[key]
        ax = axs_msd_first[i]
        
        if data['msd_corrected'] is not None and len(data['msd_corrected'].columns) > 0:
            # Plot individual tracks in light gray
            ax.plot(data['msd_corrected'], color='lightgray', alpha=0.3)
            # Plot mean MSD
            ax.plot(data['msd_corrected'].mean(axis=1), marker='o', 
                    markeredgecolor='black', markerfacecolor=PLOT_COLORS[original_index], 
                    linestyle='None', markersize=6)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
            ax.set_ylabel(r'$\langle\Delta r^{2}\rangle$ ($\mu$m$^{2}$)', fontsize=14)
            ax.set_ylim((1e-5, 1))
            ax.grid(False)
            ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            # Panel label
            ax.text(-0.15, 1.08, chr(65 + i), transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
            # Condition label
            ax.text(0.15, 0.9, current_label + '\n$n$ = ' + str(data['n_tracks']), 
                    verticalalignment='top', horizontalalignment='left', 
                    transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    fig_msd_first.savefig(OUTPUT_FIGURE_BASE / "Figure_3.png", dpi=600, bbox_inches='tight')
    print("Figure 3 saved.")
    
    
    # ============================================================================
    # --- FIGURE 4: Second 3 MSD panels (D-F becomes A-C) ---
    # ============================================================================
    print("\nSaving Figure 4 (Second 3 MSD panels, relabeled A-C)...")
    
    fig_msd_second, axs_msd_second = plt.subplots(1, 3, figsize=(12, 4))
    msd_plot_keys_second = [k for k in non_bead_keys[3:6] if k in results_dict]
    
    for i, key in enumerate(msd_plot_keys_second):
        original_index = CONDITION_KEYS.index(key)
        current_label = LABELS[original_index]
        data = results_dict[key]
        ax = axs_msd_second[i]
        
        if data['msd_corrected'] is not None and len(data['msd_corrected'].columns) > 0:
            # Plot individual tracks in light gray
            ax.plot(data['msd_corrected'], color='lightgray', alpha=0.3)
            # Plot mean MSD
            ax.plot(data['msd_corrected'].mean(axis=1), marker='o', 
                    markeredgecolor='black', markerfacecolor=PLOT_COLORS[original_index], 
                    linestyle='None', markersize=6)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\Delta t$ (s)', fontsize=14)
            ax.set_ylabel(r'$\langle\Delta r^{2}\rangle$ ($\mu$m$^{2}$)', fontsize=14)
            ax.set_ylim((1e-5, 1))
            ax.grid(False)
            ax.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            # Panel label (A, B, C instead of D, E, F)
            ax.text(-0.15, 1.08, chr(65 + i), transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
            # Condition label
            ax.text(0.15, 0.9, current_label + '\n$n$ = ' + str(data['n_tracks']), 
                    verticalalignment='top', horizontalalignment='left', 
                    transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    fig_msd_second.savefig(OUTPUT_FIGURE_BASE / "Figure_4.png", dpi=600, bbox_inches='tight')
    print("Figure 4 saved.")
    
    
    # ============================================================================
    # --- FIGURE 5: Dapp (G) + Alpha (H) + Diffusion Bar Chart (I) ---
    # ============================================================================
    print("\nSaving Figure 5 (Dapp + Alpha + Diffusion bar chart)...")
    
    import matplotlib.gridspec as gridspec
    
    fig_comb = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.4)
    
    # ----- Panel G: Dapp violin plot (top) -----
    ax_g = fig_comb.add_subplot(gs[0])
    
    # Create violin plot
    parts_g = ax_g.violinplot(dapp_data_to_plot, positions=non_bead_x_ticks, showmeans=False, showmedians=False, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts_g['bodies']):
        pc.set_facecolor(non_bead_colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add median lines manually
    for i, data_arr in enumerate(dapp_data_to_plot):
        median_val = np.median(data_arr)
        ax_g.plot([non_bead_x_ticks[i] - 0.2, non_bead_x_ticks[i] + 0.2], [median_val, median_val], 
                  color='black', linewidth=2, zorder=10)
    
    # Scatter data points
    for i, data_arr in enumerate(dapp_data_to_plot):
        ax_g.scatter(np.full_like(data_arr, non_bead_x_ticks[i]) + np.random.uniform(-0.05, 0.05, size=len(data_arr)),
                     data_arr, color=non_bead_colors[i], alpha=0.5, marker='o', edgecolor='k', s=20, zorder=5)
    
    # Add significance bars
    add_sig_bars_dapp(ax_g, dapp_data_to_plot, non_bead_labels)
    
    ax_g.yaxis.set_ticks_position('both')
    ax_g.xaxis.set_ticks_position('both')
    for spine in ax_g.spines.values():
        spine.set_linewidth(1.5)
    ax_g.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_g.set_yscale('log')
    ax_g.set_ylim(1e-5, 100)
    ax_g.set_xticks(non_bead_x_ticks)
    ax_g.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_g.set_ylabel(r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)', fontsize=14)
    ax_g.text(-0.09, 1.12, 'A', transform=ax_g.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_g.grid(False)
    
    # ----- Panel H: Alpha violin plot (middle) -----
    ax_h = fig_comb.add_subplot(gs[1])
    
    # Create violin plot
    parts_h = ax_h.violinplot(alpha_data_to_plot, positions=non_bead_x_ticks, showmeans=False, showmedians=False, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts_h['bodies']):
        pc.set_facecolor(non_bead_colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Add median lines manually
    for i, data_arr in enumerate(alpha_data_to_plot):
        median_val = np.median(data_arr)
        ax_h.plot([non_bead_x_ticks[i] - 0.2, non_bead_x_ticks[i] + 0.2], [median_val, median_val], 
                  color='black', linewidth=2, zorder=10)
    
    # Scatter data points
    for i, data_arr in enumerate(alpha_data_to_plot):
        ax_h.scatter(np.full_like(data_arr, non_bead_x_ticks[i]) + np.random.uniform(-0.05, 0.05, size=len(data_arr)),
                     data_arr, color=non_bead_colors[i], alpha=0.5, marker='o', edgecolor='k', s=20, zorder=5)
    
    # Add significance bars
    add_sig_bars_alpha(ax_h, alpha_data_to_plot, non_bead_labels)
    
    ax_h.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Normal diffusion ($\\alpha=1$)', zorder=0)
    ax_h.axhline(0.4, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7, label='Threshold for confinement ($\\alpha<0.4$)', zorder=0)
    
    ax_h.yaxis.set_ticks_position('both')
    ax_h.xaxis.set_ticks_position('both')
    for spine in ax_h.spines.values():
        spine.set_linewidth(1.5)
    ax_h.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    
    if ax_h.get_ylim()[1] < 2.5:
        ax_h.set_ylim(-0.08, 2.5)
            
    ax_h.set_xticks(non_bead_x_ticks)
    ax_h.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_h.set_ylabel(r'$\alpha$')
    ax_h.text(-0.09, 1.12, 'B', transform=ax_h.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax_h.grid(False)
    
    # ----- Panel I: Diffusion type bar chart (bottom) -----
    ax_i = fig_comb.add_subplot(gs[2])
    
    # Calculate percentages for each diffusion type
    confined = [np.mean(arr < 0.4) * 100 for arr in alpha_data_to_plot]
    subdiff = [np.mean((arr >= 0.4) & (arr < 1.0)) * 100 for arr in alpha_data_to_plot]
    superdiff = [np.mean(arr >= 1.0) * 100 for arr in alpha_data_to_plot]
    
    x = np.arange(len(non_bead_labels))
    width = 0.25
    
    bar1 = ax_i.bar(x - width, confined, width, label='Confined (<0.4)', color='#332288', edgecolor='black', linewidth=1.5)
    bar2 = ax_i.bar(x, subdiff, width, label='Sub-diffusive (0.4-1)', color='#88CCEE', edgecolor='black', linewidth=1.5)
    bar3 = ax_i.bar(x + width, superdiff, width, label='Super-diffusive (>1)', color='#44AA99', edgecolor='black', linewidth=1.5)
    
    ax_i.set_xticks(x)
    ax_i.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    ax_i.set_ylabel('Percentage (%)', fontsize=14)
    ax_i.set_ylim(0, 100)
    ax_i.legend(frameon=False, fontsize=12)
    ax_i.yaxis.set_ticks_position('both')
    ax_i.xaxis.set_ticks_position('both')
    for spine in ax_i.spines.values():
        spine.set_linewidth(1.5)
    ax_i.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    ax_i.grid(False)
    ax_i.text(-0.1, 1.08, 'C', transform=ax_i.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    fig_comb.savefig(OUTPUT_FIGURE_BASE / "Figure_5.png", dpi=600, bbox_inches='tight')
    print("Figure 5 saved.")
    
    print("\nAll figures generated successfully!")
    



    # # ============================================================================
    # # --- FIGURE 6: Alpha Box Plot (Standalone Figure) ---
    # # ============================================================================
    # fig_alpha, ax_alpha_single = plt.subplots(figsize=(4, 4.5)) # Match Figure 2 size
    
    # # Prepare Alpha data
    # alpha_data_to_plot = [results_dict[k]['alpha2'] for k in processed_keys]
    
    # bp3_single = ax_alpha_single.boxplot(
    #     alpha_data_to_plot,
    #     showfliers=False,
    #     patch_artist=True,
    #     medianprops=dict(color='black', linewidth=1.5),
    #     boxprops=dict(linewidth=1.5)
    # )

    # # Scatter points and color boxes
    # for i, data_arr in enumerate(alpha_data_to_plot):
    #     original_key = processed_keys[i]
    #     original_index = CONDITION_KEYS.index(original_key)
        
    #     # Scatter data points for visualization
    #     ax_alpha_single.scatter(np.full_like(data_arr, x_ticks[i]) + np.random.uniform(-jitter, jitter, size=len(data_arr)),
    #                             data_arr, color=PLOT_COLORS[original_index], alpha=0.8, marker='o', edgecolor='k', zorder=10)
        
    #     # Set box colors
    #     bp3_single['boxes'][i].set_facecolor(PLOT_COLORS[original_index])
    #     bp3_single['boxes'][i].set_alpha(0.6)
        
    # # === CRITICAL: Add Significance Bars to Standalone Figure 3 ===
    # add_sig_bars_alpha(ax_alpha_single, alpha_data_to_plot, non_bead_labels)

    # # Reference lines
    # ax_alpha_single.axhline( 1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Normal diffusion ($\\alpha=1$)', zorder=0 )
    # ax_alpha_single.axhline( 0.4, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.7, label='Threshold for confinement ($\\alpha<0.4$)', zorder=0 )

    # # Axis formatting
    # ax_alpha_single.yaxis.set_ticks_position('both')
    # ax_alpha_single.xaxis.set_ticks_position('both')
    # for spine in ax_alpha_single.spines.values():
    #     spine.set_linewidth(1.5)
    # ax_alpha_single.tick_params(which="both", axis="both", direction="in", labelsize=12, length=6, width=1.5)
    
    # # Note: ax.set_ylim is handled by add_sig_bars_alpha, but we ensure a minimum base limit
    # if ax_alpha_single.get_ylim()[1] < 2.5:
    #     ax_alpha_single.set_ylim(-0.08, 2.5) 
        
    # ax_alpha_single.set_xticks(x_ticks)
    # ax_alpha_single.set_xticklabels(non_bead_labels, rotation=45, ha='right', fontsize=12)
    # ax_alpha_single.set_ylabel(r'$\alpha$', fontsize=14)
    # ax_alpha_single.grid(False)

    # # Final layout and save
    # fig_alpha.tight_layout()
    # fig_alpha.savefig(OUTPUT_FIGURE_BASE / "Figure_3_Alpha_Boxplot.png", dpi=600, bbox_inches='tight')



    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {OUTPUT_FIGURE_BASE}")
    print("="*80)

if __name__ == '__main__':
    run_analysis()
