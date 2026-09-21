# -*- coding: utf-8 -*-
"""
msd_core.py
Shared MSD calculation core for the DNA/cell tracking pipeline.

This module is the SINGLE SOURCE OF TRUTH for:
  - MSD computation (_msd_iter, msd)
  - per-particle localization sigma correction (calculate_sigma_for_particle,
    calculate_sigma_for_particle_from_mass_array)
  - power-law fitting of MSD curves (power_law, powerfit)
  - per-file trajectory processing (imsd)
  - per-directory aggregation + noise floor filtering (process_directory,
    compute_bead_noise_floor, trajectory_above_noise_floor, corrected_msd_series)
  - the trajectory() start-shifting/tagging helper used by several
    visualization/QC scripts

Every script in the pipeline that computes an MSD imports from here rather
than defining its own copy: 06a, 06b, 06c (full imsd()/process_directory()
pipeline), 07a, 07b, 08a, 08b (noise-floor helpers only - these scripts
don't fit alpha), and 03, 04, 11 (msd()/power_law()/trajectory() only -
these QC/visualization scripts apply no localization correction and do
their own local power-law fitting for diagnostic purposes). This
guarantees that a given trajectory produces the same MSD no matter which
script processes it, and that any future change to the MSD/fitting/
filtering logic only has to be made in ONE place.

Standardized choices (see conversation history for rationale):
  - powerfit(): log-log linear regression (np.polyfit on log10 x, log10 y),
    alpha UNBOUNDED, R^2 computed on log-space residuals.
    (This replaces the older bounded scipy.curve_fit-in-linear-space
    version that used to live in 06c only.)
  - Noise floor filter: STRICT '>' comparison against NOISE_FLOOR_FRACTION
    (a trajectory must be above the noise floor for MORE than this
    fraction of shared lag times to be kept; exactly-at-threshold is
    rejected, matching the convention already used in 06a/06b/06c).
  - Array dtype safety: powerfit() explicitly casts its working slices to
    float64 with np.asarray(..., dtype=np.float64) before calling
    np.isfinite(). This prevents the
        TypeError: ufunc 'isfinite' not supported for the input types
    crash that occurs when a slice ends up as an object-dtype array
    (a pandas quirk that can occur after replace(0, np.nan) + dropna
    chains) - np.isfinite has no object-dtype loop, even though
    comparisons like `> 0` still work fine on such arrays.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except ImportError:
    _HAS_PINGOUIN = False

# Canonical default for the noise-floor acceptance fraction. A trajectory
# must be above the bead noise floor for STRICTLY MORE than this fraction
# of shared lag times to be kept (see process_directory() and
# trajectory_above_noise_floor() below - both use '>' not '>=').
NOISE_FLOOR_FRACTION = 0.98 #

# ============================================================================
# CORE MSD FUNCTIONS
# ============================================================================
def _msd_iter(pos, lagtimes):
    """Iterator for MSD calculation. Yields mean displacement (<dr>) and
    mean squared displacement (<dr^2>) components for each lag time.

    If every displacement at a given lag time is exactly zero (a fully
    degenerate lag - e.g. a very short, effectively-stationary segment),
    yields a row of NaN for that lag time rather than skipping it. This
    guarantees the output always has exactly len(lagtimes) rows with the
    lag-time index staying correctly aligned to each row (previously,
    skipping the row silently shifted every subsequent lag time's label by
    one, and produced a different number of rows for different
    trajectories even when called with the same max_lagtime - which broke
    any caller that stacks per-particle MSD arrays of a fixed length,
    e.g. np.vstack). NaN rows are naturally excluded downstream by
    powerfit()'s np.isfinite() validity filter, so this does not change
    which lag times are usable for fitting - it only fixes the alignment
    and shape.
    """
    n_cols = pos.shape[1] if pos.ndim > 1 else 1
    nan_row = np.full(n_cols * 2, np.nan)
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff_no_nan = np.nan_to_num(diff, nan=0)
        if not np.all(diff_no_nan == 0):
            yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))
        else:
            yield nan_row


def msd(pos, max_lagtime, sigma=None):
    """Compute MSD with optional per-particle localization correction.

    sigma : float or None
        If provided and finite, subtracts 4*sigma^2 (localization noise
        floor) from the raw MSD and clips the result at zero, producing
        'msd_corrected'. If None, 'msd_corrected' == 'msd'.
    """
    max_lagtime = min(int(max_lagtime), len(pos) - 1)
    lagtimes = np.arange(1, max_lagtime + 1)
    pos_columns = ['x', 'y']
    result_columns = [f'<{p}>' for p in pos_columns] + [f'<{p}^2>' for p in pos_columns]
    resultmsd = list(_msd_iter(pos, lagtimes))
    # _msd_iter always yields exactly len(lagtimes) rows now (degenerate
    # lag times come back as NaN rather than being dropped), so no
    # truncation of lagtimes is needed here any more.
    result = pd.DataFrame(resultmsd, columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(axis=1, skipna=False)
    if sigma is not None and np.isfinite(sigma):
        correction = 4 * (sigma ** 2)
        result['msd_corrected'] = result['msd'] - correction
        result['msd_corrected'] = result['msd_corrected'].clip(lower=0)
    else:
        result['msd_corrected'] = result['msd']
    return result


def calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_mass_scaling_flag):
    """Scale localization sigma based on particle brightness (mass).
    Brighter particles (higher mass) get a smaller sigma."""
    if not use_mass_scaling_flag or mass_particle <= 0 or mass_ref <= 0:
        return sigma_ref
    return sigma_ref * np.sqrt(mass_ref / mass_particle)


def power_law(x, alpha, A):
    """Power law function A * x^alpha (kept for reference / plotting;
    powerfit() below no longer calls this directly since it fits in
    log-log space, but downstream code may still import it)."""
    return A * np.power(x, alpha)


def trajectory(pos, pid, max_lagtime, video, pos_columns):
    """Shift a trajectory's starting point to (0, 0) and tag it with
    particle/video identifiers. Used for overlay/QC plots (originally
    duplicated identically across 03/04/09/11) - not part of the MSD
    calculation itself, just a labeling convenience so those scripts can
    concat multiple trajectories into one DataFrame."""
    max_lagtime = len(pos)
    lagtimes = np.arange(0, max_lagtime)
    result = pd.DataFrame(pos - pos[0], columns=pos_columns, index=lagtimes)
    result['particle'] = pid
    result['video'] = video
    return result


def powerfit(msd_df, start_index, end_index, fps=10, use_corrected=False):
    """
    Fit a power law to an MSD data slice via linear regression on
    log10-transformed data: log10(MSD) = alpha * log10(dt) + log10(A).

    Returns
    -------
    [alpha, A] : list of float
        alpha is UNBOUNDED (can be negative or > 2 for noisy/short slices).
    r_squared : float
        Computed on the log-transformed fit (log-space residuals).
    """
    if not isinstance(msd_df, pd.DataFrame):
        return [np.nan, np.nan], 0.0

    if use_corrected and 'msd_corrected' in msd_df.columns:
        y_raw = msd_df['msd_corrected'].values
    elif 'msd' in msd_df.columns:
        y_raw = msd_df['msd'].values
    else:
        return [np.nan, np.nan], 0.0

    x_raw = msd_df.index.values.astype('float64')

    a, b = start_index, end_index
    # Explicit float64 coercion BEFORE isfinite - prevents object-dtype
    # TypeError crashes (see module docstring).
    y_slice = np.asarray(y_raw[a:b], dtype=np.float64)
    x_slice = np.asarray(x_raw[a:b], dtype=np.float64)

    valid = np.isfinite(y_slice) & (y_slice > 0) & (x_slice > 0)
    y_fit = y_slice[valid]
    x_fit = x_slice[valid]

    if len(y_fit) < 2:
        return [np.nan, np.nan], 0.0

    try:
        log_x = np.log10(x_fit)
        log_y = np.log10(y_fit)

        coeffs, _, _, _, _ = np.polyfit(log_x, log_y, 1, full=True)
        alpha = coeffs[0]   # slope
        log_A = coeffs[1]   # intercept
        A = 10 ** log_A

        log_y_pred = alpha * log_x + log_A
        ss_res = np.sum((log_y - log_y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

        return [alpha, A], r_squared
    except Exception:
        return [np.nan, np.nan], 0.0


# ============================================================================
# OPTIONAL TRAJECTORY TRUNCATION (off by default everywhere)
# ============================================================================
# These live here so any script CAN opt in via the same code path instead of
# re-implementing it (as 05a previously did on its own). Truncation is only
# applied by imsd() if you pass truncation_cfg with enabled=True.

def find_first_mass_discontinuity(mass_arr, max_drop_frac, ignore_initial=3, avg_window=3):
    mass_arr = np.asarray(mass_arr, dtype=float)
    if len(mass_arr) < 2 or np.isnan(mass_arr).all():
        return len(mass_arr)
    for i in range(ignore_initial, len(mass_arr) - 1):
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
    original_len = len(pos)
    for i in range(original_len - 1):
        if np.isnan(pos[i]).any() or np.isnan(pos[i + 1]).any():
            return i
        step = np.linalg.norm(pos[i + 1] - pos[i])
        if step > max_step_size:
            return i
    return original_len


def handle_truncation(pos, mass_series, max_step_size, max_mass_drop,
                       keep_first_segment, enable_truncation, enable_mass_drop):
    original_len = len(pos)
    spatial_break = find_first_spatial_discontinuity(pos, max_step_size)
    mass_break = original_len

    if enable_mass_drop and mass_series is not None:
        mass_arr = mass_series.values if hasattr(mass_series, 'values') else mass_series
        mass_break = find_first_mass_discontinuity(mass_arr, max_mass_drop)

    final_break_index = min(spatial_break, mass_break)
    has_break = final_break_index < original_len

    segment_info = {
        'is_continuous': not has_break,
        'original_length': original_len,
        'first_break_at': final_break_index if has_break else None,
        'break_type': 'spatial' if spatial_break <= mass_break else 'mass',
        'used_length': original_len,
        'frames_discarded': 0,
    }

    if not enable_truncation:
        return pos, segment_info

    if has_break:
        if keep_first_segment:
            new_pos = pos[:final_break_index]
            segment_info['used_length'] = len(new_pos)
            segment_info['frames_discarded'] = original_len - len(new_pos)
            return new_pos, segment_info
        else:
            return None, segment_info

    return pos, segment_info


# ============================================================================
# PER-FILE TRAJECTORY PROCESSING
# ============================================================================
def imsd(file_path, mpp, fps, max_lagtime, sigma_ref, mass_ref, max_length,
         use_scaling_flag, use_localization_correction, r_squared_threshold,
         truncation_cfg=None):
    """
    Process a single trajectory CSV: compute per-particle MSD, apply
    localization correction, and fit alpha over two lag windows - a
    short-time window (0-8 frames, ~0.1-0.8s) and a long-time window
    (19-79 frames, ~2-8s). Trajectory acceptance (R^2 threshold and
    positive alpha) is gated ENTIRELY on the long-time window; the
    short-time window is excluded from this decision because it is
    contaminated by localization error (confirmed via Van Hove analysis)
    and is not used for any reported diffusion measurement. The
    short-time fit is still computed and returned (results_dict['alpha']/
    ['Dapp']) for potential diagnostic use.

    truncation_cfg : dict or None
        Optional dict with keys: enabled, enable_mass_drop, keep_first_segment,
        max_step_size, max_mass_drop, min_segment_length.
        If None (default), truncation is fully disabled - identical to the
        original 06b/06c behavior.

    Returns an 8-tuple: the original 7 values, plus a `counts` dict
    breaking down what happened to every particle examined in this file:
        {'n_total', 'n_too_short', 'n_msd_invalid', 'n_fit_invalid',
         'n_failed_r_squared', 'n_accepted'}
    `counts` is always returned, even when the file couldn't be read or no
    trajectory in it passed (in which case the other 7 values are None),
    so process_directory() can always aggregate a full per-condition
    breakdown of WHY trajectories were rejected - previously only the
    noise-floor filter's rejections were counted/printed; trajectories
    failing the R^2/alpha acceptance check here were silently dropped
    with no visible count anywhere.
    """
    counts = {'n_total': 0, 'n_too_short': 0, 'n_msd_invalid': 0,
              'n_fit_invalid': 0, 'n_failed_r_squared': 0, 'n_accepted': 0}
    try:
        traj = pd.read_csv(file_path)
    except Exception:
        return None, None, None, None, None, None, None, counts

    pos_columns = ['x', 'y']
    ids, msds_raw, msds_corrected = [], [], []
    alpha, alpha2, length, mass_values, sigma_values = [], [], [], [], []
    index = 0

    trunc_enabled = bool(truncation_cfg and truncation_cfg.get('enabled', False))

    for pid, ptraj in traj.groupby('particle'):
        counts['n_total'] += 1
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()

        mass_series = ptraj['mass'] if 'mass' in ptraj.columns else None

        if max_length is not None:
            pos = pos[0:max_length]
            if mass_series is not None:
                mass_series = mass_series.iloc[0:max_length]

        if trunc_enabled:
            pos, _segment_info = handle_truncation(
                pos, mass_series,
                truncation_cfg.get('max_step_size', 1.0),
                truncation_cfg.get('max_mass_drop', 0.5),
                truncation_cfg.get('keep_first_segment', False),
                enable_truncation=True,
                enable_mass_drop=truncation_cfg.get('enable_mass_drop', False),
            )
            if pos is None:
                counts['n_too_short'] += 1
                continue
            if len(pos) < truncation_cfg.get('min_segment_length', 10):
                counts['n_too_short'] += 1
                continue

        if len(pos) < 10:
            counts['n_too_short'] += 1
            continue

        mass_particle = ptraj['mass'].mean() if 'mass' in ptraj.columns else np.nan
        sigma_particle = (calculate_sigma_for_particle(mass_particle, sigma_ref, mass_ref, use_scaling_flag)
                          if use_localization_correction else None)

        msdtemp = msd(pos, max_lagtime, sigma=sigma_particle)
        if np.any(msdtemp.values == 0):
            msdtemp = msdtemp.replace(0, np.nan)
            msdtemp = msdtemp.dropna(axis='columns', how='all')
        if len(msdtemp.columns) <= 1:
            counts['n_msd_invalid'] += 1
            continue

        use_corrected_col = 'msd_corrected' if use_localization_correction and 'msd_corrected' in msdtemp.columns else 'msd'
        msd_for_fit = pd.DataFrame({'msd': msdtemp[use_corrected_col]}, index=msdtemp.index)
        msd_for_fit.index = msd_for_fit.index.values.astype('float64') / float(fps)

        alphatemp, r_squaredtemp = powerfit(msd_for_fit, 0, 8, fps, use_corrected=False)
        alphatemp2, r_squaredtemp2 = powerfit(msd_for_fit, 19, 80, fps, use_corrected=False)

        # Acceptance is gated ENTIRELY on the long-time (2-8s) window.
        # The short-time (0.1-0.8s) fit above is still computed (kept for
        # potential diagnostic use - results_dict['alpha']/['Dapp']) but
        # deliberately excluded from the accept/reject decision, since
        # that window is contaminated by localization error (confirmed via
        # Van Hove analysis) and shouldn't gate which trajectories are
        # trusted for the reported long-time diffusion measurements.
        if not (np.isfinite(alphatemp2[0]) and np.isfinite(r_squaredtemp2)):
            counts['n_fit_invalid'] += 1
            continue

        if r_squaredtemp2 >= r_squared_threshold and alphatemp2[0] > 0:
            alpha.append(alphatemp)
            alpha2.append(alphatemp2)
            msds_raw.append(msdtemp['msd'])
            msds_corrected.append(msdtemp.get('msd_corrected', msdtemp['msd']))
            mass_values.append(mass_particle)
            sigma_values.append(sigma_particle if sigma_particle is not None else np.nan)
            ids.append(index)
            index += 1
            length.append(len(pos))
            counts['n_accepted'] += 1
        else:
            counts['n_failed_r_squared'] += 1

    if not msds_raw:
        return None, None, None, None, None, None, None, counts

    resultsmsd_raw = pd.concat(msds_raw, axis=1, keys=ids)
    resultsmsd_corrected = pd.concat(msds_corrected, axis=1, keys=ids)
    lagt = resultsmsd_raw.index.values.astype('float64') / float(fps)
    resultsmsd_raw.index = lagt
    resultsmsd_corrected.index = lagt

    return (resultsmsd_raw, resultsmsd_corrected,
            pd.DataFrame(alpha), pd.DataFrame(alpha2),
            pd.DataFrame(length), pd.DataFrame(mass_values), pd.DataFrame(sigma_values),
            counts)


# ============================================================================
# PER-DIRECTORY AGGREGATION + NOISE FLOOR FILTERING
# ============================================================================
def process_directory(directory, mpp, fps, max_lagtime, sigma_ref, mass_ref,
                       max_length, use_scaling_flag, use_localization_correction,
                       r_squared_threshold, noise_floor_fraction=NOISE_FLOOR_FRACTION,
                       noise_floor_msd=None, truncation_cfg=None, verbose=True):
    """
    Process all trajectory CSVs in a directory and aggregate results.

    noise_floor_msd : pd.Series or None
        Ensemble corrected MSD of the bead (noise floor) condition.
        When provided, a trajectory is kept only if its corrected MSD is
        STRICTLY above this floor for MORE than noise_floor_fraction of
        shared lag times (i.e. `> noise_floor_fraction`, not `>=`).
        Pass None to disable filtering (used for beads themselves).
    """
    MSDcollected_raw, MSDcollected_corrected = [], []
    alphacollected, alphacollected2 = [], []
    lengthcollected, masscollected, sigmacollected = [], [], []

    total_counts = {'n_total': 0, 'n_too_short': 0, 'n_msd_invalid': 0,
                     'n_fit_invalid': 0, 'n_failed_r_squared': 0, 'n_accepted': 0}

    for file_path in directory.glob('*.csv'):
        if not file_path.name.startswith('.'):
            result = imsd(file_path, mpp, fps, max_lagtime, sigma_ref, mass_ref,
                          max_length, use_scaling_flag, use_localization_correction,
                          r_squared_threshold, truncation_cfg=truncation_cfg)
            msd_raw, msd_corr, alpha, alpha2, length, mass, sigma, counts = result
            for key in total_counts:
                total_counts[key] += counts[key]
            if msd_raw is not None:
                MSDcollected_raw.append(msd_raw)
                MSDcollected_corrected.append(msd_corr)
                alphacollected.append(alpha)
                alphacollected2.append(alpha2)
                lengthcollected.append(length)
                masscollected.append(mass)
                sigmacollected.append(sigma)

    if verbose:
        print(f"  Trajectory acceptance ({total_counts['n_total']} particles examined): "
              f"{total_counts['n_too_short']} too short, "
              f"{total_counts['n_msd_invalid']} invalid MSD, "
              f"{total_counts['n_fit_invalid']} fit failed, "
              f"{total_counts['n_failed_r_squared']} failed R^2/alpha threshold, "
              f"{total_counts['n_accepted']} accepted (pre-noise-floor).")

    if not MSDcollected_raw:
        return None

    MSDcollected_raw_df = pd.concat(MSDcollected_raw, axis=1)
    MSDcollected_corrected_df = pd.concat(MSDcollected_corrected, axis=1)

    n_cols = len(MSDcollected_corrected_df.columns)
    MSDcollected_raw_df.columns = list(range(n_cols))
    MSDcollected_corrected_df.columns = list(range(n_cols))

    Alphacollected_df = pd.concat(alphacollected, ignore_index=True)
    Alphacollected2_df = pd.concat(alphacollected2, ignore_index=True)

    if masscollected and sigmacollected:
        mass_data = pd.concat(masscollected, ignore_index=True)[0]
        sigma_data = pd.concat(sigmacollected, ignore_index=True)[0]
    else:
        mass_data, sigma_data = np.array([]), np.array([])

    n_removed_noise_floor = 0
    if noise_floor_msd is not None:
        valid_mask = []
        for col in MSDcollected_corrected_df.columns:
            traj_msd = MSDcollected_corrected_df[col]
            common_idx = traj_msd.index.intersection(noise_floor_msd.index)
            if len(common_idx) == 0:
                valid_mask.append(False)
                continue
            above = traj_msd.loc[common_idx].values > noise_floor_msd.loc[common_idx].values
            fraction_above = above.sum() / len(above) if len(above) > 0 else 0
            valid_mask.append(fraction_above > noise_floor_fraction)

        valid_series = pd.Series(valid_mask, index=MSDcollected_corrected_df.columns)
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        n_removed_noise_floor = len(valid_mask) - sum(valid_mask)

        MSDcollected_raw_df = MSDcollected_raw_df.loc[:, valid_series]
        MSDcollected_corrected_df = MSDcollected_corrected_df.loc[:, valid_series]
        Alphacollected_df = Alphacollected_df.iloc[valid_idx]
        Alphacollected2_df = Alphacollected2_df.iloc[valid_idx]
        if len(mass_data) > 0:
            mass_data = mass_data.iloc[valid_idx]
        if len(sigma_data) > 0:
            sigma_data = sigma_data.iloc[valid_idx]

        if verbose:
            print(f"  Noise floor filter: {n_removed_noise_floor} removed, {sum(valid_mask)} retained.")

    if MSDcollected_raw_df.empty:
        return None

    totaltracks = len(MSDcollected_raw_df.columns)
    Dapp_data = Alphacollected_df[1].values / 4.0
    Dapp2_data = Alphacollected2_df[1].values / 4.0
    alpha2_data = Alphacollected2_df[0].values

    return {
        'msd_raw': MSDcollected_raw_df,
        'msd_corrected': MSDcollected_corrected_df,
        'ensemble_msd_raw': MSDcollected_raw_df.mean(axis=1),
        'ensemble_msd_corrected': MSDcollected_corrected_df.mean(axis=1),
        'alpha': Alphacollected_df[0].values,
        'alpha2': alpha2_data,
        'Dapp': Dapp_data,
        'Dapp2': Dapp2_data,
        'mass': mass_data,
        'sigma': sigma_data,
        'n_tracks': totaltracks,
        'rejection_counts': {
            **total_counts,
            'n_removed_noise_floor': n_removed_noise_floor,
        },
    }


# ============================================================================
# SHARED HELPERS FOR NOISE-FLOOR-ONLY ANALYSES
# (van Hove, angular-bias, trajectory-length-sensitivity scripts) -
# these don't fit alpha/power-laws like imsd()/process_directory() do; they
# only need a trajectory's corrected MSD curve to check it against the bead
# noise floor. Folded in here so this logic (and the noise-floor comparison
# convention) has exactly one implementation instead of one per script.
# ============================================================================
def calculate_sigma_for_particle_from_mass_array(mass_data, sigma_ref, mass_ref,
                                                   use_mass_scaling_flag):
    """Array-input variant of calculate_sigma_for_particle(): takes a
    particle's per-frame mass values (e.g. ptraj['mass'].values) and uses
    their mean, rather than a single pre-averaged scalar. Mathematically
    equivalent to calling calculate_sigma_for_particle(mass_data.mean(), ...)
    - kept as a separate function only so scripts that pass the raw
    per-frame array don't need to compute the mean themselves."""
    mass_data = np.asarray(mass_data, dtype=float)
    if mass_data.size == 0:
        return sigma_ref
    return calculate_sigma_for_particle(np.nanmean(mass_data), sigma_ref, mass_ref,
                                        use_mass_scaling_flag)


def corrected_msd_series(pos, max_lagtime, sigma, fps):
    """
    Compute a single trajectory's corrected MSD as a pd.Series indexed by
    lag time in SECONDS (not frames). This is the noise-floor-comparison
    counterpart to msd() - used where only the corrected MSD curve is
    needed (not a fitted alpha/A), e.g. van Hove / angular-bias / trajectory
    length sensitivity scripts. Returns None if the trajectory is too short
    to produce any valid lag times.
    """
    if len(pos) < 2:
        return None
    result = msd(pos, max_lagtime, sigma=sigma)
    if result.empty:
        return None
    corrected = result['msd_corrected'].copy()
    corrected.index = corrected.index.values.astype('float64') / float(fps)
    return corrected


def compute_bead_noise_floor(bead_directory, mpp, fps, max_lagtime, max_length, verbose=True):
    """
    Build the ensemble noise-floor MSD curve from a directory of bead
    trajectory CSVs. No localization correction is applied (sigma=0.0)
    since the beads themselves define the noise floor.
    """
    bead_directory = Path(bead_directory)
    all_msds = []
    for file_path in bead_directory.glob('*.csv'):
        if file_path.name.startswith('.'):
            continue
        try:
            traj_df = pd.read_csv(file_path)
        except Exception:
            continue
        for pid, ptraj in traj_df.groupby('particle'):
            pos_df = ptraj.set_index('frame')[['x', 'y']] * mpp
            pos = pos_df.reindex(np.arange(pos_df.index[0], 1 + pos_df.index[-1])).to_numpy()
            pos = pos[:max_length]
            if len(pos) < 10:
                continue
            msd_series = corrected_msd_series(pos, max_lagtime, sigma=0.0, fps=fps)
            if msd_series is not None:
                all_msds.append(msd_series)

    if not all_msds:
        return None

    ensemble_floor = pd.concat(all_msds, axis=1).mean(axis=1)
    if verbose:
        print(f"  Bead noise floor established from {len(all_msds)} trajectories "
              f"(mean MSD = {ensemble_floor.mean():.4e} um^2)")
    return ensemble_floor


def trajectory_above_noise_floor(pos, sigma, fps, max_lagtime, noise_floor_msd,
                                  noise_floor_fraction=NOISE_FLOOR_FRACTION):
    """
    True if this trajectory's corrected MSD is STRICTLY above the bead
    noise floor for MORE than noise_floor_fraction of shared lag times
    (i.e. uses '>', matching process_directory()'s convention - not '>=').
    """
    msd_series = corrected_msd_series(pos, max_lagtime, sigma, fps)
    if msd_series is None:
        return False
    common_idx = msd_series.index.intersection(noise_floor_msd.index)
    if len(common_idx) == 0:
        return False
    above = msd_series.loc[common_idx].values > noise_floor_msd.loc[common_idx].values
    return above.sum() / len(above) > noise_floor_fraction


def check_long_time_fit_acceptance(pos, max_lagtime, sigma_particle, fps,
                                    r_squared_threshold, use_localization_correction,
                                    fit_start_frame=19, fit_end_frame=80):
    """
    Single source of truth for the long-time (2-8s by default) power-law
    fit acceptance test that imsd()/process_directory() use to decide
    whether a trajectory is included in the Dapp/alpha/MSD analysis.
    Exposed separately so OTHER scripts (e.g. Van Hove/kurtosis analyses)
    that need to know whether imsd() would have accepted a given
    trajectory - without redoing imsd()'s full aggregation - can apply
    the EXACT same test, rather than reimplementing it ad hoc and risking
    it silently drifting out of sync with imsd() over time.

    Mirrors imsd()'s internal acceptance logic exactly (same MSD
    computation with the given sigma correction, same long-time window
    fit via powerfit(), same thresholds): computes msd(), fits
    [fit_start_frame:fit_end_frame] (frames 19:80 = ~2-8s at fps=10, the
    imsd() default), and accepts iff the fit is finite AND
    r_squared2 >= r_squared_threshold AND alpha2 > 0.

    Returns (accepted: bool, alpha2: [alpha, A] list, r_squared2: float).
    (accepted=False, [nan, nan], nan) if the MSD/fit couldn't be computed
    at all.
    """
    msdtemp = msd(pos, max_lagtime, sigma=sigma_particle)
    if np.any(msdtemp.values == 0):
        msdtemp = msdtemp.replace(0, np.nan)
        msdtemp = msdtemp.dropna(axis='columns', how='all')
    if len(msdtemp.columns) <= 1:
        return False, [np.nan, np.nan], np.nan

    use_corrected_col = ('msd_corrected' if use_localization_correction
                          and 'msd_corrected' in msdtemp.columns else 'msd')
    msd_for_fit = pd.DataFrame({'msd': msdtemp[use_corrected_col]}, index=msdtemp.index)
    msd_for_fit.index = msd_for_fit.index.values.astype('float64') / float(fps)

    alphatemp2, r_squaredtemp2 = powerfit(msd_for_fit, fit_start_frame, fit_end_frame,
                                           fps, use_corrected=False)

    if not (np.isfinite(alphatemp2[0]) and np.isfinite(r_squaredtemp2)):
        return False, alphatemp2, r_squaredtemp2

    accepted = (r_squaredtemp2 >= r_squared_threshold) and (alphatemp2[0] > 0)
    return accepted, alphatemp2, r_squaredtemp2


# ============================================================================
# Single source of truth for the statistical comparisons used on violin
# plots (alpha, Dapp, etc.) across 05a/05b/05c. Replaces the previous
# per-script pattern of uncorrected pairwise scipy.stats.ttest_ind calls,
# which does not control the family-wise error rate across multiple
# comparisons. Tukey's HSD is the standard post-hoc test paired with a
# one-way ANOVA and is used here via scipy.stats.tukey_hsd (no extra
# dependency beyond scipy, which the pipeline already requires).

def check_normality(data_list, labels, verbose=True):
    """Shapiro-Wilk normality test, one group at a time.

    This is a diagnostic, not a gate - it doesn't change what runs
    downstream. Small sample sizes have low power to detect non-normality,
    and large sample sizes can flag trivial deviations as 'significant',
    so treat this as one input alongside actually looking at the violin
    plots, not a strict pass/fail rule.

    Returns a list of dicts: [{'label', 'n', 'W', 'p'}, ...]
    """
    results = []
    for label, data in zip(labels, data_list):
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            results.append({'label': label, 'n': len(arr), 'W': np.nan, 'p': np.nan})
            continue
        W, p = stats.shapiro(arr)
        results.append({'label': label, 'n': len(arr), 'W': W, 'p': p})

    if verbose:
        print("\nShapiro-Wilk normality check (per group):")
        print(f"  {'Label':<20} {'n':>6} {'W':>8} {'p':>10}  {'Normal? (p>0.05)'}")
        for r in results:
            if np.isnan(r['p']):
                print(f"  {r['label']:<20} {r['n']:>6}     (n<3, skipped)")
            else:
                flag = 'yes' if r['p'] > 0.05 else 'NO'
                print(f"  {r['label']:<20} {r['n']:>6} {r['W']:>8.4f} {r['p']:>10.4e}  {flag}")
    return results


def check_equal_variance(data_list, labels, verbose=True):
    """Levene's test for equal variance across all groups (the assumption
    Tukey's HSD, like the one-way ANOVA before it, relies on). Uses the
    median-centered variant (Brown-Forsythe), which is more robust to
    non-normality than the mean-centered original Levene's test.

    Returns {'statistic', 'pvalue', 'equal_variance_plausible' (p > 0.05)}.
    """
    clean = [np.asarray(d, dtype=float) for d in data_list]
    clean = [d[np.isfinite(d)] for d in clean]
    clean = [d for d in clean if len(d) >= 2]
    if len(clean) < 2:
        if verbose:
            print("\nLevene's test: not enough groups with data to run.")
        return {'statistic': np.nan, 'pvalue': np.nan, 'equal_variance_plausible': None}

    stat, p = stats.levene(*clean, center='median')
    if verbose:
        flag = 'plausible' if p > 0.05 else 'NOT plausible (variances differ)'
        print(f"\nLevene's test (equal variance across groups): "
              f"statistic={stat:.4f}, p={p:.4e} -> {flag}")
    return {'statistic': stat, 'pvalue': p, 'equal_variance_plausible': p > 0.05}


def run_anova_tukey(data_list, labels, exclude_substring='beads', verbose=True):
    """
    One-way ANOVA + Tukey HSD post-hoc across groups in data_list.

    Groups whose label contains exclude_substring (case-insensitive, e.g.
    the bead calibration control) are excluded from BOTH the ANOVA and the
    Tukey comparisons entirely, not just hidden from the output - an
    irrelevant calibration control shouldn't be part of the omnibus test.

    Returns a dict:
      {
        'anova_f': float, 'anova_p': float,
        'tukey_pvalues': {(i, j): p_adj, ...}   # keyed by ORIGINAL indices
                                                  # into data_list/labels,
                                                  # i < j, symmetric lookup
                                                  # not required (caller can
                                                  # look up either order)
        'included_indices': [...],               # original indices used
      }
    Groups with < 2 finite datapoints are also excluded (can't run a t-test
    against them). If fewer than 2 groups remain, tukey_pvalues is empty
    and anova_f/anova_p are NaN.
    """
    included_indices = []
    included_data = []
    for idx, (label, data) in enumerate(zip(labels, data_list)):
        if exclude_substring and exclude_substring.lower() in str(label).lower():
            continue
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue
        included_indices.append(idx)
        included_data.append(arr)

    result = {'anova_f': np.nan, 'anova_p': np.nan, 'tukey_pvalues': {},
              'included_indices': included_indices}

    if len(included_data) < 2:
        if verbose:
            print("\nANOVA/Tukey HSD: fewer than 2 valid groups, skipped.")
        return result

    f_stat, p_anova = stats.f_oneway(*included_data)
    result['anova_f'] = f_stat
    result['anova_p'] = p_anova

    tukey_res = stats.tukey_hsd(*included_data)
    for a, orig_i in enumerate(included_indices):
        for b, orig_j in enumerate(included_indices):
            if a < b:
                result['tukey_pvalues'][(orig_i, orig_j)] = tukey_res.pvalue[a, b]

    if verbose:
        print("\nOne-way ANOVA:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value:     {p_anova:.6e}")
        print("\nTukey HSD post-hoc (family-wise error rate corrected):")
        for (i, j), p_adj in sorted(result['tukey_pvalues'].items(), key=lambda kv: kv[1]):
            if p_adj < 0.001:
                sig = "***"
            elif p_adj < 0.01:
                sig = "**"
            elif p_adj < 0.05:
                sig = "*"
            else:
                sig = "ns"
            print(f"  {labels[i]} vs {labels[j]}: p_adj = {p_adj:.6e} ({sig})")

    return result


def log10_safe(data_list, verbose=True):
    """
    Log10-transform each group in data_list, dropping non-finite and
    non-positive values first (log10 of zero/negative is undefined).

    Intended for quantities like Dapp that are typically log-normally
    distributed (strictly positive, right-skewed) - ANOVA
    assumes roughly normal, roughly equal-variance data, which raw Dapp
    usually violates badly, but log10(Dapp) often satisfies well.

    Returns a new list of arrays (does not mutate the input). Prints a
    warning if any group loses a non-trivial fraction of points to the
    non-positive filter, since that would indicate something worth
    checking in the underlying data rather than just transforming past.
    """
    out = []
    for data in data_list:
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        n_before = len(arr)
        arr = arr[arr > 0]
        n_after = len(arr)
        if verbose and n_before > 0 and (n_before - n_after) / n_before > 0.01:
            print(f"  WARNING: log10_safe dropped {n_before - n_after}/{n_before} "
                  f"non-positive values from one group before log-transforming.")
        out.append(np.log10(arr))
    return out


def run_welch_games_howell(data_list, labels, exclude_substring='beads', verbose=True):
    """
    Welch's ANOVA + Games-Howell post-hoc across groups in data_list.

    Unlike run_anova_tukey(), this does NOT assume equal variance across
    groups - use this whenever check_equal_variance()/Levene's test flags
    unequal variance, since standard ANOVA/Tukey HSD's equal-variance
    assumption would then be violated. Same underlying questions as
    run_anova_tukey() ("is anything different overall", "which specific
    pairs differ, corrected for multiple comparisons"), just without
    relying on an assumption that's been shown not to hold.

    Requires the 'pingouin' package (pip install pingouin) - raises a
    clear ImportError here, at call time, rather than failing to import
    msd_core.py at all for scripts that never call this function.

    Groups whose label contains exclude_substring (case-insensitive, e.g.
    the bead calibration control) are excluded from BOTH the Welch ANOVA
    and the Games-Howell comparisons entirely, matching run_anova_tukey().

    Returns a dict:
      {
        'anova_f': float, 'anova_p': float,
        'pairwise_pvalues': {(i, j): p_adj, ...}   # keyed by ORIGINAL
                                                     # indices into
                                                     # data_list/labels,
                                                     # i < j
        'included_indices': [...],
      }
    Groups with < 2 finite datapoints are excluded. If fewer than 2 groups
    remain, pairwise_pvalues is empty and anova_f/anova_p are NaN.
    """
    if not _HAS_PINGOUIN:
        raise ImportError(
            "run_welch_games_howell() requires the 'pingouin' package, "
            "which is not installed. Install it with:\n"
            "    pip install pingouin"
        )

    included_indices = []
    rows = []
    index_by_group_key = {}
    for idx, (label, data) in enumerate(zip(labels, data_list)):
        if exclude_substring and exclude_substring.lower() in str(label).lower():
            continue
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue
        included_indices.append(idx)
        # Unique key per group (guards against duplicate display labels);
        # mapped back to the original index after pingouin returns results.
        group_key = f"{idx}__{label}"
        index_by_group_key[group_key] = idx
        for v in arr:
            rows.append({'value': v, 'group': group_key})

    result = {'anova_f': np.nan, 'anova_p': np.nan, 'pairwise_pvalues': {},
              'included_indices': included_indices}

    if len(included_indices) < 2:
        if verbose:
            print("\nWelch's ANOVA/Games-Howell: fewer than 2 valid groups, skipped.")
        return result

    df = pd.DataFrame(rows)

    anova_res = pg.welch_anova(data=df, dv='value', between='group')
    f_stat = float(anova_res['F'].iloc[0])
    p_anova = float(anova_res['p_unc'].iloc[0])
    result['anova_f'] = f_stat
    result['anova_p'] = p_anova

    gh_res = pg.pairwise_gameshowell(data=df, dv='value', between='group')
    for _, row in gh_res.iterrows():
        idx_a = index_by_group_key[row['A']]
        idx_b = index_by_group_key[row['B']]
        i, j = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
        result['pairwise_pvalues'][(i, j)] = float(row['pval'])

    if verbose:
        print("\nWelch's ANOVA (does not assume equal variance):")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value:     {p_anova:.6e}")
        print("\nGames-Howell post-hoc (family-wise error rate corrected, unequal variance):")
        for (i, j), p_adj in sorted(result['pairwise_pvalues'].items(), key=lambda kv: kv[1]):
            if p_adj < 0.001:
                sig = "***"
            elif p_adj < 0.01:
                sig = "**"
            elif p_adj < 0.05:
                sig = "*"
            else:
                sig = "ns"
            print(f"  {labels[i]} vs {labels[j]}: p_adj = {p_adj:.6e} ({sig})")

    return result


def extract_msd_at_lagtime(msd_corrected_df, target_lag_seconds, fps, tolerance_frames=1):
    """
    Extract each particle's corrected MSD value at (or nearest to) a fixed
    lag time, read directly off the corrected MSD curves in
    results_dict[key]['msd_corrected'] (rows = lag time in seconds,
    columns = one per trajectory).

    Why use this instead of Dapp for cross-group comparisons: Dapp is the
    prefactor of a fitted power law (MSD = A*t^alpha, Dapp = A/4), so its
    units are technically um^2/s^alpha - not directly comparable between
    particles or conditions whose fitted alpha differs, since the exponent
    on 'per second' is different. MSD at a fixed lag time has units of
    um^2 for every particle regardless of its individual alpha, so
    comparing it across groups doesn't have that units mismatch.

    Returns a 1D numpy array of MSD values (finite only) for particles
    that have data at/near target_lag_seconds. Returns an empty array if
    msd_corrected_df is empty/None or no lag time is close enough
    (within tolerance_frames * (1/fps) seconds) to target_lag_seconds.
    """
    if msd_corrected_df is None or msd_corrected_df.empty:
        return np.array([])

    idx = msd_corrected_df.index.values.astype(float)
    closest_pos = int(np.argmin(np.abs(idx - target_lag_seconds)))
    closest_lag = idx[closest_pos]

    frame_interval = 1.0 / fps
    if abs(closest_lag - target_lag_seconds) > tolerance_frames * frame_interval:
        return np.array([])

    values = msd_corrected_df.iloc[closest_pos].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def run_trend_test(data_list, x_values, labels, exclude_substring='beads',
                    log_x=True, verbose=True):
    """
    Tests whether the response variable shows a systematic trend across an
    ORDERED covariate (e.g. DNA fragment size in bp), instead of testing
    pairwise group differences the way run_anova_tukey()/
    run_welch_games_howell() do. This directly answers "is there a
    size-dependence" rather than inferring it from the pattern of which
    pairwise comparisons happen to be significant, which can be misleading
    when the groups have a natural order.

    Runs two complementary tests on POOLED per-particle data (every
    particle contributes one point tagged with its condition's x-value,
    not just one point per condition/group mean, so statistical power
    reflects the actual sample size):

    1. Linear regression (scipy.stats.linregress) of the response against
       x (log10(x) by default) - tests whether the slope differs
       significantly from zero. Assumes a roughly linear relationship
       between the response and log(x); appropriate when x spans orders
       of magnitude (e.g. bp count from 100 to 9553), which is why log_x
       defaults to True.

       How p_value is determined (scipy.stats.linregress internals):
       the null hypothesis is slope == 0 (no linear relationship). This
       is a two-sided t-test: t = slope / stderr, where stderr is the
       standard error of the slope estimate, computed as
           stderr = sqrt( (1 - r^2) / (n - 2) ) * (std(y) / std(x))
       (equivalently sqrt(SS_res / (n-2)) / sqrt(SS_xx)), and r is the
       Pearson correlation coefficient (lr.rvalue) between x and y. The
       test statistic t is referred to a Student's t-distribution with
       n - 2 degrees of freedom (n = pooled sample size, 2 parameters -
       slope and intercept - estimated from the data), and p_value is
       the two-sided tail probability P(|T| >= |t|) under that null
       distribution. Small p_value => the observed slope is unlikely
       under a true-zero-slope null => evidence of a real trend.
       lr.stderr (the slope's standard error itself) is also returned
       below and is what a 95% CI on the slope would be built from
       (slope +/- t_crit(n-2) * stderr).
    2. Spearman's rank correlation (scipy.stats.spearmanr) - a
       non-parametric monotonic-trend test that doesn't assume linearity,
       only that any real relationship is monotonic. Included since a
       trend could in principle be real but non-linear even on the log
       scale, and Spearman doesn't require the linearity assumption.

    Parameters
    ----------
    data_list : list of arrays, one per condition (already log10-
        transformed by the caller beforehand if desired, e.g. for
        Dapp/MSD which are typically log-normal - this function does not
        transform the response itself).
    x_values : list of numeric values, one per condition, SAME order and
        length as data_list/labels (e.g. bp count per condition). Entries
        for excluded groups (e.g. beads) can be any placeholder (NaN,
        None) since they're never used.
    labels : list of condition labels (for exclude_substring filtering
        and reporting).
    exclude_substring : groups whose label contains this (case-
        insensitive) are excluded entirely, e.g. 'beads', which has no
        meaningful size value.
    log_x : if True (default), regress against log10(x_values) rather
        than raw x_values.

    Returns a dict:
      {
        'regression': {'slope', 'intercept', 'r_value', 'p_value', 'stderr'},
        'spearman': {'rho', 'p_value'},
        'n_total': int,   # pooled sample size actually used
      }
    """
    pooled_x = []
    pooled_y = []
    included_labels = []
    for label, data, x in zip(labels, data_list, x_values):
        if exclude_substring and exclude_substring.lower() in str(label).lower():
            continue
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0 or not np.isfinite(x):
            continue
        included_labels.append(label)
        x_val = np.log10(x) if log_x else x
        pooled_x.extend([x_val] * len(arr))
        pooled_y.extend(arr.tolist())

    pooled_x = np.asarray(pooled_x, dtype=float)
    pooled_y = np.asarray(pooled_y, dtype=float)

    result = {
        'regression': {'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan,
                        'p_value': np.nan, 'stderr': np.nan},
        'spearman': {'rho': np.nan, 'p_value': np.nan},
        'n_total': len(pooled_y),
    }

    if len(pooled_y) < 3 or len(set(pooled_x.tolist())) < 2:
        if verbose:
            print("\nTrend test: not enough data/distinct x-values, skipped.")
        return result

    lr = stats.linregress(pooled_x, pooled_y)
    result['regression'] = {'slope': lr.slope, 'intercept': lr.intercept,
                             'r_value': lr.rvalue, 'p_value': lr.pvalue,
                             'stderr': lr.stderr}

    rho, p_spear = stats.spearmanr(pooled_x, pooled_y)
    result['spearman'] = {'rho': rho, 'p_value': p_spear}

    if verbose:
        x_label = 'log10(size)' if log_x else 'size'
        print(f"\nTrend test across ordered conditions ({', '.join(included_labels)}):")
        print(f"  n = {len(pooled_y)} pooled data points (per-particle, not per-condition)")
        print(f"  Linear regression (response vs {x_label}):")
        print(f"    slope = {lr.slope:.4e}, R^2 = {lr.rvalue**2:.4f}, p = {lr.pvalue:.6e}")
        sig = 'significant trend' if lr.pvalue < 0.05 else 'no significant trend'
        print(f"    -> {sig}")
        print(f"  Spearman rank correlation (monotonic trend, no linearity assumed):")
        print(f"    rho = {rho:.4f}, p = {p_spear:.6e}")
        sig2 = 'significant monotonic trend' if p_spear < 0.05 else 'no significant monotonic trend'
        print(f"    -> {sig2}")

    return result

# ============================================================================
# CONFINEMENT THRESHOLD CLASSIFICATION + SENSITIVITY ANALYSIS
# ============================================================================
# Single source of truth for the alpha < 0.4 "confined" classification used
# in the percentage bar charts (previously hardcoded independently in 5
# places across 05a/05b). Also provides a sensitivity analysis across a
# range of thresholds, so the choice of 0.4 specifically can be shown to be
# (or not be) robust, rather than resting on a single citation that does
# not itself use 0.4 as a hard cutoff.

DEFAULT_CONFINEMENT_THRESHOLD = 0.4


def classify_diffusion_regime(alpha_array, confined_threshold=DEFAULT_CONFINEMENT_THRESHOLD):
    """
    Classify a condition's alpha values into confined (< threshold),
    subdiffusive (threshold <= alpha < 1), and superdiffusive (alpha > 1).

    Returns (pct_confined, pct_subdiffusive, pct_superdiffusive), each a
    percentage (0-100). Returns (nan, nan, nan) if alpha_array is empty
    after dropping non-finite values.
    """
    arr = np.asarray(alpha_array, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    pct_confined = np.mean(arr < confined_threshold) * 100
    pct_subdiff = np.mean((arr >= confined_threshold) & (arr < 1.0)) * 100
    pct_superdiff = np.mean(arr > 1.0) * 100
    return pct_confined, pct_subdiff, pct_superdiff


def threshold_sensitivity_table(data_list, labels, thresholds=None,
                                 exclude_substring='beads', verbose=True):
    """
    Recompute the confined/subdiffusive/superdiffusive percentages across
    a RANGE of confinement thresholds (default 0.30-0.50 in steps of 0.05),
    to check whether the classification - and any differences between
    conditions built on it - depends on the exact threshold chosen or is
    robust across the range used elsewhere in the literature.

    Returns (results, thresholds) where results is a nested dict:
        {label: {threshold: {'confined', 'subdiffusive', 'superdiffusive'}}}
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.51, 0.05), 2)

    results = {}
    included_labels = []
    for label, data in zip(labels, data_list):
        if exclude_substring and exclude_substring.lower() in str(label).lower():
            continue
        included_labels.append(label)
        results[label] = {}
        for t in thresholds:
            confined, subdiff, superdiff = classify_diffusion_regime(data, confined_threshold=t)
            results[label][t] = {'confined': confined, 'subdiffusive': subdiff,
                                  'superdiffusive': superdiff}

    if verbose:
        print("\nThreshold sensitivity analysis (% classified as CONFINED, by threshold):")
        header = "  Condition".ljust(22) + "".join(f"a<{t:>5.2f}".rjust(10) for t in thresholds)
        print(header)
        for label in included_labels:
            row = f"  {label}".ljust(22)
            for t in thresholds:
                val = results[label][t]['confined']
                row += (f"{val:>9.1f}%" if np.isfinite(val) else f"{'n/a':>10}")
            print(row)

        # Check whether the RELATIVE ORDERING of conditions (not just the
        # absolute percentages) is stable across the threshold range -
        # this is the property that actually matters for "is my comparison
        # between conditions an artifact of the exact threshold chosen".
        orderings = []
        for t in thresholds:
            vals = [(results[l][t]['confined'], l) for l in included_labels
                    if np.isfinite(results[l][t]['confined'])]
            orderings.append(tuple(l for _, l in sorted(vals)))
        if len(set(orderings)) == 1:
            print("  -> Relative ordering of conditions by % confined is IDENTICAL "
                  "across all tested thresholds.")
        else:
            print("  -> WARNING: relative ordering of conditions by % confined CHANGES "
                  "somewhere across the tested threshold range - inspect the table above.")

    return results, thresholds


def plot_threshold_sensitivity(ax, data_list, labels, colors, thresholds=None,
                                exclude_substring='beads', highlight_threshold=DEFAULT_CONFINEMENT_THRESHOLD):
    """
    Line plot of % classified as 'confined' vs. the confinement threshold,
    one line per condition, over a fine-grained threshold range (for a
    smooth curve, denser than threshold_sensitivity_table's default).
    Demonstrates whether the RELATIVE differences between conditions are
    robust to the exact threshold chosen, rather than an artifact of
    picking 0.4 specifically.
    """
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.6, 41)

    included_indices = [i for i, label in enumerate(labels)
                        if not (exclude_substring and exclude_substring.lower() in str(label).lower())]

    for i in included_indices:
        arr = np.asarray(data_list[i], dtype=float)
        arr = arr[np.isfinite(arr)]
        pct_confined = [np.mean(arr < t) * 100 if len(arr) else np.nan for t in thresholds]
        ax.plot(thresholds, pct_confined, color=colors[i], linewidth=2, label=labels[i])

    ax.axvline(highlight_threshold, color='black', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'Threshold used ({highlight_threshold})')
    ax.set_xlabel(r'Confinement threshold ($\alpha_c$)', fontsize=14)
    ax.set_ylabel('% classified as confined', fontsize=14)
    ax.tick_params(which='both', axis='both', direction='in', labelsize=12, length=6, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(False)
    ax.legend(loc='best', frameon=False, fontsize=9)


def find_stable_threshold_range(data_list, labels, exclude_labels=None,
                                 target_threshold=DEFAULT_CONFINEMENT_THRESHOLD,
                                 resolution=0.01, bounds=(0.2, 0.6), verbose=True):
    """
    Find the widest contiguous threshold interval, containing
    target_threshold, over which the pairwise ordering of % confined
    across conditions matches the ordering AT target_threshold exactly -
    i.e. no pair of conditions swaps relative order anywhere inside the
    interval. Uses a fine grid (default step 0.01), not just a handful of
    spot-checked values, so the boundary is the actual crossing point
    rather than an approximation read off a coarser table.

    exclude_labels : optional list of condition labels to leave out of the
        stability check entirely (e.g. a condition with a much smaller
        sample size than the others, whose ordering is expected to be
        noisy regardless of threshold for reasons unrelated to the
        question this check is asking). Excluded conditions are not
        computed against anything; this does not distort the result for
        the remaining conditions.

    Returns (left_bound, right_bound): the boundaries of the stable
    interval, inclusive of the grid point nearest target_threshold.
    """
    exclude_labels = set(exclude_labels or [])
    included = [(l, d) for l, d in zip(labels, data_list) if l not in exclude_labels]
    included_labels = [l for l, _ in included]

    if len(included_labels) < 2:
        if verbose:
            print("\nStable threshold range: fewer than 2 conditions to compare, skipped.")
        return (np.nan, np.nan)

    grid = np.round(np.arange(bounds[0], bounds[1] + resolution / 2, resolution), 6)
    target_idx = int(np.argmin(np.abs(grid - target_threshold)))

    pct = {}
    for label, data in included:
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        pct[label] = np.array([np.mean(arr < t) * 100 if len(arr) else np.nan for t in grid])

    pairs = [(a, b) for i, a in enumerate(included_labels) for b in included_labels[i + 1:]]

    def signs_at(idx):
        return {(a, b): np.sign(pct[a][idx] - pct[b][idx]) for a, b in pairs}

    ref_signs = signs_at(target_idx)

    left_idx = target_idx
    for idx in range(target_idx - 1, -1, -1):
        cur_signs = signs_at(idx)
        if any(cur_signs[p] != 0 and ref_signs[p] != 0 and cur_signs[p] != ref_signs[p] for p in pairs):
            break
        left_idx = idx

    right_idx = target_idx
    for idx in range(target_idx + 1, len(grid)):
        cur_signs = signs_at(idx)
        if any(cur_signs[p] != 0 and ref_signs[p] != 0 and cur_signs[p] != ref_signs[p] for p in pairs):
            break
        right_idx = idx

    left_bound = float(grid[left_idx])
    right_bound = float(grid[right_idx])

    if verbose:
        excl_note = f" (excluding {', '.join(exclude_labels)})" if exclude_labels else ""
        print(f"\nStable threshold range around alpha_c={target_threshold}{excl_note}: "
              f"[{left_bound:.2f}, {right_bound:.2f}]")
        print(f"  Relative ordering of {', '.join(included_labels)} matches the "
              f"ordering at {target_threshold} throughout this range.")

    return left_bound, right_bound


def find_all_stable_plateaus(data_list, labels, exclude_labels=None,
                              resolution=0.01, bounds=(0.1, 0.9),
                              min_width=0.03, verbose=True):
    """
    Scan the ENTIRE threshold range (not just the neighborhood around one
    chosen value) and report every maximal "plateau" - a contiguous
    interval over which the pairwise ordering of % confined across
    conditions doesn't change at all.

    This directly answers "why not use some other threshold entirely"
    (e.g. why not 0.7 instead of 0.4): find_stable_threshold_range() only
    proves LOCAL robustness around one chosen value; it says nothing about
    whether a completely different value might sit in an equally-valid
    plateau supporting a different conclusion. This function finds every
    plateau in the scanned range so that comparison can actually be made,
    rather than assumed.

    Returns a list of dicts, each: {'start', 'end', 'width', 'ordering'}
    (ordering = the tuple of labels sorted by % confined within that
    plateau), sorted by width descending (widest/most robust plateau
    first). Plateaus narrower than min_width are omitted from the printed
    summary (still returned) since they're not wide enough to be a
    meaningful alternative regardless of ordering.
    """
    exclude_labels = set(exclude_labels or [])
    included = [(l, d) for l, d in zip(labels, data_list) if l not in exclude_labels]
    included_labels = [l for l, _ in included]

    grid = np.round(np.arange(bounds[0], bounds[1] + resolution / 2, resolution), 6)

    pct = {}
    for label, data in included:
        arr = np.asarray(data, dtype=float)
        arr = arr[np.isfinite(arr)]
        pct[label] = np.array([np.mean(arr < t) * 100 if len(arr) else np.nan for t in grid])

    def ordering_at(idx):
        vals = [(pct[l][idx], l) for l in included_labels]
        return tuple(l for _, l in sorted(vals))

    plateaus = []
    seg_start = 0
    seg_ordering = ordering_at(0)
    for idx in range(1, len(grid)):
        cur_ordering = ordering_at(idx)
        if cur_ordering != seg_ordering:
            plateaus.append({
                'start': float(grid[seg_start]), 'end': float(grid[idx - 1]),
                'width': float(grid[idx - 1] - grid[seg_start]),
                'ordering': seg_ordering,
            })
            seg_start = idx
            seg_ordering = cur_ordering
    plateaus.append({
        'start': float(grid[seg_start]), 'end': float(grid[-1]),
        'width': float(grid[-1] - grid[seg_start]),
        'ordering': seg_ordering,
    })

    plateaus_sorted = sorted(plateaus, key=lambda p: p['width'], reverse=True)

    if verbose:
        excl_note = f" (excluding {', '.join(exclude_labels)})" if exclude_labels else ""
        print(f"\nFull-range plateau scan, alpha_c in [{bounds[0]}, {bounds[1]}]{excl_note}:")
        shown = [p for p in plateaus_sorted if p['width'] >= min_width]
        if not shown:
            print(f"  No plateau wider than {min_width} found - ordering changes "
                  f"almost continuously across the scanned range.")
        for p in shown:
            print(f"  [{p['start']:.2f}, {p['end']:.2f}]  (width={p['width']:.2f}):  "
                  + " < ".join(p['ordering']))

    return plateaus_sorted


def plot_dapp_alpha_diagnostic(ax1, ax2, results_dict, condition_keys, labels, colors,
                                target_lag_seconds=8.0, fps=10,
                                exclude_substring='beads'):
    """
    Two-panel diagnostic (caller provides both axes, matching the
    ax-parameter convention used by plot_threshold_sensitivity() etc. -
    msd_core.py does not import matplotlib.pyplot itself) distinguishing
    whether an observed Dapp-vs-alpha relationship is a mechanical
    artifact of the power-law fit, or reflects genuine underlying physics.

    Left panel (ax1):  Dapp vs alpha, pooled per-particle across
                 conditions. Dapp and alpha are the intercept and slope
                 of the SAME log-log fit (MSD = 4*Dapp*t^alpha), so they
                 are not independent measurements - a negative
                 correlation here can arise purely from fit geometry (a
                 shallower line pinned through similar MSD values needs a
                 higher intercept), regardless of any real biology.
    Right panel (ax2): directly-measured MSD at target_lag_seconds vs
                 alpha, pooled the same way. This is NOT subject to the
                 same mechanical coupling, since it's read directly off
                 the corrected MSD curve rather than fit - a
                 relationship that persists here is more likely to
                 reflect real confinement-related physics rather than
                 fit geometry.

    Both panels report the Spearman correlation (monotonic, matches
    run_trend_test's convention) so the strength of the two relationships
    can be compared directly. The printed interpretation checks SIGN
    agreement between the two correlations before comparing magnitude -
    Dapp-alpha correlations are typically negative from fit geometry alone
    (see left panel note above) while a real MSD-alpha relationship is
    typically positive (more superdiffusive particles travel further by a
    fixed lag time); if the two panels disagree in sign, that's a stronger
    signal that fit-coupling is dominating the pooled Dapp-alpha trend
    than either magnitude-ratio comparison alone would show, and is
    checked first. Returns 'signs_agree' (bool, or None if either
    correlation is NaN) alongside the two correlations/p-values so callers
    can act on this programmatically rather than only via the printed text.
    """
    pooled_alpha_dapp, pooled_dapp = [], []
    pooled_alpha_msd, pooled_msd = [], []
    point_colors_dapp, point_colors_msd = [], []

    for key, label, color in zip(condition_keys, labels, colors):
        if exclude_substring and exclude_substring.lower() in str(label).lower():
            continue
        if key not in results_dict:
            continue
        data = results_dict[key]

        alpha_arr = np.asarray(data['alpha2'], dtype=float)
        dapp_arr = np.asarray(data['Dapp2'], dtype=float)
        mask = np.isfinite(alpha_arr) & np.isfinite(dapp_arr) & (dapp_arr > 0)
        ax1.scatter(alpha_arr[mask], dapp_arr[mask], c=color, alpha=0.6, s=10,
               edgecolor='none', label=label)
        pooled_alpha_dapp.extend(alpha_arr[mask].tolist())
        pooled_dapp.extend(dapp_arr[mask].tolist())
        point_colors_dapp.extend([color] * int(mask.sum()))

        msd_df = data.get('msd_corrected')
        if msd_df is not None and not msd_df.empty:
            idx = msd_df.index.values.astype(float)
            closest_pos = int(np.argmin(np.abs(idx - target_lag_seconds)))
            msd_row = msd_df.iloc[closest_pos].to_numpy(dtype=float)  # NaN kept, stays aligned to alpha_arr
            if len(msd_row) == len(alpha_arr):
                mask2 = np.isfinite(alpha_arr) & np.isfinite(msd_row) & (msd_row > 0)
                pooled_alpha_msd.extend(alpha_arr[mask2].tolist())
                pooled_msd.extend(msd_row[mask2].tolist())
                point_colors_msd.extend([color] * int(mask2.sum()))

    pooled_alpha_dapp = np.array(pooled_alpha_dapp)
    pooled_dapp = np.array(pooled_dapp)
    log_dapp = np.log10(pooled_dapp)
    rho1, p1 = stats.spearmanr(pooled_alpha_dapp, log_dapp) if len(pooled_alpha_dapp) > 2 else (np.nan, np.nan)
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\alpha$', fontsize=14)
    ax1.set_ylabel(r'$D_{\mathrm{app}}$ ($\mu$m$^2$/s$^\alpha$)', fontsize=14)
    ax1.text(0.05, 0.95, f"Spearman $\\rho$ = {rho1:.3f}\n$p$ = {p1:.2e}",
             transform=ax1.transAxes, fontsize=11, va='top', ha='left')
    ax1.text(-0.15, 1.05, 'A',transform=ax1.transAxes,fontsize=16, fontweight='bold',
        va='top', ha='left')

    pooled_alpha_msd = np.array(pooled_alpha_msd)
    pooled_msd = np.array(pooled_msd)
    log_msd = np.log10(pooled_msd)
    rho2, p2 = stats.spearmanr(pooled_alpha_msd, log_msd) if len(pooled_alpha_msd) > 2 else (np.nan, np.nan)
    ax2.scatter(pooled_alpha_msd, pooled_msd, c=point_colors_msd, alpha=0.6, s=10, edgecolor='none')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\alpha$', fontsize=14)
    ax2.set_ylabel(f'MSD at $\Delta t =$ {target_lag_seconds:.0f}s ($\\mu$m$^2$)', fontsize=14)
    ax2.text(0.05, 0.95, f"Spearman $\\rho$ = {rho2:.3f}\n$p$ = {p2:.2e}",
             transform=ax2.transAxes, fontsize=11, va='top', ha='left')
    ax2.text(-0.15, 1.05, 'B',transform=ax2.transAxes,fontsize=16, fontweight='bold',
        va='top', ha='left')

    handles, labels_ = ax1.get_legend_handles_labels()
    ax2.figure.legend(handles, labels_, loc='center left',
                       bbox_to_anchor=(1.02, 0.8), bbox_transform=ax2.transAxes,
                       fontsize=10)

    for ax in (ax1, ax2):
        ax.tick_params(which='both', axis='both', direction='in', labelsize=12, length=6, width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(False)

    print(f"\nDapp-alpha coupling diagnostic:")
    print(f"  Dapp vs alpha:          rho={rho1:.3f}, p={p1:.2e}, n={len(pooled_alpha_dapp)}")
    print(f"  MSD@{target_lag_seconds:.0f}s vs alpha (direct): rho={rho2:.3f}, p={p2:.2e}, n={len(pooled_alpha_msd)}")
    signs_agree = None
    if np.isfinite(rho1) and np.isfinite(rho2):
        signs_agree = (np.sign(rho1) == np.sign(rho2)) or (rho1 == 0) or (rho2 == 0)
        if not signs_agree:
            # A negative Dapp-alpha correlation together with a positive
            # MSD@8s-alpha correlation is exactly the signature fit-geometry
            # coupling predicts: on the same log-log fit, a steeper slope
            # (higher alpha) mechanically pulls the intercept (Dapp) down,
            # which is the OPPOSITE of the physically-expected direction
            # (more superdiffusive particles travel further by a fixed lag
            # time, i.e. positive MSD-alpha). The magnitude-ratio checks
            # below can't catch this - |rho2| can be close to |rho1| (or
            # even larger) while the two panels are still disagreeing on
            # which way the relationship goes, which is a stronger warning
            # sign than either magnitude comparison alone conveys.
            print("  -> Dapp-alpha and MSD@8s-alpha correlations have OPPOSITE SIGNS: "
                  "this is the specific signature of fit-geometry coupling dominating "
                  "the pooled Dapp-alpha trend (not just adding noise to it), rather "
                  "than that trend reflecting the same real relationship MSD@8s shows. "
                  "Treat the pooled Dapp-alpha correlation's direction as unreliable; "
                  "use MSD@8s to judge the real pooled trend. This does NOT by itself "
                  "invalidate a Dapp difference between two SPECIFIC groups whose alpha "
                  "does not differ (mean and variance) - that comparison should be "
                  "checked on its own terms, e.g. via check_equal_variance() plus "
                  "run_welch_games_howell() on alpha for those two groups.")
        elif abs(rho2) < 0.5 * abs(rho1):
            print("  -> MSD@8s correlation is much weaker than Dapp correlation: "
                  "consistent with the Dapp-alpha relationship being substantially "
                  "a fitting artifact rather than purely physical.")
        elif abs(rho2) > 0.8 * abs(rho1):
            print("  -> MSD@8s correlation is nearly as strong as Dapp correlation, "
                  "and both agree in sign: the relationship likely reflects genuine "
                  "physics, not just fit geometry.")
        else:
            print("  -> MSD@8s correlation is intermediate (same sign as Dapp, "
                  "smaller magnitude): likely a mix of real physics and "
                  "fitting-geometry contributions.")

    return {'dapp_rho': rho1, 'dapp_p': p1, 'msd_rho': rho2, 'msd_p': p2,
            'signs_agree': signs_agree}


def msd_lagtime_sensitivity_table(results_dict, condition_keys, labels,
                                   lag_times=None, fps=10, verbose=True):
    """
    Recompute pairwise significance (Games-Howell on log10-transformed
    MSD, matching the STATISTICAL ANALYSIS - MSD AT 8s convention used
    elsewhere) across a RANGE of extraction lag times (default 2.0-8.0s
    in 0.5s steps), to check whether a single fixed extraction time
    (e.g. 8s) gives conclusions that hold across the trustworthy window,
    or whether curve convergence/divergence near one particular time is
    suppressing or inflating a real difference visible elsewhere in the
    window.

    condition_keys, labels : should NOT include beads - pass only the
    conditions actually being compared (matching the convention used by
    find_stable_threshold_range() etc. elsewhere in this module).

    Returns (pair_pvalues, lag_times) where pair_pvalues is a dict
    {(label_i, label_j): [p_adj at each lag time]}.
    """
    if lag_times is None:
        lag_times = np.round(np.arange(2.0, 8.01, 0.5), 2)

    n = len(labels)
    pairs = [(labels[i], labels[j]) for i in range(n) for j in range(i + 1, n)]
    pair_pvalues = {pair: [] for pair in pairs}

    for t in lag_times:
        msd_data = [extract_msd_at_lagtime(results_dict[k]['msd_corrected'], t, fps)
                    for k in condition_keys]
        log_msd = log10_safe(msd_data, verbose=False)
        gh = run_welch_games_howell(log_msd, labels, exclude_substring=None, verbose=False)
        for (i, j), p in gh['pairwise_pvalues'].items():
            pair_pvalues[(labels[i], labels[j])].append(p)

    if verbose:
        print(f"\nMSD extraction-time sensitivity analysis (pairwise p-values, "
              f"{lag_times[0]:.1f}-{lag_times[-1]:.1f}s):")
        header = "  Pair".ljust(35) + "".join(f"{t:.1f}s".rjust(9) for t in lag_times)
        print(header)
        for pair, pvals in pair_pvalues.items():
            label_str = f"{pair[0]} vs {pair[1]}"
            row = f"  {label_str}".ljust(35)
            for p in pvals:
                if not np.isfinite(p):
                    sig = "n/a"
                elif p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                row += f"{sig:>9}"
            print(row)

        print("\n  Pairs whose significance call (p<0.05 vs not) changes across this range:")
        any_change = False
        for pair, pvals in pair_pvalues.items():
            sig_calls = [np.isfinite(p) and p < 0.05 for p in pvals]
            if len(set(sig_calls)) > 1:
                any_change = True
                print(f"    {pair[0]} vs {pair[1]}")
        if not any_change:
            print("    none - all pairwise conclusions stable across the "
                  "tested extraction-time range.")

    return pair_pvalues, lag_times


def plot_msd_lagtime_sensitivity(ax, results_dict, condition_keys, labels, colors,
                                  lag_times=None, fps=10):
    """
    Line plot of -log10(p) vs. MSD extraction lag time, one line per
    pairwise comparison, with a horizontal reference line at p=0.05
    (i.e. -log10(0.05) ~ 1.3). Complements
    msd_lagtime_sensitivity_table()'s printed output with a visual
    showing exactly where (if anywhere) a comparison crosses the
    significance threshold across the tested window.
    """
    pair_pvalues, lag_times = msd_lagtime_sensitivity_table(
        results_dict, condition_keys, labels, lag_times=lag_times, fps=fps, verbose=False
    )

    label_to_color = {label: colors[i] for i, label in enumerate(labels)}

    for (label_i, label_j), pvals in pair_pvalues.items():
        neg_log_p = [-np.log10(p) if np.isfinite(p) and p > 0 else np.nan for p in pvals]
        ax.plot(lag_times, neg_log_p, marker='o', markersize=4, linewidth=1.5,
                label=f"{label_i} vs {label_j}")

    ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1.2,
               alpha=0.6, label='p = 0.05')
    ax.set_xlabel('MSD extraction lag time (s)', fontsize=14)
    ax.set_ylabel(r'$-\log_{10}(p)$', fontsize=14)
    ax.tick_params(which='both', axis='both', direction='in', labelsize=12, length=6, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(False)
    ax.legend(loc='best', frameon=False, fontsize=8)