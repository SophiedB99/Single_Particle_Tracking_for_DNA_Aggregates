# -*- coding: utf-8 -*-
"""
Localization Error Validation Suite
Comprehensive validation of sigma calculation and correction methods
@author: sophiedeboer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import warnings
from scipy import stats
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
CY3_BRIGHT_PATH = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/MDA-MB-231/Cy3 MDA-MB-231/500 bp_most_recent/E_output_data/Collected_Cy3_MDAMB231/tm'
YOYO1_DIM_PATH = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/Aswin Data MDA-MB-231 Corrected for Drift/E_output_data/Collected_MDAMB231/tm'
BEADS_PATH = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Stuck_Beads_Reference/tm'  # UPDATE THIS

MPP = 0.102
FPS = 10
MIN_TRACK_LENGTH = 50
MAX_LAGTIME_FRAMES = 10

# Your current correction values
SIGMA_YOYO1_CURRENT = 0.036  # Update with your current value in microns
SIGMA_CY3_CURRENT = 0.020    # Update with your current value in microns
SIGMA_BEADS_REF = 0.0135     # 13.5 nm reference from stuck beads

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def calculate_ensemble_msd_and_mass(directory, mpp, fps, min_track_length=50, max_lag_frames=50):
    """Calculate ensemble MSD and collect mass values."""
    all_displacements = {}
    mass_values = []
    
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return None, None
        
    file_count = 0
    total_tracks = 0
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and not filename.startswith('.'):
            filepath = os.path.join(directory, filename)
            try:
                traj_df = pd.read_csv(filepath)
            except Exception as e:
                continue

            file_count += 1
            for particle_id, ptraj in traj_df.groupby('particle'):
                if len(ptraj) < min_track_length:
                    continue
                total_tracks += 1
                
                if 'mass' in ptraj.columns:
                    mass_values.append(ptraj['mass'].mean())
                    
                ptraj = ptraj.set_index('frame').sort_index()
                x_microns = ptraj['x'].values * mpp
                y_microns = ptraj['y'].values * mpp
                
                for lag_frames in range(1, max_lag_frames + 1):
                    if len(ptraj) > lag_frames:
                        dx = x_microns[lag_frames:] - x_microns[:-lag_frames]
                        dy = y_microns[lag_frames:] - y_microns[:-lag_frames]
                        r2 = dx**2 + dy**2
                        
                        if lag_frames not in all_displacements:
                            all_displacements[lag_frames] = []
                        all_displacements[lag_frames].extend(r2.tolist())

    if not all_displacements:
        return None, None
        
    msd_data = {
        lag_frames / fps: np.mean(displacements)
        for lag_frames, displacements in all_displacements.items()
    }

    print(f"  Loaded {file_count} files, {total_tracks} tracks, {len(mass_values)} mass values")
    
    msd_series = pd.Series(msd_data).sort_index()
    msd_series.index.name = 'dt'
    return msd_series, mass_values

# ============================================================================
# VALIDATION TEST 1: MSD INTERCEPT ANALYSIS
# ============================================================================
def validate_msd_intercept(msd_series, label, fps, max_frames=5):
    """
    Extrapolate MSD to dt=0 to estimate 4*sigma^2.
    A purely noise-dominated regime should show a non-zero intercept.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION 1: MSD Intercept Analysis - {label}")
    print(f"{'='*80}")
    
    # Use first few time points
    time_lags = np.arange(1, max_frames + 1) / fps
    msd_short = msd_series.loc[msd_series.index.intersection(time_lags)]
    
    if len(msd_short) < 3:
        print("ERROR: Not enough data points for fitting")
        return None, None
    
    # Fit MSD = 4*D*t + 4*sigma^2
    def msd_model(t, D, sigma_sq):
        return 4 * D * t + 4 * sigma_sq
    
    try:
        popt, pcov = curve_fit(msd_model, msd_short.index, msd_short.values, p0=[0.01, 0.001])
        D_fit, sigma_sq_fit = popt
        sigma_intercept = np.sqrt(sigma_sq_fit)
        
        # Calculate R^2
        residuals = msd_short.values - msd_model(msd_short.index, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_short.values - np.mean(msd_short.values))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Linear fit: MSD = 4D·t + 4σ²")
        print(f"  D (fitted) = {D_fit:.6f} μm²/s")
        print(f"  4σ² (intercept) = {4*sigma_sq_fit:.8f} μm² ({4*sigma_sq_fit*1e6:.2f} nm²)")
        print(f"  σ (from intercept) = {sigma_intercept*1000:.3f} nm")
        print(f"  R² = {r_squared:.4f}")
        
        # Check if slope is significant
        if D_fit > 0.001:
            print(f"  ⚠️  WARNING: Significant diffusion detected (D = {D_fit:.6f} μm²/s)")
            print(f"      Short-time MSD includes real motion, sigma is an UPPER BOUND")
        else:
            print(f"  ✓ Diffusion minimal, intercept method reliable")
            
        return sigma_intercept, D_fit
        
    except Exception as e:
        print(f"ERROR in fitting: {e}")
        return None, None

# ============================================================================
# VALIDATION TEST 2: MASS SCALING VERIFICATION
# ============================================================================
def validate_mass_scaling(mass_dict, sigma_dict, reference_label='Beads'):
    """
    Verify that sigma ∝ 1/sqrt(Mass) holds across all datasets.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION 2: Mass Scaling Relationship (σ ∝ 1/√Mass)")
    print(f"{'='*80}")
    
    labels = list(mass_dict.keys())
    medians = [np.median(mass_dict[label]) for label in labels]
    sigmas = [sigma_dict[label] for label in labels]
    
    # Calculate expected sigma from mass scaling
    ref_mass = np.median(mass_dict[reference_label])
    ref_mass_stddev = np.std(mass_dict[reference_label])
    ref_sigma = sigma_dict[reference_label]
    
    print(f"\nReference: {reference_label}")
    print(f"  Mass (median) = {ref_mass:.1f}")
    print(f"  Mass (median) standard deviation = {ref_mass_stddev:.2f}")
    print(f"  σ = {ref_sigma*1000:.3f} nm")
    
    print(f"\nMass Scaling Predictions:")
    for label, mass_med in zip(labels, medians):
        current_mass_stddev = np.std(mass_dict[label])
        if label == reference_label:
            continue
        predicted_sigma = ref_sigma * np.sqrt(ref_mass / mass_med)
        actual_sigma = sigma_dict[label]
        ratio = actual_sigma / predicted_sigma
        
        print(f"\n{label}:")
        print(f"  Mass (median) = {mass_med:.1f}")
        print(f"  Mass (median) standard deviation = {current_mass_stddev:.1f}")
        print(f"  σ (predicted from mass) = {predicted_sigma*1000:.3f} nm")
        print(f"  σ (actual/measured) = {actual_sigma*1000:.3f} nm")
        print(f"  Ratio (actual/predicted) = {ratio:.3f}")
        
        if ratio > 1.5:
            print(f"  ⚠️  OVER-CORRECTION: Actual sigma is {ratio:.1f}x higher than predicted")
        elif ratio < 0.7:
            print(f"  ⚠️  UNDER-CORRECTION: Actual sigma is {1/ratio:.1f}x lower than predicted")
        else:
            print(f"  ✓ Mass scaling prediction reasonable")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    inv_sqrt_mass = [1/np.sqrt(m) for m in medians]
    colors = ['red' if l == reference_label else 'blue' if 'Cy3' in l else 'green' for l in labels]
    
    ax.scatter(inv_sqrt_mass, [s*1000 for s in sigmas], c=colors, s=100, alpha=0.7)
    for i, label in enumerate(labels):
        ax.annotate(label, (inv_sqrt_mass[i], sigmas[i]*1000), fontsize=9, ha='right')
    
    # Fit line through all points
    slope, intercept, r_value, _, _ = stats.linregress(inv_sqrt_mass, [s*1000 for s in sigmas])
    x_fit = np.linspace(min(inv_sqrt_mass), max(inv_sqrt_mass), 100)
    ax.plot(x_fit, slope*x_fit + intercept, 'k--', alpha=0.5, label=f'Linear fit (R²={r_value**2:.3f})')
    
    ax.set_xlabel('1/√(Mass) [a.u.]', fontsize=12)
    ax.set_ylabel('σ [nm]', fontsize=12)
    ax.set_title('Mass Scaling Validation: σ vs 1/√Mass', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig

# ============================================================================
# VALIDATION TEST 3: CORRECTED MSD CONSISTENCY CHECK
# ============================================================================
def validate_corrected_msd_consistency(msd_yoyo1, msd_cy3, sigma_yoyo1, sigma_cy3, fps):
    """
    After correction, check if YOYO-1 and Cy3 give similar diffusion coefficients.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION 3: Corrected MSD Consistency (Same DNA should give same D)")
    print(f"{'='*80}")
    
    # Apply corrections
    msd_yoyo1_corrected = msd_yoyo1 - 4 * sigma_yoyo1**2
    msd_cy3_corrected = msd_cy3 - 4 * sigma_cy3**2
    
    # Check for negative values
    negative_yoyo1 = (msd_yoyo1_corrected < 0).sum()
    negative_cy3 = (msd_cy3_corrected < 0).sum()
    
    if negative_yoyo1 > 0:
        print(f"⚠️  ERROR: {negative_yoyo1} negative MSD values after YOYO-1 correction!")
        print(f"   This indicates OVER-CORRECTION of localization error")
    if negative_cy3 > 0:
        print(f"⚠️  ERROR: {negative_cy3} negative MSD values after Cy3 correction!")
    
    # Calculate diffusion coefficients from slope (fit 5-20 frames)
    time_range = slice(5/fps, 20/fps)
    
    def calculate_D(msd_series, time_range):
        msd_fit = msd_series.loc[msd_series.index.to_series().between(*[time_range.start, time_range.stop])]
        if len(msd_fit) < 3:
            return np.nan
        slope, _, _, _, _ = stats.linregress(msd_fit.index, msd_fit.values)
        return slope / 4  # MSD = 4*D*t
    
    D_yoyo1_uncorrected = calculate_D(msd_yoyo1, time_range)
    D_yoyo1_corrected = calculate_D(msd_yoyo1_corrected, time_range)
    D_cy3_uncorrected = calculate_D(msd_cy3, time_range)
    D_cy3_corrected = calculate_D(msd_cy3_corrected, time_range)
    
    print(f"\nDiffusion Coefficients (fitted from slope between 0.5-2.0s):")
    print(f"\nYOYO-1:")
    print(f"  D (uncorrected) = {D_yoyo1_uncorrected:.6f} μm²/s")
    print(f"  D (corrected)   = {D_yoyo1_corrected:.6f} μm²/s")
    print(f"\nCy3:")
    print(f"  D (uncorrected) = {D_cy3_uncorrected:.6f} μm²/s")
    print(f"  D (corrected)   = {D_cy3_corrected:.6f} μm²/s")
    
    ratio_corrected = D_yoyo1_corrected / D_cy3_corrected if D_cy3_corrected != 0 else np.nan
    print(f"\nRatio D_YOYO1 / D_Cy3 (corrected) = {ratio_corrected:.3f}")
    
    if 0.8 <= ratio_corrected <= 1.2:
        print(f"✓ Diffusion coefficients match well (within 20%)")
    elif ratio_corrected < 0.8:
        print(f"⚠️  YOYO-1 appears slower: possible OVER-CORRECTION")
    elif ratio_corrected > 1.2:
        print(f"⚠️  YOYO-1 appears faster: possible UNDER-CORRECTION")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Uncorrected
    ax1.plot(msd_yoyo1.index, msd_yoyo1.values*1e6, 'o-', label='YOYO-1', alpha=0.7)
    ax1.plot(msd_cy3.index, msd_cy3.values*1e6, 's-', label='Cy3', alpha=0.7)
    ax1.axhline(4*sigma_yoyo1**2*1e6, color='green', linestyle='--', alpha=0.5, label=f'4σ² YOYO-1')
    ax1.axhline(4*sigma_cy3**2*1e6, color='orange', linestyle='--', alpha=0.5, label=f'4σ² Cy3')
    ax1.set_xlabel('Time lag [s]', fontsize=12)
    ax1.set_ylabel('MSD [nm²]', fontsize=12)
    ax1.set_title('Uncorrected MSD', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Corrected
    ax2.plot(msd_yoyo1_corrected.index, msd_yoyo1_corrected.values*1e6, 'o-', label='YOYO-1 (corrected)', alpha=0.7)
    ax2.plot(msd_cy3_corrected.index, msd_cy3_corrected.values*1e6, 's-', label='Cy3 (corrected)', alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time lag [s]', fontsize=12)
    ax2.set_ylabel('MSD [nm²]', fontsize=12)
    ax2.set_title('Corrected MSD (should overlap)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ratio_corrected, negative_yoyo1

# ============================================================================
# VALIDATION TEST 4: SHORT-TIME MSD PLATEAU CHECK
# ============================================================================
def validate_plateau_check(msd_series, label, fps, max_frames=10):
    """
    Check if MSD plateaus at short times (indicator of pure noise).
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION 4: Short-Time MSD Slope Analysis - {label}")
    print(f"{'='*80}")
    
    time_lags = np.arange(1, max_frames + 1) / fps
    msd_short = msd_series.loc[msd_series.index.intersection(time_lags)]
    
    # Calculate frame-to-frame differences
    msd_diffs = msd_short.diff().dropna()
    mean_diff = msd_diffs.mean()
    std_diff = msd_diffs.std()
    cv = std_diff / mean_diff if mean_diff != 0 else np.nan
    
    print(f"MSD increments (frame-to-frame changes):")
    print(f"  Mean = {mean_diff*1e6:.3f} nm²")
    print(f"  Std  = {std_diff*1e6:.3f} nm²")
    print(f"  CV (std/mean) = {cv:.3f}")
    
    if cv < 0.3:
        print(f"✓ MSD increments are relatively constant (CV < 0.3)")
        print(f"  Suggests noise-dominated regime")
    else:
        print(f"⚠️  MSD increments vary significantly (CV = {cv:.3f})")
        print(f"  Suggests real diffusion contributing to short-time MSD")
    
    # Statistical test: is the MSD increasing significantly?
    slope, _, _, p_value, _ = stats.linregress(msd_short.index, msd_short.values)
    print(f"\nLinear trend test:")
    print(f"  Slope = {slope:.8f} μm²/s")
    print(f"  p-value = {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ⚠️  Significant positive slope detected (p < 0.05)")
        print(f"     MSD is NOT flat - contains real motion")
    else:
        print(f"  ✓ No significant slope (p >= 0.05)")
        print(f"     MSD plateau suggests noise dominance")
    
    return cv, p_value

# ============================================================================
# MAIN VALIDATION SUITE
# ============================================================================
def run_full_validation():
    """Run all validation tests."""
    
    print("\n" + "="*80)
    print("LOCALIZATION ERROR VALIDATION SUITE")
    print("="*80 + "\n")
    
    # Load all datasets
    print("Loading datasets...")
    msd_yoyo1, mass_yoyo1 = calculate_ensemble_msd_and_mass(YOYO1_DIM_PATH, MPP, FPS, MIN_TRACK_LENGTH)
    msd_cy3, mass_cy3 = calculate_ensemble_msd_and_mass(CY3_BRIGHT_PATH, MPP, FPS, MIN_TRACK_LENGTH)
    
    # Optional: Load beads if available
    msd_beads, mass_beads = None, None
    if os.path.exists(BEADS_PATH):
        msd_beads, mass_beads = calculate_ensemble_msd_and_mass(BEADS_PATH, MPP, FPS, MIN_TRACK_LENGTH)
    
    # Store results
    results = {}
    
    # TEST 1: MSD Intercept Analysis
    sigma_yoyo1_intercept, D_yoyo1 = validate_msd_intercept(msd_yoyo1, "YOYO-1", FPS)
    sigma_cy3_intercept, D_cy3 = validate_msd_intercept(msd_cy3, "Cy3", FPS)
    if msd_beads is not None:
        sigma_beads_intercept, D_beads = validate_msd_intercept(msd_beads, "Beads", FPS)
        results['beads_intercept_sigma'] = sigma_beads_intercept
    
    results['yoyo1_intercept_sigma'] = sigma_yoyo1_intercept
    results['cy3_intercept_sigma'] = sigma_cy3_intercept
    
    # TEST 2: Mass Scaling Validation
    mass_dict = {'YOYO-1': mass_yoyo1, 'Cy3': mass_cy3}
    sigma_dict = {'YOYO-1': SIGMA_YOYO1_CURRENT, 'Cy3': SIGMA_CY3_CURRENT}
    
    if msd_beads is not None and mass_beads:
        mass_dict['Beads'] = mass_beads
        sigma_dict['Beads'] = SIGMA_BEADS_REF
        fig_mass = validate_mass_scaling(mass_dict, sigma_dict, reference_label='Beads')
    else:
        fig_mass = validate_mass_scaling(mass_dict, sigma_dict, reference_label='Cy3')
    
    # TEST 3: Corrected MSD Consistency
    fig_consistency, d_ratio, neg_count = validate_corrected_msd_consistency(
        msd_yoyo1, msd_cy3, SIGMA_YOYO1_CURRENT, SIGMA_CY3_CURRENT, FPS
    )
    results['d_ratio'] = d_ratio
    results['negative_msd_count'] = neg_count
    
    # TEST 4: Plateau Check
    cv_yoyo1, p_yoyo1 = validate_plateau_check(msd_yoyo1, "YOYO-1", FPS)
    cv_cy3, p_cy3 = validate_plateau_check(msd_cy3, "Cy3", FPS)
    results['yoyo1_cv'] = cv_yoyo1
    results['yoyo1_p_value'] = p_yoyo1
    
    # SUMMARY
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n1. INTERCEPT METHOD SIGMAS:")
    print(f"   YOYO-1: {sigma_yoyo1_intercept*1000 if sigma_yoyo1_intercept else 'N/A':.3f} nm")
    print(f"   Cy3:    {sigma_cy3_intercept*1000 if sigma_cy3_intercept else 'N/A':.3f} nm")
    print(f"   (Compare to your current: YOYO-1={SIGMA_YOYO1_CURRENT*1000:.3f} nm, Cy3={SIGMA_CY3_CURRENT*1000:.3f} nm)")
    
    print(f"\n2. MASS SCALING:")
    print(f"   See plot for deviations from theoretical σ ∝ 1/√Mass")
    
    print(f"\n3. CONSISTENCY:")
    print(f"   D_YOYO1 / D_Cy3 = {d_ratio:.3f}")
    print(f"   Negative MSD values after correction: {neg_count}")
    if neg_count > 0:
        print(f"   ⚠️  CRITICAL: Negative MSD indicates OVER-CORRECTION")
    
    print(f"\n4. SHORT-TIME DYNAMICS:")
    print(f"   YOYO-1 CV={cv_yoyo1:.3f}, p={p_yoyo1:.4f}")
    if p_yoyo1 < 0.05 or cv_yoyo1 > 0.3:
        print(f"   ⚠️  YOYO-1 short-time MSD includes real motion")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    
    if neg_count > 0:
        print("❌ Your correction is OVER-CORRECTED (negative MSD values)")
        print("   → Use the INTERCEPT METHOD sigma instead")
        print(f"   → Suggested: σ_YOYO1 = {sigma_yoyo1_intercept*1000:.3f} nm (from intercept)")
    elif d_ratio < 0.8:
        print("⚠️  YOYO-1 appears too slow after correction")
        print("   → Reduce σ_YOYO1 correction value")
    elif d_ratio > 1.2:
        print("⚠️  YOYO-1 appears too fast after correction")
        print("   → Increase σ_YOYO1 correction value")
    else:
        print("✓ Correction appears reasonable based on D ratio")
        print("  But verify with other tests above")
    
    plt.show()
    
    return results

if __name__ == "__main__":
    results = run_full_validation()