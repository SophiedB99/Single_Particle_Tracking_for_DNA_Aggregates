# -*- coding: utf-8 -*-
"""
Step 1: Calculate Localization Error (Sigma) - CORRECTED VERSION
FIXED: Data assignment and output logic swapped to reflect that Cy3 is BRIGHTER than YOYO-1.
@author: sophiedeboer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
# ============================================================================
# CONFIGURATION - CORRECTED VARIABLE ROLES
# ============================================================================
# Path for the Cy3 data (New BRIGHTER Reference)
CY3_BRIGHT_PATH = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/MDA-MB-231/Cy3 MDA-MB-231/500 bp_most_recent/E_output_data/Collected_Cy3_MDAMB231/tm'
# Path for the YOYO-1 data (New DIMMER Target for Sigma Calculation)
YOYO1_DIM_PATH = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Aswin Data/Aswin Data MDA-MB-231 Corrected for Drift/E_output_data/Collected_MDAMB231/tm'
# Conversion factors
MPP = 0.102
FPS = 10
# Analysis parameters
MIN_TRACK_LENGTH = 50
MAX_LAGTIME_FRAMES = 10 # Number of short time lags to average over
# ============================================================================
# FUNCTIONS
# ============================================================================
def determine_dim_sigma(raw_msd_dim, corrected_msd_bright, fps, max_lagtime_frames):
    """
    Calculates sigma for a dim condition by comparing its raw MSD to the 
    corrected MSD of a physically identical bright condition.

    Theory: 4σ²_dim ≈ MSD_raw_dim - MSD_raw_bright
    (Assuming MSD_raw_bright ≈ MSD_true since σ_bright is very small)
    """
    if raw_msd_dim is None or corrected_msd_bright is None:
        return np.nan

    # Align data
    merged_msd = pd.merge(
        raw_msd_dim.rename('raw_dim'),
        corrected_msd_bright.rename('raw_bright'),
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Slice to short-time regime
    time_lags_seconds = np.arange(1, max_lagtime_frames + 1) / fps
    comparison_slice = merged_msd.loc[merged_msd.index.intersection(time_lags_seconds)]

    if comparison_slice.empty:
        print(f"Warning: No overlapping MSD data in first {max_lagtime_frames} frames.")
        return np.nan

    # Calculate noise difference: MSD_dim - MSD_bright
    noise_component = comparison_slice['raw_dim'] - comparison_slice['raw_bright']

    # Check for physical validity
    if (noise_component < 0).any():
        n_negative = (noise_component < 0).sum()
        print(f"️ WARNING: {n_negative}/{len(noise_component)} time points show negative noise difference!")
        print(" This means the 'dim' target has *better* localization than the 'bright' reference at those points.")

    # Filter for positive values only (only positive difference contributes to σ_dim calculation)
    positive_noise = noise_component[noise_component > 0]

    if positive_noise.empty:
        print("  ERROR: All noise values are non-positive.")
        print("   Possible causes: Both conditions have similar localization or data is too noisy.")
        return np.nan

    # Average noise component (4 * sigma_dim^2)
    mean_noise_diff = positive_noise.mean()

    # Calculate sigma: σ_dim = sqrt(mean_noise_diff / 4)
    sigma_dim = np.sqrt(mean_noise_diff / 4.0)

    print(f"\n Noise analysis over {len(positive_noise)} valid time points:")
    print(f" Mean noise difference: {mean_noise_diff*1e6:.2f} nm²")
    print(f" Std dev: {positive_noise.std()*1e6:.2f} nm²")
    print(f" Range: [{positive_noise.min()*1e6:.2f}, {positive_noise.max()*1e6:.2f}] nm²")
    
    return sigma_dim

def calculate_ensemble_msd_and_mass(directory, mpp, fps, min_track_length=50):
    """Calculate ensemble MSD and collect mass values."""
    all_displacements = {}
    mass_values = []
    max_lag_frames = 50
    
    print(f"Loading trajectories from: {os.path.basename(directory)}")
    
    if not os.path.exists(directory):
        print(f" CRITICAL ERROR: Directory not found: {directory}")
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
                
                # Extract mass
                if 'mass' in ptraj.columns:
                    particle_mass = ptraj['mass'].mean()
                    mass_values.append(particle_mass)
                
                # Prepare positions
                ptraj = ptraj.set_index('frame').sort_index()
                x_microns = ptraj['x'].values * mpp
                y_microns = ptraj['y'].values * mpp
                
                # Calculate MSD
                for lag_frames in range(1, max_lag_frames + 1):
                    if len(ptraj) > lag_frames:
                        dx = x_microns[lag_frames:] - x_microns[:-lag_frames]
                        dy = y_microns[lag_frames:] - y_microns[:-lag_frames]
                        r2 = dx**2 + dy**2
                        
                        if lag_frames not in all_displacements:
                            all_displacements[lag_frames] = []
                        all_displacements[lag_frames].extend(r2.tolist())

    if not all_displacements:
        print("️ WARNING: No valid tracks found.")
        return None, None

    msd_data = {
        lag_frames / fps: np.mean(displacements) 
        for lag_frames, displacements in all_displacements.items()
    }
    
    print(f"  Analyzed {file_count} files, {total_tracks} tracks")
    if mass_values:
        print(f"  Collected {len(mass_values)} mass values (median: {np.median(mass_values):.0f})")
    else:
        print(" ️ No mass values found")

    msd_series = pd.Series(msd_data).sort_index()
    msd_series.index.name = 'dt' # Corrected index name assignment
    return msd_series, mass_values

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # --- Data Assignment based on User's Correction ---
    # Target: YOYO-1 (Dimmer)
    YOYO1_MSD_PATH = YOYO1_DIM_PATH
    YOYO1_LABEL = "YOYO-1 (Dim Target)"
    # Reference: Cy3 (Brighter)
    CY3_MSD_PATH = CY3_BRIGHT_PATH
    CY3_LABEL = "Cy3 (Bright Ref)"
    
    print("\n" + "="*80)
    print("COMPARATIVE LOCALIZATION ERROR (SIGMA) CALCULATION - LOGIC REVERSED")
    print("Calculating σ_YOYO-1 based on MSD_YOYO-1 (Raw) vs MSD_Cy3 (Reference)")
    print("="*80 + "\n")

    # Step 1: Analyze YOYO-1 (the new DIM/Target condition)
    print(f" Step 1: Analyzing DIM Target condition ({YOYO1_LABEL})...")
    raw_msd_yoyo1, mass_values_yoyo1 = calculate_ensemble_msd_and_mass(
        directory=YOYO1_MSD_PATH, mpp=MPP, fps=FPS, min_track_length=MIN_TRACK_LENGTH
    )

    # Step 2: Analyze Cy3 (the new BRIGHT/Reference condition)
    print(f"\n Step 2: Analyzing BRIGHT Reference condition ({CY3_LABEL})...")
    raw_msd_cy3, mass_values_cy3 = calculate_ensemble_msd_and_mass(
        directory=CY3_MSD_PATH, mpp=MPP, fps=FPS, min_track_length=MIN_TRACK_LENGTH
    )

    # --- Mass Statistics ---
    print("\n" + "="*80)
    print("MASS STATISTICS")
    print("="*80)
    
    mass_yoyo1_median = np.nan
    mass_yoyo1_mean = np.nan
    mass_cy3_median = np.nan
    mass_cy3_mean = np.nan
    
    if mass_values_yoyo1:
        mass_yoyo1_median = np.median(mass_values_yoyo1)
        mass_yoyo1_mean = np.mean(mass_values_yoyo1)
        print(f"\n DIM Target ({YOYO1_LABEL}):")
        print(f" Median mass: {mass_yoyo1_median:.1f}")
        print(f" Mean mass: {mass_yoyo1_mean:.1f}")
        print(f" Std dev: {np.std(mass_values_yoyo1):.1f}")
        print(f" N = {len(mass_values_yoyo1)} particles")
    else:
        print(f"\n️ No mass data for DIM Target ({YOYO1_LABEL})")

    if mass_values_cy3:
        mass_cy3_median = np.median(mass_values_cy3)
        mass_cy3_mean = np.mean(mass_values_cy3)
        print(f"\n BRIGHT Reference ({CY3_LABEL}):")
        print(f" Median mass: {mass_cy3_median:.1f}")
        print(f" Mean mass: {mass_cy3_mean:.1f}")
        print(f" Std dev: {np.std(mass_values_cy3):.1f}")
        print(f" N = {len(mass_values_cy3)} particles")

        if not np.isnan(mass_yoyo1_median):
            brightness_ratio = mass_cy3_median / mass_yoyo1_median
            print(f"\n Brightness ratio (Cy3/YOYO-1): {brightness_ratio:.2f}x")
            if brightness_ratio < 2:
                print(" ️ WARNING: Brightness difference is small (<2x)!")
                print(" Localization difference might not be significant.")
    else:
        print(f"\n️ No mass data for BRIGHT Reference ({CY3_LABEL})")

    # --- Calculate sigma ---
    if raw_msd_yoyo1 is not None and raw_msd_cy3 is not None:
        print("\n" + "="*80)
        print(" Step 3: Calculating σ_YOYO-1 from MSD difference...")
        print("="*80)
        
        # NOTE: raw_msd_yoyo1 is the 'dim' data (higher noise)
        # raw_msd_cy3 is the 'bright' data (lower noise reference)
        sigma_yoyo1 = determine_dim_sigma(
            raw_msd_dim=raw_msd_yoyo1, 
            corrected_msd_bright=raw_msd_cy3, 
            fps=FPS, 
            max_lagtime_frames=MAX_LAGTIME_FRAMES
        )

        if not np.isnan(sigma_yoyo1):
            sigma_yoyo1_nm = sigma_yoyo1 * 1000
            msd_correction_value = 4 * sigma_yoyo1**2
            
            print("\n" + "="*80)
            print(" RESULT: LOCALIZATION ERROR FOR DIM CONDITION (YOYO-1)")
            print("="*80)
            print(f"σ_YOYO-1 = {sigma_yoyo1_nm:.3f} nm (or {sigma_yoyo1:.6f} μm)")
            print(f"4σ² = {msd_correction_value:.8f} μm² (= {msd_correction_value*1e6:.2f} nm²)")
            print("="*80 + "\n")

            # --- Save results ---
            output_dir = os.path.dirname(os.path.dirname(CY3_MSD_PATH))
            summary_path = os.path.join(output_dir, 'SIGMA_AND_MASS_REFERENCE_VALUES_YOYO1_DIM.txt')
            
            with open(summary_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("COMPARATIVE LOCALIZATION ERROR AND MASS REFERENCE VALUES\n")
                f.write("(YOYO-1 [DIM TARGET] vs Cy3 [BRIGHT REF] for 500 bp DNA)\n")
                f.write("="*80 + "\n\n")
                f.write("# LOGIC REVERSED: Cy3 is brighter reference, YOYO-1 is dim target\n\n")

                f.write("# --- Sigma (Localization Error for YOYO-1/Dim particles) ---\n")
                f.write(f"SIGMA_YOYO1_MICRONS = {sigma_yoyo1:.6f} # microns\n")
                f.write(f"# MSD Correction Term: 4σ² = {msd_correction_value:.8f} μm²\n")
                f.write(f"# In nanometers: σ = {sigma_yoyo1_nm:.3f} nm, 4σ² = {msd_correction_value*1e6:.2f} nm²\n\n")

                f.write("# --- Mass Reference (use MEDIAN for robust scaling) ---\n")
                if not np.isnan(mass_yoyo1_median):
                    f.write(f"MASS_YOYO1_REFERENCE = {mass_yoyo1_median:.1f} # Median mass from YOYO-1 condition\n")
                    f.write(f"# Alternative (mean): {mass_yoyo1_mean:.1f}\n")
                    f.write(f"# Use this as your reference when scaling sigma for individual YOYO-1 particles\n\n")
                else:
                    f.write("MASS_YOYO1_REFERENCE =??? # WARNING: Mass data not available!\n\n")

                f.write("# --- Cy3 Reference Mass Statistics ---\n")
                if not np.isnan(mass_cy3_median):
                    f.write(f"# Cy3 median mass: {mass_cy3_median:.1f}\n")
                    f.write(f"# Cy3 mean mass: {mass_cy3_mean:.1f}\n")
                    if not np.isnan(mass_yoyo1_median):
                         f.write(f"# Brightness ratio (Cy3/YOYO-1): {mass_cy3_median/mass_yoyo1_median:.2f}x\n\n")
                else:
                    f.write("# WARNING: Mass data not available for Cy3 reference!\n\n")

                f.write("# --- Usage in Van Hove / MSD Analysis ---\n")
                f.write("# For per-particle sigma correction for YOYO-1:\n")
                f.write("# σ_particle = SIGMA_YOYO1_MICRONS * sqrt(MASS_YOYO1_REFERENCE / mass_particle)\n")
                f.write("# MSD_corrected = MSD_raw - 4 * σ_particle²\n")
                f.write("# Δr²_corrected = Δr²_raw - 4 * σ_particle²\n\n")
                
                f.write("# --- Analysis Parameters ---\n")
                f.write(f"# Averaged over first {MAX_LAGTIME_FRAMES} frames ({MAX_LAGTIME_FRAMES/FPS:.1f} s)\n")
                f.write(f"# Minimum track length: {MIN_TRACK_LENGTH} frames\n")
                f.write(f"# MPP: {MPP}, FPS: {FPS}\n")
                f.write("="*80 + "\n")
            
            print(f" Summary saved to: {summary_path}")

            # Diagnostic plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot 1: MSD comparison
            ax1 = axes[0]
            ax1.plot(raw_msd_cy3.index, raw_msd_cy3.values, 'o-', label=CY3_LABEL, color='red', markersize=4)
            ax1.plot(raw_msd_yoyo1.index, raw_msd_yoyo1.values, 's-', label=YOYO1_LABEL, color='blue', markersize=4)
            ax1.axhline(y=msd_correction_value, color='gray', linestyle='--', linewidth=2, label=f'4σ² = {msd_correction_value*1e6:.1f} nm²')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel('Time lag Δt (s)', fontsize=12)
            ax1.set_ylabel('MSD (μm²)', fontsize=12)
            ax1.set_title(f'MSD Comparison: {CY3_LABEL} vs {YOYO1_LABEL}', fontsize=13)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Mass distributions
            ax2 = axes[1]
            if mass_values_cy3:
                ax2.hist(mass_values_cy3, bins=30, alpha=0.6, label=f'{CY3_LABEL} (median={mass_cy3_median:.0f})', color='red', edgecolor='black')
            if mass_values_yoyo1:
                ax2.hist(mass_values_yoyo1, bins=30, alpha=0.6, label=f'{YOYO1_LABEL} (median={mass_yoyo1_median:.0f})', color='blue', edgecolor='black')
            ax2.set_xlabel('Particle Mass (brightness)', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Mass Distribution Comparison', fontsize=13)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Noise difference vs time lag
            ax3 = axes[2]
            merged_msd_plot = pd.merge(
                raw_msd_yoyo1.rename('dim'),
                raw_msd_cy3.rename('bright'),
                left_index=True,
                right_index=True,
                how='inner'
            )
            # Noise is plotted as: (Dim Target) - (Bright Ref)
            noise = merged_msd_plot['dim'] - merged_msd_plot['bright']
            
            ax3.plot(noise.index, noise.values * 1e6, 'o-', color='purple', markersize=4)
            ax3.axhline(y=msd_correction_value*1e6, color='red', linestyle='--', linewidth=2, label=f'Mean 4σ² = {msd_correction_value*1e6:.1f} nm²')
            ax3.axvline(x=MAX_LAGTIME_FRAMES/FPS, color='gray', linestyle=':', linewidth=2, label=f'Avg window = {MAX_LAGTIME_FRAMES/FPS:.1f}s')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax3.set_xlabel('Time lag Δt (s)', fontsize=12)
            ax3.set_ylabel('Noise Difference (nm²)', fontsize=12)
            ax3.set_title('MSD_YOYO1 - MSD_Cy3 (should be positive)', fontsize=13)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'SIGMA_MASS_DIAGNOSTIC_PLOTS_REVERSED_LOGIC.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f" Diagnostic plots saved to: {plot_path}")
            plt.show()

        else:
            print("\n ERROR: Sigma could not be determined.")
            print("Check the diagnostic output above for clues.")
    else:
        print("\n CRITICAL FAILURE: Could not calculate MSD for one or both conditions.")