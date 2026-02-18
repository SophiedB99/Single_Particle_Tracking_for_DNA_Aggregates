 # -*- coding: utf-8 -*-
"""
Step 1: Calculate Localization Error (Sigma) from Stuck Beads
This gives you the reference sigma to use for MSD correction

@author: sophiedeboer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your stuck beads data (immobilized particles)
STUCK_BEADS_PATH = 'F:/A_Data/A_DNATranslocation/Experiments_2025/DNA Tracking/Corrected_Data/Control Experiments_Beads/0.22 um/E_output_data/Collected_0.22um/tm'

# Conversion factors
MPP = 0.102  # Microns per pixel (UPDATE if different for stuck beads)
FPS = 10    # Frames per second (UPDATE if different for stuck beads)

# ============================================================================
# FUNCTIONS
# ============================================================================

def calculate_localization_error_per_trajectory(positions, mpp):
    """
    Calculate localization error (sigma) from position fluctuations of a stuck bead.
    
    For a truly immobile particle, all position fluctuations are due to localization error.
    In 2D: <Δx²> + <Δy²> = 2σ² (for each dimension independently)
    
    Parameters
    ----------
    positions : DataFrame with 'x' and 'y' columns in pixels
    mpp : microns per pixel conversion factor
    
    Returns
    -------
    sigma_x, sigma_y, sigma_combined : localization uncertainties in microns
    """
    # Convert to microns
    x = positions['x'].values * mpp
    y = positions['y'].values * mpp
    
    # Remove mean (center the trajectory)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    # Calculate standard deviation (this IS the localization error)
    sigma_x = np.std(x_centered, ddof=1)
    sigma_y = np.std(y_centered, ddof=1)
    
    # Combined 2D localization error
    sigma_combined = np.sqrt((sigma_x**2 + sigma_y**2) / 2)
    
    return sigma_x, sigma_y, sigma_combined

def determine_dim_sigma(raw_msd_dim, corrected_msd_bright, fps, max_lagtime_frames):
    """
    Calculates sigma for a dim condition by comparing its raw MSD 
    to the corrected MSD (true motion) of a physically identical bright condition.
    """
    if raw_msd_dim is None or corrected_msd_bright is None:
        return np.nan

    # 1. Align data based on time index
    merged_msd = pd.merge(
        raw_msd_dim.rename('raw_dim'), 
        corrected_msd_bright.rename('corrected_bright'),
        left_index=True, right_index=True, how='inner'
    )

    # 2. Slice to the short-time regime (e.g., first 10 frames)
    comparison_slice = merged_msd.iloc[0:max_lagtime_frames]

    # 3. Calculate Noise Component: 4*sigma^2 = MSD_raw_dim - MSD_true_bright
    noise_component = comparison_slice['raw_dim'] - comparison_slice['corrected_bright']
    
    # Filter for physical results (Noise must be positive)
    positive_noise = noise_component[noise_component > 0]
    
    if positive_noise.empty:
        return np.nan

    # 4. Average the noise component
    mean_noise_4sigma2 = positive_noise.mean()

    # 5. Calculate sigma: sigma = sqrt( (4*sigma^2) / 4 )
    sigma_dim = np.sqrt(mean_noise_4sigma2 / 4.0)
    
    return sigma_dim

def calculate_localization_error_msd_method(positions, mpp, fps, max_lagtime=10):
    """
    Alternative method: Calculate sigma from MSD of stuck beads.
    
    For stuck beads: MSD(Δt) ≈ 4σ² (constant, independent of lag time)
    
    Parameters
    ----------
    positions : DataFrame with 'x' and 'y' columns
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : number of lag times to average over
    
    Returns
    -------
    sigma : localization uncertainty in microns
    """
    # Reindex to handle missing frames
    pos = positions.set_index('frame')[['x', 'y']]
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy() * mpp
    
    # Calculate MSD for short lag times
    msds = []
    for lag in range(1, min(max_lagtime + 1, len(pos))):
        displacements = pos[lag:] - pos[:-lag]
        msd = np.nanmean(displacements**2)
        msds.append(msd)
    
    # For stuck beads, MSD should be constant ≈ 4σ²
    mean_msd = np.nanmean(msds)
    sigma = np.sqrt(mean_msd / 4)
    
    return sigma

def analyze_stuck_beads(directory, mpp, fps, min_track_length=50):
    """
    Analyze all stuck bead trajectories to determine localization error.
    
    Returns
    -------
    results_df : DataFrame with sigma values for each trajectory
    """
    results = []
    
    print(f"Analyzing stuck beads in: {directory}\n")
    
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return None
    
    file_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and not filename.startswith('.'):
            filepath = os.path.join(directory, filename)
            
            try:
                traj_df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                continue
            
            file_count += 1
            
            for pid, ptraj in traj_df.groupby('particle'):
                # Filter short trajectories
                if len(ptraj) < min_track_length:
                    continue
                
                # Method 1: Direct variance method
                sigma_x, sigma_y, sigma_combined = calculate_localization_error_per_trajectory(
                    ptraj, mpp)
                
                # Method 2: MSD method
                sigma_msd = calculate_localization_error_msd_method(ptraj, mpp, fps)
                
                # Get particle properties
                mass_mean = ptraj['mass'].mean() if 'mass' in ptraj.columns else np.nan
                ep_mean = ptraj['ep'].mean() if 'ep' in ptraj.columns else np.nan
                
                results.append({
                    'file': filename,
                    'particle': pid,
                    'n_frames': len(ptraj),
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y,
                    'sigma_combined': sigma_combined,
                    'sigma_msd': sigma_msd,
                    'mass': mass_mean,
                    'ep': ep_mean
                })
    
    if not results:
        print("WARNING: No valid stuck bead trajectories found!")
        return None
    
    results_df = pd.DataFrame(results)
    
    print(f"Analyzed {file_count} files")
    print(f"Found {len(results_df)} stuck bead trajectories (length ≥ {min_track_length} frames)")
    print("\n" + "="*80)
    print("LOCALIZATION ERROR STATISTICS")
    print("="*80)
    print(f"Mean sigma_x:        {results_df['sigma_x'].mean():.6f} ± {results_df['sigma_x'].std():.6f} μm")
    print(f"Mean sigma_y:        {results_df['sigma_y'].mean():.6f} ± {results_df['sigma_y'].std():.6f} μm")
    print(f"Mean sigma_combined: {results_df['sigma_combined'].mean():.6f} ± {results_df['sigma_combined'].std():.6f} μm")
    print(f"Mean sigma_msd:      {results_df['sigma_msd'].mean():.6f} ± {results_df['sigma_msd'].std():.6f} μm")
    print(f"\nMedian sigma_combined: {results_df['sigma_combined'].median():.6f} μm")
    print(f"Median sigma_msd:      {results_df['sigma_msd'].median():.6f} μm")
    print(f"\nMean mass:  {results_df['mass'].mean():.1f}")
    print(f"Mean EP:    {results_df['ep'].mean():.6f}")
    print("="*80)
    
    # Recommended value for MSD correction
    sigma_recommended = results_df['sigma_combined'].median()
    correction_value = 4 * sigma_recommended**2
    
    print(f"\n📌 RECOMMENDED VALUES FOR MSD CORRECTION:")
    print(f"   SIGMA_STUCK = {sigma_recommended:.6f}  # microns")
    print(f"   MSD correction = 4 × σ² = {correction_value:.8f} μm²")
    print("="*80 + "\n")
    
    return results_df

def plot_localization_error_diagnostics(results_df, output_dir):
    """Create diagnostic plots for localization error analysis"""
    
    if results_df is None or len(results_df) == 0:
        print("No data to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Sigma distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sigma_x histogram
    ax1 = axes[0, 0]
    ax1.hist(results_df['sigma_x'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(results_df['sigma_x'].median(), color='red', linestyle='--', linewidth=2, 
                label=f'Median: {results_df["sigma_x"].median():.4f} μm')
    ax1.set_xlabel('σ_x (μm)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('X-direction Localization Error', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Sigma_y histogram
    ax2 = axes[0, 1]
    ax2.hist(results_df['sigma_y'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(results_df['sigma_y'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {results_df["sigma_y"].median():.4f} μm')
    ax2.set_xlabel('σ_y (μm)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Y-direction Localization Error', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Combined sigma histogram
    ax3 = axes[1, 0]
    ax3.hist(results_df['sigma_combined'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(results_df['sigma_combined'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {results_df["sigma_combined"].median():.4f} μm')
    ax3.set_xlabel('σ_combined (μm)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Combined 2D Localization Error', fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    

    ax4 = axes[1, 1]
    ax4.scatter(results_df['mass'], results_df['sigma_combined'], alpha=0.6, s=50, edgecolor='black')
    ax4.set_xlabel('Mass (brightness)', fontsize=12)
    ax4.set_ylabel(r'$\sigma_{\text{combined}}$ ($\mu\mathrm{m}$)', fontsize=12)
    ax4.set_title('Localization Error vs Brightness (Theoretical Fit)', fontsize=14)
    ax4.grid(alpha=0.3)
    
    # --- Corrected Fitting Logic ---
    if results_df['mass'].notna().sum() > 5:
        # Get reference values from the stuck beads data
        sigma_ref = results_df['sigma_combined'].median()
        mass_ref = results_df['mass'].median()
        
        # 1. Define the theoretical scaling curve
        def power_law_sigma(mass, A):
            return A / np.sqrt(mass)
            
        # 2. Calculate the expected constant A
        A_expected = sigma_ref * np.sqrt(mass_ref)
        
        # 3. Define the x-range for the fit line
        x_trend = np.linspace(results_df['mass'].min(), results_df['mass'].max(), 100)
        x_trend = x_trend[x_trend > 0] 
        
        # 4. Calculate the theoretical y-values
        y_trend = power_law_sigma(x_trend, A_expected)
        
        # Plot the theoretical curve (red line)
        ax4.plot(x_trend, y_trend, color='red', linestyle='--', linewidth=2, 
                 label=r'$\sigma = \sigma_{\text{ref}} \sqrt{M_{\text{ref}}/M}$')
        ax4.legend(fontsize=10)

 
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Localization_Error_Diagnostics.jpg'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Individual trajectory examples
    print("\nGenerating example trajectory plots...")

def plot_example_trajectories(directory, mpp, n_examples=4, output_dir='./'):
    """Plot example stuck bead trajectories to visualize localization error"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    count = 0
    for filename in os.listdir(directory):
        if count >= n_examples:
            break
        
        if filename.endswith('.csv') and not filename.startswith('.'):
            filepath = os.path.join(directory, filename)
            
            try:
                traj_df = pd.read_csv(filepath)
            except:
                continue
            
            for pid, ptraj in traj_df.groupby('particle'):
                if len(ptraj) < 50:
                    continue
                
                # Calculate sigma for this trajectory
                sigma_x, sigma_y, sigma_combined = calculate_localization_error_per_trajectory(
                    ptraj, mpp)
                
                # Plot trajectory
                x = (ptraj['x'].values - ptraj['x'].mean()) * mpp * 1000  # Convert to nm
                y = (ptraj['y'].values - ptraj['y'].mean()) * mpp * 1000
                
                axes[count].plot(x, y, 'o-', alpha=0.5, markersize=3)
                axes[count].set_xlabel('Δx (nm)', fontsize=11)
                axes[count].set_ylabel('Δy (nm)', fontsize=11)
                axes[count].set_title(f'Bead {count+1}: σ = {sigma_combined*1000:.1f} nm', fontsize=12)
                axes[count].grid(alpha=0.3)
                axes[count].axis('equal')
                
                # Add circle showing 2σ radius
                circle = plt.Circle((0, 0), 2*sigma_combined*1000, color='red', 
                                   fill=False, linestyle='--', linewidth=2, 
                                   label=f'2σ = {2*sigma_combined*1000:.1f} nm')
                axes[count].add_patch(circle)
                axes[count].legend()
                
                count += 1
                if count >= n_examples:
                    break
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Example_Stuck_Bead_Trajectories.jpg'), 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("LOCALIZATION ERROR CALCULATION FROM STUCK BEADS")
    print("="*80 + "\n")
    
    # Analyze stuck beads
    results_df = analyze_stuck_beads(
        directory=STUCK_BEADS_PATH,
        mpp=MPP,
        fps=FPS,
        min_track_length=50  # Minimum trajectory length to include
    )
    
    if results_df is not None and len(results_df) > 0:
        # Save results to CSV
        output_dir = os.path.dirname(STUCK_BEADS_PATH)
        output_path = os.path.join(output_dir, 'localization_error_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}\n")
        
        # Generate diagnostic plots
        plot_localization_error_diagnostics(results_df, output_dir)
        plot_example_trajectories(STUCK_BEADS_PATH, MPP, n_examples=4, 
                                  output_dir=output_dir)
        
        # Extract recommended values
        SIGMA_STUCK = results_df['sigma_combined'].median()
        MASS_STUCK = results_df['mass'].median()
        MASS_STUCK_STDDEV = results_df['mass'].std()
        
        print("\n" + "="*80)
        print("📋 COPY THESE VALUES TO YOUR MSD ANALYSIS CODE:")
        print("="*80)
        print(f"SIGMA_STUCK_MICRONS = {SIGMA_STUCK:.6f}  # microns")
        print(f"MASS_STUCK_REFERENCE = {MASS_STUCK:.1f}  # median mass")
        print(f"Standard Deviation of MASS_STUCK_REFERENCE = {MASS_STUCK_STDDEV:.2f}  # standard deviation")
        print(f"\nMSD Correction = 4 × σ² = {4*SIGMA_STUCK**2:.8f} μm²")
        print("="*80 + "\n")
        
        # Also save to a text file for easy reference
        summary_path = os.path.join(output_dir, 'SIGMA_VALUES.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LOCALIZATION ERROR REFERENCE VALUES\n")
            f.write("="*80 + "\n\n")
            f.write(f"SIGMA_STUCK_MICRONS = {SIGMA_STUCK:.6f}  # microns\n")
            f.write(f"MASS_STUCK_REFERENCE = {MASS_STUCK:.1f}  # median mass\n\n")
            f.write(f"MSD Correction = 4 × σ² = {4*SIGMA_STUCK**2:.8f} μm²\n\n")
            f.write("="*80 + "\n")
            f.write("STATISTICS:\n")
            f.write("="*80 + "\n")
            f.write(f"Number of beads analyzed: {len(results_df)}\n")
            f.write(f"Mean sigma_x: {results_df['sigma_x'].mean():.6f} ± {results_df['sigma_x'].std():.6f} μm\n")
            f.write(f"Mean sigma_y: {results_df['sigma_y'].mean():.6f} ± {results_df['sigma_y'].std():.6f} μm\n")
            f.write(f"Median sigma_combined: {SIGMA_STUCK:.6f} μm\n")
            f.write(f"Median mass: {MASS_STUCK:.1f}\n")
            f.write(f"Mean EP: {results_df['ep'].mean():.6f}\n")
        
        print(f"Summary saved to: {summary_path}")
    else:
        print("\n❌ ERROR: No stuck bead data could be analyzed!")
        print("Please check:")
        print(f"  1. Path exists: {STUCK_BEADS_PATH}")
        print(f"  2. CSV files contain 'x', 'y', 'frame', 'particle' columns")
        print(f"  3. Trajectories are at least 50 frames long")