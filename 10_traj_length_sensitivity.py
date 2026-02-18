import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import string
import warnings

# Suppress warnings related to performance and pandas indexing
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning, message='The behavior of obj<<LATEX_0>> with a float-dtype index is deprecated.')


# --- Utility Functions (Provided in Original Code) ---
def _msd_iter(pos, lagtimes):
    """Internal iterator for calculating squared displacements."""
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff_no_nan = np.nan_to_num(diff, nan=0)
        if not np.all(diff_no_nan == 0):
            yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))

def trajectory(pos, pid, max_lagtime, video, pos_columns):
    """Shift the starting point of the trajectory to (0,0)."""
    max_lagtime = len(pos)
    lagtimes = np.arange(0, max_lagtime)
    result = pd.DataFrame(pos - pos[0], columns=pos_columns, index=lagtimes)
    result['particle'] = pid
    result['video'] = video
    return result

def msd(pos, max_lagtime):
    """Compute the mean displacement and mean squared displacement of one trajectory."""
    max_lagtime = min(max_lagtime, len(pos) - 1)
    lagtimes = np.arange(1, max_lagtime + 1)
    pos_columns = ['x', 'y']
    result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                     ['<{}^2>'.format(p) for p in pos_columns]
    resultmsd = list(_msd_iter(pos, lagtimes))
    resultmsd = resultmsd[:len(lagtimes)]
    result = pd.DataFrame(resultmsd, columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(1)
    return result

def power_law(x, alpha, A):
    """General function for power law fitting: MSD = A * t^alpha."""
    return A * np.power(x, alpha)

def powerfit(msd, a, b, fps=10):
    """ Function for fitting power law to a specific index range of the MSD data.
    a and b are integer POSITIONAL indices of the data points to use for the fit.
    """
    y = msd['msd']
    # Convert index (lag frames) to time (seconds)
    x = msd.index.values.astype('float64') / float(fps)
    
    # Use positional slicing for y (Series) and x (Numpy array)
    try:
        y_data = y.iloc[a:b]
        x_data = x[a:b]
    except IndexError:
        raise ValueError("Fit range indices are out of bounds.")

    if len(x_data) < 2:
        raise ValueError("Insufficient data points for fitting.")

    pars, cov = curve_fit(f=power_law, xdata=x_data, ydata=y_data, p0=[0, 0], bounds=(-np.inf, np.inf))
    
    # Calculate R-squared
    residuals = y_data - power_law(x_data, *pars)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    
    if ss_tot == 0 or np.isnan(ss_tot):
        r_squared = 0.0
    else:
        r_squared = 1.0 - (ss_res / ss_tot)
        
    return pars, r_squared

def imsd(filename, mpp, fps, video, directory, max_lagtime=100):
    """ Function for computing the time averaged mean square displacement for all particles in a file. """
    statistic = 'msd'
    try:
        traj = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return [], [], [], [], []

    pos_columns = ['x', 'y']
    ids = []
    trajec = []
    msds = []
    alpha = []
    alpha2 = []
    length = []
    index = 0

    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
        
        # Calculate MSD for this trajectory (using up to 800 frames max)
        msdtemp = msd(pos[:800], max_lagtime)
        
        # Data Cleaning
        if np.any(msdtemp.values == 0):
            msdtemp = msdtemp.replace(0, np.nan)
        msdtemp = msdtemp.dropna(axis='rows', how='all')
        
        if msdtemp.empty or 'msd' not in msdtemp.columns or len(msdtemp) < 20:
            continue
        
        # --- Perform power law fits ---
        try:
            # Fit regime 1 (Early time - 0 to 8 frames / 0 to 0.8s)
            # Use positional indices: 0 to 8
            alphatemp, r_squaredtemp = powerfit(msdtemp, 0, 8, fps=fps) 
            
            # Fit regime 2 (Late time - 20 to 400 frames / 2s to 40s)
            # Use positional indices: 20 to 400
            alphatemp2, r_squaredtemp2 = powerfit(msdtemp, 20, 400, fps=fps)
            
        except (RuntimeError, ValueError) as e:
            # print(f"Fit failed for particle {pid}: {e}")
            continue

        # --- Apply Trajectory Acceptance Criteria ---
        if r_squaredtemp >= 0.9 and alphatemp[0] > 0:
            # print(f"Particle {pid}, Alpha: {alphatemp[0]:.2f}, R-squared: {r_squaredtemp:.2f}")
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
        
        # Convert index (lag frames) to time (seconds)
        lagt = resultsmsd.index.values.astype('float64') / float(fps)
        resultsmsd.set_index(lagt, inplace=True)
        resultsmsd.index.name = 'lag time [s]'
        
        resultstraj = pd.concat(trajec)
        time = resultstraj.index.values.astype('float64') / float(fps)
        resultstraj.set_index(time, inplace=True)
        resultstraj.index.name = 'Time [s]'
        
        alpha_df = pd.DataFrame(alpha)
        alpha2_df = pd.DataFrame(alpha2)
        length_df = pd.DataFrame(length)
        return resultsmsd, resultstraj, alpha_df, alpha2_df, length_df
    else:
        return [], [], [], [], []

# --- Optimal Length Determination Function (Implemented) ---
def analyze_length_convergence(length_df, msd_df, fps, powerfit_func, max_trajectory_length_frames):
    """ 
    Analyzes the convergence of the anomalous exponent (alpha) as a function of 
    minimum trajectory length required, using the ensemble mean MSD.
    """
    length_sec = length_df.to_numpy().flatten() / fps
    trajectory_indices = msd_df.columns.to_list()

    # Define a set of minimum length thresholds (in seconds)
    length_thresholds_sec = np.array([5, 10, 15, 20, 30, 40, 50, 60, 70, 80])
    
    # Filter thresholds based on the max length available
    max_sec = max_trajectory_length_frames / fps
    length_thresholds_sec = length_thresholds_sec[length_thresholds_sec <= max_sec]

    results = []
    print("\n--- Analyzing Alpha Convergence ---")
    for min_len_sec in length_thresholds_sec:
        min_len_frames = int(min_len_sec * fps)
        
        # 1. Select trajectories longer than the current minimum length
        valid_indices = [idx for idx, length in enumerate(length_df.to_numpy().flatten()) if length >= min_len_frames]
        cols_to_use = [col for col in trajectory_indices if col in valid_indices]

        if len(cols_to_use) < 5:
            # print(f"Skipping threshold {min_len_sec}s: Too few trajectories ({len(cols_to_use)})")
            continue

        msd_subset = msd_df[cols_to_use]
        mean_msd = msd_subset.mean(axis=1).dropna()

        if mean_msd.empty or len(mean_msd) < 8:
            continue

        # 2. Define the fitting range: 0 to 0.8s (indices 0 to 8)
        # We rely on the positional indices a=0, b=8 as defined in your imsd function for the early regime fit.
        a_idx, b_idx = 0, 8
        if len(mean_msd) < b_idx:
            # Ensure the MSD itself is long enough for the standard fit range
            continue
        
        # 3. Fit the mean MSD curve for this length subset
        try:
            # Use positional slicing: a_idx, b_idx are positional indices for the array/series
            pars, r_squared = powerfit_func(mean_msd.to_frame(name='msd'), a_idx, b_idx, fps)
            D_eff = pars[1] / 4.0 # D = A / 4 for 2D diffusion
            results.append({
                'min_length_sec': min_len_sec,
                'N_trajectories': len(cols_to_use),
                'alpha': pars[0],
                'D_eff': D_eff,
                'R_squared': r_squared
            })
        except (RuntimeError, ValueError) as e:
            # print(f"Could not fit data for min length {min_len_sec}s: {e}")
            continue
            
    return pd.DataFrame(results)

# --- Path Resolver Function ---
def resolve_paths(data_set, i, pl, bp):
    """ Resolves the specific file path based on the dataset type. """
    filepath_base = data_set['filepath_base']
    pl_clean = pl.strip('/')
    bp_clean = bp.strip('/')
    
    if data_set['name'] == 'Sophie DNA (Plasmids)':
        base_dir = os.path.join(filepath_base, pl_clean, 'E_output_data', bp_clean)
    elif data_set['name'] == 'Aswin DNA (Ladders)':
        # Aswin Ladders path structure relies only on filepath_base and bp
        base_dir = os.path.join(filepath_base, 'E_output_data', bp_clean)
    elif data_set['name'] == 'Aswin DNA (Cell Types)':
        base_dir = os.path.join(filepath_base, pl_clean, 'E_output_data', bp_clean)
    elif data_set['name'] == 'Immobile Controls':
        # Immobile Controls path structure relies only on filepath_base and bp
        base_dir = os.path.join(filepath_base, 'E_output_data', bp_clean)
    else:
        base_dir = os.path.join(filepath_base, bp_clean) # Fallback

    # Directories for input and output
    directory = os.path.join(base_dir, 'tm')       # Input: Trajectory measurements
    directory2 = os.path.join(base_dir, 'MSDindividual') # Output: Individual MSD
    directory3 = os.path.join(base_dir, 'Figures') # Output: Figures
    
    return base_dir, directory, directory2, directory3

# --- Main Processor Function ---
def run_analysis(data_set, global_fig_axes):
    """ Executes the analysis for a single data set configuration. """
    print(f"\n{'#'*50}")
    print(f"STARTING DATASET: {data_set['name']}")
    print(f"{'#'*50}")

    # Fixed Parameters
    fps = 10 
    max_lagtime = 100
    mpp = 0.102
    mkr = ['^', 'o', 's']
    colors = ['#e41a1c','#4daf4a','#377eb8']
    
    hist_axes = global_fig_axes[data_set['hist_ax']]

    # Loop over the individual conditions (plasmids/sizes) within the dataset
    for i, (bp, pl, sz) in enumerate(zip(data_set['bpc_folders'], 
                                        data_set.get('plasmid_folders', [''] * len(data_set['bpc_folders'])), 
                                        data_set['size_labels'])):
        
        # --- 1. Resolve and Create Directories ---
        base_dir, directory, directory2, directory3 = resolve_paths(data_set, i, pl, bp)
        print(f"\nProcessing Condition {i+1}: {sz}")
        
        # Create necessary folders
        for dirs in ['MSDcollected', 'Trajcollected', 'MSDindividual', 'Figures']:
            os.makedirs(os.path.join(base_dir, dirs), exist_ok=True)
        
        # Check if the input directory exists
        if not os.path.exists(directory):
            print(f"Input directory not found: {directory}. Skipping.")
            continue
            
        # --- 2. Data Collection ---
        currentlength = 0
        video = 0
        Trajcollected = []
        alphacollected = []
        alphacollected2 = []
        lengthcollected = []
        msd_list = [] # List to store individual MSD DataFrames
        
        for filename in os.listdir(directory):
            if not filename.startswith('.'):
                meansquaredis, traj, alpha, alpha2, length = imsd(
                    os.path.join(directory, filename), mpp, fps, video, directory2, max_lagtime
                )
                
                if isinstance(meansquaredis, pd.DataFrame) and not meansquaredis.empty:
                    num_tracks_in_file = len(meansquaredis.columns)
                    MSDcolumns = [j + currentlength for j in range(num_tracks_in_file)]
                    meansquaredis.columns = MSDcolumns # Rename columns before adding
                    msd_list.append(meansquaredis)
                    Trajcollected.append(traj)
                    alphacollected.append(alpha)
                    alphacollected2.append(alpha2)
                    video += 1
                    currentlength += num_tracks_in_file
                    lengthcollected.append(length)

        # --- 3. Final Consolidation ---
        if not msd_list:
            print(f"No valid MSD data collected for {sz}.")
            continue
            
        # Concatenate MSDs along columns (individual tracks)
        MSDcollected_df = pd.concat(msd_list, axis=1) 
        MSDcollected_df = MSDcollected_df.dropna(axis='columns', how='all')
        
        Trajcollected_df = pd.concat(Trajcollected).dropna(axis='columns', how='all')
        Alphacollected_df = pd.concat(alphacollected)
        Alphacollected2_df = pd.concat(alphacollected2)
        lengthcollected_df = pd.concat(lengthcollected, ignore_index=True)

        totaltracks = len(MSDcollected_df.columns)
        max_len_frames = lengthcollected_df.to_numpy().max()
        print(f"Total Tracks analyzed for {sz}: {totaltracks}")

        # --- 4. Alpha Convergence Analysis (Optimal Length Determination) ---
        convergence_df = analyze_length_convergence(
            lengthcollected_df, MSDcollected_df, fps, powerfit, max_len_frames
        )
        
        print(f"\nAlpha Convergence Results for {sz}:")
        optimal_length = None
        if not convergence_df.empty:
            print(convergence_df)

            # --- Determine the Optimal Length based on stability ---
            if len(convergence_df) > 1:
                alphas = convergence_df['alpha'].values
                # Fractional change between successive thresholds (using percentage change)
                d_alphas = np.abs(np.diff(alphas) / alphas[:-1])
                tolerance = 0.05 # 5% fractional change tolerance
                
                for j in range(len(d_alphas) - 1, -1, -1):
                    if d_alphas[j] <= tolerance:
                        # The optimal length is the minimum threshold that resulted in stable alpha
                        optimal_length = convergence_df['min_length_sec'].iloc[j + 1]
                        break
                
                if optimal_length:
                    print(f"\nOptimal Trajectory Length (Alpha stability < {tolerance*100}%): {optimal_length:.1f} seconds")
                else:
                    print("\nAlpha did not stabilize within the tested range. Consider longer trajectories.")
            else:
                print("Not enough convergence data points (fewer than 2 successful fittings).")


        # --- 5. Bias analysis: MSD vs trajectory length (Plotting) ---
        length_sec = lengthcollected_df.to_numpy().flatten() / fps
        length_bins_list = data_set['msd_bin_params']['bins']
        bin_labels = data_set['msd_bin_params']['labels']

        # Ensure length of final_bins is one greater than length of bin_labels
        # We start with length_bins_list (N-1 elements), and dynamically add the max edge (total N elements)
        final_bins = length_bins_list + [np.max(length_sec) + 1]

        # Ensure length_sec matches the number of successfully processed trajectories
        length_assignment = pd.cut(length_sec[:totaltracks], bins=final_bins, labels=bin_labels, right=False)

        fig_msd_length, ax_msd_length = plt.subplots(figsize=(5,4))

        for label in bin_labels:
            bin_cols = [col for col, bin_label in zip(MSDcollected_df.columns, length_assignment) if bin_label == label]
            if len(bin_cols) == 0:
                continue
            
            msd_bin_mean = MSDcollected_df[bin_cols].mean(axis=1)
            ax_msd_length.plot(msd_bin_mean, marker='o', linestyle='-', label=f'{label} (n={len(bin_cols)})')

        # Plot all trajectories mean for comparison
        ax_msd_length.plot(MSDcollected_df.mean(axis=1), marker=mkr[i % len(mkr)], markeredgecolor='k', markerfacecolor=colors[i % len(colors)], linestyle='None', label=f"All trajectories (N={totaltracks})")
        
        ax_msd_length.set_xscale('log')
        ax_msd_length.set_yscale('log')
        ax_msd_length.set_xlabel(r'$\Delta t$ (s)')
        ax_msd_length.set_ylabel(r'$\langle \overline{\Delta r^2} \rangle$ ($\mu m^2$)')
        ax_msd_length.set_title(f"MSD sensitivity to trajectory length: {sz}")
        ax_msd_length.legend(frameon=False, fontsize=8)
        ax_msd_length.tick_params(direction='in', which='both')
        ax_msd_length.grid(False)
        
        fig_msd_length.savefig(os.path.join(directory3, f'MSD_Length_Bias_{sz.replace(" ", "_")}.png'))
        plt.close(fig_msd_length)

        # --- 6. Plotting the histogram of trajectory lengths ---
        # NOTE: Using positional index i for the axes array
        ax = hist_axes[i] 

        if data_set['hist_ax'] in ['ax6', 'ax9']:
            hist_bins = np.arange(5, 80, 3)
            xlim, ylim = (0, 90), (0, 0.3)
        else: # ax7, ax8, ax9
            hist_bins = np.arange(0, 50, 3)
            xlim, ylim = (0, 40), (0, 0.3)
            
        lengthcollected_hist = lengthcollected_df.to_numpy() / fps
        ax.hist(lengthcollected_hist, bins=hist_bins, edgecolor='black', color=colors[i % len(colors)], density=True, rwidth=1)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(which="both", axis="both", direction="in")
        ax.set(ylabel=r'$P(\text{Length})$', xlabel=r'Length (s)')
        ax.grid(False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(-0.23, 1.2, sz, transform=ax.transAxes, size=8, weight='bold')

        # --- 7. Print Summary Statistics ---
        print('\n--- Summary Statistics (Individual Tracks) ---')
        if not Alphacollected_df.empty:
            print('Mean alpha (Regime 1) is:')
            print(Alphacollected_df.mean().to_string(header=False))
            print('Standard error of the alpha (Regime 1) is:')
            print(Alphacollected_df.sem().to_string(header=False))
            print('Mean alpha_2 (Regime 2) is:')
            print(Alphacollected2_df.mean().to_string(header=False))
            print('Standard error of the alpha_2 (Regime 2) is:')
            print(Alphacollected2_df.sem().to_string(header=False))

# --- Data Definition Structure (Corrected) ---
# NOTE ON CORRECTION: The definition of 'bins' now contains N-1 edges for N labels, 
# as the final edge is added dynamically using np.max(length_sec) + 1.
DATASETS = [
    {
        'name': 'Sophie DNA (Plasmids)',
        'filepath_base': 'F:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Sophie Corrected for Gaussian Filter',
        'bpc_folders': ['/Collected_pEGFP-C1', '/Collected_BPK4410', '/Collected_hCas9D10A'],
        'plasmid_folders': ['/pEGFP-C1', '/BPK4410', '/hCas9D10A'],
        'size_labels': ['4713 bp', '7614 bp', '9553 bp'],
        'hist_ax': 'ax6',
        'msd_bin_params': {
            'bins': [0, 10, 20, 40], # N-1=4 edges (0, 10, 20, 40)
            'labels': ['0-10s', '10-20s', '20-40s', '40s+'] # N=4 labels
        }
    },
    {
        'name': 'Aswin DNA (Ladders)',
        'filepath_base': 'F:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Aswin Data/Aswin Data CHO Corrected for Drift',
        'bpc_folders': ['/Collected_100bp', '/Collected_250bp', '/Collected_500bp'],
        'plasmid_folders': [''] * 3, # Used for path resolution compatibility
        'size_labels': ['100 bp', '250 bp', '500 bp'],
        'hist_ax': 'ax7',
        'msd_bin_params': {
            'bins': [0, 10, 20], # N-1=3 edges (0, 10, 20)
            'labels': ['0-10s', '10-20s', '20s+'] # N=3 labels
        }
    },
    {
        'name': 'Aswin DNA (Cell Types)',
        'filepath_base': 'F:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Aswin Data',
        'bpc_folders': ['/Collected_MCF10A', '/Collected_MCF7', '/Collected_MDAMB231'],
        'plasmid_folders': ['/Aswin Data MCF 10A Corrected for Drift', '/Aswin Data MCF 7 Corrected for Drift', '/Aswin Data MDA-MB-231 Corrected for Drift'],
        'size_labels': ['MCF 10A', 'MCF 7', 'MDA-MB-231'],
        'hist_ax': 'ax8',
        'msd_bin_params': {
            'bins': [0, 10, 20], # N-1=3 edges (0, 10, 20)
            'labels': ['0-10s', '10-20s', '20s+'] # N=3 labels
        }
    },
    {
        'name': 'Immobile Controls',
        'filepath_base': 'F:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Control Experiments_Plasmid Aggregates on Glass',
        'bpc_folders': ['/Collected_pEGFP-C1', '/Collected_BPK4410', '/Collected_hCas9D10A'],
        'plasmid_folders': [''] * 3, # Used for path resolution compatibility
        'size_labels': ['Immobile 4713 bp', 'Immobile 7614 bp', 'Immobile 9553 bp'],
        'hist_ax': 'ax9',
        'msd_bin_params': {
            'bins': [0, 10, 20, 40], # N-1=4 edges (0, 10, 20, 40)
            'labels': ['0-10s', '10-20s', '20-40s', '40s+'] # N=4 labels
        }
    }
]

# --- Initialization and Execution ---
if __name__ == '__main__':
    # Initialize all necessary figures
    fig5, ax6 = plt.subplots(1, 3, figsize=(3.375 * 2, 3.375 * 2.2 / 3 * 0.75))
    fig6, ax7 = plt.subplots(1, 3, figsize=(3.375 * 2, 3.375 * 2.2 / 3 * 0.75))
    fig7, ax8 = plt.subplots(1, 3, figsize=(3.375 * 2, 3.375 * 2.2 / 3 * 0.75))
    fig8, ax9 = plt.subplots(1, 3, figsize=(3.375 * 2, 3.375 * 2.2 / 3 * 0.75))

    # Map the plot axis name to the actual figure axes
    global_fig_axes = {
        'ax6': ax6,
        'ax7': ax7,
        'ax8': ax8,
        'ax9': ax9,
    }

    # Execute the analysis loop for all defined datasets
    for dataset in DATASETS:
        run_analysis(dataset, global_fig_axes)

    # Display final plots
    plt.show()