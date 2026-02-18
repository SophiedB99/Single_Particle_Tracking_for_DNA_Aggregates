# -*- coding: utf-8 -*-
"""
MSD Analysis - Works for both diffusing and stuck beads
Includes power-law fitting AND localization precision measurement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from scipy.stats import kde


# Global style

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 15,
    "axes.labelsize": 13,
    "axes.linewidth": 1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 600,
})


def _msd_iter(pos, lagtimes):
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff[diff == 0] = np.nan
        yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings"""
    ax = plt.gca() if ax is None else ax
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    lc.set_array(np.asarray(c))
    ax.add_collection(lc)
    return lc

def msd(pos, max_lagtime):
    """Compute the mean displacement and mean squared displacement"""
    max_lagtime = min(max_lagtime, len(pos) - 1)
    lagtimes = np.arange(1, max_lagtime + 1)
    pos_columns = ['x', 'y']
    result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                     ['<{}^2>'.format(p) for p in pos_columns]
    resultmsd = _msd_iter(pos, lagtimes)
    result = pd.DataFrame(resultmsd, columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(1)
    return result

def power_law(x, alpha, A):
    return A * np.power(x, alpha)

def trajectory(pos, pid, max_lagtime, video, pos_columns):
    """Shift the starting point of the trajectory to (0,0)."""
    max_lagtime = len(pos)
    lagtimes = np.arange(0, max_lagtime)
    result = pd.DataFrame(pos - pos[0], columns=pos_columns, index=lagtimes)
    result['particle'] = pid
    result['video'] = video
    return result

def powerfit(msd, fps=10, min_lag=20, max_lag=80):
    """Fit a power law to MSD data"""
    y = msd['msd'].values
    x = msd.index.values.astype('float64') / float(fps)
    
    n_points = len(x)
    fit_start = min(min_lag, n_points - 20)
    fit_end = min(max_lag, n_points)
    
    if fit_end - fit_start < 20:
        return None, 0.0, None
    
    try:
        pars, cov = curve_fit(
            f=power_law,
            xdata=x[fit_start:fit_end],
            ydata=y[fit_start:fit_end],
            p0=[1, 0.01],
            bounds=(-np.inf, np.inf),
            maxfev=10000
        )
        
        residuals = y[fit_start:fit_end] - power_law(x[fit_start:fit_end], *pars)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y[fit_start:fit_end] - np.mean(y[fit_start:fit_end])) ** 2)
        
        if ss_tot == 0:
            return None, 0.0, None
            
        r_squared = 1.0 - (ss_res / ss_tot)
        perr = np.sqrt(np.diag(cov))
        
        return pars, r_squared, perr
        
    except Exception as e:
        return None, 0.0, None

def analyze_stuck_bead(msd_df, fps=10, plateau_start=10, plateau_end=50):
    """Measure localization precision from plateau MSD"""
    y = msd_df['msd'].values
    
    n_points = len(y)
    plateau_start = min(plateau_start, n_points - 10)
    plateau_end = min(plateau_end, n_points)
    
    if plateau_end - plateau_start < 5:
        return None, None, None
    
    plateau_region = y[plateau_start:plateau_end]
    valid_plateau = plateau_region[~np.isnan(plateau_region)]
    
    if len(valid_plateau) < 3:
        return None, None, None
    
    plateau_msd = np.mean(valid_plateau)
    std_msd = np.std(valid_plateau)
    
    if plateau_msd > 0:
        localization_error = np.sqrt(plateau_msd / 4.0)
    else:
        localization_error = None
    
    return plateau_msd, std_msd, localization_error

def imsd(filename, mpp, fps, video, directory, plotindividual=False, max_lagtime=100, 
         r2_threshold=0.3, analyze_stuck=False):
    """
    Process MSD data with both power-law fitting and stuck bead analysis.
    
    Parameters:
        analyze_stuck: If True, also compute localization precision for each bead
        r2_threshold: Minimum R² for accepting a fit (0.6 recommended)
    """
    statistic = 'msd'
    traj = pd.read_csv(filename)
    pos_columns = ['x', 'y']

    ids = []
    trajec = []
    msds = []
    alpha = []
    msdsuper = []
    msdsub = []
    alphasuper = []
    alphasub = []
    localization_data = []  # [plateau_msd, loc_error] for stuck beads
    index = 0
    
    total_particles = 0
    skip_reasons = {
        'empty_msd': 0,
        'too_short': 0,
        'fit_failed': 0,
        'low_r2': 0,
        'invalid_alpha': 0,
        'passed': 0
    }

    for pid, ptraj in traj.groupby('particle'):
        total_particles += 1
        
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()

        msdtemp = msd(pos, max_lagtime)
        
        if msdtemp is None or msdtemp.empty or 'msd' not in msdtemp.columns:
            skip_reasons['empty_msd'] += 1
            continue
        
        if msdtemp['msd'].isna().all() or (msdtemp['msd'] == 0).all():
            skip_reasons['empty_msd'] += 1
            continue
        
        traj_length = len(msdtemp)
        if traj_length < 80:
            skip_reasons['too_short'] += 1
            continue

        # Power law fit
        try:
            result = powerfit(msdtemp, fps)
            if result[0] is None:
                skip_reasons['fit_failed'] += 1
                continue
            alphatemp, r_squaredtemp, perr = result
        except Exception as e:
            skip_reasons['fit_failed'] += 1
            continue

        # Check fit quality
        if alphatemp[0] <= 0:
            skip_reasons['invalid_alpha'] += 1
            continue
            
        if r_squaredtemp < r2_threshold:
            skip_reasons['low_r2'] += 1
            continue

        # Particle passed!
        skip_reasons['passed'] += 1
        alpha.append(alphatemp)
        msds.append(msdtemp)
        trajec.append(trajectory(pos, pid, max_lagtime, video, pos_columns))
        ids.append(index)
        index += 1

        # Classify as super/sub diffusive
        if alphatemp[0] > 1:
            msdsuper.append(msdtemp)
            alphasuper.append(alphatemp)
        elif alphatemp[0] < 1:
            msdsub.append(msdtemp)
            alphasub.append(alphatemp)

        # Optional: also compute localization precision
        if analyze_stuck:
            plateau_msd, std_msd, loc_error = analyze_stuck_bead(msdtemp, fps)
            if loc_error is not None and 0 < loc_error < 1.0:
                localization_data.append([plateau_msd, loc_error])
                print(f"✓ Particle {pid}: α={alphatemp[0]:.3f}, R²={r_squaredtemp:.3f}, σ={loc_error*1000:.2f} nm")
            else:
                localization_data.append([np.nan, np.nan])
                print(f"✓ Particle {pid}: α={alphatemp[0]:.3f}, R²={r_squaredtemp:.3f}")
        else:
            print(f"✓ Particle {pid}: α={alphatemp[0]:.3f}, R²={r_squaredtemp:.3f}")

        if plotindividual:
            y = msdtemp['msd'].values
            x = msdtemp.index.values.astype('float64') / float(fps)
            fig, ax = plt.subplots()
            ax.loglog(x, y, 'ro', label='Data')
            fit_range = slice(20, min(80, len(x)))
            ax.loglog(x[fit_range], power_law(x[fit_range], *alphatemp), 'b-', 
                     label=f'Fit: α={alphatemp[0]:.3f}')
            ax.set_ylim((1e-4, 10))
            ax.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
                   xlabel=r'$\mathrm{\tau}$ (s)')
            ax.legend()
            plt.savefig(directory + f'/particle_{video}_{pid}.pdf')
            plt.close()

    print(f"\n{'─'*60}")
    print(f"File: {os.path.basename(filename)}")
    print(f"{'─'*60}")
    print(f"Total particles: {total_particles}")
    print(f"  ✓ Passed (R²≥{r2_threshold}): {skip_reasons['passed']}")
    print(f"  ✗ Empty/invalid MSD: {skip_reasons['empty_msd']}")
    print(f"  ✗ Too short (< 80 frames): {skip_reasons['too_short']}")
    print(f"  ✗ Fit failed: {skip_reasons['fit_failed']}")
    print(f"  ✗ Low R² (< {r2_threshold}): {skip_reasons['low_r2']}")
    print(f"  ✗ Invalid α (≤ 0): {skip_reasons['invalid_alpha']}")
    print(f"{'─'*60}\n")

    if not msds:
        return None, None, None, None, None, None, None, None

    # Concatenate results (same as before)
    resultsmsd = pd.concat(msds, keys=ids)
    resultsmsd = resultsmsd.swaplevel(0, 1)[statistic].unstack()
    lagt = resultsmsd.index.values.astype('float64') / float(fps)
    resultsmsd.set_index(lagt, inplace=True)
    resultsmsd.index.name = 'lag time [s]'
    
    resultstraj = pd.concat(trajec)
    time = resultstraj.index.values.astype('float64') / float(fps)
    resultstraj.set_index(time, inplace=True)
    resultstraj.index.name = 'Time [s]'
    
    alpha = pd.DataFrame(alpha)
    
    if msdsuper:
        resultssuper = pd.concat(msdsuper, keys=range(len(msdsuper)))
        resultssuper = resultssuper.swaplevel(0, 1)[statistic].unstack()
        lagt = resultssuper.index.values.astype('float64') / float(fps)
        resultssuper.set_index(lagt, inplace=True)
        resultssuper.index.name = 'lag time [s]'
        alphasuper = pd.DataFrame(alphasuper)
    else:
        resultssuper = pd.DataFrame()
        alphasuper = pd.DataFrame()
        
    if msdsub:
        resultssub = pd.concat(msdsub, keys=range(len(msdsub)))
        resultssub = resultssub.swaplevel(0, 1)[statistic].unstack()
        lagt = resultssub.index.values.astype('float64') / float(fps)
        resultssub.set_index(lagt, inplace=True)
        resultssub.index.name = 'lag time [s]'
        alphasub = pd.DataFrame(alphasub)
    else:
        resultssub = pd.DataFrame()
        alphasub = pd.DataFrame()
    
    # Localization data
    if localization_data:
        loc_df = pd.DataFrame(localization_data, columns=['plateau_msd', 'localization_error'])
    else:
        loc_df = None
    #MSD(τ) = <(x(t+τ) - x(t))²> + <(y(t+τ) - y(t))²>
    #For stuck bead:
    #MSD(τ) ≈ <(noise_x)²> + <(noise_y)²>
    #MSD(τ) ≈ 2σ² + 2σ² = 4σ²
    #So: σ = √(MSD_plateau / 4)
        
    return resultsmsd, resultstraj, alpha, resultssuper, resultssub, alphasuper, alphasub, loc_df


"""
Main code
"""

Filepath = 'F:/A_Data/A_DNATranslocation/Experiments 2025/DNA Tracking/Corrected_Data/Control Experiments_Beads/0.22 um'
bp = 'Collected_0.22um'
bps = 'Beads - Diameter 0.22μm'

for dirs in ['/MSDcollected', '/Trajcollected', '/MSDindividual', '/Figures']:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)

directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'

mpp = 0.16
fps = 10
max_lagtime = 100
currentlength = 0
currentlengthsuper = 0
currentlengthsub = 0
video = 0

MSDcollected_df = pd.DataFrame()
MSDsuper_df = pd.DataFrame()
MSDsub_df = pd.DataFrame()
Trajcollected = []
alphacollected = []
alphasuper = []
alphasub = []
localization_collected = []

# SETTINGS - Adjust these!
R2_THRESHOLD = 0.3  # Lower for stuck beads, higher (0.9) for good diffusion
ANALYZE_STUCK = True  # Set to True to also measure localization precision

"""
Process all files
"""

for filename in os.listdir(directory):
    if not filename.startswith('.'):
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        meansquaredis, traj, alpha, resultssuper, resultssub, alphasupert, alphasubt, loc_data = imsd(
            os.path.join(directory, filename), mpp, fps, video, directory2, 
            plotindividual=False, max_lagtime=max_lagtime,
            r2_threshold=R2_THRESHOLD, analyze_stuck=ANALYZE_STUCK
        )
        
        if meansquaredis is None:
            print(f"⚠️ No valid data from {filename}")
            continue
        
        print(f"✓ Found {len(meansquaredis.columns)} valid trajectories")
        
        MSDcolumns = [i + currentlength for i in range(1, len(meansquaredis.columns) + 1)]
        if currentlength == 0:
            MSDcollected_df = pd.DataFrame(meansquaredis, columns=np.arange(0, len(meansquaredis.columns)))
        else:
            MSDcollected_df[MSDcolumns] = meansquaredis
            
        if isinstance(resultssuper, pd.DataFrame) and not resultssuper.empty:
            MSDsupercolumns = [i + currentlengthsuper for i in range(1, len(resultssuper.columns) + 1)]
            alphasuper.append(alphasupert)
            if currentlengthsuper == 0:
                MSDsuper_df = pd.DataFrame(resultssuper, columns=np.arange(0, len(resultssuper.columns)))
            else:
                MSDsuper_df[MSDsupercolumns] = resultssuper
            currentlengthsuper = len(MSDsuper_df.columns)
            
        if isinstance(resultssub, pd.DataFrame) and not resultssub.empty:
            MSDsubcolumns = [i + currentlengthsub for i in range(1, len(resultssub.columns) + 1)]
            alphasub.append(alphasubt)
            if currentlengthsub == 0:
                MSDsub_df = pd.DataFrame(resultssub, columns=np.arange(0, len(resultssub.columns)))
            else:
                MSDsub_df[MSDsubcolumns] = resultssub
            currentlengthsub = len(MSDsub_df.columns)
            
        Trajcollected.append(traj)
        alphacollected.append(alpha)
        if loc_data is not None:
            localization_collected.append(loc_data)
        video += 1
        currentlength = len(MSDcollected_df.columns)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

MSDcollected_df = MSDcollected_df.dropna(axis='columns', how='all')
Trajcollected_df = pd.concat(Trajcollected)
Trajcollected_df = Trajcollected_df.dropna(axis='columns', how='all')
Alphacollected_df = pd.concat(alphacollected)

if alphasuper:
    Alphacollected_super_df = pd.concat(alphasuper)
else:
    Alphacollected_super_df = pd.DataFrame()
    
if alphasub:
    Alphacollected_sub_df = pd.concat(alphasub)
else:
    Alphacollected_sub_df = pd.DataFrame()

MSDsuper_df = MSDsuper_df.dropna(axis='columns', how='all')
MSDsub_df = MSDsub_df.dropna(axis='columns', how='all')

totaltracks = len(MSDcollected_df.columns)
totalsub = len(MSDsub_df.columns)
totalsup = len(MSDsuper_df.columns)

print(f"Total valid trajectories: {totaltracks}")
print(f"Superdiffusive (α > 1): {totalsup}")
print(f"Subdiffusive (α < 1): {totalsub}")
print(f"Normal diffusion (α ≈ 1): {totaltracks - totalsup - totalsub}")

if not Alphacollected_super_df.empty:
    max_alpha = max(Alphacollected_super_df[0])
    print(f'Max α = {max_alpha:.3f}')

if not Alphacollected_sub_df.empty:
    min_alpha = min(Alphacollected_sub_df[0])
    print(f'Min α = {min_alpha:.3f}')

if localization_collected:
    Localization_df = pd.concat(localization_collected, ignore_index=True)
    Localization_df = Localization_df.dropna()
    if not Localization_df.empty:
        print(f"\nLocalization Precision:")
        print(f"  Mean: {Localization_df['localization_error'].mean()*1000:.2f} ± {Localization_df['localization_error'].std()*1000:.2f} nm")
        print(f"  Median: {Localization_df['localization_error'].median()*1000:.2f} nm")

"""
ALL YOUR ORIGINAL PLOTS
"""

# Plot 1: MSD with alpha coloring
fig1, ax1 = plt.subplots()
ys = np.transpose(MSDcollected_df.to_numpy())
xs = [np.transpose(MSDcollected_df.index.to_numpy())] * ys.shape[0]
norm1 = mcolors.Normalize(0, 2)
lc1 = multiline(xs, ys, Alphacollected_df[0].to_numpy(), cmap='viridis', norm=norm1, ax=ax1)
ax1cb = fig1.colorbar(lc1)
ax1cb.ax.tick_params(axis='y', direction='in')
ax1cb.ax.set_ylim(0, 2)
ax1cb.set_ticks(np.arange(0, 2.2, step=0.4))
ax1cb.set_label(r'$\alpha$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
        xlabel=r'$\mathrm{\tau}$ (s)')
ax1.set_ylim((1e-4, 10))
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which="both", axis="both", direction="in")
ax1.set_aspect(0.5, adjustable='box')
ax1.text(0.05, 0.95, bps + '\n$n$ = ' + str(totaltracks), verticalalignment='top',
         horizontalalignment='left', multialignment="left", transform=ax1.transAxes)
fig1.savefig(directory3 + '/MSD.png', dpi=300, bbox_inches='tight')

# Plot 2: Subdiffusive
if not MSDsub_df.empty:
    fig2, ax2 = plt.subplots()
    ysub = np.transpose(MSDsub_df.to_numpy())
    xsub = [np.transpose(MSDsub_df.index.to_numpy())] * ysub.shape[0]
    lc2 = multiline(xsub, ysub, Alphacollected_sub_df[0].to_numpy(), cmap='viridis', norm=norm1, ax=ax2)
    ax2cb = fig2.colorbar(lc2)
    ax2cb.ax.tick_params(axis='y', direction='in')
    ax2cb.ax.set_ylim(0, 2)
    ax2cb.set_ticks(np.arange(0, 2.2, step=0.4))
    ax2cb.set_label(r'$\alpha$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
            xlabel=r'$\mathrm{\tau}$ (s)')
    ax2.set_ylim((1e-4, 10))
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
    ax2.tick_params(which="both", axis="both", direction="in")
    ax2.set_aspect(0.5, adjustable='box')
    ax2.text(0.05, 0.95, bps + '\nSubdiffusive \n$n$ = ' + str(totalsub),
             verticalalignment='top', horizontalalignment='left',
             multialignment="left", transform=ax2.transAxes)
    fig2.savefig(directory3 + '/MSDsub.png', dpi=300, bbox_inches='tight')

# Plot 3: Superdiffusive
if not MSDsuper_df.empty:
    fig3, ax3 = plt.subplots()
    ysup = np.transpose(MSDsuper_df.to_numpy())
    xsup = [np.transpose(MSDsuper_df.index.to_numpy())] * ysup.shape[0]
    lc3 = multiline(xsup, ysup, Alphacollected_super_df[0].to_numpy(), cmap='viridis', norm=norm1, ax=ax3)
    ax3cb = fig3.colorbar(lc3)
    ax3cb.ax.tick_params(axis='y', direction='in')
    ax3cb.ax.set_ylim(0, 2)
    ax3cb.set_ticks(np.arange(0, 2.2, step=0.4))
    ax3cb.set_label(r'$\alpha$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
            xlabel=r'$\mathrm{\tau}$ (s)')
    ax3.set_ylim((1e-4, 10))
    ax3.yaxis.set_ticks_position('both')
    ax3.xaxis.set_ticks_position('both')
    ax3.tick_params(which="both", axis="both", direction="in")
    ax3.set_aspect(0.5, adjustable='box')
    ax3.text(0.05, 0.95, bps + '\nSuperdiffusive \n$n$ = ' + str(totalsup),
             verticalalignment='top', horizontalalignment='left',
             multialignment="left", transform=ax3.transAxes)
    fig3.savefig(directory3 + '/MSDsup.png', dpi=300, bbox_inches='tight')


# Plot 4a: Trajectories with incremental additions
figs = []
num_to_plot = [1, 2, 3]  # number of trajectories to include in each figure

# Loop through desired n values
for n_idx, n_traj in enumerate(num_to_plot, start=1):
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k', linestyle='dashdot')
    ax.axvline(x=0, color='k', linestyle='dashdot')

    t = np.arange(0, 40.1, 0.1)
    norm = plt.Normalize(t.min(), t.max())

    # Get first n trajectories
    for i, (id, traj) in enumerate(Trajcollected_df.groupby(['particle', 'video'])):
        if i >= n_traj:
            break

        x = traj['x'].head(400)
        y = traj['y'].head(400)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(t[:len(segments)])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        del x, y

    # Add colorbar
    fig.colorbar(line, ax=ax, label='$t$ (s)').ax.tick_params(axis='y', direction='in')

    # Aesthetics
    bounds = 0.2
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    ax.set(ylabel=r'y ($\mathrm{\mu}$m)', xlabel=r'x ($\mathrm{\mu}$m)')
    ax.set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='both', axis="both", direction="in")

    # Text label (updates n)
    ax.text(0.05, 0.95, bps + f'\n$n$ = {n_traj}',
            verticalalignment='top', horizontalalignment='left',
            multialignment="left", transform=ax.transAxes)

    # Save each figure separately
    savepath = f"{directory3}/Traj_n{n_traj}.png"
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    # plt.close(fig)


# Plot 4: Trajectories
fig4, ax4 = plt.subplots()
ax4.axhline(y=0, color='k', linestyle='dashdot')
ax4.axvline(x=0, color='k', linestyle='dashdot')
t = np.arange(0, 40.1, 0.1)
for id, traj in Trajcollected_df.groupby(['particle', 'video']):
    x = traj['x'].head(400)
    y = traj['y'].head(400)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)#cmap='plasma'
    lc.set_array(t[:len(segments)])
    lc.set_linewidth(2)
    line = ax4.add_collection(lc)
    del x, y
fig4.colorbar(line, ax=ax4, label='$t$ (s)').ax.tick_params(axis='y', direction='in')
bounds = 0.5
ax4.set_xlim(-bounds, bounds)
ax4.set_ylim(-bounds, bounds)
ax4.set(ylabel=r'y ($\mathrm{\mu}$m)', xlabel=r'x ($\mathrm{\mu}$m)')
ax4.set_aspect('equal', adjustable='box')
plt.minorticks_on()
ax4.yaxis.set_ticks_position('both')
ax4.xaxis.set_ticks_position('both')
ax4.tick_params(which='both', axis="both", direction="in")
ax4.text(0.05, 0.95, bps + '\n$n$ = ' + str(totaltracks), verticalalignment='top',
         horizontalalignment='left', multialignment="left", transform=ax4.transAxes)
fig4.savefig(directory3 + '/Traj.png', dpi=300, bbox_inches='tight')

# Plot 5: Histogram diffusion coefficient
fig5, ax5 = plt.subplots(figsize=(4, 4))
alpha_his250 = Alphacollected_df[1].to_numpy()
weights = np.ones_like(alpha_his250)/len(alpha_his250)
ax5.hist(np.log10(Alphacollected_df[1].to_numpy()), bins=np.arange(-6, 0, 0.25),
         edgecolor='black', color='white', weights=weights, rwidth=1)
ax5.set_xlim(-4, 0)
ax5.set_ylim(0, 0.3)
ax5.yaxis.set_ticks_position('both')
ax5.xaxis.set_ticks_position('both')
ax5.tick_params(which='both', axis="both", direction="in")
ax5.text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
         multialignment="left", transform=ax5.transAxes)
ax5.set(xlabel=r'$\log (D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)) ',
        ylabel=r'Probability')
ax5.set_yticks(np.arange(0, 0.35, step=0.05))
plt.tight_layout()
fig5.savefig(directory3 + '/HistoD.png', dpi=300, bbox_inches='tight')

# Plot 6: Histogram alpha
fig6, ax6 = plt.subplots(figsize=(4, 4))
alpha_his = Alphacollected_df[0].to_numpy()
weights = np.ones_like(alpha_his)/len(alpha_his)
ax6.hist(Alphacollected_df[0], bins=np.arange(0, 2, 0.1), edgecolor='black',
         color='white', weights=weights, rwidth=1)
ax6.set_xlim(0, 2)
ax6.set_ylim(0, 0.25)
ax6.yaxis.set_ticks_position('both')
ax6.xaxis.set_ticks_position('both')
ax6.tick_params(which='both', axis="both", direction="in")
ax6.text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
         multialignment="left", transform=ax6.transAxes)
ax6.set(xlabel=r'$\alpha_2$ ', ylabel=r'$P(\alpha_2)')
ax6.set_yticks(np.arange(0, 0.3, step=0.05))
plt.tight_layout()
fig6.savefig(directory3 + '/Histoalpha.png', dpi=300, bbox_inches='tight')

# Plot 7: 2D histogram alpha vs D
fig7, ax7 = plt.subplots(figsize=(4, 3.5))
k = kde.gaussian_kde([np.log10(alpha_his250), alpha_his])
xi, yi = np.mgrid[-4:0.1:0.05, 0:2.1:0.025]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
his1 = ax7.contourf(xi, yi, zi.reshape(xi.shape), 200, cmap='RdGy_r', alpha=1)
ax7cb = fig7.colorbar(his1)
contours = ax7.contour(xi, yi, zi.reshape(xi.shape), 6, colors='black')
ax7.scatter(np.log10(alpha_his250), alpha_his, marker='.', color='k', s=1)
ax7.set(xlabel=r'$\log (D_{\mathrm{app}})$ ', ylabel=r'$\alpha_2$')
ax7.set_yticks(np.arange(0, 2.01, step=0.5))
ax7.set_xticks(np.arange(-4, 0.01, step=1))
ax7.set_xlim(-4, 0)
ax7.set_ylim(0, 2)
ax7.yaxis.set_ticks_position('both')
ax7.xaxis.set_ticks_position('both')
ax7.tick_params(which='both', axis="both", direction="in")
ax7.text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
         multialignment="left", transform=ax7.transAxes, color='white')
ax7cb.ax.tick_params(axis='y', direction='in')
ax7cb.ax.set_ylim(0, 1)
ax7cb.set_ticks(np.arange(0, 1.1, step=0.2))
plt.tight_layout()
fig7.savefig(directory3 + '/AlphaVsD.png', dpi=300, bbox_inches='tight')

# BONUS Plot 8: Localization error histogram (if stuck bead analysis was done)
if localization_collected and ANALYZE_STUCK:
    Localization_df = pd.concat(localization_collected, ignore_index=True)
    Localization_df = Localization_df.dropna()
    if not Localization_df.empty:
        fig8, ax8 = plt.subplots(figsize=(5, 4))
        loc_errors_nm = Localization_df['localization_error'].to_numpy() * 1000
        ax8.hist(loc_errors_nm, bins=20, edgecolor='black', color='lightcoral', alpha=0.7)
        ax8.axvline(x=np.mean(loc_errors_nm), color='darkred', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(loc_errors_nm):.2f} nm')
        ax8.axvline(x=np.median(loc_errors_nm), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(loc_errors_nm):.2f} nm')
        ax8.axvline(x=np.median(loc_errors_nm), color= 'none', linestyle='-', linewidth=2,
                    label= '$n$ = ' + str(totaltracks))
        ax8.set(xlabel='Localization error σ (nm)', ylabel='Count',
                title= bps)
        ax8.legend()
        ax8.yaxis.set_ticks_position('both')
        ax8.xaxis.set_ticks_position('both')
        ax8.tick_params(which='both', axis="both", direction="in")
        plt.tight_layout()
        fig8.savefig(directory3 + '/LocalizationError.png', dpi=300, bbox_inches='tight')

plt.show()

"""
Save the files
"""
MSDcollected_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/MSDcollected.csv')
if not MSDsub_df.empty:
    MSDsub_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/MSDsub.csv')
if not MSDsuper_df.empty:
    MSDsuper_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/MSDsuper.csv')
Trajcollected_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/Trajcollected/Trajcollected.csv')
Alphacollected_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/Alpha.csv')

if localization_collected and ANALYZE_STUCK:
    Localization_df = pd.concat(localization_collected, ignore_index=True)
    Localization_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/LocalizationPrecision.csv', index=False)

print(f"\n{'='*60}")
print("✓ Analysis complete!")
print(f"✓ Figures saved to: {directory3}")
print(f"✓ Data saved to: {Filepath}/E_output_data/{bp}/")
print(f"{'='*60}")