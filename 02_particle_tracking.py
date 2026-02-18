# -*- coding: utf-8 -*-

# For compatibility with Python 2 and 3
from __future__ import division, unicode_literals, print_function #used for compatibility with both python 2 and 3
# pip install PIMS
import pims #a library for working with multidimensional image sequences
import trackpy as tp #is a library for particle tracking
import os #provides a way to interact with the operating system
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
import pandas as pd
from pandas import DataFrame, Series



###############################################
###############################################
#The first step in the data analysis##########
##############################################
##############################################

# O:/A_Data/A_DNATranslocation/Experiments 2024/Immobile Plasmid Control Experiments/20.11.2024/B_ProcessedData/BPK4410/Experiment 1.1

Filepath = r'F:\A_Data\A_DNATranslocation\Experiments 2025\DNA Tracking\Corrected_Data\Control Experiments_Beads\0.22 um' #where to save data
bp = '/Experiment8'  # Experiment to process
name = 'Experiment8'
#directory = Filepath + '/B_ProcessedData' + str(bp) # The input directory after background image corrections
directory = r'F:\A_Data\A_DNATranslocation\Experiments 2025\DNA Tracking\Corrected_Data\Control Experiments_Beads\0.22 um\B_ProcessedData' + str(bp) #where to obtain data
frames   = pims.ImageSequence(directory)


for dirs in ['/f', '/t', '/tf', '/im_plot', '/em_plot', '/trajectories', '/tm', '/hist_subpix']:
    if not os.path.exists(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + dirs)

plot_name = 'plot'

# print(frames)
# print(frames[0]) #this prints the numpy array of the different intensities in my first frame for this image sequence
# plt.imshow(frames[0]);


f = tp.locate(frames[0],11, minmass =1000) #The minimum integrated brightness. Default is 100 for integer images and 1 for float images, but a good value is often much higher. 
#This is a crucial parameter for eliminating spurious features.
#Estimate the size of the features (in pixels). The size must be an odd integer, and it is better to err on the large side, as we'll see below.
f.head() #returns the first few rows of the DataFrame or Series.
#print('Particle', f['particle'].nunique())


###############################
###############################
"""visualization of tracked particles through annotation of particles (red circles):"""
###############################
############################### 

tp.annotate(f, frames[0])

###############################
###############################
"""Subpixel bias histogram:"""
###############################
###############################    

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=10)
ax.set(xlabel='mass', ylabel='count');
tp.subpx_bias(f) #a quick way to check for subpixel accuracy is to check that the decimal part of the x and/or y positions are evenly distributed. 
#Trackpy provides a convenience plotting function for this called subpx_bias
#if you use a mask size that is too small (i.e. the particle size), you get a dip in the middle of the histogram
plt.savefig(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/hist_subpix/hist_' + plot_name + '.png', dpi=300, bbox_inches='tight')
plt.show()

###############################
###############################
"""Locating features in ALL FRAMES (400)"""
###############################
###############################   


f = tp.batch(frames[0:800],11, minmass=1000, maxsize=4,noise_size=1, smoothing_size=17, processes = 1, threshold=None) 
# Or after locating in all frames
tp.subpx_bias(f)
#noise_size = Width of Gaussian blurring kernel, in pixels Default is 1
#maxsize = maximum radius-of-gyration of brightness, default None
tp.quiet()  # Turn off progress reports for best performance
x_frac = f['x'] % 1
y_frac = f['y'] % 1

print("KS test X:", kstest(x_frac, 'uniform', args=(0,1)))
print("KS test Y:", kstest(y_frac, 'uniform', args=(0,1)))

###############################
###############################
"""Linking features in ALL FRAMES (400) to make PARTICLE TRAJECTORIES"""
###############################
############################### 

t = tp.link(f, 2, memory=3 ) #2 is chosen as maximum displacement as the farthest a particle can travel between frames
t.head() #returns the first few rows of the DataFrame or Series.

###############################
###############################
#Filter Spurious Trajectories
###############################
############################### 

t1 = tp.filter_stubs(t, 150) #keeps only trajectories that last for a given number of frames. (tracks, threshold) threshold is the minimum number
#of points(video frames) to survive
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

plt.figure()


###############################
###############################
"""Filter Particles based on mass vs size (to assess whether they are faded (ecc) out yes or no)"""
###############################
############################### 
tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass. Groups data by particle and averages the mass over time,
#meaning a single frame might have mass < 23000, even if the averaged mass is above it.
t2 = t1[((t1['mass'] > 10) & (t1['size'] < 3.6) & (t1['ecc'] < 0.3))] #low mass, size (too large) or non-circular (eccentric) 
#& (t1['particle'] != 1079)

#############IMPORTANT TO UNDERSTAND THAT###########################
#for the filter steps, the filtering is done per row, not pertrajectory. 
#That means if a trajectory had 200 points initially, but 80 of those frames had slightly higher eccentricity or size → those rows get dropped.


# overlapping_particles = [28, 29, 30, 35, 39, 46, 53, 61, 63, 64, 71, 72, 988, 17, 19, 22, 77, 80, 86, 88, 91, 93, 116, 209, 604]

# t2 = t2[~t2['particle'].isin(overlapping_particles)]
#& (t1['particle'] != 20) ###### for excluding a particular particle

# To get the average localization error for the frame:
average_error = t2['ep'].mean()
print(f"Average localization error: {average_error:.4f} pixels")

###############################
###############################
"""Perform the annotation again, and this time we should see an improved selection"""
###############################
############################### 
tp.annotate(t2[t2['frame'] == 0], frames[0]);


###############################
###############################
"""PLOT THE TRAJECTORIES (Tf)"""
###############################
###############################

fig, ax = plt.subplots()
tp.plot_traj(t2, ax= ax);
fig.savefig(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/trajectories/tf_trajectory_' + plot_name + '.png', dpi=300, bbox_inches='tight')
plt.show()

###############################
###############################
"""COMPUTE THE DRIFT (d)"""
###############################
###############################


d = tp.compute_drift(t2) #computes the drift of particles and returns values in pixels.
d.plot()
plt.savefig(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/trajectories/tf_drift_' + plot_name + '.png', dpi=300, bbox_inches='tight')
plt.show()

###############################
###############################
"""PLOT THE TRAJECTORIES HAVING SUBTRACTED THE DRIFT (Tm)"""
###############################
###############################

tm = tp.subtract_drift(t2.copy(), d) #done through MSD calculations as well. 
tm = tm.reset_index(drop = True) #ensures particle is a column, not an index label
# Ensure the figure is created explicitly
d = tp.compute_drift(tm) #computes the drift of particles and returns values in pixels.
d.plot()
fig, ax = plt.subplots()  # Create a new figure and axis for the plot
tp.plot_traj(tm, ax=ax)
fig.savefig(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/trajectories/tm_trajectory_' + plot_name + '.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# """code to determine which trajectory has the least points"""
# Count number of coordinates per particle
not_filtered_counts = tm.groupby('particle').size()
# Filter counts to only include particles with more than 150 points
counts = not_filtered_counts[not_filtered_counts > 150]

# Find the particle ID with the fewest coordinates
min_particle = counts.idxmin()
min_count = counts.min()

print(f"Particle with fewest coordinates: {min_particle} ({min_count} points)")


# tm['dx'] = tm.groupby('particle')['x'].diff()
# tm['dy'] = tm.groupby('particle')['y'].diff()
# tm['step'] = np.sqrt(tm['dx']**2 + tm['dy']**2)
# bad_particles = tm.groupby('particle')['step'].max()
# # threshold = 0.5
# # bad_ids = bad_particles[bad_particles > threshold].index.tolist()
# print(f"Particle with overalap: {bad_particles}")

# max_particle = bad_particles.idxmax()
# max_step = bad_particles.max()
# print(f"Particle {max_particle} had the largest step: {max_step:.3f} µm")
# print(bad_particles.sort_values(ascending=False).head(10))

###############################
###############################
"""Saving"""
###############################
###############################


filename = 'beads_movie_' + name

      

"""Store Data"""
f.to_csv(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/f/f_' + filename + '.csv')
t.to_csv(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/t/t _' + filename + '.csv')
#t1.to_csv(Filepath + '/E_output_data/' + str(bp) + '/tf/tf_' + filename + '.csv') #with spurious particle trajectory removal
t2.to_csv(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/tf/tf_' + filename + '.csv') #with mass/size/ecc analysis particle selective removal
tm.to_csv(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/tm/tm_' + filename + '.csv') #with drift substraction

im = tp.imsd(tm, 16/100, 10)  # microns per pixel = 100/285., frames per second = 24
# print(im)
# print(im.index)

fig, ax = plt.subplots()

#convert the index to an array
index_array = im.index.to_numpy()
# print(index_array)
# print(im.to_numpy())

# Plot the entire DataFrame
ax.plot(index_array, im.to_numpy(), 'k-', alpha=0.1)  # black lines, semitransparent
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
        xlabel='lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/im_plot/im_' + plot_name + '.png', dpi=300, bbox_inches='tight')

em = tp.emsd(tm, 16/100, 10)
fig, ax = plt.subplots()
ax.plot(em.index, em, 'o')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
        xlabel='lag time $t$')
ax.set(ylim=(1e-4, 10));


# Plotting im
fig, ax = plt.subplots()
index_array = im.index.to_numpy()
ax.plot(index_array, im.to_numpy(), 'k-', alpha=0.1)  # black lines, semitransparent

# Plotting em on the same graph
em = tp.emsd(tm, 16/100, 10)
ax.plot(em.index, em, 'o')

# Set plot properties
ax.set(xlabel='lag time $t$',
        ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
        xscale='log', yscale='log',
        ylim=(1e-4, 10))


# Show the plot
plt.savefig(Filepath + '/E_output_data/pEGFP-C1' + str(bp) + '/em_plot/em_' + plot_name + '.png', dpi=300, bbox_inches='tight')
plt.show()

''' Data where we show comparison of emsd of ATP depleted and "normal" emsd '''




'''COLOR MAP SHOWING LINEAR GRADIENT OF PARTICLE TRACK OVER TIME'''

# #Linear Gradient of Colors
# #Generate example linear gradient
# num_points = len(tm)
# #print('NUMBER OF POINTS', num_points)
# gradient = np.linspace(0, 90, num_points)


# #CHECK WHETHER GRADIENT VALUES ARE EVENLY DISTRIBUTED
# # hist, bins = np.histogram(gradient, bins=20)
# # print("Histogram:", hist)

# # Filter the particle trajectories based on the specified ranges
# filtered_data = tm[(tm['x'] >= 220) & (tm['x'] <= 350) & (tm['y'] >= 150) & (tm['y'] <= 250)]

# # Filter the gradient data accordingly
# filtered_gradient = gradient[filtered_data.index]

# # Define colormap for the gradient
# cmap = plt.get_cmap('seismic')

# # Plot the particle tracks with a linear gradient
# fig, ax = plt.subplots()
# for particle_id, traj in filtered_data.groupby('particle'):
#     ax.plot(traj['x'], traj['y'], color=cmap(gradient[particle_id]), alpha=0.5)  # Adjust alpha as needed
# ax.set_xlabel('X')
# ax.set_ylabel('Y')

# # Optionally, add a colorbar
# scalar_mappable = plt.cm.ScalarMappable(cmap=cmap)
# scalar_mappable.set_array(filtered_gradient)
# plt.colorbar(scalar_mappable, ax=ax, label='Time (s)')

# plt.show()