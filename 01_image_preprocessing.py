# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:19:50 2019

@author: amuralidharan
"""
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
print('Imported __future__')
import pims
import os
import numpy as np
print('Numpy imported')
from scipy.ndimage.filters import uniform_filter1d, correlate1d
from PIL import Image
from skimage import color
import imageio
import matplotlib.pyplot as plt
print('All imports finished')


def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


def gaussian_kernel(sigma, truncate = 4.0):
    "1D discretized gaussian"
    lw = int(truncate * sigma + 0.5)      #calculate the length of the gaussian_kernel
    x = np.arange(-lw, lw+1)              #convert into a list
    result = np.exp(x**2/(-2*sigma**2))   #calculate the gaussian curve
    return result / np.sum(result)        #normalize by the summation


"""Gaussian filter"""
def lowpass(image, sigma=1, truncate=4):
    sigma = validate_tuple(sigma, image.ndim) #convert sigma to 2 dimension
    result = np.array(image, dtype=float)  #convert image to np.float
    for axis, _sigma in enumerate(sigma):     #convolve gaussian about the x and y axis 
        if _sigma > 0:
            correlate1d(result, gaussian_kernel(_sigma, truncate), axis,
                        output=result, mode='constant', cval=0.0)
    return result


"""Boxcar average"""
def boxcar(image, size):
    size = validate_tuple(size, image.ndim)  #convert size in 2 dimension
    if not np.all([x & 1 for x in size]):    #check if size is odd integer
        raise ValueError("Smoothing size must be an odd integer. Round up.")
    result = image.copy()                    #store image as results
    for axis, _size in enumerate(size):      #boxcar average the image about the x and y axis 
        if _size > 1:
            uniform_filter1d(result, _size, axis, output=result,
                             mode='nearest', cval=0)
    return result


def bandpass(image, lshort, llong, threshold=None, truncate=4):
    lshort = validate_tuple(lshort, image.ndim)             #size of the gaussian kernel
    llong = validate_tuple(llong, image.ndim)               #size of the boxcar average
    if np.any([x >= y for (x, y) in zip(lshort, llong)]): 
        raise ValueError("The smoothing length scale must be larger than " +
                         "the noise length scale.")
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):          #Clip bandpass for integer type images
            threshold = 1
        else:
            threshold = 1/250.                              #Clip bandpass for float images 
    gauss      = lowpass(image, lshort, truncate)           #Perfrom gaussian filter
    background = boxcar(image, llong)                       #Calculate background
    result1    = gauss - background                         #Substract background
    result2    = np.where(result1 >= threshold, result1, 0) #Apply bandpass filter 
    return result2, result1, background, gauss              #Return processed image, gaussian fit, and background


#experiment = ['Experiment1.1', 'Experiment1.2', 'Experiment1.3'] # Enter the folder name

N = 800 # Number of frames 
filepath = r'F:\A_Data\A_DNATranslocation\Experiments 2025\DNA Tracking\Corrected_Data\Control Experiments_Beads\1 um\A_rawtif' #where the original TIFs are located
#directory = os.path.join(filepath, str(experiment))
#print(os.listdir(directory))
filepath2 = r'F:\A_Data\A_DNATranslocation\Experiments 2025\DNA Tracking\Corrected_Data\Control Experiments_Beads\1 um\B_ProcessedData' #where the processed files are saved
#directory2 = os.path.join(filepath2, str(experiment))

# Automatically detect experiment folders
experiment_folders = [d for d in os.listdir(filepath) if d.startswith('Exp') and os.path.isdir(os.path.join(filepath, d))]


for bp in experiment_folders:
    directory = os.path.join(filepath, bp)
    directory2 = os.path.join(filepath2, bp)
    
    output_check_file = os.path.join(directory2, f'mov_{bp}_0.tiff')
    if os.path.exists(output_check_file):
        print(f'Skipping {bp} — already processed.')
        continue  # Skip this folder

    if not os.path.exists(directory2):
        os.makedirs(directory2)

    print(f'Processing experiment: {bp}')

    # Load the image sequence
    frames = pims.ImageSequence(os.path.join(directory, '*.tif'))  # Assuming .tif images
    total_frames = len(frames)
    print(f'Total frames in video: {total_frames}')

    frames_to_process = min(N, total_frames)

    for i in range(frames_to_process):
        image = frames[i]
        a, b, c, d = bandpass(image, 1, 15, threshold=40)
        # print("Background values:", c)
        # plt.imshow(c, cmap='gray')
        # plt.title("Background")
        # plt.colorbar()
        # plt.show()
        im = Image.fromarray(a.astype(np.uint16))  # Cast to proper type before saving if needed
        im.save(os.path.join(directory2, f'mov_{bp}_{i}.tiff'))