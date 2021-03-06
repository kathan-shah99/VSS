#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:38:20 2021

@author: riken
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

# Define color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a boolean OR with the "|" character to identify
# pixels below the thresholds
thresholds = (image[:, :, 0] < rgb_threshold[0]) | (
    image[:, :, 1] < rgb_threshold[1]) | (image[:, :, 2] < rgb_threshold[2])
color_select[thresholds] = [0, 0, 0]

# Display the image
titles = ['Input Image', 'Color Selected Image']
images = [image, color_select]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
