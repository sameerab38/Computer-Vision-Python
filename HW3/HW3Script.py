import numpy as np
import cv2
import Tkinter as tki

from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog

from Tkinter import Label,Tk
from PIL import Image, ImageTk

import os
from matplotlib import pyplot as plt

from scipy.spatial import distance as dist
import glob

import random

import argparse


# initialize the histogram_grey_list dictionary to store the image name
# and corresponding grey histograms and the histogram_color_list dictionary
# to store the color image histograms
histogram_grey_list = {}
histogram_color_list = {}

# Problem 1:
# Convert all color images into grey images and store in greyImages variable.
'''for greyImage in glob.glob("ST2MainHall4" + "/*.jpg"):
	image = cv2.imread(greyImage, 0)
	imageCount = imageCount + 1
	cv2.imwrite("ST2MainHall4_Grey/ST2MainHall400" + str(imageCount) + "/.jpg", image)
	greyImages.append(image)'''


'''# Store all color images in the colorImages variable.
for colorImage in glob.glob("ST2MainHall4" + "/*.jpg"):
	colorImages.append(cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB))'''
def hist_bins(image):
	nparray = np.array(image)
	flattenedarray = nparray.flatten()
	return flattenedarray


# Problem 2:
# For grey histograms:
for imagePath in glob.glob("ST2MainHall4" + "/*.jpg"):
	# extract the image filename (assumed to be unique)
	filename = imagePath[imagePath.rfind("/") + 1:]
	# Read the image as grey image
	image = cv2.imread(imagePath, 0)
	# Use canny edge detector to select edge points
	cannyImageMask = cv2.Canny(image, 100, 250)
	# To create the 8-bit mask
	cannyImageMask = np.uint8(cannyImageMask)
	# Apply bitwise and on the image to mask it
	maskedImage = cv2.bitwise_and(image,image,mask = cannyImageMask)
	# Compute gradients in x and y direction
	sobelXDir = cv2.Sobel(maskedImage, cv2.CV_64F, 1, 0, ksize=5)
	sobelYDir = cv2.Sobel(maskedImage, cv2.CV_64F, 0, 1, ksize=5)	
	# Compute magnitude and theta angle using the gradients
	magnitude, theta = cv2.cartToPolar(sobelXDir, sobelYDir, angleInDegrees = True)
	# To turn theta into hist index
	theta = np.round(np.divide(theta, 10))
	theta = np.uint8(theta)
	# flatten the theta and magnitude arrays
	flattenedTheta = hist_bins(theta)
	flattenedMagnitude = hist_bins(magnitude)
	# Build 36-bin histograms
	hist, bins = np.histogram(flattenedTheta, range(37), weights=flattenedMagnitude)
	histogram_grey_list[filename] = hist
	plt.plot(hist)
	plt.show()
	

# Problem 4 a):
# Histogram intersection
def hist_intersection(h1, h2):
	sum1 = 0.0
	sum2 = 0.0
	for i in range(0, h1.size):
		if h1[i] == 0 and h2[i] == 0:
			sum1 = sum1+1
			sum2 = sum2+1
		else:
			sum1 = sum1+min(h1[i],h2[i])
			sum2 = sum2+max(h1[i],h2[i])
	return (sum1/sum2)

# Problem 4 b):
# Chi-squared measure
def chi_squared_measure(h1, h2):
	sum1 = 0.0
	for i in range(0, h1.size):
		if h1[i] + h2[i] > 250:
			sum1 = sum1+ (((h1[i]-h2[i])**2)/(h1[i]+h2[i]))
	return (sum1)


def normalise_chi_squared_measure(matrix):
    minVal = min([min(value) for value in matrix])
    maxVal = max([max(value) for value in matrix])
    distanceFunction = 255 - 255*((matrix - minVal)/float(maxVal-minVal))
    return distanceFunction


# Problem 5:
# Compare all grey histograms using hist_intersection and store the values in a matrix

keylist_grey = histogram_grey_list.keys()
keylist_grey.sort()

w, h = len(histogram_grey_list), len(histogram_grey_list)
matrix = [[0 for x in range(w)] for y in range(h)]

for i in range(0, len(histogram_grey_list)):
	histogram1 = histogram_grey_list.get(keylist_grey[i])
	for j in range(0, len(histogram_grey_list)):
		histogram2 = histogram_grey_list.get(keylist_grey[j])
		matrix[i][j] = hist_intersection(histogram1, histogram2)
		

# Create heatmap for Histogram intersection comparison
# data to plot, all normal Python lists
x = list()
y = list()
for i in range(0, 99):
	x.append(i)
	y.append(i)



# setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

# convert matrix (list of lists) to a numpy array for plotting
matrix = np.array(matrix)

# plug the data into pcolormesh
plt.pcolormesh(x, y, matrix)
plt.colorbar() #need a colorbar to show the quality of match scale
plt.savefig("histIntCompGrey.png")
plt.show()

# Compare all histograms Chi-squared measure and store the values in a matrix
w, h = len(histogram_grey_list), len(histogram_grey_list)
matrix = np.zeros((w, h))

for i in range(0, len(histogram_grey_list)):
	histogram1 = histogram_grey_list.get(keylist_grey[i])
	for j in range(i, len(histogram_grey_list)):
		histogram2 = histogram_grey_list.get(keylist_grey[j])
		value = chi_squared_measure(histogram1, histogram2)
		matrix[i][j] = value
		matrix[j][i] = value
		if i == j:
			matrix[i][j] = 0


matrix = normalise_chi_squared_measure(matrix)

# Create heatmap for Chi-squared measure comparison
x = list()
y = list()
for i in range(0, 99):
	x.append(i)
	y.append(i)


# setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

# convert matrix (list of lists) to a numpy array for plotting
matrix = np.array(matrix)

# plug the data into pcolormesh
plt.pcolormesh(x, y, matrix)
plt.colorbar() #need a colorbar to show the quality of match scale
plt.savefig("histChiSqrGrey.png")
plt.show()

# Problem 3:
# For color histograms:
# Split into three color channels and combine them and store in different variables say x and y.
# Now, repeat the above gray images procedure on these.
for imagePath in glob.glob("ST2MainHall4" + "/*.jpg"):
	# extract the image filename (assumed to be unique)
	filename = imagePath[imagePath.rfind("/") + 1:]
	# Read the image as color image
	image = cv2.imread(imagePath, 1)
	# Use canny edge detector to select edge points
	cannyImageMask = cv2.Canny(image, 100, 250)
	# To create the 8-bit mask
	cannyImageMask = np.uint8(cannyImageMask)
	# Apply bitwise and on the image to mask it
	maskedImage = cv2.bitwise_and(image,image,mask = cannyImageMask)
	# Split the image into individual color channels
	b, g, r = cv2.split(maskedImage)
	# compute gradient got each channel's x and y components
	bx = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=5)
	gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)
	rx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=5)
	by = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=5)
	gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5)
	ry = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=5)
	# Compute vector components in x and y direction to determine edge orientation
	vectorXDir = bx + gx + rx
	vectorYDir = by + gy + ry
	magnitude, theta = cv2.cartToPolar(vectorXDir, vectorYDir, angleInDegrees=True)
	# To turn theta into hist index
	theta = np.round(np.divide(theta, 10))
	theta = np.uint8(theta)
	# flatten the theta and magnitude arrays
	flattenedTheta = hist_bins(theta)
	flattenedMagnitude = hist_bins(magnitude)
	# Construct Histogram
	hist, bins = np.histogram(flattenedTheta, range(37), weights=flattenedMagnitude)
	histogram_color_list[filename] = hist
	plt.plot(hist)
	plt.show()


# Problem 5 - second half:
# Compare all color histograms using hist_intersection and store the values in a matrix
keylist_color = histogram_color_list.keys()
keylist_color.sort()

w, h = len(histogram_color_list), len(histogram_color_list)
matrix = [[0 for x in range(w)] for y in range(h)]

for i in range(0, len(histogram_color_list)):
	histogram1 = histogram_color_list.get(keylist_color[i])
	for j in range(0, len(histogram_color_list)):
		histogram2 = histogram_color_list.get(keylist_color[j])
		matrix[i][j] = hist_intersection(histogram1, histogram2)


# Create heatmap for Histogram intersection comparison
# data to plot, all normal Python lists
x = list()
y = list()
for i in range(0, 99):
	x.append(i)
	y.append(i)



# setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

# convert matrix (list of lists) to a numpy array for plotting
matrix = np.array(matrix)

# plug the data into pcolormesh
plt.pcolormesh(x, y, matrix)
plt.colorbar() #need a colorbar to show the quality of match scale
plt.savefig("histIntCompColor.png")
plt.show()


# Compare all histograms Chi-squared measure and store the values in a matrix
w, h = len(histogram_color_list), len(histogram_color_list)
matrix = np.zeros((w,h))
for i in range(0, len(histogram_color_list)):
	histogram1 = histogram_color_list.get(keylist_color[i])
	for j in range(i, len(histogram_color_list)):
		histogram2 = histogram_color_list.get(keylist_color[j])
		value = chi_squared_measure(histogram1, histogram2)
		matrix[i][j] = value
		matrix[j][i] = value
		if i == j:
			matrix[i][j] = 0



matrix = normalise_chi_squared_measure(matrix)

# Create heatmap for Chi-squared measure comparison
x = list()
y = list()
for i in range(0, 99):
	x.append(i)
	y.append(i)



# setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

# convert matrix (list of lists) to a numpy array for plotting
matrix = np.array(matrix)

# plug the data into pcolormesh
plt.pcolormesh(x, y, matrix)
plt.colorbar() #need a colorbar to show the quality of match scale
plt.savefig("histChiSqrColor.png")
plt.show()

# Extra credit:
# Problem 2 and problem 3
histogram_color_list_eigen = {}

for imagePath in glob.glob("ST2MainHall4" + "/*.jpg"):
	# extract the image filename (assumed to be unique)
	filename = imagePath[imagePath.rfind("/") + 1:]
	# Read the image as color image
	image = cv2.imread(imagePath, 1)
	edge_magnitude, edge_orientation = coloredges(image)
	# To turn theta into hist index
	edge_orientation = np.round(np.divide(edge_orientation, 10))
	edge_orientation = np.uint8(edge_orientation)	


def coloredges(image):
	cannyImageMask = cv2.Canny(image, 100, 250)
	# To create the 8-bit mask
	cannyImageMask = np.uint8(cannyImageMask)
	# Apply bitwise and on the image to mask it
	maskedImage = cv2.bitwise_and(image,image,mask = cannyImageMask)
	# Split the image into individual color channels
	b, g, r = cv2.split(maskedImage)
	# compute gradient got each channel's x and y components
	bx = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=5)
	gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)
	rx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=5)
	by = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=5)
	gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5)
	ry = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=5)
	Jx = np.multiply(rx, rx) + np.multiply(gx, gx) + np.multiply(bx, bx)
	Jy = np.multiply(ry, ry) + np.multiply(gy, gy) + np.multiply(by, by)
	Jxy = np.multiply(rx, ry) + np.multiply(gx, gy) + np.multiply(bx, by)
	# compute first (greatest) eigenvalue of 2x2 matrix J'*J.
	# note that the abs() is only needed because some values may be slightly
	# negative due to round-off error.
	d = np.sqrt(abs(np.multiply(Jx, Jx) - 2*(np.multiply(Jx, Jy)) + np.multiply(Jy, Jy) + 4*(np.multiply(Jxy, Jxy))))
	ev1 = (Jx + Jy + d) / 2
	# the 2nd eigenvalue would be:  ev2 = (Jx + Jy - D) / 2;
	edge_magnitude = np.sqrt(ev1)
	#compute edge orientation (from eigenvector tangent)	
	edge_orientation = np.arctan2(-Jxy, (ev1 - Jy))
	return edge_magnitude, edge_orientation



# References
# https://stackoverflow.com/questions/8251111/collecting-results-from-a-loop-that-returns-numpy-arrays
# https://docs.opencv.org/3.4.1/da/d22/tutorial_py_canny.html
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
# https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#ga4e0972be5de079fed4e3a10e24ef5ef0
# https://stackoverflow.com/questions/16661790/difference-between-plt-close-and-plt-clf
# https://docs.opencv.org/3.3.1/d1/db7/tutorial_py_histogram_begins.html
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
# https://matplotlib.org/examples/pylab_examples/quiver_demo.html
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
# https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html
# https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
# https://www.mathworks.com/matlabcentral/fileexchange/28114-fast-edges-of-a-color-image--actual-color--not-converting-to-grayscale-


