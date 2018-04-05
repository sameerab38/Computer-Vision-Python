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

# Exercise 1.1
# 1. Load a colour (RGB) image I in a lossless data format, such as bmp , png , or tiff , and display it on a screen.
root = Tk()
root.filename = tkFileDialog.askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("ppm files","*.ppm"),("all files","*.*")))
# os.getcwd() - returns current working directory.

im = Image.open(root.filename)
tki = ImageTk.PhotoImage(im)
myvar = Label(root,image = tki)
myvar.image = tki
myvar.pack()

# 2. Display the histograms of all three colour channels of I.

# Save it as a color image
img = cv2.imread(root.filename,1)

# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
# channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
# mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. 
# histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
# ranges : this is our RANGE. Normally, it is [0,256].

# individual histograms
histb = cv2.calcHist([img],[0],None,[256],[0,256]) # Histogram for blue channel
plt.plot(histb, color = 'b')
plt.show()

histg = cv2.calcHist([img],[1],None,[256],[0,256]) # Histogram for green channel
plt.plot(histg, color = 'g')
plt.show()

histr = cv2.calcHist([img],[2],None,[256],[0,256]) # Histogram for red channel
plt.plot(histr, color = 'r')
plt.show()

# 3

def intensity(b, g, r): 
  print ((np.int(b) + np.int(g) + np.int(r)) / 3)

# event - Type of the mouse event. e.g: EVENT_MOUSEMOVE
# x - x coordinate of the mouse event
# y - y coordinate of the mouse event
# flags - Specific condition whenever a mouse event occurs. eg: EVENT_FLAG_ALTLKEY
# userdata - Any pointer passes to the "setMouseCallback" function as the 3rd parameter

def rectangle_window(event,x,y,flags,param):
    i = random.randint(1, 20)
    if event == cv2.EVENT_MOUSEMOVE:
        pixels = [y,x]
	print 'Pixel position:'
        print(pixels)
        bgrchannels = img[y,x]
	print 'Intensities of b g r channels:'
        print (bgrchannels)
        [b,g,r] = bgrchannels
	print 'average intensity:'
        intensity(b,g,r)
    if event == cv2.EVENT_LBUTTONDOWN:
        crop_img = img[y-5:y+5, x-5:x+5] # To produce a 11 x 11 image on top of it
	cv2.imwrite('crop_img'+str(random.randint(1, 20))+'.png',crop_img)
	#Show the mean and standard deviation
	mean, std = cv2.meanStdDev(crop_img)
	print 'mean of the 3 color channels b, g, r:'
	print mean
	print "Standard Deviation of 3 color channels b, g, r:"
	print std
	img_scaled = cv2.resize(crop_img,None,fx=4, fy=4, interpolation = cv2.INTER_LINEAR)
	constant= cv2.copyMakeBorder(img_scaled,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,255,0)) # To create a border	
	cv2.imshow("cropped", constant)
	cv2.waitKey(1000)


cv2.namedWindow('image')
cv2.setMouseCallback('image',rectangle_window)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
	
cv2.destroyAllWindows()

# Produce histograms for the cropped image
img1 = cv2.imread('crop_img6_Img7.png',1)
histb = cv2.calcHist([img1],[0],None,[256],[0,256]) # Histogram for blue channel
plt.plot(histb, color = 'b')
plt.show()

histg = cv2.calcHist([img1],[1],None,[256],[0,256]) # Histogram for green channel
plt.plot(histg, color = 'g')
plt.show()

histr = cv2.calcHist([img1],[2],None,[256],[0,256]) # Histogram for red channel
plt.plot(histr, color = 'r')
plt.show()

# Produce histograms for the cropped image
img2 = cv2.imread('crop_img9_Img9.png',1)
histb = cv2.calcHist([img2],[0],None,[256],[0,256]) # Histogram for blue channel
plt.plot(histb, color = 'b')
plt.show()

histg = cv2.calcHist([img2],[1],None,[256],[0,256]) # Histogram for green channel
plt.plot(histg, color = 'g')
plt.show()

histr = cv2.calcHist([img2],[2],None,[256],[0,256]) # Histogram for red channel
plt.plot(histr, color = 'r')
plt.show()

# 4
# Refer to the report

# Exercise 1.2
# A

	'''for h in range(0, height):
	    for w in range(0, width):
		rgbchannels = image[h,w]
        	[r,g,b] = rgbchannels
		#value = [(np.int(r)/32)*64 + (np.int(g)/32)*8 + np.int(b)/32]
		#value = [(np.int(r)>>5)<<6 + (np.int(g)>>5)<<3 + np.int(b)>>5]
		value = ((r>>5)<<6) + ((g>>5)<<3) + (b>>5)  		
		colors.append(value)'''

def hist_bins(image):
	r,g,b = cv2.split(image)
	# colors = list()
	value = (((np.int16(r))>>5)<<6) + (((np.int16(g))>>5)<<3) + ((np.int16(b))>>5)
	nparray = np.array(value)
	flattenedarray = nparray.flatten()
	return flattenedarray

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}


# loop over the image paths
for imagePath in glob.glob("ST2MainHall42" + "/*.jpg"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	# By default, OpenCV stores images in BGR format rather than RGB.
	# Matplotlib is used to display results, and it assumes the image is in RGB format.
	# To remedy this, a call to cv2.cvtColor is made to convert the image from BGR to RGB.
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	colorlist = hist_bins(images[filename])
	hist, bins = np.histogram(colorlist, range(513))
	index[filename] = hist
	plt.plot(hist)
	plt.show()
	
# B
# a) Histogram intersection
# cv2.compareHist(H1, H2, method) - This method is different from the formula given in the HW2 description.
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

# b) Chi-squared measure
def chi_squared_measure(h1, h2):
	sum1 = 0.0
	for i in range(0, h1.size):
		if h1[i] + h2[i] > 5:
			sum1 = sum1+ (((h1[i]-h2[i])**2)/(h1[i]+h2[i]))
	return (sum1)

# C
# Compare all histograms using hist_intersection and store the values in a matrix

keylist = index.keys()
keylist.sort()

w, h = len(index), len(index)
matrix = [[0 for x in range(w)] for y in range(h)]

for i in range(0, len(index)):
	histogram1 = index.get(keylist[i])
	for j in range(0, len(index)):
		histogram2 = index.get(keylist[j])
		matrix[i][j] = hist_intersection(histogram1, histogram2)
		# matrix[j][i] = value
		



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
plt.show()

# Compare all histograms Chi-squared measure and store the values in a matrix
w, h = len(index), len(index)
matrix = [[0 for x in range(w)] for y in range(h)]

for i in range(0, len(index)):
	histogram1 = index.get(keylist[i])
	for j in range(i, len(index)):
		histogram2 = index.get(keylist[j])
		value = chi_squared_measure(histogram1, histogram2)
		matrix[i][j] = value
		matrix[j][i] = value
		if i == j:
			matrix[i][j] = 0


# print matrix

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
plt.show()

# References
# https://pythonspot.com/tk-file-dialogs/
# https://stackoverflow.com/questions/22802989/displaying-the-selected-image-using-tkinter
# https://docs.opencv.org/3.3.1/d1/db7/tutorial_py_histogram_begins.html
# https://stackoverflow.com/questions/23596511/how-to-save-mouse-position-in-variable-using-opencv-and-python
# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
# https://www.programcreek.com/python/example/89359/cv2.meanStdDev
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# https://wiki.python.org/moin/BitwiseOperators
# https://wiki.python.org/moin/BitManipulation
# https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
# https://www.tutorialspoint.com/python/dictionary_items.htm
# https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
# https://gist.github.com/teechap/9c066a9ab054cc322877

