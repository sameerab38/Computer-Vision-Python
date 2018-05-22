import numpy as np
import cv2
import Tkinter as tki

from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog

from Tkinter import Label,Tk
from PIL import Image, ImageTk

import os
from matplotlib import pyplot as plt

import matplotlib as mpl

from scipy.spatial import distance as dist
import glob

import random

import xlsxwriter

import argparse
import matplotlib.cm as cm


# Problem 1:
# Apply functions listed in “Structural Analysis and Shape Descriptors” section of OpenCV documentation
binaryImages = {}
contourVertices = {}
conctContourVertices = {}
cnvDefects = {}
numberOfDeficits = {}
areaOfDeifcits = {}
imageArea = {}
hullArea = {}
imagePerimeter = {}
imageMoments = {}
hullMoments = {}
distances = {}
epsilon = 1.414

def allVertices(contours):
	vertices = {}
	for i in range(0, len(contours)):
		cnt = contours[i]
		vertices[i] = cv2.approxPolyDP(cnt, epsilon, True)
	return vertices

def hullLines(hullVertices, img):
	for i in range(0, len(hullVertices)-1):
		cv2.line(img,(hullVertices[i][0][0],hullVertices[i][0][1]),(hullVertices[i+1][0][0],hullVertices[i+1][0][1]),(255,0,0))
		if (i == len(hullVertices)-2):
			cv2.line(img,(hullVertices[0][0][0],hullVertices[0][0][1]),(hullVertices[len(hullVertices)-1][0][0],hullVertices[len(hullVertices)-1][0][1]),(255,0,0))
	return img

def allContours(contours):
	st = contours[0]
	for i in range(1, len(contours)):
		st = np.concatenate((st, contours[i]))
	return st

def rgba2rgb( RGBA_color , RGB_background=(255,255,255)):
    	alpha = RGBA_color[3]
    	return ((1 - alpha) * RGB_background[0] + alpha * RGBA_color[0],(1 - alpha) * RGB_background[1] + alpha * RGBA_color[1],(1 - alpha) * RGB_background[2] + alpha * RGBA_color[2])

def getXAndYArrays(conct):
	x = []
	y=[]
	for i in range(0,len(conct)):
		x.append(conct[i][0])
		y.append(conct[i][1])
	return x,y

def chamferMatching(dist, contourVertices):
	score = 0
	scoreArray = np.array([[]])
	x,y = getXAndYArrays(contourVertices)
	minx = np.amin(x)
	maxx = np.amax(x)
	width = 1 + maxx - minx
	height = 128 # Though it is maxy - miny, we consider the height of the distance tansform
	# Create a matrix of zeros with height = 128 and width = maxx - minx
	matrix = [[0 for x in range(width)] for y in range(height)]
	# Loop and place 1s in the contour vertex positions
	for v in range(0,len(contourVertices)):
		matrix[contourVertices[v][1]][contourVertices[v][0]-minx-1] = 1
	# Now slide the matrix window through dist window and get element wise multiplication and product.
	for d in range(0, len(dist)-len(matrix[0])):
		choppedDist = dist[0:128, d:d+len(matrix[0])]
		mul = np.multiply(choppedDist,matrix)
		add = mul.sum()
		scoreArray = np.append(scoreArray, add)
	score = min(scoreArray)
	return score

def curvatureCalc(contourArray):
	curvatures = np.array([[]])
	for c in range(0, len(contourArray)):
		curv = 0
		t = contourArray[[(c-2)%(len(contourArray)), (c)%(len(contourArray)), (c+2)%(len(contourArray))]] # xt = a0 + a1*t + a2*(t**2), yt = b0 + b1*t + b2*(t**2)
		xtN2 = t[0][0]
		ytN2 = t[0][1]
		a0 = t[1][0]
		b0 = t[1][1]
		xt2 = t[2][0]
		yt2 = t[2][1]
		a1 = ((xt2 - a0)-(xtN2 - a0))/4
		a2 = ((xt2 - a0)+(xtN2 - a0))/8
		b1 = ((yt2 - b0)-(ytN2 - b0))/4
		b2 = ((yt2 - b0)+(ytN2 - b0))/8
		if(a1 != 0 and b1 != 0):		
			curv = 2*(a1*b2 - b1*a2)/(a1**2+b1**2)**1.5
		curvatures = np.append(curvatures, curv)
	return curvatures


# Find and display image boundaries/contours.
for imagePath in glob.glob("GaitImages" + "/*.png"):
	# extract the image filename (assumed to be unique)
	filename = imagePath[imagePath.rfind("/") + 1:]
	# Read the image
	image = cv2.imread(imagePath, 0)
	imageCopy = image.copy()
	imageCopy = cv2.cvtColor(imageCopy,cv2.COLOR_GRAY2BGR)
	imageCopy2 = imageCopy.copy()
	imageCopy3 = imageCopy.copy()
	imageCopy4 = imageCopy.copy()
	imageCopy5 = imageCopy.copy()
	# convert to binary image
	ret, binaryImage = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
	binaryImages[filename] = binaryImage
	# find contours
	contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	contourVertices[filename] = contours
	# Problem 3:
	# Concatinate all the contours and store into single array
	conct = allContours(contours)
	# Reshape
	conct = np.reshape(conct, (-1,2))
	# Compute curvature with k=2
	curvature = curvatureCalc(conct)
	conctContourVertices[filename] = conct
	# Compute local maxima of the curvature
	pc = curvature[np.where(curvature > 0.5)]
	area2 = np.pi*20
	# Get vertex arrays
	x,y = getXAndYArrays(conct)
	x1,y1 = getXAndYArrays(conct[np.where(curvature > 0.5)])
	area1 = np.pi*6
	# Plot the hotness. Show the hotness index and save the plot.
	plt.scatter(y,x, s=area1, c=curvature, cmap="jet", alpha=0.5)
	plt.colorbar()
	plt.scatter(y1,x1, s=area2, c=pc, alpha=0.5)
	plt.xlabel("y")
	plt.ylabel("x")
	plt.title(filename)
	# plt.show()
	plt.savefig("ContourCurvatureColorImages/" + filename)
	plt.gcf().clear()
	# boundaries: draw contours (parameters: image, contours array, contours to draw, color, thickness)
	cv2.drawContours(imageCopy, contours, -1, (0, 255, 0))
	cv2.imwrite("ContourImages/" + filename, imageCopy)
	# Problem 4:
	# Compute Eucledian distance transform for each boundary and save the results
	blankImage = np.zeros((128,128,3), np.uint8)
	cv2.drawContours(blankImage, contours, -1, (0, 255, 0))
	blankImage = cv2.cvtColor(blankImage,cv2.COLOR_BGR2GRAY)
	ret2, binaryImage2 = cv2.threshold(blankImage, 5, 255, cv2.THRESH_BINARY)
	distance = cv2.distanceTransform(255-binaryImage2, cv2.cv.CV_DIST_L2, 3)
	distances[filename] = distance
	plot = plt.imshow(distance)
	plot.set_cmap("bone")
	# plt.show()
	plt.savefig("DistanceTransform/"+filename)
	plt.gcf().clear()
	# Find polygonal approximation of computed boundaries
	vertices = allVertices(contours)
	# visualize the vertices
	cv2.drawContours(imageCopy2, vertices.values(), 0, (0, 0, 255))
	cv2.imwrite("VerticesImages/" + filename, imageCopy2)
	# Compute convex hull
	hullVertices = cv2.convexHull(contours[0], True, returnPoints = True)
	hullIndices = cv2.convexHull(contours[0], True, returnPoints = False)
	# visualize the hull
	cv2.drawContours(imageCopy3, hullVertices, -1, (255, 0, 0))
	# draw lines, given vertices
	imageCopy3 = hullLines(hullVertices, image)
	cv2.imwrite("HullImages/" + filename, imageCopy3)
	# Compute deficits of convexity for the shapes
	defects = cv2.convexityDefects(contours[0],hullIndices) # [start_point,end_point,far_point,distance]
	cnvDefects[filename] = defects
	# Compute area, perimeter
	area = cv2.contourArea(allContours(contours))
	imageArea[filename] = area
	imagePerimeter[filename] = np.float32(cv2.arcLength(contours[0],True))
	# Compute convex hull area
	convexHullArea = cv2.contourArea(hullVertices)
	hullArea[filename] = convexHullArea
	# Compute number of convexity deficits
	numberOfDeficits[filename] = len(defects)
	# Compute convexity deficits area = difference between the convex hull area and the object area
	areaOfDeifcits[filename] = convexHullArea - area
	# Compute all first and second order image moments for the original image
	allMoments = cv2.moments(contours[0]) # Returns a dictionary of all moment values calculated
	imageMoments[filename] = allMoments
	# Compute all first and second order image moments for the convex hull
	allMomentsHull = cv2.moments(hullVertices)
	hullMoments[filename] = allMomentsHull

# Problem 2:
# Sort all the keys
imageAreaKeylist = imageArea.keys().sort
imagePerimeterKeylist = imagePerimeter.keys().sort()
imageMomentsKeylist = imageMoments.keys().sort()
numberOfDeficitsKeyist = numberOfDeficits.keys().sort()
areaOfDeifcitsKeyist = areaOfDeifcits.keys().sort()

# Create a table with computed values (area, perimeter, moments: m01, m10, m11, m02, m20) for all frames:
# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('ComputedValues.xlsx')
worksheet = workbook.add_worksheet()
# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

values = ("Image Name", "Area", "Perimeter", "M01", "M10", "M11", "M02", "M20", "NumberOfCDs", "AreaOfCDs")
for value in (values):
	worksheet.write(row, col, value)
	col += 1

# Display the computed features
row = 1
col = 0
# Save the image area
for key, value in (imageArea.iteritems()):
	worksheet.write(row, col, key)
	worksheet.write(row, col + 1, value)
	row += 1

row = 1
col = 2
# Save the image perimeter
for key, value in (imagePerimeter.iteritems()):
	worksheet.write(row, col, value)
	row += 1

row = 1
col = 3
# Save Moments M01
for key, value in (imageMoments.iteritems()):
	worksheet.write(row, col, value["m01"])
	row += 1

row = 1
col = 4
# Save Moments M10
for key, value in (imageMoments.iteritems()):
	worksheet.write(row, col, value["m10"])
	row += 1

row = 1
col = 5
# Save Moments M11
for key, value in (imageMoments.iteritems()):
	worksheet.write(row, col, value["m11"])
	row += 1

row = 1
col = 6
# Save Moments M02
for key, value in (imageMoments.iteritems()):
	worksheet.write(row, col, value["m02"])
	row += 1

row = 1
col = 7
# Save Moments M20
for key, value in (imageMoments.iteritems()):
	worksheet.write(row, col, value["m20"])
	row += 1

# For deficits of convexity compute the number and their total area (include  the number of deficits, and the area the deficits adds to the original contour, into our Data Table)
row = 1
col = 8
# Save Number Of Convexity Deficits
for key, value in (numberOfDeficits.iteritems()):
	worksheet.write(row, col, value)
	row += 1

row = 1
col = 9
# Save Area Of Convexity Deficits
for key, value in (areaOfDeifcits.iteritems()):
	worksheet.write(row, col, value)
	row += 1

workbook.close()


# Problem 5:
keylist1 = conctContourVertices.keys()
keylist2 = distances.keys()
keylist1.sort()
keylist2.sort()

# Store the chamfer match values in an array for each pair
w, h = len(distances), len(distances)
matrix = [[0 for x in range(w)] for y in range(h)]

for i in range(0, len(distances)):
	dist = distances.get(keylist2[i])
	for j in range(i, len(conctContourVertices)):
		contourVertices = conctContourVertices.get(keylist1[j])
		score = chamferMatching(dist, contourVertices)
		matrix[i][j] = score
		matrix[j][i] = score

# Create heatmap for Chamfer Matching comparison
x = list()
y = list()
for i in range(0, 126):
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
# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=structural%20analysis%20shape%20descriptors
# https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html
# https://www.programcreek.com/python/example/89328/cv2.approxPolyDP
# http://xlsxwriter.readthedocs.io/tutorial01.html
# https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops
# https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
# https://pythonspot.com/matplotlib-scatterplot/
# https://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
# https://stackoverflow.com/questions/42940043/set-colorbar-color-in-matplotlib
# https://stackoverflow.com/questions/24029401/python-attributeerror-module-object-has-no-attribute-dist-l2
# https://github.com/eyantrainternship/eYSIP_2015_Marker_based_Robot_Localisation/wiki/Distance-Transformation
# https://stackoverflow.com/questions/12881926/create-a-new-rgb-opencv-image-using-python
# https://gis.stackexchange.com/questions/140245/remove-values-in-list-using-logical-operators
# https://stackoverflow.com/questions/10970821/extract-part-of-2d-list-matrix-list-of-lists-in-python
# https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.matrix.sum.html
# https://stackoverflow.com/questions/3499026/find-a-minimum-value-in-an-array-of-floats



