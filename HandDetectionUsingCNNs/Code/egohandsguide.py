# Author: sbammidi
# Date created: 04/25/2018

# This script is used to convert the polygon.mat files into individual .txt files with bounding box coordinate values
# Run this recursively on 48 folders to produce results by changing the folder name in the image path.
import numpy as np
import Image

import cv2

import os
from matplotlib import pyplot as plt
import scipy.io as sio
import matplotlib as mpl
from os.path import basename
from scipy.spatial import distance as dist
import glob

import random

import xlsxwriter

import matplotlib.cm as cm


# len(mat.get("polygons")[0]) = 100
# len(mat.get("polygons")[0,0]) = 4
# a total of 100 images for each folder
'''mat.get("polygons")[0,0]["myleft"] # This is for the first image
mat.get("polygons")[0,0]["myright"]
mat.get("polygons")[0,0]["yourleft"]
mat.get("polygons")[0,0]["yourright"]'''

rxryfilename = {}
for imagePath in glob.glob("ProjectDataSet/egohands_data/_LABELLED_SAMPLES/PUZZLE_OFFICE_T_S" + "/*.jpg"):
	filename = imagePath[imagePath.rfind("/") + 1:]
	filename = "PUZZLE_OFFICE_T_S"+filename
	im = Image.open(imagePath)
	im1 = im.resize((416,416),Image.ANTIALIAS)
	im1.save("ProjectDataSet/egohands_data/_LABELLED_SAMPLES/PUZZLE_OFFICE_T_S/images/"+filename)
	rx = float(float(im1.size[0])/float(im.size[0]))
	ry = float(float(im1.size[1])/float(im.size[1]))
	rxryfilename[filename] = rx, ry

def getXAndYArrays(poly, rx, ry):
	x = []
	y=[]
	minx = 0
	miny = 0
	maxx = 0
	maxy = 0
	for i in range(0,len(poly)):
		if(len(poly) > 1):
			x.append(poly[i][0])
		if(len(poly) > 1):
			y.append(poly[i][1])
	if(len(x) > 0):
		minx = min(x)
		maxx = max(x)
	if(len(y) > 0):
		miny = min(y)
		maxy = max(y)
	return minx * rx,miny * ry,maxx * rx,maxy * ry

mat = sio.loadmat("ProjectDataSet/egohands_data/_LABELLED_SAMPLES/PUZZLE_OFFICE_T_S/polygons.mat")
fileNames = {}

for imagePath in glob.glob("ProjectDataSet/egohands_data/_LABELLED_SAMPLES/PUZZLE_OFFICE_T_S/images"+"/*.jpg"):
	filename = imagePath[imagePath.rfind("/") + 1:]
	fn = os.path.splitext(filename)[0]
	fileNames[filename] = fn

fileNamesKeyList = fileNames.keys()
fileNamesKeyList.sort()

# Read the images and store the file names without the jpg extension in a list
txtnames = []
for i in range(0,len(fileNames)):
	name = fileNames.get(fileNamesKeyList[i])
	txtnames = np.append(txtnames, name)

for l in range(0, len(mat.get("polygons")[0])):
	strings = []
	rx, ry = rxryfilename.get(txtnames[l]+".jpg")
	mlminx, mlminy, mlmaxx, mlmaxy = getXAndYArrays(mat.get("polygons")[0,l]["myleft"], rx, ry)
	string = str(mlminx)+" "+str(mlminy)+" "+str(mlmaxx)+" "+str(mlmaxy)
	strings = np.append(strings, str(0))
	strings = np.append(strings, string)
	mrminx, mrminy, mrmaxx, mrmaxy = getXAndYArrays(mat.get("polygons")[0,l]["myright"], rx, ry)
	string = str(mrminx)+" "+str(mrminy)+" "+str(mrmaxx)+" "+str(mrmaxy)
	strings = np.append(strings, str(0))
	strings = np.append(strings, string)
	ylminx, ylminy, ylmaxx, ylmaxy = getXAndYArrays(mat.get("polygons")[0,l]["yourleft"], rx, ry)
	string = str(ylminx)+" "+str(ylminy)+" "+str(ylmaxx)+" "+str(ylmaxy)
	strings = np.append(strings, str(0))
	strings = np.append(strings, string)
	yrminx, yrminy, yrmaxx, yrmaxy = getXAndYArrays(mat.get("polygons")[0,l]["yourright"], rx, ry)
	string = str(yrminx)+" "+str(yrminy)+" "+str(yrmaxx)+" "+str(yrmaxy)
	strings = np.append(strings, str(0))
	strings = np.append(strings, string)	
	outFile = open("ProjectDataSet/egohands_data/_LABELLED_SAMPLES/PUZZLE_OFFICE_T_S/oldtxts/"+txtnames[l]+".txt", "w")
	for line in strings:
		outFile.write(line)
		outFile.write("\n")
	outFile.close()



