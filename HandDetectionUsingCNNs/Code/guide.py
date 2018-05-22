# Author: sbammidi
# Date created: 04/25/2018
# This script is used to convert the annotation .mat files into individual .txt files with bounding box coordinate values
import numpy as np
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
import Image

'''rxryfilename = {}
for imagePath in glob.glob("ProjectDataSet/training_dataset/training_data/images" + "/*.jpg"):
	filename = imagePath[imagePath.rfind("/") + 1:]
	im = Image.open(imagePath)
	im1 = im.resize((416,416),Image.ANTIALIAS)
	# im1.save("ProjectDataSet/training_dataset/training_data/images1/"+filename)
	rx = float(float(im1.size[0])/float(im.size[0]))
	ry = float(float(im1.size[1])/float(im.size[1]))
	rxryfilename[filename] = rx, ry'''

# For Oxford training_dataset, test_dataset and validation_dataset
# Get all the .mat files from yolo_exp/darknet/ProjectDataSet/training_dataset/training_data and read them and convert them to bounding box text file format
convertedTrainingFiles = {}

for imagePath in glob.glob("ProjectDataSet/training_dataset/training_data/annotations" + "/*.mat"):
	filename = imagePath[imagePath.rfind("/") + 1:]
	filename = os.path.splitext(filename)[0]
	mat = sio.loadmat(imagePath)
	howmany = len(mat.get('boxes')[0])
	strings = []
	for current in range(0, howmany):
		ymin = min(mat.get('boxes')[0,current]['a'][0,0][0,0],mat.get('boxes')[0,current]['b'][0,0][0,0],mat.get('boxes')[0,current]['c'][0,0][0,0],mat.get('boxes')[0,current]['d'][0,0][0,0])
		ymax = max(mat.get('boxes')[0,current]['a'][0,0][0,0],mat.get('boxes')[0,current]['b'][0,0][0,0],mat.get('boxes')[0,current]['c'][0,0][0,0],mat.get('boxes')[0,current]['d'][0,0][0,0])
		xmin = min(mat.get('boxes')[0,current]['a'][0,0][0,1],mat.get('boxes')[0,current]['b'][0,0][0,1],mat.get('boxes')[0,current]['c'][0,0][0,1],mat.get('boxes')[0,current]['d'][0,0][0,1])
		xmax = max(mat.get('boxes')[0,current]['a'][0,0][0,1],mat.get('boxes')[0,current]['b'][0,0][0,1],mat.get('boxes')[0,current]['c'][0,0][0,1],mat.get('boxes')[0,current]['d'][0,0][0,1])
		string = str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)
		strings = np.append(strings, str(0))
		strings = np.append(strings, string)
	outFile = open("ProjectDataSet/training_dataset/training_data/images1/oldtxts/"+filename+".txt", "w")
	for line in strings:
		outFile.write(line)
		outFile.write("\n")
	outFile.close()


for imagePath in glob.glob("ProjectDataSet/test_dataset/test_data/annotations" + "/*.mat"):
	filename = imagePath[imagePath.rfind("/") + 1:]
	filename = os.path.splitext(filename)[0]
	mat = sio.loadmat(imagePath)
	howmany = len(mat.get('boxes')[0])
	strings = []
	for current in range(0, howmany):
		ymin = min(mat.get('boxes')[0,current]['a'][0,0][0,0],mat.get('boxes')[0,current]['b'][0,0][0,0],mat.get('boxes')[0,current]['c'][0,0][0,0],mat.get('boxes')[0,current]['d'][0,0][0,0])
		ymax = max(mat.get('boxes')[0,current]['a'][0,0][0,0],mat.get('boxes')[0,current]['b'][0,0][0,0],mat.get('boxes')[0,current]['c'][0,0][0,0],mat.get('boxes')[0,current]['d'][0,0][0,0])
		xmin = min(mat.get('boxes')[0,current]['a'][0,0][0,1],mat.get('boxes')[0,current]['b'][0,0][0,1],mat.get('boxes')[0,current]['c'][0,0][0,1],mat.get('boxes')[0,current]['d'][0,0][0,1])
		xmax = max(mat.get('boxes')[0,current]['a'][0,0][0,1],mat.get('boxes')[0,current]['b'][0,0][0,1],mat.get('boxes')[0,current]['c'][0,0][0,1],mat.get('boxes')[0,current]['d'][0,0][0,1])
		string = str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)
		strings = np.append(strings, str(0))
		strings = np.append(strings, string)
	outFile = open("ProjectDataSet/test_dataset/test_data/images/oldtxts/"+filename+".txt", "w")
	for line in strings:
		outFile.write(line)
		outFile.write("\n")
	outFile.close()


for imagePath in glob.glob("ProjectDataSet/validation_dataset/validation_data/annotations" + "/*.mat"):
	filename = imagePath[imagePath.rfind("/") + 1:]
	filename = os.path.splitext(filename)[0]
	mat = sio.loadmat(imagePath)
	howmany = len(mat.get('boxes')[0])
	strings = []
	for current in range(0, howmany):
		ymin = min(mat.get('boxes')[0,current]['a'][0,0][0,0],mat.get('boxes')[0,current]['b'][0,0][0,0],mat.get('boxes')[0,current]['c'][0,0][0,0],mat.get('boxes')[0,current]['d'][0,0][0,0])
		ymax = max(mat.get('boxes')[0,current]['a'][0,0][0,0],mat.get('boxes')[0,current]['b'][0,0][0,0],mat.get('boxes')[0,current]['c'][0,0][0,0],mat.get('boxes')[0,current]['d'][0,0][0,0])
		xmin = min(mat.get('boxes')[0,current]['a'][0,0][0,1],mat.get('boxes')[0,current]['b'][0,0][0,1],mat.get('boxes')[0,current]['c'][0,0][0,1],mat.get('boxes')[0,current]['d'][0,0][0,1])
		xmax = max(mat.get('boxes')[0,current]['a'][0,0][0,1],mat.get('boxes')[0,current]['b'][0,0][0,1],mat.get('boxes')[0,current]['c'][0,0][0,1],mat.get('boxes')[0,current]['d'][0,0][0,1])
		string = str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)
		strings = np.append(strings, str(0))
		strings = np.append(strings, string)
	outFile = open("ProjectDataSet/validation_dataset/validation_data/images/oldtxts/"+filename+".txt", "w")
	for line in strings:
		outFile.write(line)
		outFile.write("\n")
	outFile.close()

'''mat = scipy.io.loadmat('VOC2010_1373.mat')
howmany = len(mat.get('boxes')[0])
current = 0
# iterate on current = (0 to howmany-1)
min (mat.get('boxes')[0,0]['a'][0,0][0,0]'''



 

