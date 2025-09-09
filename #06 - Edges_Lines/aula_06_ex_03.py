
#import
import sys
import numpy as np
import cv2
import math

srcPts1 = []
srcPts2 = []


def printImageFeatures(image):
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)


# Read the image from argv
image1 = cv2.imread("./images/lena.jpg" , cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./imagename_tf.jpg", cv2.IMREAD_GRAYSCALE)

if  np.shape(image1) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

if  np.shape(image2) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image1)

printImageFeatures(image2)

cv2.imshow('Original1', image1)

cv2.imshow('Original2', image2)

gray1=image1 #porque a imagem já é em escala de cinza
gray2=image2 #porque a imagem já é em escala de cinza

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(image1,None)
kp2, des2 = sift.detectAndCompute(image2,None)

image1=cv2.drawKeypoints(gray1,kp1,image1)

cv2.imshow('SIFT1', image1) 
cv2.imwrite('sift_keypoints.jpg',image1)

image2=cv2.drawKeypoints(gray2,kp2,image2)

cv2.imshow('SIFT2', image2) 
cv2.imwrite('sift_keypoints.jpg',image2)


cv2.waitKey(0)
cv2.destroyAllWindows()
