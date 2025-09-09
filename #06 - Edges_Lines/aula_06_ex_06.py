
#import
import sys
import numpy as np
import cv2
import math

global srcPts
global dstPts

windowName = "Image Window"

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



def select_src(event, x, y, flags, params):
	global srcPts
	if event == cv2.EVENT_LBUTTONDOWN and len(srcPts) < 4:

			srcPts.append((x,y))
			cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
			cv2.putText(image,str(len(srcPts)), (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
			cv2.imshow("Original", image)




def select_dst(event, x, y, flags, params):
    global dstPts
    global srcPts
    if event == cv2.EVENT_LBUTTONDOWN and len(dstPts) < 4:

        dstPts.append((x,y))
        cv2.circle(imageTransformed, (x, y), 2, (255, 0, 0), 2)
        cv2.putText(imageTransformed,str(len(dstPts)), (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.imshow("Warped", imageTransformed)

        if len(dstPts) > 3 and len(srcPts) > 3:
            np_srcPts = np.array(srcPts).astype(np.float32)
            np_dstPts = np.array(dstPts).astype(np.float32)

            h, status = cv2.findHomography(np_dstPts, np_srcPts)

            im_out = cv2.warpPerspective(imageTransformed, h, (imageTransformed.shape[1],imageTransformed.shape[0]))


            cv2.imshow('Final', im_out)

            print(h)

            print(f"X diff {h[0][2]}")
            print(f"Y diff {h[1][2]}")
            print(f"Angle  {math.degrees(math.acos(h[0][0] % (math.pi/2)))}")

srcPts = []
dstPts = []


# Read the image from argv
#image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );
image = cv2.imread( "./images/homography_4.jpg", cv2.IMREAD_GRAYSCALE)
imageTransformed = cv2.imread('./images/homography_1.jpg', cv2.IMREAD_GRAYSCALE)


if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

cv2.imshow("Original", image)
cv2.imshow('Warped', imageTransformed)

cv2.setMouseCallback("Original", select_src)
cv2.setMouseCallback("Warped", select_dst)

while cv2.waitKey(1000) != 27:
    pass