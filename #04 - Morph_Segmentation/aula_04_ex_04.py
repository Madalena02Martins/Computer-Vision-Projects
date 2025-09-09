
#import
import sys
import numpy as np
import cv2

# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
image = cv2.imread( "./images/art3.bmp", cv2.IMREAD_UNCHANGED)
image2 = cv2.imread( "./images/art2.bmp", cv2.IMREAD_UNCHANGED)

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)


image_array = image.shape
height = image_array[0]
width = image_array[1]

if len(image_array) == 2:
    channels = 1
else:
    channels = image_array[2]

# print some features
print("Image Copy Size: (%d,%d,%d)" % (height, width, channels))
print("Image Copy Type: %s" % (image.dtype))
print("Number of elements : %d" % image.size)

#cv2.imshow("Image", image)

if  np.shape(image2) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)


image2_array = image2.shape
height2 = image2_array[0]
width2 = image2_array[1]

if len(image2_array) == 2:
    channels2 = 1
else:
    channels2 = image2_array[2]

# print some features
print("Image Copy Size: (%d,%d,%d)" % (height2, width2, channels2))
print("Image Copy Type: %s" % (image2.dtype))
print("Number of elements : %d" % image2.size)

#cv2.imshow("Image2", image2)

# Taking a matrix of size 5 as the kernel 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

img_opening1 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#cv2.imshow("Opennig", img_opening1)

# Taking a matrix of size 5 as the kernel 
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,9))

img_opening2 = cv2.morphologyEx(image2, cv2.MORPH_OPEN, kernel2)

#cv2.imshow("Opennig2", img_opening2)

# Taking a matrix of size 5 as the kernel 
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))

img_opening3 = cv2.morphologyEx(image2, cv2.MORPH_OPEN, kernel3)

#cv2.imshow("Opennig3", img_opening3)

# Concatenar horizontalmente para formar duas linhas (2 imagens por linha)
row1 = cv2.hconcat([image, img_opening1])
row2 = cv2.hconcat([image2, img_opening2, img_opening3])


# Display all the images
cv2.imshow("Separar regiões circulares", row1)
cv2.imshow("Separar segmentos de linha vertical e horizontal", row2)

cv2.waitKey(0)
cv2.destroyAllWindows()
