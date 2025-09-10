
#import
import sys
import numpy as np
import cv2

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
#image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );
#image = cv2.imread( "../images/lena.jpg", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "../images/Lena_Ruido.png", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "../images/DETI_Ruido.png", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "../images/fce5noi3.bmp", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "../images/fce5noi4.bmp", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "../images/fce5noi6.bmp", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "../images/sta2.bmp", cv2.IMREAD_GRAYSCALE)
image = cv2.imread( "./images/sta2noi1.bmp", cv2.IMREAD_GRAYSCALE)

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

# Reduzir a imagem para 50% do tamanho original
scale_percent = 70  # Percentual do redimensionamento
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Redimensionar a imagem
resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# cv2.imshow('Orginal', image)

# Average filter image
imageAFilter3x3_1 = cv2.blur(resized_image,(3, 3))
imageAFilter5x5_1 = cv2.blur(resized_image,(5, 5))
imageAFilter7x7_1 = cv2.blur(resized_image,(7, 7))

imageAFilter3x3_2 = cv2.blur(imageAFilter3x3_1,(3, 3))
imageAFilter5x5_2 = cv2.blur(imageAFilter5x5_1,(5, 5))
imageAFilter7x7_2 = cv2.blur(imageAFilter7x7_1,(7, 7))

imageAFilter3x3_3 = cv2.blur(imageAFilter3x3_2,(3, 3))
imageAFilter5x5_3 = cv2.blur(imageAFilter5x5_2,(5, 5))
imageAFilter7x7_3 = cv2.blur(imageAFilter7x7_2,(7, 7))

# Concatenar horizontalmente para formar duas linhas (2 imagens por linha)
row1 = cv2.hconcat([resized_image, imageAFilter3x3_1, imageAFilter3x3_2,imageAFilter3x3_3])
row2 = cv2.hconcat([resized_image, imageAFilter5x5_1, imageAFilter5x5_2,imageAFilter5x5_3])
row3 = cv2.hconcat([resized_image, imageAFilter7x7_1, imageAFilter7x7_2,imageAFilter7x7_3])

# Display all the images
cv2.imshow("ImageAFilter3x3", row1)
cv2.imshow("ImageAFilter5x5", row2)
cv2.imshow("ImageAFilter7x7", row3)

cv2.waitKey(0)
cv2.destroyAllWindows()


