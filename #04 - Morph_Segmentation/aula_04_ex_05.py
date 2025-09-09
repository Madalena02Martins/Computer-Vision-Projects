
#import
import sys
import numpy as np
import cv2

# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
image = cv2.imread( "./images/art4.bmp", cv2.IMREAD_UNCHANGED)

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

# Taking a matrix of size 5 as the kernel 
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22,22)) #Funciona melhor para valores ímpares do que para valores pares

img_closing1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)

# Taking a matrix of size 5 as the kernel 
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)) #Até ao 13 ainda existem circulos pequenos

img_closing2 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel2)

# Taking a matrix of size 5 as the kernel 
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)) #Os valores 15, 17, 19 e 21 são os melhores valores
#para separar os circulos grandes dos pequenos

img_closing3 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel3)

# Taking a matrix of size 5 as the kernel 
kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30)) #A partir do valor 30 todos os circulos desaparecem
#para separar os circulos grandes dos pequenos

img_closing4 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel4)

# Concatenar horizontalmente para formar duas linhas (2 imagens por linha)
row1 = cv2.hconcat([image, img_closing1, img_closing2, img_closing3,img_closing4])

# Display all the images
cv2.imshow("Separar regioes circulares", row1)

cv2.waitKey(0)
cv2.destroyAllWindows()
