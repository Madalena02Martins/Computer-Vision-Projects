
#import
import sys
import numpy as np
import cv2

# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
image = cv2.imread( "./images/wdg2.bmp", cv2.IMREAD_UNCHANGED)

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



# Loop through the image
for i in range(0,height):
    for j in range(0,width):
        if channels == 1:  # Grayscale
            px = image[i, j]
            if px < 120:
                image[i, j] = 0
            else:
                image[i, j] = 255
#Inverter image
image_inverte = 255-image

# Taking a matrix of size 5 as the kernel 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))#(11,1)

img_erode = cv2.erode(image_inverte, kernel, iterations=1) 

img_erode2 = cv2.erode(img_erode, kernel, iterations=1) 


# Taking a matrix of size 5 as the kernel 
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))#(11,1)

img_erode3 = cv2.erode(image_inverte, kernel2, iterations=1) 

img_erode4 = cv2.erode(img_erode3, kernel2, iterations=1)  

# Taking a matrix of size 5 as the kernel 
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 

# Definir a origem (ponto de acesso) do kernel para o pixel central da primeira linha (0,1)
anchor = (1, 0)

img_erode5 = cv2.erode(image_inverte, kernel3, anchor, iterations=1) 

img_erode6 = cv2.erode(img_erode5, kernel3, anchor, iterations=1)  

# Concatenar horizontalmente para formar duas linhas (2 imagens por linha)
row1 = cv2.hconcat([image, image_inverte, img_erode])
row2 = cv2.hconcat([img_erode2, img_erode3, img_erode4])
row3 = cv2.hconcat([img_erode5, img_erode6])

# Concatenar verticalmente para formar a matriz 2x3
matrix_2x3 = cv2.vconcat([row1, row2])

# Display all the images
cv2.imshow("Matriz 2x3", matrix_2x3)
cv2.imshow("Matriz 1x3", row3)

cv2.waitKey(0)
cv2.destroyAllWindows()
