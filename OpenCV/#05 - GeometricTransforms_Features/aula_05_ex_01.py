
#import
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
image = cv2.imread( "./images/lena.jpg", cv2.IMREAD_UNCHANGED)

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

# Reduzir a imagem para 50% do tamanho original
scale_percent = 70  # Percentual do redimensionamento
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Redimensionar a imagem
resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

ret,thresh1 = cv2.threshold (resized_image,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold (resized_image,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold (resized_image,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold (resized_image,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold (resized_image,127,255,cv2.THRESH_TOZERO_INV)

# titles = [ 'Imagem original' , 'BINÁRIO' , 'INV_BINÁRIO' , 'TRUNC' , 'TOZERO' , 'INV_TOZERO' ]
# images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
 
# plt.show()

# Concatenar horizontalmente para formar duas linhas (2 imagens por linha)
row1 = cv2.hconcat([resized_image, thresh1, thresh2])
row2 = cv2.hconcat([thresh3, thresh4, thresh5])

# Concatenar verticalmente para formar a matriz 2x3
matrix_2x3 = cv2.vconcat([row1, row2])

# Display all the images
cv2.imshow("Matriz 2x3", matrix_2x3)

cv2.waitKey(0)
cv2.destroyAllWindows()
