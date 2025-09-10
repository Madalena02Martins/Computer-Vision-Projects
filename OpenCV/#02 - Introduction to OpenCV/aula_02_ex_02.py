
#import
import numpy as np
import cv2

# Read the image
image_bmp = cv2.imread("./images/deti.bmp", cv2.IMREAD_UNCHANGED)
image_jpg = cv2.imread("./images/deti.jpg", cv2.IMREAD_UNCHANGED)


if  np.shape(image_bmp) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)
     
if  np.shape(image_jpg) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

# Verifica se as dimensões das imagens são as mesmas
if image_bmp.shape != image_jpg.shape:
    print("Images do not have the same size or number of channels.")
    exit(-1)

# Image characteristics
image_bmp_array = image_bmp.shape
height = image_bmp_array[0]
width = image_bmp_array[1]

# Check if the image is grayscale or RGB
if len(image_bmp_array) == 2:  # Grayscale
    channels = 1
else:  # RGB
    channels = image_bmp_array[2]

print("Image Size: (%d,%d,%d)" % (height, width, channels))
print("Image Type: %s" % (image_bmp.dtype))


# Image characteristics
image_jpg_array = image_jpg.shape
height = image_jpg_array[0]
width = image_jpg_array[1]

if len(image_jpg_array) == 2:
    channels_copy = 1
else:
    channels_copy = image_jpg_array[2]

print("Image Copy Size: (%d,%d,%d)" % (height, width, channels))
print("Image Copy Type: %s" % (image_jpg.dtype))

# Show the image
cv2.imshow ("bmp/jpg",np.hstack([image_bmp, image_jpg]))

# Realiza a subtração entre as duas imagens
difference = cv2.subtract(image_bmp, image_jpg)

# Exibe a imagem resultante da subtração
cv2.imshow("Difference", difference)

# Salva a imagem de diferença (opcional)
cv2.imwrite("difference_image.jpg", difference)
print("Difference image saved as 'difference_image.jpg'")

# Realiza a subtração com numpy
# Subtração direta com numpy pode resultar em valores fora do intervalo 0-255
difference = image_bmp.astype(np.int16) - image_jpg.astype(np.int16)

# Garante que os valores fiquem no intervalo de 0 a 255
difference_clipped = np.clip(difference, 0, 255).astype(np.uint8)

# Exibe a imagem resultante da subtração
cv2.imshow("Difference_numpy", difference_clipped)

# Realiza a subtração com numpy diretamente
# Como estamos usando uint8, valores negativos se tornam altos (de 0 a 255)
difference = image_bmp - image_jpg

# Exibe a imagem resultante da subtração
cv2.imshow("Difference (Numpy without Clipping)", difference)

# Realiza a subtração com opencv mas em módulo
#result = cv2.absdiff(image_bmp, image_jpg)

# Exibe a imagem resultante da subtração
#cv2.imshow("Difference (OpenCV em módulo)", result)


# Wait
cv2.waitKey(0)

# Fechar todas as janelas
cv2.destroyAllWindows()
