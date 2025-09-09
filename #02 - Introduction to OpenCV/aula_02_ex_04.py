
#import
import numpy as np
import cv2

# Read the image
image = cv2.imread("./images/orchid.bmp", cv2.IMREAD_UNCHANGED)

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

# Image characteristics
image_array = image.shape
height = image_array[0]
width = image_array[1]

# Check if the image is grayscale or RGB
if len(image_array) == 2:  # Grayscale
    channels = 1
else:  # RGB
    channels = image_array[2]

# Exemplo de redimensionamento
scale_percent = 70  # 70% do tamanho original
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

print("Image Size: (%d,%d,%d)" % (height, width, channels))
print("Image Type: %s" % (image.dtype))

#Redimensionar a imagem original
image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Using cv2.COLOR_BGR2GRAY color space
image_GRAY = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

# Replicar o canal de cinza para 3 canais
image_GRAY_3_channels = np.stack((image_GRAY,)*3, axis=-1)

# Using cv2.COLOR_RGB2HLS color space
image_HLS = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HLS)

# Using cv2.COLOR_RGBXYZ color space
image_XYZ = cv2.cvtColor(image_resized, cv2.COLOR_RGB2XYZ)

# Using cv2.COLOR_RGB2HSV color space
image_HSV = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)

# Using cv2.COLOR_RGB2BGR color space
image_BGR = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)

# Concatenar horizontalmente para formar duas linhas (2 imagens por linha)
row1 = cv2.hconcat([image_resized, image_GRAY_3_channels, image_HLS])
row2 = cv2.hconcat([image_XYZ, image_HSV, image_BGR])

# Concatenar verticalmente para formar a matriz 2x3
matrix_2x3 = cv2.vconcat([row1, row2])

# Exibir a matriz
cv2.imshow("Matriz 2x3", matrix_2x3)
# Wait
cv2.waitKey(0)

# Fechar todas as janelas
cv2.destroyAllWindows()
