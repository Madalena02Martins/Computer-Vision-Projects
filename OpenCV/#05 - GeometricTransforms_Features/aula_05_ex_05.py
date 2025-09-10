
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
#image = cv2.imread(sys.argv[1] , cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("./images/lena.jpg", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread("./images/wdg2.bmp", cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "./images/cln1.bmp", cv2.IMREAD_GRAYSCALE) #scale 100%
image = cv2.imread( "./images/Bikesgray.jpg", cv2.IMREAD_GRAYSCALE) #scale 50%

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

# Sobel Operatot 3 x 3
# imageSobel3x3_X = cv2.Sobel(image, cv2.CV_64F, 1, 0, 3)
# cv2.imshow("Sobel 3 x 3 - X", imageSobel3x3_X)
# image8bits = np.uint8(np.absolute(imageSobel3x3_X))
# cv2.imshow("8 bits - Sobel 3 x 3 - X", image8bits)


# Reduzir a imagem para 50% do tamanho original
scale_percent = 50  # Percentual do redimensionamento # 40% para Lena_Ruido.png e DETI_Ruido.png, restantes 70%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Redimensionar a imagem
resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# cv2.imshow('Orginal', image)

# Average filter image
imageCFilter_1= cv2.Canny(resized_image,1,255, None, 3)
imageCFilter_2= cv2.Canny(resized_image,220,225, None, 3)
imageCFilter_3= cv2.Canny(resized_image,1,128, None, 3)

#edge = cv2.Canny(img, t_lower, t_upper) 

# Concatenar horizontalmente para formar duas linhas (2 imagens por linha)
row1 = cv2.hconcat([resized_image, imageCFilter_1, imageCFilter_2,imageCFilter_3])

# Display all the images
cv2.imshow("ImageCFilter", row1)

# Iniciar captura de vídeo
capture = cv2.VideoCapture(0)

# Ajustar a resolução e a taxa de quadros (opcional)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# capture.set(cv2.CAP_PROP_FPS, 60)

while True:
    # Capturar frame da câmera
    ret, frame = capture.read()

    if not ret:
        print("Erro ao capturar a imagem da câmera.")
        break

    # Aplicar os filtros Canny
    imageCFilter_1 = cv2.Canny(frame, 100, 255)
    imageCFilter_2 = cv2.Canny(frame, 220, 225)
    imageCFilter_3 = cv2.Canny(frame, 100, 128)

    # Mostrar os resultados
    cv2.imshow('Filtro Canny 1 (1, 255)', imageCFilter_1)
    cv2.imshow('Filtro Canny 2 (220, 225)', imageCFilter_2)
    cv2.imshow('Filtro Canny 3 (1, 128)', imageCFilter_3)

    # Sair do loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar a captura de vídeo e fechar as janelas
capture.release()

cv2.destroyAllWindows()



