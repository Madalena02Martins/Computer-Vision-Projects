import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the images
image1 = cv2.imread("./images/ireland-06-01.tif", cv2.IMREAD_UNCHANGED)
image2 = cv2.imread("./images/ireland-06-02.tif", cv2.IMREAD_UNCHANGED)
image3 = cv2.imread("./images/ireland-06-03.tif", cv2.IMREAD_UNCHANGED)
image4 = cv2.imread("./images/ireland-06-04.tif", cv2.IMREAD_UNCHANGED)
image5 = cv2.imread("./images/ireland-06-05.tif", cv2.IMREAD_UNCHANGED)
image6 = cv2.imread("./images/ireland-06-06.tif", cv2.IMREAD_UNCHANGED)

list_image = [image1, image2, image3, image4, image5, image6]

# Verificação de leitura das imagens
for idx, image in enumerate(list_image):
    if image is None:
        print(f"Image file {idx+1} could not be opened!")
        exit(-1)
        
# Concatenar horizontalmente com linhas brancas
row1 = cv2.hconcat([list_image[0], list_image[1], list_image[2]])
row2 = cv2.hconcat([list_image[3], list_image[4], list_image[5]])

# Exibir a matriz de histogramas
cv2.imshow('Imagens linha1', row1)
cv2.imshow('Imagens linha2', row2)

# Definir o tamanho fixo do histograma
histImageWidth = 512
histImageHeight = 512
histSize = 256  # from 0 to 255
histRange = [0, 256]  # Intensity range

# List to store histogram images
list_histImage = []

for image in list_image:
    # Compute the histogram
    hist_item = cv2.calcHist([image], [0], None, [histSize], histRange)

    # Create an image to display the histogram
    histImage = np.zeros((histImageHeight, histImageWidth, 1), np.uint8)

    # Width of each histogram bin
    binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))

    # Normalize values to [0, histImageHeight]
    cv2.normalize(hist_item, hist_item, 0, histImageHeight, cv2.NORM_MINMAX)

    # Draw the bars of the normalized histogram
    for i in range(histSize):
        cv2.rectangle(histImage, (i * binWidth, 0), ((i + 1) * binWidth, int(hist_item[i][0])), (125), -1)

    # Flip the histogram vertically to make it appear correctly
    histImage = np.flipud(histImage)
    
    # Append to the list of histograms
    list_histImage.append(histImage)

# Criar a linha branca de separação
white_line = np.ones((histImageHeight, 1, 1), np.uint8) * 255  # 10 pixels de largura

# Concatenar horizontalmente com linhas brancas
row1 = cv2.hconcat([list_histImage[0], white_line, list_histImage[1], white_line, list_histImage[2]])
row2 = cv2.hconcat([list_histImage[3], white_line, list_histImage[4], white_line, list_histImage[5]])

# Exibir a matriz de histogramas
cv2.imshow('Histograms linha1', row1)
cv2.imshow('Histograms linha2', row2)

# Exibir os gráficos com matplotlib numa matriz 2x3
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Matriz 2x3 de gráficos
axes = axes.ravel()  # Para facilitar o loop

for i, image in enumerate(list_image):
    hist_item = cv2.calcHist([image], [0], None, [histSize], histRange)
    axes[i].plot(hist_item, color='r')  # Desenha o histograma a vermelho
    axes[i].set_xlim(histRange)
    axes[i].set_title(f'Image {i+1} Histogram')

plt.tight_layout()  # Ajusta o layout para evitar sobreposição
plt.show()

cv2.waitKey(0)  # Espera até que uma tecla seja pressionada
cv2.destroyAllWindows()  # Fecha todas as janelas
