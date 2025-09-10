import sys
import numpy as np
import cv2

def printImageFeatures(image):
    # Características da imagem
    if len(image.shape) == 2:
        height, width = image.shape
        nchannels = 1
    else:
        height, width, nchannels = image.shape

    # Imprimir as características da imagem
    print("Altura da imagem: %d" % height)
    print("Largura da imagem: %d" % width)
    print("Canais da imagem: %d" % nchannels)
    print("Número de elementos: %d" % image.size)

# Carregar a imagem em tons de cinza
image = cv2.imread("./images/Bikesgray.jpg", cv2.IMREAD_GRAYSCALE)  # Reduzida a 50%

if image is None:
    print("Não foi possível abrir o ficheiro de imagem!")
    exit(-1)

printImageFeatures(image)

# Reduzir a imagem para 50% do tamanho original
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Redimensionar a imagem
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Aplicar o filtro Canny para detectar bordas
edges = cv2.Canny(resized_image, 50, 150)

# Detectar linhas usando a Transformada de Hough Clássica
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

# Detectar linhas usando a Transformada de Hough Probabilística
linesP = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=70, minLineLength=30, maxLineGap=10)

# Copiar a imagem para desenhar as linhas detectadas
image_with_lines = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
image_with_linesP = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

# Desenhar as linhas detectadas pela Transformada de Hough Clássica
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Desenhar as linhas detectadas pela Transformada de Hough Probabilística
if linesP is not None:
    for lineP in linesP:
        x1, y1, x2, y2 = lineP[0]
        cv2.line(image_with_linesP, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Mostrar a imagem original com as bordas detectadas (Canny)
cv2.imshow('Bordas (Canny)', edges)

# Mostrar a imagem com as linhas detectadas pela Transformada de Hough Clássica
cv2.imshow('Linhas Detectadas (HoughLines)', image_with_lines)

# Mostrar a imagem com as linhas detectadas pela Transformada de Hough Probabilística
cv2.imshow('Linhas Detectadas (HoughLinesP)', image_with_linesP)

# Esperar até que uma tecla seja pressionada para fechar
cv2.waitKey(0)
cv2.destroyAllWindows()
