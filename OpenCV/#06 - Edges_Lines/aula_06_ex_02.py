
#import
import sys
import numpy as np
import cv2
import math

srcPts1 = []
srcPts2 = []


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

def select_src1(event, x, y, flags, params):
    global srcPts1
    if event == cv2.EVENT_LBUTTONDOWN and len(srcPts1) < 3:
        srcPts1.append((x, y))
        cv2.circle(image1, (x, y), 2, (255, 0, 0), 2)
        cv2.putText(image1, str(len(srcPts1)), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.imshow("Original1", image1)

def select_src2(event, x, y, flags, params):
    global srcPts2
    if event == cv2.EVENT_LBUTTONDOWN and len(srcPts2) < 3:
        srcPts2.append((x, y))
        cv2.circle(image2, (x, y), 2, (255, 0, 0), 2)
        cv2.putText(image2, str(len(srcPts2)), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.imshow("Original2", image2)

# Read the image from argv
image1 = cv2.imread("./images/lena.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./imagename_tf.jpg", cv2.IMREAD_GRAYSCALE)

if np.shape(image1) == ():
    # Failed Reading
    print("Image file could not be open!")
    exit(-1)

if np.shape(image2) == ():
    # Failed Reading
    print("Image file could not be open!")
    exit(-1)

printImageFeatures(image1)
printImageFeatures(image2)

cv2.imshow('Original1', image1)
cv2.setMouseCallback('Original1', select_src1)

cv2.imshow('Original2', image2)
cv2.setMouseCallback('Original2', select_src2)

# Esperando que o usuário selecione os pontos
cv2.waitKey(0)

# Verificar se foram selecionados 3 pontos em cada imagem
if len(srcPts1) == 3 and len(srcPts2) == 3:
    # Convertendo os pontos para np.array
    np_srcPts1 = np.array(srcPts1).astype(np.float32)
    np_srcPts2 = np.array(srcPts2).astype(np.float32)

    # Estimando a matriz de transformação afim
    transformation_matrix = cv2.getAffineTransform(np_srcPts1, np_srcPts2)

    # Aplicando a transformação à imagem original
    warp_dst = cv2.warpAffine(image1, transformation_matrix, (image1.shape[1], image1.shape[0]))

    # Mostrando a imagem transformada
    cv2.imshow("Imagem Transformada", warp_dst)

    # Imprimindo a matriz de transformação
    print("Matriz de Transformação:")
    print(transformation_matrix)

    # Calculando os parâmetros de transformação (Escala, Rotação)
    a, b = transformation_matrix[0, 0], transformation_matrix[0, 1]
    c, d = transformation_matrix[1, 0], transformation_matrix[1, 1]

    # Cálculo dos parâmetros
    scale_x = math.sqrt(a**2 + b**2)
    scale_y = math.sqrt(c**2 + d**2)
    rotation_angle = (math.atan2(b, a) * 180) / math.pi

    print(f"Escala em x: {scale_x}")
    print(f"Escala em y: {scale_y}")
    print(f"Ângulo de rotação (em graus): {rotation_angle}")

    # Calculando e mostrando a diferença entre a imagem transformada e a original distorcida
    image_diff = cv2.absdiff(warp_dst, image2)
    cv2.imshow("Diferenca entre as imagens", image_diff)

else:
    print("Erro: Por favor, selecione exatamente 3 pontos em cada imagem.")

cv2.waitKey(0)
cv2.destroyAllWindows()
