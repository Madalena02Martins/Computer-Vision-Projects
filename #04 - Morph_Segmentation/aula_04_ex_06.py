
#import
import sys
import numpy as np
import cv2

# Variável global para armazenar o ponto semente escolhido pelo usuário
seedPoint = None #(430,30) - para um pixel em especifico previamente definido

# Função de callback do mouse para capturar o ponto semente
def select_seed(event, x, y, flags, param):
    global seedPoint
    seedPoint = (x, y)
    # Define os valores de variação de intensidade
    loDiff = 5
    upDiff = 5

    # Define o novo valor a ser preenchido
    newVal = (125,125,125)  # Branco, para visualização
    if event == cv2.EVENT_LBUTTONDOWN:  # Captura o clique com o botão esquerdo
        # Executa o floodFill
        cv2.floodFill(image,None, seedPoint, newVal, (loDiff,upDiff))
        cv2.imshow(f'Segmented Image', image)


# Carregar as imagens
image1 = cv2.imread("./images/lena.jpg", cv2.IMREAD_UNCHANGED)
image2 = cv2.imread("./images/wdg2.bmp", cv2.IMREAD_UNCHANGED)
image3 = cv2.imread("./images/tools_2.png", cv2.IMREAD_UNCHANGED)

list_image = [image1, image2, image3]
for idx, image in enumerate(list_image):
    if np.shape(image) == ():
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
    print(f"Image {idx+1} Size: ({height},{width},{channels})")
    print(f"Image {idx+1} Type: {image.dtype}")
    print(f"Number of elements in Image {idx+1}: {image.size}")

    # Nome da janela específico para cada imagem
    cv2.imshow(f'Segmented Image', image)

    # Definir a função de callback para capturar o ponto semente
    cv2.setMouseCallback(f'Segmented Image', select_seed)#Para capturar um pixel através do clique do rato
    
    # Espera que o usuário selecione um ponto semente
    cv2.waitKey(0)
    

# Fecha todas as janelas
cv2.destroyAllWindows()
