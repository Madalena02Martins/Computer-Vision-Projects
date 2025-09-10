import numpy as np
import cv2

# Usar um caminho absoluto para teste
image_path = "./images/lena.jpg"

# Ler a imagem
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Image file could not be opened")
else:
    print("Image file opened successfully")

# Características da imagem
if image is not None:
    height, width = image.shape
    print(f"Image Size: ({height},{width})")
    print(f"Image Type: {image.dtype}")

    # Criar uma janela de visualização (opcional)
    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)

    # Mostrar a imagem
    cv2.imshow("Display window", image)

    # Esperar
    cv2.waitKey(0)

    # Fechar a janela
    #cv2.destroyWindow("Display window")

    # Fechar todas as janelas
    cv2.destroyAllWindows()
