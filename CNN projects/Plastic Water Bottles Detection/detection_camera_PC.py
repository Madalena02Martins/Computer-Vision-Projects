import cv2
import numpy as np
from ultralytics import YOLO

# Caminho para seu modelo treinado
weights_path = r"C:\Users\madal\OneDrive\Ambiente de Trabalho\GitHub Computer Vision\OpenCV Projects\CNN projects\Plastic Water Bottles Detection\runs\detect\train\weights\best.pt"

# Carrega o modelo
model = YOLO(weights_path)

# Inicializa a câmera do PC (câmera padrão, geralmente a 0)
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()

while True:
    # Captura o frame da câmera
    ret, frame = cap.read()
    
    if not ret:
        print("Falha ao capturar frame")
        break

    # Executa a inferência
    results = model(frame, stream=True)

    # Para cada resultado (se houver mais de uma escala ou saída, normalmente há apenas uma)
    for result in results:
        annotated_frame = result.plot()  # Desenha as detecções na imagem

    # Exibe a imagem com as detecções
    cv2.imshow("Deteccoes - Camera do PC", annotated_frame)

    # Sai se pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Para de capturar e fecha as janelas do OpenCV
cap.release()
cv2.destroyAllWindows()
