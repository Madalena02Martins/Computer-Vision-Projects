import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO  

# Caminho para seu modelo treinado
weights_path = r"C:\Users\madal\OneDrive\Ambiente de Trabalho\GitHub Computer Vision\OpenCV Projects\CNN projects\Plastic Water Bottles Detection\runs\detect\train\weights\best.pt"

# Carrega o modelo
model = YOLO(weights_path)

# Configuração da câmera RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Definindo o stream de vídeo (colorido)
pipeline.start(config)

while True:
    # Espera por novos frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    # Verifica se o frame foi capturado corretamente
    if not color_frame:
        continue

    # Converte o frame para formato numpy (OpenCV utiliza numpy)
    frame = np.asanyarray(color_frame.get_data())

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
pipeline.stop()
cv2.destroyAllWindows()
