from ultralytics import YOLO

import torch
if __name__ == "__main__":
    print(torch.cuda.is_available())

    # Caminho para o arquivo data.yaml
    data_yaml = 'C:\\Users\\madal\\OneDrive\\Ambiente de Trabalho\\GitHub Computer Vision\\OpenCV Projects\\CNN projects\\Plastic Water Bottles Detection\\data.yaml'

    # Iniciar o treinamento
    model = YOLO('yolov8n.yaml')  # Pode ser "yolov8s.yaml", "yolov8m.yaml", etc.
    model.train(data=data_yaml, epochs=70, batch=64) #batch=32 para gráfica dedicada de 8GB


