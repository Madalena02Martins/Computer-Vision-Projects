import cv2
import numpy as np
import glob

# Tamanho do tabuleiro de xadrez
board_h = 9  # Número de quadrados na altura
board_w = 6  # Número de quadrados na largura
square_size = 1.0  # Ajuste para o tamanho real de cada quadrado do tabuleiro, em unidades métricas

# Prepare os pontos 3D do objeto
objp = np.zeros((board_h * board_w, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
objp *= square_size  # Multiplica para ajustar ao tamanho real do tabuleiro

# Arrays para armazenar pontos do objeto e pontos da imagem
objpoints = []  # Pontos 3D no espaço do mundo real
imgpoints = []  # Pontos 2D no plano da imagem

# Alternativa: Usar câmera ou imagens de arquivo
use_camera = True  # Altere para False para usar as imagens de arquivo fornecidas

if not use_camera:
    # Leitura das imagens fornecidas
    images = sorted(glob.glob('.//images//left*.jpg'))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontre os cantos do tabuleiro de xadrez
        ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

        # Se os cantos forem encontrados, adicione pontos de objeto e pontos de imagem
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Melhora a precisão dos cantos
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints[-1] = corners2

            # Desenha e exibe os cantos
            img = cv2.drawChessboardCorners(img, (board_w, board_h), corners2, ret)
            cv2.imshow('Chessboard', img)
            # cv2.waitKey(500)
else:
    # Uso da câmera para capturar imagens ao vivo
    capture = cv2.VideoCapture(0)
    captured_images = 0  # Contador de imagens capturadas
    max_images = 13  # Número de imagens para calibração

    while captured_images < max_images:
        ret, frame = capture.read()
        # Aplicar flip horizontal
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta cantos do tabuleiro de xadrez
        ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            # Desenha e exibe os cantos
            frame = cv2.drawChessboardCorners(frame, (board_w, board_h), corners2, ret)
            captured_images += 1
            cv2.imshow('Chessboard', frame)
            # cv2.waitKey(500)

        # Mostra o vídeo ao vivo
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

# Calibração da câmera
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Exibe os parâmetros de calibração
    print("Intrinsics: ")
    print(intrinsics)
    print("Distortion : ")
    print(distortion)
    for i in range(len(tvecs)):
        print(f"Translations({i}) :")
        print(tvecs[i])
        print(f"Rotation({i}) :")
        print(rvecs[i])

    # Salva os parâmetros em um arquivo .npz
    np.savez('camera_calibration.npz', intrinsics=intrinsics, distortion=distortion)
    print("Parâmetros de calibração salvos em 'camera_calibration.npz'")
else:
    print("Falha ao calibrar a câmera: nenhum ponto de objeto ou ponto de imagem detectado.")
