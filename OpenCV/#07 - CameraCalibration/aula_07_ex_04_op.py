import cv2
import numpy as np
import glob

# Tamanho do tabuleiro de xadrez
board_h = 9  # Número de quadrados na altura
board_w = 6  # Número de quadrados na largura
square_size = 1.0  # Tamanho real de cada quadrado do tabuleiro, em unidades métricas

# Prepare os pontos 3D do objeto
objp = np.zeros((board_h * board_w, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
objp *= square_size  # Ajusta para o tamanho real do tabuleiro

# Pontos 3D para um cubo a ser projetado na imagem
cube_points = np.float32([
    [0, 0, 0], [0, square_size, 0], [square_size, square_size, 0], [square_size, 0, 0],
    [0, 0, -square_size], [0, square_size, -square_size], [square_size, square_size, -square_size], [square_size, 0, -square_size]
])

# Arrays para armazenar pontos do objeto e pontos da imagem
objpoints = []  # Pontos 3D no espaço do mundo real
imgpoints = []  # Pontos 2D no plano da imagem

# Alternativa: Usar câmera ou imagens de arquivo
use_camera = True  # Altere para False para usar as imagens de arquivo fornecidas

# Captura ou leitura das imagens
if not use_camera:
    images = sorted(glob.glob('.//images//left*.jpg'))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
else:
    capture = cv2.VideoCapture(0)
    captured_images = 0
    max_images = 13

    while captured_images < max_images:
        ret, frame = capture.read()
        
        # Aplicar flip horizontal
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

# Calibração da câmera
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez('calibrated_camera_params.npz', intrinsics=intrinsics, distortion=distortion)
    print("Parâmetros de calibração salvos em 'calibrated_camera_params.npz'")

# Leitura dos parâmetros de calibração
with np.load('calibrated_camera_params.npz') as data:
    intrinsics = data['intrinsics']
    distortion = data['distortion']
print("Parâmetros carregados:")
print("Intrinsics:\n", intrinsics)
print("Distortion:\n", distortion)

# Visualização do padrão ao vivo com projeção do cubo
capture = cv2.VideoCapture(0)
while True:
    ret, img = capture.read()

    # Aplicar flip horizontal
    img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar cantos do tabuleiro de xadrez
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        ret, rvec, tvec = cv2.solvePnP(objp, corners2, intrinsics, distortion)

        # Projetar os pontos 3D do cubo para o plano da imagem
        cube_2d, _ = cv2.projectPoints(cube_points, rvec, tvec, intrinsics, distortion)

        # Desenhar o cubo wireframe na imagem
        cube_2d = cube_2d.reshape(-1, 2).astype(int)
        # Conectar as arestas do cubo
        for i, j in zip([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4]):
            cv2.line(img, tuple(cube_2d[i]), tuple(cube_2d[j]), (0, 255, 0), 2)  # Arestas da base e do topo em verde

        # Conectar a base com o topo
        for i, j in zip([0, 1, 2, 3], [4, 5, 6, 7]):
            cv2.line(img, tuple(cube_2d[i]), tuple(cube_2d[j]), (0, 255, 0), 2)  # Arestas verticais

        # Exibe o feed ao vivo para ajuste do tabuleiro de xadrez
        cv2.imshow('Ajuste o tabuleiro de xadrez e pressione "s" para capturar', img)

        # Pressionar 's' para capturar a imagem quando o tabuleiro estiver ajustado
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break


    # Exibir o feed ao vivo
    cv2.imshow('Live Feed', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

# Converter para escala de cinza e detectar cantos do tabuleiro
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

# Se o tabuleiro for detectado, calcular a calibração externa
if ret:
    imgpoints.append(corners)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, intrinsics, distortion)

    # Converter os valores de rvec de radianos para graus
    rvec_degrees = np.degrees(rvec)

    # Exibir a posição e orientação da câmera em relação ao tabuleiro
    print("Rotação (em graus):\n", rvec_degrees)
    print("Translação:\n", tvec)
else:
    print("Falha ao detectar o padrão de calibração na imagem.")

