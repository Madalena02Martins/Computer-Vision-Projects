import numpy as np
import cv2
import glob

# Board Size
board_h = 9  # altura do padrão
board_w = 6  # largura do padrão

# # Tamanho real de cada quadrado no padrão (em cm, mm ou outra unidade)
# square_size = 5  # tamanho real do quadrado, por exemplo, 2.5 cm

# # Preparar pontos do objeto, como (0,0,0), (1,0,0), ..., multiplicando pelo tamanho real
# objp = np.zeros((board_h * board_w, 3), np.float32)
# objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # pontos 3D no espaço real
imgpoints = []  # pontos 2D no plano da imagem

def FindAndDisplayChessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
    
    # Se encontrados, mostrar imagem com cantos
    if ret:
        # Definir critérios para refinamento
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    
    return ret, corners

# Preparar os pontos do objeto, como (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Lê todas as imagens
images = sorted(glob.glob('.//images//left*.jpg'))

# Processamento de cada imagem
for fname in images:
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

cv2.destroyAllWindows()

# Obter o tamanho da imagem
if images:
    img_shape = cv2.imread(images[0]).shape[:2][::-1]

# Calibrar a câmera
ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None)

# Mostrar os resultados
print("Intrinsics:")
print(intrinsics)
print("Distortion:")
print(distortion)
for i in range(len(tvecs)):
    print(f"Translations({i}):")
    print(tvecs[i])
    print(f"Rotation({i}):")
    print(rvecs[i])

# Salvar intrinsics e distorção
np.savez('camera.npz', intrinsics=intrinsics, distortion=distortion)
