import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Arrays to store object points and image points from all the images.
objpoints = []    # 3D points in real-world space 
left_points = []   # 2D points in image plane (left image)
right_points = []  # 2D points in image plane (right image)

# Function to find and display chessboard corners
def FindAndDisplayChessboard(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    # If found, display the image with corners
    if ret:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(250)
    
    return ret, corners

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Read left and right images
left_images = sorted(glob.glob('.//images//left*.jpg'))
right_images = sorted(glob.glob('.//images//right*.jpg'))

# Processamento de cada imagem
for fname in left_images:
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)
    if ret:
        objpoints.append(objp)
        left_points.append(corners)

cv2.destroyAllWindows()

# Obter o tamanho da imagem
if left_images or right_images:
    img_shape = cv2.imread(left_images[0]).shape[:2][::-1] or cv2.imread(right_images[0]).shape[:2][::-1]

# Processamento de cada imagem
for fname in right_images:
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)
    if ret:
        right_points.append(corners)

cv2.destroyAllWindows()

# # Calibração estéreo
# ret1, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F = cv2.stereoCalibrate(
#     objpoints, left_points, right_points,
#     None, None, None, None,
#     img_shape,
#     flags=cv2.CALIB_SAME_FOCAL_LENGTH
# )

#Observação#

# Calibrar a câmera esquerda individualmente
ret_left, intrinsics1, distortion1, _, _ = cv2.calibrateCamera(objpoints, left_points, img_shape, None, None)
print("Calibração da câmera esquerda concluída.")

# Calibrar a câmera direita individualmente
ret_right, intrinsics2, distortion2, _, _ = cv2.calibrateCamera(objpoints, right_points, img_shape, None, None)
print("Calibração da câmera direita concluída.")

# Calibração estéreo usando parâmetros intrínsecos estimados
# Definimos a flag para usar a estimativa intrínseca como ponto de partida.
flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC

ret1, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, left_points, right_points,
    intrinsics1, distortion1, intrinsics2, distortion2,
    img_shape,
    flags=flags
)

# Mostrar os resultados
print("Intrinsics1:")
print(intrinsics1)
print("Distortion1:")
print(distortion1)
print("Intrinsics2:")
print(intrinsics2)
print("Distortion2:")
print(distortion2)
print("R:")
print(R)
print("T:")
print(T)
print("E:")
print(E)
print("F:")
print(F)


np.savez("stereoParams.npz",
         intrinsics1=intrinsics1,
         distortion1=distortion1,
         intrinsics2=intrinsics2,
         distortion2=distortion2,
         R=R, T=T, E=E, F=F)


cv2.destroyAllWindows()


