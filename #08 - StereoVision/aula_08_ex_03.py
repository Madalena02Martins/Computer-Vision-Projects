import numpy as np
import cv2
import glob

# Função para remover a distorção da imagem
def undistort_image(img, intrinsics, distortion):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, intrinsics, distortion, None, new_camera_matrix)
    return undistorted_img

# Carregar os parâmetros de calibração
calibration_params = np.load("stereoParams.npz")
intrinsics1 = calibration_params['intrinsics1']
distortion1 = calibration_params['distortion1']
intrinsics2 = calibration_params['intrinsics2']
distortion2 = calibration_params['distortion2']

# Selecionar um par estéreo de imagens
left_images = sorted(glob.glob('.//images//left*.jpg'))
right_images = sorted(glob.glob('.//images//right*.jpg'))

if left_images and right_images:
    # Carregar um par estéreo
    left_img = cv2.imread(left_images[0])
    right_img = cv2.imread(right_images[0])

    # Remover a distorção das imagens
    undistorted_left = undistort_image(left_img, intrinsics1, distortion1)
    undistorted_right = undistort_image(right_img, intrinsics2, distortion2)

    # Exibir as imagens originais e não distorcidas
    cv2.imshow('Left Image - Original', left_img)
    cv2.imshow('Left Image - Undistorted', undistorted_left)
    cv2.imshow('Right Image - Original', right_img)
    cv2.imshow('Right Image - Undistorted', undistorted_right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Não foram encontradas imagens estéreo para processamento.")


