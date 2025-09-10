import numpy as np
import cv2
import glob

# Função para remover a distorção da imagem
def undistort_image(img, intrinsics, distortion):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, intrinsics, distortion, None, new_camera_matrix)
    return undistorted_img

# Função callback para manipular eventos do mouse
def mouse_handler(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordenadas selecionadas: ({x}, {y})")
        params['points'].append((x, y))

        # Calcular a linha epipolar para o ponto selecionado
        img, F, is_left = params['image'], params['F'], params['is_left']
        p = np.asarray([x, y], dtype=np.float32).reshape(-1, 1, 2)
        side = 1 if is_left else 2  # 1 = da esquerda para a direita; 2 = da direita para a esquerda
        epiline = cv2.computeCorrespondEpilines(p, side, F).reshape(-1, 3)[0]
        a, b, c = epiline

        # Definir cor aleatória
        color = np.random.randint(0, 255, 3).tolist()

        # Calcular os pontos para desenhar a linha
        h, w = img.shape[:2]
        x0, y0 = 0, int(-c / b)  # Interseção com o eixo y
        x1, y1 = w, int(-(c + a * w) / b)  # Interseção com o lado direito

        # Desenhar a linha epipolar
        cv2.line(img, (x0, y0), (x1, y1), color, 2)
        cv2.imshow(params['window_name'], img)

# Carregar os parâmetros de calibração
calibration_params = np.load("stereoParams.npz")
intrinsics1 = calibration_params['intrinsics1']
distortion1 = calibration_params['distortion1']
intrinsics2 = calibration_params['intrinsics2']
distortion2 = calibration_params['distortion2']
F = calibration_params['F']

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

    # Criar cópias para desenhar as linhas epipolares
    epilines_left = undistorted_left.copy()
    epilines_right = undistorted_right.copy()

    # Configurar callbacks de mouse
    left_params = {'points': [], 'image': epilines_right, 'F': F, 'is_left': True, 'window_name': 'Right Image - Epilines'}
    right_params = {'points': [], 'image': epilines_left, 'F': F, 'is_left': False, 'window_name': 'Left Image - Epilines'}

    cv2.imshow('Left Image - Undistorted', undistorted_left)
    cv2.imshow('Right Image - Undistorted', undistorted_right)

    cv2.setMouseCallback('Left Image - Undistorted', mouse_handler, left_params)
    cv2.setMouseCallback('Right Image - Undistorted', mouse_handler, right_params)

    # Aguarda interação e fecha janelas
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
else:
    print("Não foram encontradas imagens estéreo para processamento.")
