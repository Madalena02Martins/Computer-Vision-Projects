import numpy as np
import cv2

def getAruCOMarker(image):
    # Converte para escala de cinza e define dicionário ArUco
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    # Detecta marcadores ArUco
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        print("Detected markers:\n", ids)

    return ids is not None, image, corners

# Define os pontos 3D da base do marcador (coincidindo com as esquinas)
marker_points_3d = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
], dtype=np.float32)

# Define um cubo 3D para renderizar em torno do marcador
size = 1  # Escala do cubo (pode ajustar para o tamanho desejado)
cube_3d = np.float32([
    [0, 0, 0], [0, size, 0], [size, size, 0], [size, 0, 0],  # Base do cubo
    [0, 0, -size], [0, size, -size], [size, size, -size], [size, 0, -size]  # Topo do cubo
])

# Leitura da captura de vídeo
char = ord(' ')
capture = cv2.VideoCapture(0)

while char != ord('q'):
    ret, img = capture.read()

    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    # Detecção de marcadores
    has_marker, img, corners = getAruCOMarker(img)

    if has_marker and len(corners) > 0:
        with np.load('camera.npz') as data:
            intrinsics = data['intrinsics']
            distortion = data['distortion']
        print("Parâmetros carregados:")
        print("Intrinsics:\n", intrinsics)
        print("Distortion:\n", distortion)

        # Processa cada marcador encontrado
        for marker_corners in corners:
            # Extrai as esquinas do marcador detectado
            image_points = np.array(marker_corners[0], dtype=np.float32)

            # Estima a pose do marcador
            retval, rvec, tvec = cv2.solvePnP(marker_points_3d, image_points, intrinsics, distortion)
            
            if retval:
                # Projeta os pontos 3D do cubo para o plano da imagem
                cube_2d, _ = cv2.projectPoints(cube_3d, rvec, tvec, intrinsics, distortion)
                cube_2d = cube_2d.reshape(-1, 2).astype(int)

                # Conectar as arestas do cubo
                for i, j in zip([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4]):
                    cv2.line(img, tuple(cube_2d[i]), tuple(cube_2d[j]), (0, 255, 0), 2)  # Arestas da base e do topo em verde

                # Conectar a base com o topo
                for i, j in zip([0, 1, 2, 3], [4, 5, 6, 7]):
                    cv2.line(img, tuple(cube_2d[i]), tuple(cube_2d[j]), (0, 255, 0), 2)  # Arestas verticais

                print(f"Translations:")
                print(tvec)
                print(f"Rotation:")
                print(rvec)
                

    # Exibe a imagem com o cubo projetado
    cv2.imshow('3D Cubes on ArUco Markers', img)
    char = cv2.waitKey(33) & 0xFF

# Finalizando o processo
capture.release()
cv2.destroyAllWindows()
