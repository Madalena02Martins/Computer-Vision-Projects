import cv2
import numpy as np

# Load the image
capture = cv2.VideoCapture(0)
char = ord(' ')

while char != ord('q'):
    _, image = capture.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    # Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)
    # Print the detected markers
    print("Detected markers:\n", ids)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
    cv2.imshow('Detected Markers', image)
    char = cv2.waitKey(33)

cv2.destroyAllWindows()