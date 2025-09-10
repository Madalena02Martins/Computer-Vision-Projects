import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Arrays to store object points and image points from all the images.
objpoints = []    # 3D points in real-world space
left_imgpoints = []   # 2D points in image plane (left image)
right_imgpoints = []  # 2D points in image plane (right image)

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
        cv2.waitKey(500)
    
    return ret, corners

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Read left and right images
left_images = sorted(glob.glob('.//images//left*.jpg'))
right_images = sorted(glob.glob('.//images//right*.jpg'))

# Check that both image lists have the same number of images
if len(left_images) != len(right_images):
    print("Error: The number of left and right images do not match.")
else:
    for left_fname, right_fname in zip(left_images, right_images):
        # Read left and right images
        left_img = cv2.imread(left_fname)
        right_img = cv2.imread(right_fname)

        # Find corners in the left image
        ret_left, left_corners = FindAndDisplayChessboard(left_img)
        # Find corners in the right image
        ret_right, right_corners = FindAndDisplayChessboard(right_img)

        # If both images have corners detected, store points
        if ret_left and ret_right:
            objpoints.append(objp)             # 3D points
            left_imgpoints.append(left_corners)   # 2D points for the left image
            right_imgpoints.append(right_corners) # 2D points for the right image


        # # Define matrices with detected corners
        # left_corners = np.array(left_imgpoints, dtype=np.float32)
        # right_corners = np.array(right_imgpoints, dtype=np.float32)
        # objPoints = np.array(objpoints, dtype=np.float32)

        # Print the shape of the matrices for verification
        print(f"Left corners shape: {left_corners}")
        print(f"Right corners shape: {right_corners}")
        print(f"Object points shape: {objpoints}")

cv2.destroyAllWindows()


