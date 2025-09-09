
#import
import sys
import numpy as np
import cv2
import math

srcPts1 = []
srcPts2 = []


def printImageFeatures(image):
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)


# Read the image from argv
image1 = cv2.imread("./images/lena.jpg" , cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./imagename_tf.jpg", cv2.IMREAD_GRAYSCALE)

if  np.shape(image1) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

if  np.shape(image2) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image1)

printImageFeatures(image2)

cv2.imshow('Original1', image1)

cv2.imshow('Original2', image2)

gray1=image1 #porque a imagem já é em escala de cinza
gray2=image2 #porque a imagem já é em escala de cinza

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(image1,None)
kp2, des2 = sift.detectAndCompute(image2,None)

image1=cv2.drawKeypoints(gray1,kp1,image1)

cv2.imshow('SIFT1', image1) 
cv2.imwrite('sift_keypoints.jpg',image1)

image2=cv2.drawKeypoints(gray2,kp2,image2)

cv2.imshow('SIFT2', image2) 
cv2.imwrite('sift_keypoints.jpg',image2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Remove not so good matches
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Draw matches
im_matches = cv2.drawMatches(image1,kp1,image2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("matches",im_matches)

# Evaluate transform
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

# Use Affine transformation estimation
M_affine = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])  # Needs only 3 points
print("Affine Transformation Matrix:")
print(M_affine)

# Componentes da matriz M_affine
a = M_affine[0, 0]
b = M_affine[0, 1]
c = M_affine[1, 0]
d = M_affine[1, 1]

# Cálculo dos parâmetros
scale_x = math.sqrt(a**2 + b**2)
scale_y = math.sqrt(c**2 + d**2)
rotation_angle = (math.atan2(b, a) * 180) / math.pi

print(f"Escala em x: {scale_x}")
print(f"Escala em y: {scale_y}")
print(f"Ângulo de rotação (em graus): {rotation_angle}")

# Apply the affine transformation to the first image
warp_dst = cv2.warpAffine(image1, M_affine, (image2.shape[1], image2.shape[0]))
cv2.imshow("Warped Image", warp_dst)

# Subtract warped image from image2 to evaluate transformation
difference = cv2.absdiff(warp_dst, image2)
cv2.imshow("Difference Image", difference)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Increase this value to increase accuracy

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # Ratio test
        good_matches.append(m)

# Only consider a percentage of the best matches
numGoodMatches = int(len(good_matches) * 0.1)
good_matches = good_matches[:numGoodMatches]

if len(good_matches) > 3:  # Precisamos de pelo menos 3 correspondências para calcular a transformação
    # Conversão para np.array
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estima a transformação afim com cv2.getAffineTransform ou cv2.estimateAffinePartial2D
    M_affine, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    print("Affine Transformation Matrix:")
    print(M_affine)
    
    # Se M_affine foi calculada corretamente
    if M_affine is not None:
        # Parâmetros da matriz de transformação afim
        a = M_affine[0, 0]
        b = M_affine[0, 1]
        c = M_affine[1, 0]
        d = M_affine[1, 1]

        # Cálculo dos parâmetros de escala e rotação
        scale_x = math.sqrt(a**2 + b**2)
        scale_y = math.sqrt(c**2 + d**2)
        rotation_angle = (math.atan2(b, a) * 180) / math.pi

        # Mostra os resultados
        print(f"Escala em x: {scale_x}")
        print(f"Escala em y: {scale_y}")
        print(f"Ângulo de rotação (em graus): {rotation_angle}")
        
        # Aplicação da transformação afim à imagem
        warp_dst = cv2.warpAffine(image1, M_affine, (image2.shape[1], image2.shape[0]))

        # Mostra a imagem transformada
        cv2.imshow("Imagem Transformada - FLANN ", warp_dst)

        # Exibe a subtração entre a imagem transformada e a imagem de destino
        diff = cv2.absdiff(warp_dst, image2)
        cv2.imshow("Diferenca entre imagens - FLANN", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()
