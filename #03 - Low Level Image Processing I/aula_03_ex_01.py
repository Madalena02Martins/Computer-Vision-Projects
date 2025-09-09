
#import
import sys
import numpy as np
import cv2

# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
image = cv2.imread( "./images/lena.jpg", cv2.IMREAD_UNCHANGED )

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)


image_array = image.shape
height = image_array[0]
width = image_array[1]

if len(image_array) == 2:
    channels = 1
else:
    channels = image_array[2]

# print some features
print("Image Copy Size: (%d,%d,%d)" % (height, width, channels))
print("Image Copy Type: %s" % (image.dtype))
print("Number of elements : %d" % image.size)

# Display the image
#cv2.namedWindow("Original Image")
cv2.imshow("Original Image", image)

# Loop through the image
for i in range(0, height):
    if i % 20 == 0:
        if channels == 1:
            cv2.line(image, (0,i), ( width, i), (255, 255, 255), 1)  # cor branca para grayscale, espessura 1
        else:
            cv2.line(image, (0,i), (width,i), (125, 125, 125), 1)  # cor verde para colorida, espessura 1

for j in range(0, width):
    if j % 20 == 0:
        if channels == 1:
            cv2.line(image, (j,0), (j ,height), (255, 255, 255), 1)  # cor branca para grayscale, espessura 1
        else:
            cv2.line(image, (j,0), (j ,height), (125, 125, 125), 1)  # cor verde para colorida, espessura 1


cv2.imshow("Grid Image", image)
cv2.imwrite("image_with_grid.jpg", image)


# draw horizontal lines
# for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
# 	y = int(round(y))
# 	cv.line(img, (0, y), (w, y), color=color, thickness=thickness)


cv2.waitKey(0)
cv2.destroyAllWindows()
