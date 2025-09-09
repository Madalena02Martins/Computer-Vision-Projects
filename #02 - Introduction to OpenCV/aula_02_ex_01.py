
#import
import numpy as np
import cv2

# Read the image
image = cv2.imread("./images/lena.jpg", cv2.IMREAD_UNCHANGED)
image_copy=image.copy()

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

# Image characteristics
image_array = image.shape
height = image_array[0]
width = image_array[1]

# Check if the image is grayscale or RGB
if len(image_array) == 2:  # Grayscale
    channels = 1
else:  # RGB
    channels = image_array[2]

print("Image Size: (%d,%d,%d)" % (height, width, channels))
print("Image Type: %s" % (image.dtype))


# Loop through the image
for i in range(0,height):
    for j in range(0,width):
        if channels == 1:  # Grayscale
            px = image_copy[i, j]
            if px < 128:
                image_copy[i, j] = 0
        else:  # RGB
            px = image_copy[i, j, 0]
            if px < 128:
                image_copy[i, j] = [0, 0, 0]
# Image characteristics
image_array_copy = image_copy.shape
height = image_array_copy[0]
width = image_array_copy[1]

if len(image_array_copy) == 2:
    channels_copy = 1
else:
    channels_copy = image_array_copy[2]

print("Image Copy Size: (%d,%d,%d)" % (height, width, channels))
print("Image Copy Type: %s" % (image.dtype))


# Show the image
#cv2.imshow( "Display window", image )
#cv2.imshow( "Display window2", image_copy)
cv2.imshow ("Original/Alterada",np.hstack([image, image_copy]))
# Wait
cv2.waitKey(0)

# Fechar todas as janelas
cv2.destroyAllWindows()
