
#import
import numpy as np
import cv2

# Read the image
image = cv2.imread("./images/lena.jpg", cv2.IMREAD_UNCHANGED)

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



def mouse_handler(event, x, y, flags, params):
    if event == cv2.EVENT_RBUTTONDOWN:
        print("right click")
        cv2.circle(image, (x,y), 10,(255, 0, 0), -1)
        cv2.imshow('Image', image)  # Atualizar a imagem com o círculo desenhado



# Exibe a imagem 
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_handler)
# Wait
cv2.waitKey(0)

# Fechar todas as janelas
cv2.destroyAllWindows()
