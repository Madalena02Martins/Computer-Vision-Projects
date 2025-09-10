
# Import
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread("./images/TAC_PULMAO.bmp", cv2.IMREAD_UNCHANGED)

if image is None:
    # Failed Reading
    print("Image file could not be opened!")
    exit(-1)
    
# Image characteristics
if len (image.shape) > 2:
	print ("The loaded image is NOT a GRAY-LEVEL image !")
	exit(-1)

# Display the original image
cv2.imshow("Original Image", image)

# print some features
height, width = image.shape
nchannels = 1
print("Image Size: (%d,%d)" % (height, width))
print("Image Type: %d" % nchannels)
print("Number of elements : %d" % image.size)

print("Image Size: (%d,%d)" % (height, width))

# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Display the equalized image
cv2.imshow("Equalized Image", equalized_image)

# Compute histograms for original and equalized images
histSize = 256
histRange = [0, 256]
histImageWidth = 512
histImageHeight = 512
color = (125)  # Gray color for the bars

# Compute the histogram for the original image
hist_original = cv2.calcHist([image], [0], None, [histSize], histRange)

# Compute the histogram for the equalized image
hist_equalized = cv2.calcHist([equalized_image], [0], None, [histSize], histRange)

# Normalize the histograms to fit within the display window
cv2.normalize(hist_original, hist_original, 0, histImageHeight, cv2.NORM_MINMAX)
cv2.normalize(hist_equalized, hist_equalized, 0, histImageHeight, cv2.NORM_MINMAX)

# Create blank images to display the histograms (black background)
histImage_original = np.zeros((histImageHeight, histImageWidth, 1), np.uint8)
histImage_equalized = np.zeros((histImageHeight, histImageWidth, 1), np.uint8)

# Draw the histogram bars for the original image
binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))

for i in range(histSize):
    cv2.rectangle(histImage_original, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_original[i][0])),
                  (125), -1)

# Draw the histogram bars for the equalized image
for i in range(histSize):
    cv2.rectangle(histImage_equalized, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_equalized[i][0])),
                  (125), -1)

# Display the histograms in separate windows
cv2.imshow('Original Image Histogram', histImage_original)  # Gray histogram on black background
cv2.imshow('Equalized Image Histogram', histImage_equalized)  # Gray histogram on black background

# Plot original and equalized histograms
plt.figure(figsize=(12, 6))

# Original Image Histogram
plt.subplot(2, 2, 1)
plt.title("Original Image Histogram")
plt.plot(hist_original, color='r')
plt.xlim(histRange)

# Equalized Image Histogram
plt.subplot(2, 2, 2)
plt.title("Equalized Image Histogram")
plt.plot(hist_equalized, color='g')
plt.xlim(histRange)

# Show images
plt.subplot(2, 2, 3)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 4)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
