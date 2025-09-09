
# Import
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image (change the filename as needed)
#image = cv2.imread("./images/deti.bmp", cv2.IMREAD_UNCHANGED)
image = cv2.imread("./images/input.png", cv2.IMREAD_UNCHANGED)

if image is None:
    # Failed Reading
    print("Image file could not be opened!")
    exit(-1)

# Convert to grayscale if the image is not already in gray scale
if len(image.shape) > 2:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the original image
cv2.imshow("Original Image", image)

# Get the min and max pixel intensity values
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)

# Apply Contrast Stretching
# final(x,y) = (original(x,y) - min) / (max - min) * 255
contrast_stretched_image = np.clip((image - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

# Display the contrast stretched image
cv2.imshow("Contrast Stretched Image", contrast_stretched_image)

# Compute histograms for original and contrast stretched images
histSize = 256
histRange = [0, 256]
histImageWidth = 512
histImageHeight = 512
color = (125)  # Gray color for the bars

# Compute the histogram for the original image
hist_original = cv2.calcHist([image], [0], None, [histSize], histRange)

# Compute the histogram for the contrast stretched image
hist_stretched = cv2.calcHist([contrast_stretched_image], [0], None, [histSize], histRange)

# Normalize the histograms to fit within the display window
cv2.normalize(hist_original, hist_original, 0, histImageHeight, cv2.NORM_MINMAX)
cv2.normalize(hist_stretched, hist_stretched, 0, histImageHeight, cv2.NORM_MINMAX)

# Create blank images to display the histograms (black background)
histImage_original = np.zeros((histImageHeight, histImageWidth, 1), np.uint8)
histImage_stretched = np.zeros((histImageHeight, histImageWidth, 1), np.uint8)

# Draw the histogram bars for the original image
binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))

for i in range(histSize):
    cv2.rectangle(histImage_original, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_original[i][0])),
                  (125), -1)

# Draw the histogram bars for the contrast stretched image
for i in range(histSize):
    cv2.rectangle(histImage_stretched, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_stretched[i][0])),
                  (125), -1)

# Display the histograms in separate windows
cv2.imshow('Original Image Histogram', histImage_original)  # Gray histogram on black background
cv2.imshow('Contrast Stretched Image Histogram', histImage_stretched)  # Gray histogram on black background

# Plot original and contrast stretched histograms
plt.figure(figsize=(12, 6))

# Original Image Histogram
plt.subplot(2, 2, 1)
plt.title("Original Image Histogram")
plt.plot(hist_original, color='r')
plt.xlim(histRange)

# Contrast Stretched Image Histogram
plt.subplot(2, 2, 2)
plt.title("Contrast Stretched Image Histogram")
plt.plot(hist_stretched, color='g')
plt.xlim(histRange)

# Show images
plt.subplot(2, 2, 3)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 4)
plt.imshow(contrast_stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
