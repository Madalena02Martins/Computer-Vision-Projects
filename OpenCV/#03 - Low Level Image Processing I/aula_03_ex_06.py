
# Import
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the RGB image
image = cv2.imread("./images/deti.jpg", cv2.IMREAD_COLOR)

if image is None:
    # Failed Reading
    print("Image file could not be opened!")
    exit(-1)

# Split the image into its B, G, R components
B, G, R = cv2.split(image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create images for each channel
# For blue channel (set G and R to 0)
blue_image = cv2.merge([B, np.zeros_like(B), np.zeros_like(B)])

# For green channel (set B and R to 0)
green_image = cv2.merge([np.zeros_like(G), G, np.zeros_like(G)])

# For red channel (set B and G to 0)
red_image = cv2.merge([np.zeros_like(R), np.zeros_like(R), R])

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Blue Channel", blue_image)
cv2.imshow("Green Channel", green_image)
cv2.imshow("Red Channel", red_image)

# Calculate histograms for each channel
histSize = 256
histRange = [0, 256]

# Blue channel histogram
hist_B = cv2.calcHist([B], [0], None, [histSize], histRange)

# Green channel histogram
hist_G = cv2.calcHist([G], [0], None, [histSize], histRange)

# Red channel histogram
hist_R = cv2.calcHist([R], [0], None, [histSize], histRange)

# Grayscale histogram
hist_gray = cv2.calcHist([gray_image], [0], None, [histSize], histRange)

# Grayscale histogram
hist_img = cv2.calcHist([image], [0], None, [histSize], histRange)

# Define histogram parameters
histSize = 256
histRange = [0, 256]
histImageWidth = 512
histImageHeight = 512
color = (125)  # Gray color for the bars

# Function to create and display histogram
def draw_histogram(channel_hist, window_name):
    # Normalize the histogram to fit within the display window
    cv2.normalize(channel_hist, channel_hist, 0, histImageHeight, cv2.NORM_MINMAX)
    
    # Create a blank image for the histogram
    hist_image = np.zeros((histImageHeight, histImageWidth, 1), np.uint8)
    
    # Width of each histogram bar
    binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))
    
    # Draw the histogram bars
    for i in range(histSize):
        cv2.rectangle(hist_image, (i * binWidth, histImageHeight),
                      ((i + 1) * binWidth, histImageHeight - int(channel_hist[i][0])),  # Fix here
                      (125), -1)
    
    # Display the histogram
    cv2.imshow(window_name, hist_image)

# Calculate and display histograms for each channel
hist_B = cv2.calcHist([B], [0], None, [histSize], histRange)
draw_histogram(hist_B, "Blue Channel Histogram")

hist_G = cv2.calcHist([G], [0], None, [histSize], histRange)
draw_histogram(hist_G, "Green Channel Histogram")

hist_R = cv2.calcHist([R], [0], None, [histSize], histRange)
draw_histogram(hist_R, "Red Channel Histogram")

hist_gray = cv2.calcHist([gray_image], [0], None, [histSize], histRange)
draw_histogram(hist_gray, "Grayscale Image Histogram")

hist_img = cv2.calcHist([image], [0], None, [histSize], histRange)
draw_histogram(hist_img, "Image Histogram")

# Plot histograms using matplotlib
plt.figure(figsize=(12, 10))

# Original Image Histogram
plt.subplot(3, 2, 1)
plt.title("Image Histogram")
plt.plot(hist_img, color='m')
plt.xlim(histRange)

# Grayscale Histogram
plt.subplot(3, 2, 2)
plt.title("Grayscale Image Histogram")
plt.plot(hist_gray, color='k')
plt.xlim(histRange)

# Blue Histogram
plt.subplot(3, 2, 3)
plt.title("Blue Channel Histogram")
plt.plot(hist_B, color='b')
plt.xlim(histRange)

# Green Histogram
plt.subplot(3, 2, 4)
plt.title("Green Channel Histogram")
plt.plot(hist_G, color='g')
plt.xlim(histRange)

# Red Histogram
plt.subplot(3, 2, 5)
plt.title("Red Channel Histogram")
plt.plot(hist_R, color='r')
plt.xlim(histRange)

# Leave the last subplot empty or add another histogram if desired
plt.subplot(3, 2, 6)  # This could remain empty or you can plot something here

# Apply tight_layout
plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
