""" Drawing Contour Detection using Threshold """

import cv2 as cv
import numpy as np

img = cv.imread("../../assets/Photos/cat.jpg")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create a blank image with the same dimensions as the input image
blank = np.zeros(img.shape, dtype="uint8")

# Apply Gaussian blur to the grayscale image
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)

# Apply thresholding to the grayscale image
# Pixels with intensity >= 125 are set to 255 (white), and others to 0 (black)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Thresholded image", thresh)

# Find contours in the Canny edge-detected image
contours, hierarchies = cv.findContours(
    thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours found')

# Draw contours on the blank image
# Contours are drawn in red color (0, 0, 255) with a thickness of 1 pixel
"""
(blank): Image where contours will be drawn.
(contours): List of contours to be drawn.
(-1): Draw all contours in the list.
((0, 0, 255)): Color of contours in BGR format (red).
(1): Thickness of contour lines (1 pixel).
"""
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("blank", blank)

cv.waitKey(0)

# ==================================
# NOTE:
# Generally, it is better to perform first Canny edge detection method first instead of thresholding.
# The Canny edge detection is preferred over direct thresholding for contour detection due to:
# 1. Noise Robustness: Gaussian smoothing reduces noise.
# 2. Accurate Edge Localization: Identifies maximum gradient points.
# 3. Automatic Thresholding: Eliminates manual threshold selection.
# 4. Complete Edge Detection: Produces continuous contours.
# 5. Variable Edge Widths: Detects edges of varying strengths.
# Overall, Canny provides robust contour detection addressing noise, thresholding, and edge continuity.
