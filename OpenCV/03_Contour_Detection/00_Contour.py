""" Contour Detection using cv.Canny """

import cv2 as cv

img = cv.imread("../../assets/Photos/cat.jpg")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
# NOTE: 
# This helps to produce cleaner and more accurate edge maps 
# by reducing noise and enhancing the continuity of edges in the image.
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

# Apply Canny edge detection to the blurred image
# NOTE:
# The first threshold (50) is the minimum gradient intensity for a pixel to be considered an edge.
# Pixels with gradients below this value are ignored.

# The second threshold (125) is the maximum gradient intensity to be instantly accepted as an edge.
# Pixels with gradients above this value are immediately considered edges.

# Pixels with gradients between the thresholds are potential edges,
# and their acceptance depends on connectivity to strong edges.
canny = cv.Canny(blur, 50, 125)
cv.imshow("Canny edges", canny)

# Find contours in the Canny edge-detected image
# NOTE:
# Contours are outlines of objects in an image.
# They are represented as a list of points defining the object's boundary.
# Contours are used for object detection, shape recognition, and image segmentation.
"""
 cv.RETR_LIST retrieves all contours without any hierarchy
 cv.RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours
 cv.RETR_EXTERNAL retrieves only the external contours without any internal contours

 cv.CHAIN_APPROX_NONE gets all points of the edges line
 cv.CHAIN_APPROX_SIMPLE gets endpoints of the edges line by compressing horizontal, vertical, and diagonal segments
"""
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours found')

cv.waitKey(0)