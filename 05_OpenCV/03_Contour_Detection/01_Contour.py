""" Contour Detection using Threshold """

import cv2 as cv

img = cv.imread("../../assets/Photos/cat.jpg")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
# NOTE: 
# This helps to produce cleaner and more accurate edge maps 
# by reducing noise and enhancing the continuity of edges in the image.
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

# Apply thresholding to the grayscale image (gray).
# Thresholding converts a grayscale image into a binary image.
# NOTE:
# The '125' is the threshold value.
# Pixels with intensity values below 125 are set to 0 (black),
# while pixels with intensity values above or equal to 125 are set to 255 (white).

# The 'ret' variable stores the threshold value used.
# This can be useful if the thresholding method is adaptive,
# but in this case, it's not used further.

# The resulting binary image is stored in the 'thresh' variable.
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Thresholded image", thresh)

# Find contours in the new thresholded grayed image
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
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours found')

cv.waitKey(0)