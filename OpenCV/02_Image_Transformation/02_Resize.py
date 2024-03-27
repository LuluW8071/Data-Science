""" Image Transformation: Resize, Flip & Crop """

import cv2 as cv

# Load and display image
img = cv.imread("../../assets/Photos/cat_large.jpg")
cv.imshow("cat", img)

# Resizing
"""
Interpolation by deafult is cv.INTER_AREA
Other interpolation methods are:
cv.INTER_LINEAR and cv.INTER_CUBIC(High Quality)
"""

resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow("cat_resized", resized)

# Flipping
"""
-1: Horizontal & Vertical flip,
1: Horizontal flip, 
0: Vertical flip
"""
flip = cv.flip(resized, 1)
cv.imshow("cat_flipped", flip)

# Cropping
# Cropping is just a slicing technique using pixels
cropped = resized[200:400, 200:400]
cv.imshow("cropped_cat", cropped)

cv.waitKey(0)
