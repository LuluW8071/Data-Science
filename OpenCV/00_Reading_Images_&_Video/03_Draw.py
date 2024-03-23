import numpy as np 
import cv2 as cv

# Creating Blank Image with color black #000000
blank_image = np.zeros((500, 500, 3), dtype="uint8")    # width, height, color_channels[RGB=3]

# Display the blank image
# cv.imshow("Blank", blank_image)

# ===============================================
# 1. Draw a green square at a certain location
blank_image[100:200, 200:300] = 0, 255, 0  # (B, G, R) = (0, 255, 0) represents green
# cv.imshow("Blank", blank_image)

# 2. Draw a rectangle using cv.rectangle
"""
cv.rectangle(image, pt1, pt2, color, thickness=None, lineType=None, shift=None)
pt1 and pt2 are the diagonal points of the rectangle
color is represented in (B, G, R) format
if thickness = +ve value : border increase else when -ve : filled inside
"""
cv.rectangle(blank_image, 
             (0, 0), 
             (100, 100), 
             (0, 0, 255),               # Drawing a red rectangle
             thickness=-2)              # Filled Rectangle : -ve
cv.imshow("Rectangle", blank_image)

cv.rectangle(blank_image, 
             (350, 350), 
             (blank_image.shape[1]//2, blank_image.shape[0]//2), 
             (0, 255, 255),              # Drawing a yellow rectangle
             thickness=2)                # Border thickness : +ve
cv.imshow("Rectangle", blank_image)

cv.waitKey(0)

# Run the script using command
""" >>> python 03_Draw.py """  