""" Split B,G,R Color Channels & Merge with Blank image """

import cv2 as cv
import numpy as np

# Load and display the image
img = cv.imread("../../assets/Photos/lady.jpg")
cv.imshow("Lady", img)

# Create a blank black image
blank = np.zeros(img.shape[:2], dtype='uint8')

# Split the image into its B, G, R channels
b, g, r = cv.split(img)

# Merge blank color channels with respective B, G, and R channels successively
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

# Display merged blank color channel images
cv.imshow("Blue", blue)
cv.imshow("Green", green)
cv.imshow("Red", red)

# Merging and displaying B, G, R values into one image
merge = cv.merge([b, g, r])
cv.imshow("Merged image", merge)

cv.waitKey(0)