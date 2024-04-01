""" Display img using matplotlib and opencv """

import cv2 as cv
import matplotlib.pyplot as plt

# Load an image
img = cv.imread('../../assets/Photos/park.jpg')

# Display image using openCV
cv.imshow('OpenCV', img)

# Display image using maplotlib.pyplot
"""
NOTE:
When working with matplotlib, it displays the image as color inverted.
This is because 
- in OpenCV, default color space is (red, green, blue)
- in matplotlib, default color space is (blue, green, red)
"""

plt.imshow(img)
plt.show()

# ================================
# NOTE:
# In order to display the correct color values in matplpotlib
# Change BCR to RGB values using openCV color shift function
# ================================
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()

cv.waitKey(0)
