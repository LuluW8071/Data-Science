""" Split & Merge Color Channels """

import cv2 as cv
import numpy as np 

# Read and display the image
img = cv.imread("../../assets/Photos/park.jpg")
cv.imshow("BGR image", img)

# Split the image into its B, G, R channels
b, g, r = cv.split(img)

# Display each individual channel
cv.imshow("Blue", b)
cv.imshow("Green", g)
cv.imshow("Red", r)

"""
NOTE: Analyzing the splitted image 
- In the Blue image, the sky appears as:
    Highly whited areas indicate regions filled with blue pixels
- In the Green image, the grass appears as:
    Highly whited areas indicate regions filled with green pixels
- In the Red image, objects with reddish tones appear as:
    Highly whited areas indicate regions filled with red pixels

Darker areas indicate regions with fewer or no blue, green or red pixels
"""

# Merging B, G, R values into one
merge = cv.merge([b, g, r])
cv.imshow("Merged image", merge)

# Print the shape of the original image and its channels
print('Original Image Shape:', img.shape)    # (427, 640, 3) ---> (height, width, color_channel)
print(b.shape,g.shape,r.shape, sep="\n")     # (427, 640) : grayscaled have color_channel = 1
print('Merged Image Shape:', merge.shape)    # (427, 640, 3) same as original
cv.waitKey(0)