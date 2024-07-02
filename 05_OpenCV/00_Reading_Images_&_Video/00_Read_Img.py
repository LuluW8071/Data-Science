""" Read Images using OpenCV """

# Import openCV library
import cv2 as cv

# Read image using cv.imread()
# NOTE: Gives assertion error if it failes to read image
img1 = cv.imread("../../assets/Photos/cat.jpg")
img2 = cv.imread("../../assets/Photos/cat_large.jpg")

# Display image using cv.imshow('Winname', var used to store img data)
cv.imshow('Cat', img1)

# ==============================================================================================
# NOTE: 
# This is a large image and 
# it is possible that it won't contain size of actual size to your actual monitor window
# cv.imshow('Cat', img2)
# ==============================================================================================

# Delay to wait for key to be pressed
cv.waitKey(0)

# Run the script using command
""" >>> python 00_Read_Img.py """  