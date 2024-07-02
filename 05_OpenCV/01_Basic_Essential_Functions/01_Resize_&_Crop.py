""" Resize and Cropping """

import cv2 as cv 

img = cv.imread("../../assets/Photos/Cat_large.jpg")
cv.imshow("Cat", img)

# Resize the image
resized = cv.resize(img, (800, 640))  # Resize the image to a width of 800 pixels and a height of 640 pixels
cv.imshow("Cat_resize", resized)

# Crop a region of interest (ROI) from the resized image
cropped = resized[300:800, 300:640]   # Define the region to crop, starting from (300, 300) and ending at (800, 640)
cv.imshow("Cropped_cat", cropped)

cv.waitKey(0)