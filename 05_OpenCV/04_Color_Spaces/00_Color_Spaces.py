""" Color Spaces: BGR, HSV and LAB """

import cv2 as cv

# Load an image
img = cv.imread("../../assets/Photos/park.jpg")
cv.imshow('BGR Image', img)

# Convert BGR image to GreyScale image
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('GrayScale', grey)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV to BGR', hsv_bgr)

# BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# LAB to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB to BGR', lab_bgr)

# ==========================================
# NOTE:
# You cannot convert greyscaled image to HSV and LAB 
# To do that convert greyscaled to BGR and then to HSV and LAB
# ==========================================

cv.waitKey(0)