""" Image Transformation: Rotation """

import cv2 as cv 
import numpy as np

img = cv.imread("../../assets/Photos/park.jpg")
cv.imshow("Park", img)

# Rotate Function
def rotate(img, angle, rotPoint=None):
    """Rotate the image by a specified angle around a given rotation point.
    
    Args:
        img: The input image.
        angle: The angle of rotation in degrees. 
               +ve values rotate the image anti-clockwise 
               -ve values rotate it clockwise
        rotPoint: The point around which the image will be rotated. 
                  By default, it is set to the center of the image.
    
    Returns:
        Rotated image.
    """
    (height, width) = img.shape[:2]

    # If rotation point is not provided, assume it as the center of the image
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    # Get the rotation matrix
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dim = (width, height)

    # Apply the affine transformation to rotate the image
    return cv.warpAffine(img, rotMat, dim)

# Rotating the image
rotate_img_clk = rotate(img, -45)           # Negative angle for clockwise rotation
rotate_img_anticlk = rotate(img, 45)        # Positive angle for anti-clockwise rotation
cv.imshow("AntiClockwise_Rotation", rotate_img_anticlk)
cv.imshow("Clockwise_Rotation", rotate_img_clk)

# Rotating an already rotated image
rotate_rotated = rotate(rotate_img_anticlk, -45)  # Rotating the anti-clockwise rotated image by -45 degrees
cv.imshow("Rotate AntiClockwise_Rotation", rotate_rotated)

# NOTE:
# Black sides may appear when rotating an already rotated image,
# as the rotation may cause the image to lose its original aspect ratio,
# resulting in black regions to fill the gaps.

cv.waitKey(0)