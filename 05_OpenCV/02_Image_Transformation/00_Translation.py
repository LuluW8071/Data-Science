""" Image Transformation: Translation """

# Import libraries
import cv2 as cv 
import numpy as np

# Load image
img = cv.imread("../../assets/Photos/park.jpg")
cv.imshow("Park", img)

# Translate Function
def translate(img, x, y):
    """ Translate the image by shifting it by x pixels horizontally and y pixels vertically.
    
    Args:
        img: The input image.
        x: No. of pixels to shift the image horizontally.
           +ve values shift the image to the right.
           -ve values shift the image to the left.
        y: No. of pixels to shift the image vertically.
           +ve values shift the image downwards.
           -ve values shift the image upwards.
        
    Returns:
        Translated image.
    """

    # Define the translation matrix using numpy
    # "float32" is default type of image data
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    
    # Get the dimensions of the input image
    dim = (img.shape[1], img.shape[0])
    
    # Apply the affine transformation to translate the image
    return cv.warpAffine(img, transMat, dim)

# Translating the image
translated = translate(img, 100, 200)
cv.imshow("Translate", translated)

cv.waitKey(0)