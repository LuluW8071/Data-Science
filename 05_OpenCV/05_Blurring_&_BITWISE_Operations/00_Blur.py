import cv2 as cv 

img = cv.imread("../../assets/Photos/cats.jpg")
cv.imshow("cats", img)

# Averaging:
"""
In image processing, averaging is a type of filtering technique used for smoothing or blurring images.
It involves defining a kernel window (also known as a filter or mask) over a specific portion of the image.
Each pixel in the image is then replaced with the average value of its neighboring pixels within the kernel window.
This process helps to reduce noise and fine details in the image, resulting in a smoother appearance.
The size and shape of the kernel window determine the extent of smoothing or blurring applied to the image.
Commonly used averaging filters include the box filter, also known as the mean filter, which uses a square kernel window.
"""
average = cv.blur(img, (5, 5))              # Parameters: (image, kernel_size)
cv.imshow("Average Blur", average)

# Gaussian Blur:
"""
Gaussian blur is a popular blurring technique that applies weighted averaging based on a Gaussian distribution.
Unlike average blur, Gaussian blur assigns more weight to the central pixels of the kernel window and less weight to the outer pixels.
This results in a smoother blur effect that preserves edges and details better than simple averaging.
"""
gauss = cv.GaussianBlur(img, (5, 5), 0)     # Parameters: (image, kernel_size, sigmaX)
cv.imshow("Gaussian Blur", gauss)

# Median Blur:
"""
Median blur is a blurring technique that replaces each pixel in the image with the median value of its neighboring pixels within a defined kernel window.
This technique is effective in removing salt-and-pepper noise and preserving edges in the image.
Median blur is particularly useful when dealing with images corrupted by impulse noise.
"""
median = cv.medianBlur(img, 3)              # Parameters: (image, kernel_size)
cv.imshow("Median Blur", median)

# Bilateral Filter:
"""
The bilateral filter is a non-linear filtering technique that preserves edges while reducing noise in images.
It achieves this by applying two Gaussian filters: one in the spatial domain and another in the intensity domain.
The spatial Gaussian filter smooths pixels based on their spatial distance from each other within the kernel window.
The intensity Gaussian filter smooths pixels based on the difference in intensity values between them.
By considering both spatial and intensity information, the bilateral filter effectively removes noise while preserving edges.
The parameters for the bilateral filter include the diameter of the pixel neighborhood (5 in this case),
and the sigma values for the spatial and intensity Gaussian functions (both set to 15 in this example).
"""
bilateral = cv.bilateralFilter(img, 5, 15, 15)      # Parameters: (image, diameter, sigmaColor, sigmaSpace)
cv.imshow("Bilateral Blur", bilateral)

cv.waitKey(0)
