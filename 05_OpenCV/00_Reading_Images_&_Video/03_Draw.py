import numpy as np 
import cv2 as cv

# Creating Blank Image with color black #000000
blank_image = np.zeros((500, 500, 3), dtype="uint8")    # width, height, color_channels[RGB=3]

# ===============================================
# 1. Draw a green square at a certain location
# Define the green color in (B, G, R) format
green_color = (0, 255, 0)
# Fill the region from (100, 200) to (200, 300) with the green color
blank_image[100:200, 200:300] = green_color 

# 2. Draw a rectangle using cv.rectangle
# Draw a filled red rectangle from (0, 0) to (100, 100)
cv.rectangle(blank_image, 
             (0, 0), 
             (100, 100), 
             (0, 0, 255),               # Red color
             thickness=-2)              # Filled Rectangle : -ve thickness

# Draw a yellow rectangle with a border from (350, 350) to (250, 250) (half of the image size)
cv.rectangle(blank_image, 
             (350, 350), 
             (blank_image.shape[1]//2, blank_image.shape[0]//2), 
             (0, 255, 255),              # Yellow color
             thickness=2)                # Border thickness : +ve

# 3. Draw a circle using cv.circle
# Draw a circle with center at (250, 250) and radius 50
cv.circle(blank_image,
          (250, 250),
          50,
          (0, 255, 0),                  # Green color
          thickness=2)

# 4. Draw a line using cv.line
# Draw a line from (100,200) to the center of the image
cv.line(blank_image,
        (100,200),
        (blank_image.shape[1]//2, blank_image.shape[0]//2),
        (0, 255, 0),                    # Green color
        thickness=2)

# Display the created shapes
cv.imshow("Shapes", blank_image)

# 5. Display the text on shapes using cv.putText
# Display the text 'Display text using openCV' at position (225, 225) with font scale 1.0 and thickness 2
cv.putText(blank_image, 
           'Display text using openCV', 
           (225, 225), 
           cv.FONT_HERSHEY_TRIPLEX, 
           1.0, 
           (0, 255, 255),                # Yellow color
           thickness=2)

# Display the created shapes and text
cv.imshow("Draw Shapes and Text", blank_image)

cv.waitKey(0)

# Run the script using command
""" >>> python 03_Draw.py """  
