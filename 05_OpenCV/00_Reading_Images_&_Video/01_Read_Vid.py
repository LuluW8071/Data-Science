""" Read Videos using OpenCV """

# Import openCV library
import cv2 as cv

# NOTE:
# Open a video capture object. By default, it attempts to access the default webcam (for index 0). 
# You can specify a different webcam index if multiple cameras are connected. 
# Alternatively, you can provide a path to a video file to read from that file instead.

# NOTE: Gives assertion error if it failes to read image
# capture_vid = cv.VideoCapture(0)
capture_vid = cv.VideoCapture("../../assets/Videos/dog.mp4")

# ==============================================================================================
# NOTE: 
# To stream video we use here while true loop
# and capture.read()
# ==============================================================================================
while True:
    # Read a frame from the video capture object
    # 'isTrue' is a boolean variable indicating whether the frame was read successfully
    # 'frame' contains the image data of the current frame
    isTrue, frame = capture_vid.read()
    
    # Display the frame in a window named 'Video'
    cv.imshow('Video', frame)
    
    # Check for the key pressed, if 'd' is pressed, break out of the loop
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

# Release the video capture object after the loop to free up resources
capture_vid.release()

# Close all OpenCV windows
cv.destroyAllWindows()

# Delay to wait for a key to be pressed
cv.waitKey(0)

# Run the script using command
""" >>> python 01_Read_Vid.py """  