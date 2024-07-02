""" Resize Images using OpenCV """

import cv2 as cv

# Create a function to resize the iamges and frame
def rescaleFrame(frame, scale = 0.75):
    """
    Resize the input frame.

    Parameters:
        frame (numpy.ndarray): Input image/frame to be resized.
        scale (float): Scale factor for resizing the image (default is 0.75).

    Returns:
        numpy.ndarray: Resized image/frame.
    """
    # Calculate the new dimensions based on the specified scale
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    # Resize the image using the calculated dimensions
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

# =============================================
# Read the image and video file/live_video
img = cv.imread("../../assets/Photos/cat_large.jpg")
capture = cv.VideoCapture("../../assets/Videos/dog.mp4")
# capture = cv.VideoCapture(0)              # Live WebCam Vid
# =============================================

# Display the original image
# cv.imshow("Cat", img)

# Display the resized image
cv.imshow("Cat", rescaleFrame(img, scale = 0.25))

# Loop to read and resize frames from the video file
while True:
    # Read a frame from the video capture object
    isTrue, frame = capture.read()
    
    # Resize the frame
    frame_resized = rescaleFrame(frame)
    
    # Display the resized frame
    # cv.imshow("Video_Resized", frame_resized)
    cv.imshow("Live_Video_Resized", frame_resized)
    
    # Check for the key pressed, if 'd' is pressed, break out of the loop
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

    # Display the resized frame
    # cv.imshow("Video_Resized", frame_resized)
    cv.imshow("Live_Video_Resized", frame_resized)
    
    # Check for the key pressed, if 'd' is pressed, break out of the loop
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

# Release the video capture object after the loop to free up resources
capture.release()

# Close all OpenCV windows
cv.destroyAllWindows()

# Wait for a key press to close the image window
cv.waitKey(0)

# Run the script using command
""" >>> py .\02_Resize.py """