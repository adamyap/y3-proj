import cv2
import numpy as np

# Connect to webcam (0 = default cam)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Set the resolution to 720p  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set the frame rate to 60fps
cap.set(cv2.CAP_PROP_FPS, 30)

def print_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('HSV:', hsv_image[y, x])

# Create a window
cv2.namedWindow('video')

# Set mouse callback function to print HSV value of pixel
cv2.setMouseCallback('video', print_hsv)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
    cv2.imshow('video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
