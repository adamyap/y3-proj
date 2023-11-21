import cv2
import numpy as np
import time

# Global variables for the frame read from the webcam and the color range
frame = None
lower_color = None
upper_color = None

def get_color(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frame[y,x]
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)
        print("HSV value:", hsv_pixel[0][0])

def main():
    global frame, lower_color, upper_color
    # Connect to webcam (0 = default cam)
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # Set the resolution to 720p
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set the frame rate to 60fps
    cam.set(cv2.CAP_PROP_FPS, 60)

    # Check if the webcam has been opened successfully
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_color)

    while True:
        # Start the timer
        start_time = time.time()

        # Read a frame from the webcam
        ret, frame = cam.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only the selected colors
        lower_red1 = np.array([0, 70, 130])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 130])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box for each contour
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)

        # Display the frame in a window
        frame = cv2.resize(frame,(640,360))
        cv2.imshow('image', frame)

        # Break the loop if space key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
