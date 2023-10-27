# initial python file(s) just to fuck about in & learn this shi

import cv2
import numpy as np

def main():
    # Connect to webcam (0 = defautl cam)
    cam = cv2.VideoCapture(0)

    # Check if the webcam has been opened successfully
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        # Read a frame from the webcam
        ret, frame = cam.read()
        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a mask for the target color
        # Target color range (RGB: 189, 47, 61)
        lower_color = np.array([160, 30, 25])  # Lower threshold for acceptable colors
        upper_color = np.array([255, 100, 100])  # Upper threshold for acceptable colors
        mask = cv2.inRange(frame_rgb, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around the detected areas
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame in a window
        cv2.imshow('Webcam Feed + Object Detection', frame)
        # Break the loop if space key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()