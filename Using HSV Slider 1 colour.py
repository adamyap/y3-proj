
"""
Created on Mon Oct 30 15:06:18 2023

@author: 44747
"""

import cv2
import numpy as np

def nothing(x):
    pass

def main():
    cv2.namedWindow('Color Calibration')
    # Create trackbars for color change
    # Hue range is from 0-179
    cv2.createTrackbar('H_low', 'Color Calibration', 0, 179, nothing)
    cv2.createTrackbar('S_low', 'Color Calibration', 0, 255, nothing)
    cv2.createTrackbar('V_low', 'Color Calibration', 0, 255, nothing)
    cv2.createTrackbar('H_high', 'Color Calibration', 179, 179, nothing)
    cv2.createTrackbar('S_high', 'Color Calibration', 255, 255, nothing)
    cv2.createTrackbar('V_high', 'Color Calibration', 255, 255, nothing)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get trackbar positions
        H_low = cv2.getTrackbarPos('H_low', 'Color Calibration')
        S_low = cv2.getTrackbarPos('S_low', 'Color Calibration')
        V_low = cv2.getTrackbarPos('V_low', 'Color Calibration')
        H_high = cv2.getTrackbarPos('H_high', 'Color Calibration')
        S_high = cv2.getTrackbarPos('S_high', 'Color Calibration')
        V_high = cv2.getTrackbarPos('V_high', 'Color Calibration')
        
        lower_color = np.array([H_low, S_low, V_low])
        upper_color = np.array([H_high, S_high, V_high])
        
        # Create a mask using the HSV frame
        mask = cv2.inRange(hsv, lower_color, upper_color)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding box for each detected contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the original frame with bounding box
        cv2.imshow('Webcam Feed + Object Detection', frame)
        # Display the mask
        cv2.imshow('Mask', mask)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()