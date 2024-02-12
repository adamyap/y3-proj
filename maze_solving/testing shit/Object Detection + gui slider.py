# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:48:44 2023

@author: 44747
"""

import cv2
import numpy as np

def nothing(x):
    pass

def main():
    cv2.namedWindow('Color Calibration')
    cv2.createTrackbar('R_low', 'Color Calibration', 0, 255, nothing)
    cv2.createTrackbar('G_low', 'Color Calibration', 0, 255, nothing)
    cv2.createTrackbar('B_low', 'Color Calibration', 0, 255, nothing)
    cv2.createTrackbar('R_high', 'Color Calibration', 255, 255, nothing)
    cv2.createTrackbar('G_high', 'Color Calibration', 255, 255, nothing)
    cv2.createTrackbar('B_high', 'Color Calibration', 255, 255, nothing)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        R_low = cv2.getTrackbarPos('R_low', 'Color Calibration')
        G_low = cv2.getTrackbarPos('G_low', 'Color Calibration')
        B_low = cv2.getTrackbarPos('B_low', 'Color Calibration')
        R_high = cv2.getTrackbarPos('R_high', 'Color Calibration')
        G_high = cv2.getTrackbarPos('G_high', 'Color Calibration')
        B_high = cv2.getTrackbarPos('B_high', 'Color Calibration')
        
        lower_color = np.array([B_low, G_low, R_low])
        upper_color = np.array([B_high, G_high, R_high])
        
        mask = cv2.inRange(frame, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Webcam Feed + Object Detection', frame)
        cv2.imshow('Mask', mask)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()