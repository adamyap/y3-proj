# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:14:46 2023

@author: 44747
"""

import cv2
import numpy as np
import json
import time

def nothing(x):
    pass

# Function to create trackbars for color calibration
def create_color_trackbars(window_name):
    cv2.namedWindow(window_name)
    saved_values = load_saved_values(window_name)
    cv2.createTrackbar('H_low', window_name, saved_values.get('H_low', 0), 179, nothing)
    cv2.createTrackbar('S_low', window_name, saved_values.get('S_low', 0), 255, nothing)
    cv2.createTrackbar('V_low', window_name, saved_values.get('V_low', 0), 255, nothing)
    cv2.createTrackbar('H_high', window_name, saved_values.get('H_high', 179), 179, nothing)
    cv2.createTrackbar('S_high', window_name, saved_values.get('S_high', 255), 255, nothing)
    cv2.createTrackbar('V_high', window_name, saved_values.get('V_high', 255), 255, nothing)

def get_trackbar_values(window_name):
    H_low = cv2.getTrackbarPos('H_low', window_name)
    S_low = cv2.getTrackbarPos('S_low', window_name)
    V_low = cv2.getTrackbarPos('V_low', window_name)
    H_high = cv2.getTrackbarPos('H_high', window_name)
    S_high = cv2.getTrackbarPos('S_high', window_name)
    V_high = cv2.getTrackbarPos('V_high', window_name)
    return np.array([H_low, S_low, V_low]), np.array([H_high, S_high, V_high])

def load_saved_values(color):
    try:
        with open('trackbar_values.json', 'r') as f:
            data = json.load(f)
        return data.get(color, {})
    except FileNotFoundError:
        return {}

def save_current_values(colors):
    data = {}
    for color in colors:
        lower_color, upper_color = get_trackbar_values(color)
        data[color] = {
            'H_low': int(lower_color[0]),
            'S_low': int(lower_color[1]),
            'V_low': int(lower_color[2]),
            'H_high': int(upper_color[0]),
            'S_high': int(upper_color[1]),
            'V_high': int(upper_color[2])
        }
    with open('trackbar_values.json', 'w') as f:
        json.dump(data, f, indent=4)

def main():
    colors = ['White', 'Black', 'Red', 'Yellow', 'Green', 'Magenta', 'Blue']
    for color in colors:
        create_color_trackbars(color)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize the FPS counter
    fps = 0
    fps_time = time.time()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Calculate FPS
            new_fps_time = time.time()
            fps = 1 / (new_fps_time - fps_time)
            fps_time = new_fps_time

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Process each color
            for color in colors:
                lower_color, upper_color = get_trackbar_values(color)

                mask = cv2.inRange(hsv, lower_color, upper_color)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display FPS on frame
            font_scale = 0.7
            cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Webcam Feed + Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord(' '):  # If spacebar is pressed
                break
    finally:
        save_current_values(colors)  # Save the values when the program exits
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()