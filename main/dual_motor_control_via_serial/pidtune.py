import cv2
import numpy as np
from image_rectification import rectify
from pidtune import *
import time
import serial
import tkinter as tk
import threading

def locate_ball(frame,edges):
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only red colors
    lower_red1 = np.array([0, 80, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 60])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Define the minimum area for a filled contour
    min_area = 100

    # Filter contours based on the area
    ball_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create an empty mask to draw the filled contours
    mask = np.zeros_like(edges)

    # Draw the filled contours on the mask
    cv2.drawContours(mask, ball_contours, -1, (255), thickness=cv2.FILLED)

    #cv2.imshow('Ball Mask',mask)

    # Merge all the contours into one
    if ball_contours:
        merged_contour = np.concatenate([contour.reshape(-1, 2) for contour in ball_contours])

        # Calculate the convex hull of the merged contour
        hull = cv2.convexHull(merged_contour)
    else:
        hull = None

    return hull

def run_image_processing():
    global KpX, KiX, KdX, KpY, KiY, KdY
    global start_solve
    global ser
    try:
        # Initialize serial connection
        ser = serial.Serial('COM3', 9600)
        # Connect to webcam (0 = default cam)
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        # Set the resolution to 720p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Set the frame rate to 60fps
        cap.set(cv2.CAP_PROP_FPS, 30)

    except:
        print("Error: Could not connect to the camera or the Arduino")
        pass
    # Start the timer
    start_time = time.time()
    interval_time = time.time()
    # Flag to indicate if the image has been captured
    image_captured = False

    # Backup image
    backup_image = cv2.imread('green.jpg')
    cv2.resize(backup_image,(1280,720))

    prev_center = None
    velocity = [0, 0]
    N = 1 # Number of frames to consider for the moving average
    velocities = []
    prev_time = time.time()
    x_integral = 0
    y_integral = 0
    x_distance_prev = 0
    y_distance_prev = 0
    start_solve = False
    while True:
        # Resize the frame to desired size
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = rectify(frame)
        except:
            frame = rectify(backup_image)

        if not image_captured and time.time() - start_time >= 1.5:
            image = frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 15, 50)

            processed_image = image.copy()
            cv2.imshow('Working Image', image)
            image_captured = True

        
        elif image_captured:
            image = processed_image.copy()
            ball_contours = locate_ball(frame,edges)

            # If the target point is defined, draw it on the image
            if target_point is not None:
                cv2.circle(image, target_point, 3, (0, 255, 0), thickness=2)

            if ball_contours is not None:

                # Find the smallest enclosing circle
                (x_ball, y_ball), radius = cv2.minEnclosingCircle(ball_contours)
                center = (int(x_ball), int(y_ball))
                radius = int(radius)

                # Draw the smallest enclosing circle on the frame
                cv2.circle(image, center, 13, (0, 0, 255), thickness=2)

                # Draw crosshair at the center
                cv2.line(image, (center[0] - 5, center[1]), (center[0] + 5, center[1]), (0, 0, 255), 1)
                cv2.line(image, (center[0], center[1] - 5), (center[0], center[1] + 5), (0, 0, 255), 1)

                # Calculate velocity
                if prev_center is not None:
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    dt = time.time() - prev_time  # Time difference in seconds
                    velocities.append((dx/dt, dy/dt))
                    if len(velocities) > N:
                        velocities.pop(0)
                    velocity = [sum(v[i] for v in velocities) / len(velocities) for i in range(2)]
                prev_center = center
                prev_time = time.time()

                # Draw velocity vector
                scale = 0.1  # Adjust this value to change the scale of the velocity vector
                end_point = (int(center[0] + velocity[0]*scale), int(center[1] + velocity[1]*scale))
                cv2.arrowedLine(image, center, end_point, (0, 255, 0), 2)
            
            if start_solve == True:

                # The nearest point on the path is then
                x_nearest, y_nearest = target_point

                # Draw a line from the point to the nearest point on the path
                cv2.line(image, (center[0], center[1]), (x_nearest, y_nearest), (0, 255, 0), 1)

                # Calculate the Euclidean distance to nearest point
                euclidean_distance = np.sqrt((x_nearest - center[0])**2 + (y_nearest - center[1])**2)

                # Display the Euclidean distance next to the line
                cv2.putText(image, f'Distance: {euclidean_distance:.2f}', (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # Calculate the x and y distances
                x_distance = (x_nearest - center[0])
                y_distance = -(y_nearest - center[1])

                # Display the x and y distances
                cv2.putText(image, f'X Distance: {x_distance:.2f}', (center[0], center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(image, f'Y Distance: {y_distance:.2f}', (center[0], center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                dt = time.time() - interval_time
                x_integral += x_distance * dt
                if x_integral > 60: #Saturation code for Ki values
                    x_integral = 60
                elif x_integral < -60:
                    x_integral = -60
                y_integral += y_distance * dt
                if y_integral > 60:
                    y_integral = 60
                elif y_integral < -60:
                    y_integral = -60
                x_derivative = (x_distance-x_distance_prev)/dt
                y_derivative = (y_distance-y_distance_prev)/dt
                PDx = PIDcontrol(KpX, KiX, KdX, x_distance, x_integral, x_derivative)
                PDy = PIDcontrol(KpY, KiY, KdY, y_distance, y_integral, y_derivative)
                print(x_distance_prev,y_distance_prev)
                x_distance_prev = x_distance
                y_distance_prev = y_distance
                motorx = max(min(PDx, 60), -60)
                motory = max(min(PDy, 60), -60)
                print('p',x_distance,y_distance)
                print('i ',x_integral,y_integral)
                print('d ',x_derivative,y_derivative)
                print(motorx,motory)
                send_position(motorx,motory)
                interval_time = time.time()
            
            cv2.namedWindow("Working Image")
            cv2.imshow('Working Image', image)
            cv2.setMouseCallback("Working Image", mouse_callback)

        #cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == 27: #ESC key
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def start_solve():
    global start_solve
    start_solve = True

def mouse_callback(event, x, y, flags, param):
    # If the left mouse button was clicked, update the target point
    if event == cv2.EVENT_LBUTTONDOWN:
        global target_point
        target_point = (x, y)

def PIDcontrol(Kp,Ki,Kd,distance,integral,derivative):
    e = (Kp.get())*distance + (Ki.get())*integral + (Kd.get())*(derivative)
    return e

def send_position(angle1, angle2):
    if -60 <= angle1 <= 60 and -60 <= angle2 <= 60:
        angle1 = angle1 + 60
        angle2 = angle2 + 60
        ser.write(f"x,{angle1}".encode())
        ser.write(f"y,{angle2}".encode())
        print(f"Sent command: {angle1},{angle2}")

def create_gui():
    global KpX, KiX, KdX, KpY, KiY, KdY
    root = tk.Tk()
    root.title("Main")

    button1 = tk.Button(root, text="Start Image Processing", command=lambda: threading.Thread(target=run_image_processing).start())
    button1.pack()
    button2 = tk.Button(root, text="Start Solve", command=start_solve)
    button2.pack()

    # Create sliders for KpX, KiX, KdX, KpY, KiY, KdY
    KpX = tk.DoubleVar(value=0.1)
    tk.Scale(root, from_=0, to=1, resolution=0.01, length=400, orient=tk.HORIZONTAL, label="KpX", variable=KpX).pack()

    KiX = tk.DoubleVar(value=0)
    tk.Scale(root, from_=0, to=1, resolution=0.01, length=400, orient=tk.HORIZONTAL, label="KiX", variable=KiX).pack()

    KdX = tk.DoubleVar(value=0.02)
    tk.Scale(root, from_=0, to=1, resolution=0.01, length=400, orient=tk.HORIZONTAL, label="KdX", variable=KdX).pack()

    KpY = tk.DoubleVar(value=0.1)
    tk.Scale(root, from_=0, to=1, resolution=0.01, length=400, orient=tk.HORIZONTAL, label="KpY", variable=KpY).pack()

    KiY = tk.DoubleVar(value=0)
    tk.Scale(root, from_=0, to=1, resolution=0.01, length=400, orient=tk.HORIZONTAL, label="KiY", variable=KiY).pack()

    KdY = tk.DoubleVar(value=0.02)
    tk.Scale(root, from_=0, to=1, resolution=0.01, length=400, orient=tk.HORIZONTAL, label="KdY", variable=KdY).pack()

    root.mainloop()

# Initialize the target point
target_point = None

if __name__ == "__main__":
    create_gui()
    

