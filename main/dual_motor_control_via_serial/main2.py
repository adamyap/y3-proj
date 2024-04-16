import cv2
import numpy as np
from image_rectification import rectify
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from pidtune import *
import time
import serial
import tkinter as tk
import threading

timer_running = False
start_time = None
timer_label = None

def start_timer():
    global timer_running, start_time
    if not timer_running:
        start_time = time.time()
        timer_running = True
        update_timer()

def stop_timer():
    global timer_running
    timer_running = False

def update_timer():
    if timer_running:
        elapsed_time = time.time() - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        timer_label.config(text=formatted_time)
        timer_label.after(1000, update_timer)

def define_path(contours,edges):
    # Define the minimum area for a filled contour
    min_area = 2000
    min_perimeter = 5000

    # Filter contours based on the area
    filled_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.arcLength(cnt, True) > min_perimeter]

    # Create an empty mask to draw the filled contours
    mask = np.zeros_like(edges)

    # Draw the filled contours on the mask
    for cnt in filled_contours:
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

    #cv2.imshow('path',mask)

    # Skeletonize the mask
    skeleton = skeletonize(mask // 255)
    skeleton = img_as_ubyte(skeleton)  # Convert the image back to 8-bit
    
    # Get the coordinates of the skeleton path
    y, x = np.where(skeleton == 255)

    # Create an array of evenly spaced indices
    indices = np.linspace(0, len(x) - 1, 100,dtype=int)

    # Sample the x and y arrays with these indices
    sample_x = x[indices]
    sample_y = y[indices]

    points = list(zip(sample_x, sample_y))

    # Find contours in the skeleton
    contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.001*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        for point in approx:
            distances = [np.sqrt((point[0][0] - p[0])**2 + (point[0][1] - p[1])**2) for p in points]
            if np.min(distances) > 10:
                points.append(tuple(point[0]))

    return points, contours

def order_points(image,points):

    lower_yellow = np.array([20, 180, 80])
    upper_yellow = np.array([30, 255, 255])

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a binary mask where the color is within the range
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, assuming it's the yellow object
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the center of the largest contour
    M = cv2.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    ordered_points = [[cX,cY]]

    while points:
        last_point = ordered_points[-1]
        distances = [np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2) for point in points]
        nearest_point = points.pop(np.argmin(distances))
        ordered_points.append(nearest_point)

    return ordered_points

def define_hole(contours,edges):
    # Define the min and max area for a filled contour
    min_area = 100
    max_area = 10000

    # Filter contours based on the area
    circle_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area]

    # Create an empty mask to draw the filled contours
    mask = np.zeros_like(edges)

    # Draw the filled contours on the mask
    for cnt in circle_contours:
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

    #cv2.imshow('hole',mask)

    # Set up the SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = True
    params.minCircularity = 0.75

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(mask)

    return keypoints

def define_wall(contours):
    # Define the minimum area for a filled contour
    max_area = 1000

    # Filter contours based on the area
    wall_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area]

    return wall_contours

def locate_ball(frame,edges):
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only red colors
    lower_red1 = np.array([0, 120, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 60])
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
    global x_integral
    global y_integral
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
    KpX,KiX,KdX = 1.1,5.5,0.9
    KpY,KiY,KdY = 2.0,5.5,-1.6
    x_integral = 0
    y_integral = 0
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
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # Define walls
            #wall_contours = define_wall(contours)
            #for cnt in wall_contours:
            #    cv2.drawContours(image, [cnt], -1, (255, 155, 0), 1)
            # Define and draw holel
            keypoints = define_hole(contours,edges)
            cv2.drawKeypoints(image, keypoints, image, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # Define and draw path
            path_points,path_contours = define_path(contours,edges)
            ordered_points = order_points(image, path_points)
            # Separate path points into x and y arrays
            x_path = np.array([point[0] for point in ordered_points])
            y_path = np.array([point[1] for point in ordered_points])
            # Label each point on the path
            for i,point in enumerate(ordered_points):
                cv2.circle(image, tuple(map(int, point)), 3, (255, 255, 255), -1)
                cv2.putText(image, str(i), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            # Draw the path contour
            for contour in path_contours:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)

            processed_image = image.copy()
            cv2.imshow('Working Image', image)
            image_captured = True
        
        elif image_captured:
            ball_contours = locate_ball(frame,edges)
            if ball_contours is not None:
                image = processed_image.copy()

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
                x_nearest, y_nearest = x_path[0], y_path[0]

                # Draw a line from the point to the nearest point on the path
                cv2.line(image, (center[0], center[1]), (x_nearest, y_nearest), (0, 255, 0), 1)

                # Calculate the Euclidean distance to nearest point
                euclidean_distance = np.sqrt((x_nearest - center[0])**2 + (y_nearest - center[1])**2)

                # Calculate the Euclidean distance to all points
                euclidean_distances = np.array([np.sqrt((x - center[0])**2 + (y - center[1])**2) for x, y in zip(x_path, y_path)])

                # Display the Euclidean distance next to the line
                cv2.putText(image, f'Distance: {euclidean_distance:.2f}', (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # Calculate the x and y distances
                x_distance = (x_nearest - center[0])
                y_distance = -(y_nearest - center[1])

                # Display the x and y distances
                cv2.putText(image, f'X Distance: {x_distance:.2f}', (center[0], center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(image, f'Y Distance: {y_distance:.2f}', (center[0], center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # If the Euclidean distance is less than a certain threshold, consider the point as visited
                close_points = np.where(euclidean_distances < 40)[0]
                if close_points.size > 0:
                    first_close_point = close_points[0]
                    for i in range(first_close_point + 1):
                        cv2.circle(processed_image,(x_path[i], y_path[i]),3,(70,70,70),-1)
                    # Remove the point from the path
                    x_path = np.delete(x_path, slice(0, first_close_point + 1))
                    y_path = np.delete(y_path, slice(0, first_close_point + 1))

                dt = time.time() - interval_time
                interval_time = time.time()
                x_integral += x_distance * dt
                y_integral += y_distance * dt
                PDx = PIDcontrol(KpX, KiX, KdX, x_distance, x_integral, velocity[0])
                PDy = PIDcontrol(KpY, KiY, KdY, y_distance, y_integral, velocity[1])
                motorx = max(min(PDx, 450), -450)
                motory = max(min(PDy, 450), -450)
                print(x_distance,y_distance)
                print(x_integral,y_integral)
                print(motorx,motory)
                send_position('A', motory)
                send_position('B', motorx)
                interval_time = time.time()
            
            cv2.imshow('Working Image', image)

        #cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == 27: #ESC key
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def start_solve():
    global x_integral
    global y_integral
    global start_solve
    start_solve = True
    x_integral = 0
    y_integral = 0

def PIDcontrol(Kp,Ki,Kd,distance,integral,velocity):
    return Kp*distance + Ki*integral + Kd*(-velocity)

def send_position(motor, position):
    """
    Sends a motor position command to the Arduino.
    :param motor: 'A' or 'B', indicating which motor to control
    :param position: The desired position as an integer
    """
    command = f"{motor}{position}\n"  # Format the command string
    ser.write(command.encode())  # Encode and send the command
    print(f"Sent command: {command}")

def create_gui():
    root = tk.Tk()
    root.title("Main")

    root.geometry("200x200")
    
    timer_label = tk.Label(root, text="00:00:00", font=("Helvetica", 16))
    timer_label.pack()

    button1 = tk.Button(root, text="Start Image Processing", command=lambda: threading.Thread(target=run_image_processing).start())
    #button1 = tk.Button(root, text="Start Image Processing", command=lambda: threading.Thread(target=run_image_processing).start())
    button1.pack()
    button2 = tk.Button(root, text="Start Solve", command=lambda: [start_solve(), start_timer()])
    button2.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
    

