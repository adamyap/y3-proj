import cv2
import numpy as np
from image_rectification import rectify
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import time
import serial
import tkinter as tk
import threading
import random

# Timer global variables
timer_running = False
timer_start_time = 0
timer_label = None

def start_solve():
    global x_integral, y_integral, start_solve
    start_solve = True
    x_integral = 0
    y_integral = 0

def update_timer_display(start_time, stop=False):
    global timer_label
    elapsed_time = time.time() - start_time
    formatted_time = "{:0>8}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    if stop:
        print(f"Final time: {formatted_time}")
    if timer_label:
        timer_label.config(text=formatted_time)
    if not stop:
        timer_label.after(1000, update_timer_display, start_time)  # Update every second

def is_endpoint_reached(ball_center, x_path, y_path, threshold_distance=50):
    # Assuming the endpoint is the last point in the path
    distance_to_endpoint = np.sqrt((ball_center[0] - x_path[-1])**2 + (ball_center[1] - y_path[-1])**2)
    return distance_to_endpoint < threshold_distance

def define_path(contours, edges):
    min_area = 2000
    min_perimeter = 5000
    filled_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.arcLength(cnt, True) > min_perimeter]
    mask = np.zeros_like(edges)
    for cnt in filled_contours:
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)
    skeleton = skeletonize(mask // 255)
    skeleton = img_as_ubyte(skeleton)
    y, x = np.where(skeleton == 255)
    indices = np.linspace(0, len(x) - 1, 100, dtype=int)
    sample_x = x[indices]
    sample_y = y[indices]
    points = list(zip(sample_x, sample_y))
    contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for point in approx:
            distances = [np.sqrt((point[0][0] - p[0])**2 + (point[0][1] - p[1])**2) for p in points]
            if np.min(distances) > 10:
                points.append(tuple(point[0]))
    return points, contours

def order_points(image, points):
    lower_yellow = np.array([20, 180, 80])
    upper_yellow = np.array([30, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    ordered_points = [[cX, cY]]
    while points:
        last_point = ordered_points[-1]
        distances = [np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2) for point in points]
        nearest_point = points.pop(np.argmin(distances))
        ordered_points.append(nearest_point)
    return ordered_points

def locate_ball(frame, edges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 60, 70])
    upper_red1 = np.array([5, 200, 210])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 190, 200])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    ball_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    mask = np.zeros_like(edges)
    if ball_contours:
        merged_contour = np.concatenate([contour.reshape(-1, 2) for contour in ball_contours])
        hull = cv2.convexHull(merged_contour)
    else:
        hull = None
    return hull

def run_image_processing():
    global x_integral, y_integral, start_solve, ser, timer_running, timer_start_time
    try:
        ser = serial.Serial('COM3', 9600)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception as e:
        print("Initialization Error:", e)
        return

    image_captured = False
    prev_center = None
    velocities = []
    prev_time = time.time()
    KpX, KiX, KdX = 0.1, 0.2, 0.03
    KpY, KiY, KdY = 0.2, 0.2, 0.03
    x_distance_prev, y_distance_prev = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = cv2.imread('green.jpg')  # Fallback to backup image if capture failed
        frame = rectify(frame)

        if not image_captured and time.time() - timer_start_time >= 1.5:
            # Processing to initialize path and obstacles
            image_captured = True

        if image_captured:
            hull = locate_ball(frame, frame.copy())  # Assume frame is edge-detected
            if hull is not None:
                # Ball movement logic, velocity calculation, etc.
                current_center = (int(hull[0][0]), int(hull[0][1]))
                if not timer_running and current_center != prev_center:
                    timer_start_time = time.time()
                    timer_running = True
                    update_timer_display(timer_start_time)
                
                # Endpoint logic
                if is_endpoint_reached(current_center):
                    timer_running = False
                    update_timer_display(timer_start_time, stop=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def PIDcontrol(Kp, Ki, Kd, distance, integral, derivative):
    return Kp * distance + Ki * integral + Kd * derivative

def send_position(angle1, angle2):
    if -60 <= angle1 <= 60 and -60 <= angle2 <= 60:
        command = f"x,{angle1 + 60}\ny,{angle2 + 60}\n"
        ser.write(command.encode())
        print("Sent command:", command)

def create_gui():
    global timer_label
    root = tk.Tk()
    root.title("Ball Tracking System")
    root.geometry("200x200")

    timer_label = tk.Label(root, text="00:00:00", font=("Helvetica", 16))
    timer_label.pack()

    button1 = tk.Button(root, text="Start Image Processing", command=lambda: threading.Thread(target=run_image_processing, daemon=True).start())
    button1.pack()

    button2 = tk.Button(root, text="Start Solve", command=start_solve)
    button2.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
