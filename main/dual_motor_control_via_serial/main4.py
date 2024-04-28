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

timer_running = False
start_time = None
timer_label = None
end_reached = False

def start_timer():
    global timer_running, start_time, timer_label
    if not timer_running:
        start_time = time.time()
        timer_running = True
        update_timer()
        
def update_timer():
    global timer_label, start_time, timer_running, end_reached
    if timer_running:
        if end_reached:
            timer_running = False
            return
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time / 60)
        seconds = int(elapsed_time) % 60
        tenths_of_seconds = int((elapsed_time - int(elapsed_time)) * 10)  # Adjust to show tenths of a second
        formatted_time = f"{minutes:02}:{seconds:02}.{tenths_of_seconds:01}"  # Adjust format for 0.1s resolution
        timer_label.config(text=formatted_time)
        timer_label.after(100, update_timer)  # Update every 100 ms for 0.1s resolution

def define_path(contours,edges):
    # min area for filled path
    min_area = 2000
    min_perimeter = 5000
    # filter contours based on area and perimeter to highlight path
    filled_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.arcLength(cnt, True) > min_perimeter]
    mask = np.zeros_like(edges)
    for cnt in filled_contours:
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)
    # skeletonize mask to get continuous line of points
    skeleton = skeletonize(mask // 255)
    skeleton = img_as_ubyte(skeleton)
    # get points from skeleton
    y, x = np.where(skeleton == 255)
    # create array of evenly-spaced indices
    indices = np.linspace(0, len(x) - 1, 100,dtype=int) #100: number of checkpoints
    # sample x & y array with indices
    sample_x = x[indices]
    sample_y = y[indices]
    points = list(zip(sample_x, sample_y))

    # for contours in skeleton, find geometric approximation of path
    contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.001*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        for point in approx:
            # if point is not close to an existing point, add it to the list
            distances = [np.sqrt((point[0][0] - p[0])**2 + (point[0][1] - p[1])**2) for p in points]
            if np.min(distances) > 10:
                points.append(tuple(point[0]))

    return points, contours

def order_points(image,points):
    lower_yellow = np.array([20, 180, 80])
    upper_yellow = np.array([30, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # center of largest contour
    M = cv2.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    ordered_points = [[cX,cY]]
    # find nearest point to the center, order them by distance
    while points:
        last_point = ordered_points[-1]
        distances = [np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2) for point in points]
        nearest_point = points.pop(np.argmin(distances))
        ordered_points.append(nearest_point)

    return ordered_points

def define_hole(contours,edges):
    min_area = 100
    max_area = 10000
    circle_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area]
    mask = np.zeros_like(edges)
    for cnt in circle_contours:
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = True
    params.minCircularity = 0.75
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)

    return keypoints

def locate_ball(frame,edges):
    lower_red1 = np.array([0, 60, 70])
    upper_red1 = np.array([5, 200, 210])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 190, 200])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    ball_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, ball_contours, -1, (255), thickness=cv2.FILLED)
    # merge all contours
    if ball_contours:
        merged_contour = np.concatenate([contour.reshape(-1, 2) for contour in ball_contours])
        # find convex hull of contour
        hull = cv2.convexHull(merged_contour)
    else:
        hull = None

    return hull

def define_end(frame):
    lower_blue = np.array([100, 150, 100])
    upper_blue = np.array([110, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def run_image_processing():
    global x_integral
    global y_integral
    global start_solve
    global ser
    global end_reached
    global interval_time
    try:
        ser = serial.Serial('COM3', 9600)
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # 0 - default cam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        send_position(0,0)
    except:
        print("Error: Could not connect to the camera or the Arduino")
        pass

    start_time = time.time()
    checkpoint_time = time.time()
    image_captured = False

    backup_image = cv2.imread('green.jpg')
    cv2.resize(backup_image,(1280,720))

    prev_center = None
    velocity = [0, 0]
    N = 1 # no. frames for speed moving average
    velocities = []
    prev_time = time.time()
    KpX,KiX,KdX = 0.25,0.25,0.030 #0.1 0.2 0.03
    KpY,KiY,KdY = 0.31,0.25,0.030 #0.2 0.2 0.03
    x_integral = 0
    y_integral = 0
    x_distance_prev = 0
    y_distance_prev = 0
    start_solve = False
    interval_time = time.time()


    while not end_reached:
        # Resize the frame to desired size
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = rectify(frame)
        except:
            frame = rectify(backup_image)

        # take initial picture with delay to allow cam to start & settle
        if not image_captured and time.time() - start_time >= 1.5:
            image = frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 15, 50)
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # draw holes
            keypoints = define_hole(contours,edges)
            cv2.drawKeypoints(image, keypoints, image, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # draw & order path
            path_points,path_contours = define_path(contours,edges)
            ordered_points = order_points(image, path_points)
            # separate x & y coords of path
            x_path = np.array([point[0] for point in ordered_points])
            y_path = np.array([point[1] for point in ordered_points])
            # label each point on path to show order
            for i,point in enumerate(ordered_points):
                cv2.circle(image, tuple(map(int, point)), 3, (255, 255, 255), -1)
                cv2.putText(image, str(i), (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            # draw path contour
            for contour in path_contours:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
            end_contours = define_end(image)
            # draw end contour
            for cnt in end_contours:
                cv2.drawContours(image, [cnt], -1, (255, 0, 0), -1)

            # save resultant processed image so we don't overwrite original frame in following code
            processed_image = image.copy()
            cv2.imshow('Working Image', image)
            image_captured = True
        
        elif image_captured:
            ball_contours = locate_ball(frame,edges)
            if ball_contours is not None:
                image = processed_image.copy()

                # find centre of ball
                M = cv2.moments(ball_contours)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # draw a circle around ball to represent on image
                cv2.circle(image, center, 13, (0, 0, 255), thickness=2)
                # draw crosshair at centre of ball
                cv2.line(image, (center[0] - 5, center[1]), (center[0] + 5, center[1]), (0, 0, 255), 1)
                cv2.line(image, (center[0], center[1] - 5), (center[0], center[1] + 5), (0, 0, 255), 1)

                # calculate ball velocity
                if prev_center is not None:
                    prev_center = (int(center[0] + prev_center[0])*0.5 , int(center[1] + prev_center[1])*0.5) #qiuck fix
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    dt = time.time() - prev_time  # time between frames
                    velocities.append((dx/dt, dy/dt))
                    if len(velocities) > N:
                        velocities.pop(0)
                    velocity = [sum(v[i] for v in velocities) / len(velocities) for i in range(2)]
                prev_center = center
                prev_time = time.time()
                # draw a green vector with scale & direction to represent velocity
                scale = 0.1
                end_point = (int(center[0] + velocity[0]*scale), int(center[1] + velocity[1]*scale))
                cv2.arrowedLine(image, center, end_point, (0, 255, 0), 2)
            
            if start_solve == True:
                # check if ball is at the end of the course
                for contour in end_contours:
                    if cv2.pointPolygonTest(contour, (center[0], center[1]), False) >= 0:
                        print("Congratulations! The ball has reached the end of the maze.")
                        end_reached = True
                        break
                if not end_reached:
                    if x_path.size == 0 or y_path.size == 0:
                        print("Congratulations! The ball has reached the end of the maze.")
                        end_reached = True
                        break

                # find the nearest point on the path to the ball
                x_nearest, y_nearest = x_path[0], y_path[0]
                # draw line to this point
                cv2.line(image, (center[0], center[1]), (x_nearest, y_nearest), (0, 255, 0), 1)
                # distance from ball to nearest point
                euclidean_distance = np.sqrt((x_nearest - center[0])**2 + (y_nearest - center[1])**2)
                # distance from ball to all nearest point
                euclidean_distances = np.array([np.sqrt((x - center[0])**2 + (y - center[1])**2) for x, y in zip(x_path, y_path)])
                # show distance next to drawn line
                cv2.putText(image, f'Distance: {euclidean_distance:.2f}', (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                x_distance = (x_nearest - center[0])
                y_distance = -(y_nearest - center[1])
                # show individual x & y distances next to drawn line
                cv2.putText(image, f'X Distance: {x_distance:.2f}', (center[0], center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(image, f'Y Distance: {y_distance:.2f}', (center[0], center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # if distance to point below threshold, consider point as visited
                if time.time() - checkpoint_time >= 0.1:
                    close_points = np.where(euclidean_distances < 35)[0]
                    if close_points.size > 0:
                        first_close_point = close_points[0]
                        for i in range(first_close_point + 1):
                            cv2.circle(processed_image,(x_path[i], y_path[i]),3,(70,70,70),-1)
                        x_path = np.delete(x_path, slice(0, first_close_point + 1))
                        y_path = np.delete(y_path, slice(0, first_close_point + 1))
                    checkpoint_time = time.time()

                dt = time.time() - interval_time
                if dt > 0.01:
                    x_integral += x_distance * dt
                    # set integration saturation vals to prevent windup
                    if x_integral > 500:
                        x_integral = 500
                    elif x_integral < -500:
                        x_integral = -500
                    y_integral += y_distance * dt
                    if y_integral > 500:
                        y_integral = 500
                    elif y_integral < -500:
                        y_integral = -500
                    x_derivative = (x_distance-x_distance_prev)/dt
                    y_derivative = (y_distance-y_distance_prev)/dt
                    PDx = PIDcontrol(KpX, KiX, KdX, x_distance, x_integral, x_derivative)
                    PDy = PIDcontrol(KpY, KiY, KdY, y_distance, y_integral, y_derivative)
                    x_distance_prev = x_distance
                    y_distance_prev = y_distance
                    motorx = max(min(PDx, 60), -60)
                    motory = max(min(PDy, 60), -60)
                    # add jitter to reduce stick/slip friction
                    if x_derivative < 10 and y_derivative < 10:
                        jitter = 2.5
                        motorx += random.uniform(-jitter, jitter)
                        motory += random.uniform(-jitter, jitter)
                    send_position(motorx,motory)
                    interval_time = time.time()
            
            cv2.imshow('Working Image', image)

        if cv2.waitKey(1) & 0xFF == 27: #ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

def start_solved():
    global x_integral
    global y_integral
    global start_solve
    global interval_time
    start_solve = True
    x_integral = 0
    y_integral = 0
    interval_time = time.time()

def PIDcontrol(Kp,Ki,Kd,distance,integral,derivative):
    e = (Kp)*distance + (Ki)*integral + (Kd)*(derivative)
    return e

def send_position(angle1, angle2):
    if -60 <= angle1 <= 60 and -60 <= angle2 <= 60:
        angle1 = angle1 + 60
        angle2 = angle2 + 60
        ser.write(f"x,{angle1}".encode())
        ser.write(f"y,{angle2}".encode())

def create_gui():
    global timer_label
    root = tk.Tk()
    root.title("Main")
    root.geometry("200x200")
    
    timer_label = tk.Label(root, text="00:00:000", font=("Helvetica", 16))
    timer_label.pack()

    button1 = tk.Button(root, text="Start Image Processing", command=lambda: threading.Thread(target=run_image_processing).start())
    button1.pack()
    button2 = tk.Button(root, text="Start Solve", command=lambda: [start_solved(), start_timer()])
    button2.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
    

