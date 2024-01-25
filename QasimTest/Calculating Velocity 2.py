import cv2
import numpy as np
import time

# Global variables for tracking
frame = None
prev_positions = {"Red": None, "Blue": None}  # Store previous positions
prev_times = {"Red": None, "Blue": None}  # Store previous timestamps

def get_color(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:  # If left button is clicked
        pixel = frame[y, x]  # Get the BGR of the pixel
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
        print("HSV value:", hsv_pixel[0][0])  # Print value to console

def calculate_velocity(prev_pos, current_pos, prev_time, current_time, min_distance=5):
    if prev_pos is None or prev_time is None:
        return 0  # No previous data to calculate velocity
    distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
    if distance < min_distance:
        return 0  # Movement is too small, ignore it
    time_elapsed = current_time - prev_time
    return distance / time_elapsed if time_elapsed > 0 else 0

recent_velocities = {"Red": [], "Blue": []}
velocity_history_length = 5  # For moving average

def moving_average(velocities, new_velocity, max_length):
    velocities.append(new_velocity)
    if len(velocities) > max_length:
        velocities.pop(0)
    return sum(velocities) / len(velocities) if velocities else 0

def process_contours(contours, color, color_name):
    global frame, prev_positions, prev_times, recent_velocities
    cX, cY = 0, 0  # Initialize centroid coordinates to zero
    
    current_time = time.time()  # Define current_time at the start of the function
    
    if contours:
        merged_contour = np.concatenate([contour.reshape(-1, 2) for contour in contours])
        hull = cv2.convexHull(merged_contour)
        cv2.drawContours(frame, [hull], -1, color, thickness=cv2.FILLED)
        
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
        else:
            # If moments calculation fails, simply return without further processing
            return

    if prev_positions[color_name] is not None and prev_times[color_name] is not None:
        # Now using calculate_velocity to avoid redundancy
        velocity = calculate_velocity(prev_positions[color_name], (cX, cY), prev_times[color_name], current_time, min_distance=50)
        if velocity > 0:  # If velocity is significant, log it
            print(f"{color_name} moved from {prev_positions[color_name]} to ({cX}, {cY}) with velocity: {velocity:.2f} px/s")
            recent_velocities[color_name].append(velocity)
            # Assuming moving_average function is defined elsewhere
            smoothed_velocity = moving_average(recent_velocities[color_name], velocity, velocity_history_length)
            print(f"Smoothed {color_name} Velocity: {smoothed_velocity:.2f} px/s")

    # Update the positions and times for the next iteration
    prev_positions[color_name] = (cX, cY)
    prev_times[color_name] = current_time


def main():
    global frame
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 60)

    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_color)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 70, 130]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 130]), np.array([180, 255, 255])
        lower_blue, upper_blue = np.array([100, 150, 150]), np.array([140, 255, 255])

        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        process_contours(contours_red, (0, 0, 255), "Red")
        process_contours(contours_blue, (255, 0, 0), "Blue")

        cv2.imshow('image', cv2.resize(frame, (640, 360)))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
