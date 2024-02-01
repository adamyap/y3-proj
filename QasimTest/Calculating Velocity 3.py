##Setup Libraries
import cv2
import numpy as np
import time

#
## Global variables for tracking positions and velocities
#
frame = None
prev_positions = {"Red": None, "Blue": None}  # Store previous positions
prev_times = {"Red": None, "Blue": None}  # Store previous timestamps

#
## Prints the HSV color value of the pixel when mouse clicked
#
def get_color(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:         # If left button is clicked
        pixel = frame[y, x]                    # Get the BGR of the pixel
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)  #Convert BGR to HSV
        print("HSV value:", hsv_pixel[0][0])    #Print valye to console

#
## Calculates the velocity of objects and displays the colour and coord and velocity in pixels/s
#
def calculate_velocity(prev_pos, current_pos, prev_time, current_time, min_distance=5):

    if prev_pos is None or prev_time is None:
        return 0    # No previous data to calculate velocity
    
    #Defines distance using the euclidean distance formula
    distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2) 
    if distance < min_distance:  # If distance is less than the minimum defined distance (5)
        return 0  # Ignore movmeent as too small - Allows smoothing
    
    #Calculates time elapsed
    time_elapsed = current_time - prev_time
    if time_elapsed > 0:
        #Calculates velocity - distance/ time
        velocity = distance / time_elapsed
        return velocity
    else:
        return 0

# Stores velocities to make smoother
recent_velocities = {"Red": [], "Blue": []}
velocity_history_length = 5  # Number of recent velocities for the moving average

#
## Calculates the moving average of the velocities
#
def moving_average(velocities, new_velocity, max_length):

    velocities.append(new_velocity)
    # Keep only the last max_length velocities
    if len(velocities) > max_length:
        velocities.pop(0)  # Get rid of oldest measurement in list
    # Calculate the moving average
    if velocities:
        return sum(velocities) / len(velocities)  # number of velocities in list
    else:
        return 0 # If not moving then Velocity =0

#
## Processes contours for the colour, centre and reports smoothed velocity
#

def process_contours(contours, color, color_name):
    global frame, prev_positions, prev_times, recent_velocities
    cX, cY = 0, 0  # Default values 0,0
    
    # Ensure current_time is defined at the beginning of the function
    current_time = time.time()
    
    if contours:
        #Colour the shape of object 
        merged_contour = np.concatenate([contour.reshape(-1, 2) for contour in contours])
        hull = cv2.convexHull(merged_contour)
        cv2.drawContours(frame, [hull], -1, color, thickness=cv2.FILLED)
        #Find the centre of object
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])  #Average of Coordinates
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)  # Mark centre
        
    if prev_positions[color_name] is not None and prev_times[color_name] is not None:
        # Calculate the instantaneous velocity for record-keeping
        velocity = calculate_velocity(prev_positions[color_name], (cX, cY), prev_times[color_name], current_time)
        # Add the current velocity to the list for the moving average calculation
        recent_velocities[color_name].append(velocity)
        
        if len(recent_velocities[color_name]) >= velocity_history_length:
            # Calculate the smoothed velocity
            smoothed_velocity = moving_average(recent_velocities[color_name], velocity, velocity_history_length)
            # Console output showing the movement with smoothed velocity
            print(f"{color_name} moved from {prev_positions[color_name]} to ({cX}, {cY}) with a smoothed velocity of {smoothed_velocity:.2f} px/s")
    
    # Update the previous positions and times for the next iteration
    prev_positions[color_name] = (cX, cY)
    prev_times[color_name] = current_time

#    
## Set up webcam, configure settings and track colored objects    
#
 
def main():
    global frame
    # Connect to webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 0 is for default laptop cam 

    # Set Resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set the frame rate to 60fps
    cam.set(cv2.CAP_PROP_FPS, 60)

    # Check if the webcam has been opened successfully
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    #setup open cv window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_color)

    while True:
        # Read frame from the webcam
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the HSV ranges for red 
        lower_red1 = np.array([0, 70, 130])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 130])
        upper_red2 = np.array([180, 255, 255])
        
        #Define the HSV ranges for blue
        lower_blue = np.array([100, 150, 150])
        upper_blue = np.array([140, 255, 255])

        # Create masks for red and blue- marks object in the image
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours for red and blue
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process red and blue contours
        process_contours(contours_red, (0, 0, 255), "Red")
        process_contours(contours_blue, (255, 0, 0), "Blue")

        # Display the frame in a window
        frame = cv2.resize(frame, (640, 360))
        cv2.imshow('image', frame)

        # Break the loop if space key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
