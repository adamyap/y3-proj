import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from image_rectification import rectify

# Load the image
image = cv2.imread('maze_edited_warp.png')
image = cv2.resize(image, (720, 540))
image = rectify(image)

# Convert the frame to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Threshold the HSV image to get only blue colors
lower_blue = np.array([110, 200, 200])
upper_blue = np.array([140, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Skeletonize the mask
skeleton = skeletonize(mask // 255)
skeleton = img_as_ubyte(skeleton)  # Convert the image back to 8-bit

# Get the coordinates of the skeleton path
y, x = np.where(skeleton == 255)

# Find contours in the skeleton
contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw each contour on the image
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)

# Variables to store click positions and ball position
start_point = None
ball_position = None

def click_event(event, x_start, y_start, flags, param):
    global start_point, ball_position

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x_start, y_start)  # Store the click position
        ball_position = start_point  # Initialize ball position

    elif event == cv2.EVENT_LBUTTONUP:
        start_point = None  # Reset the start point

# Setup the display window and callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_event)

# Main loop for arrow key handling
while True:
    image_copy = image.copy()

    if ball_position is not None:
        # Calculate distances to the path and find the nearest point
        distances = np.sqrt((x - ball_position[0]) ** 2 + (y - ball_position[1]) ** 2)
        idx = np.argmin(distances)
        nearest_x, nearest_y = x[idx], y[idx]

        # Show the distance and draw a line connecting the ball to the nearest path point
        distance = distances[idx]
        cv2.putText(image_copy, f"Distance: {distance:.2f}px", (ball_position[0] + 10, ball_position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(image_copy, (nearest_x, nearest_y), 5, (0, 0, 255), -1)  # Ball at the nearest point
        cv2.line(image_copy, ball_position, (nearest_x, nearest_y), (255, 0, 0), 2)  # Line connecting ball to path

    key = cv2.waitKey(50)  # Delay for ball movement effect

    # Arrow key handling to move the ball along the path
    if key == 27:  # ESC key to exit
        break
    elif key == 81:  # Left arrow key
        ball_position = (ball_position[0] - 1, ball_position[1])
    elif key == 82:  # Up arrow key
        ball_position = (ball_position[0], ball_position[1] - 1)
    elif key == 83:  # Right arrow key
        ball_position = (ball_position[0] + 1, ball_position[1])
    elif key == 84:  # Down arrow key
        ball_position = (ball_position[0], ball_position[1] + 1)

    # Display the image
    cv2.imshow("Image", image_copy)

cv2.destroyAllWindows()
