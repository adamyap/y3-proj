import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from image_rectification import rectify

# Load the image
image = cv2.imread('maze_edited_warp.png')
image = cv2.resize(image,(720,540))
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

def click_event(event, x_start, y_start, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Create a copy of the path coordinates
        x_copy, y_copy = x.copy(), y.copy()

        # Create a copy of the original image
        image_copy = image.copy()

        # Calculate the Euclidean distance from the starting point to each point on the path
        distances = np.sqrt((x_copy - x_start)**2 + (y_copy - y_start)**2)

        # Find the index of the minimum distance
        idx = np.argmin(distances)

        # The nearest point on the path is then
        x_nearest, y_nearest = x_copy[idx], y_copy[idx]

        # Draw a line from the clicked point to the nearest point on the path
        cv2.line(image_copy, (x_start, y_start), (x_nearest, y_nearest), (0, 0, 255), 2)

        # Calculate the Euclidean distance
        euclidean_distance = np.sqrt((x_nearest - x_start)**2 + (y_nearest - y_start)**2)

        # Display the Euclidean distance next to the line
        cv2.putText(image_copy, f'Distance: {euclidean_distance:.2f}', (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate the x and y distances
        x_distance = x_nearest - x_start
        y_distance = -(y_nearest - y_start)

        # Display the x and y distances
        cv2.putText(image_copy, f'X Distance: {x_distance:.2f}', (x_start, y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image_copy, f'Y Distance: {y_distance:.2f}', (x_start, y_start - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the image
        cv2.imshow('Image', image_copy)
        cv2.waitKey(1)


# Set the mouse callback function for the window
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click_event)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
