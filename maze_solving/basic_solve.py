import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

# Load the image
image = cv2.imread('maze_solving\maze_edited.png')
image = cv2.resize(image,(720,540))

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

        # Create a copy of the image
        image_copy = image.copy()

        # Define the step size
        step_size = 5

        # While there are still points on the path
        while len(x_copy) > 0 and len(y_copy) > 0:

            # Calculate the Euclidean distance from the starting point to each point on the path
            distances = np.sqrt((x_copy - x_start)**2 + (y_copy - y_start)**2)

            # Only consider the points that are within the step size from the current position
            close_points = distances <= step_size

            # If there are no close points, break the loop
            if not np.any(close_points):
                break

            # Find the index of the minimum distance
            idx = np.argmin(distances)

            # The nearest point on the path is then
            x_nearest, y_nearest = x_copy[idx], y_copy[idx]

            # Now the virtual object can teleport to (x_nearest, y_nearest)
            # We can represent the virtual object as a blue dot on the image
            cv2.circle(image, (x_nearest, y_nearest), radius=5, color=(169, 169, 169), thickness=-1)

            # Display the image
            cv2.imshow('Image', image)
            cv2.waitKey(1)

            # Remove the nearest point from the path
            x_copy = np.delete(x_copy, idx)
            y_copy = np.delete(y_copy, idx)

            # Update the clicked point to the current position
            x_start, y_start = x_nearest, y_nearest
        cv2.waitKey(0)

# Set the mouse callback function for the window
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click_event)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
