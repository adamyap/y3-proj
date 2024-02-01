import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from image_rectification import rectify

# Load the image
image = cv2.imread('maze_edited.png')
image = cv2.resize(image,(720,540))

image = rectify(image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 10, 50)

# Find contours in the edges
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def define_path(contours):
    # Define the minimum area for a filled contour
    min_area = 10001

    # Filter contours based on the area
    filled_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create an empty mask to draw the filled contours
    mask = np.zeros_like(edges)

    # Draw the filled contours on the mask
    cv2.drawContours(mask, filled_contours, -1, (255), thickness=cv2.FILLED)

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
    return x,y

def define_hole(contours):
    # Define the min and max area for a filled contour
    min_area = 100
    max_area = 10000

    # Filter contours based on the area
    circle_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area]

    # Create an empty mask to draw the filled contours
    mask = np.zeros_like(edges)

    # Draw the filled contours on the mask
    cv2.drawContours(mask, circle_contours, -1, (255), thickness=cv2.FILLED)

    # # Perform Hough Circle Transform to detect circles
    # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=15, minRadius=0, maxRadius=0)

    # # Ensure at least some circles were found
    # if circles is not None:
    #     # Convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")

    #     # Loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # Draw the circle in the output image
    #         cv2.circle(image, (x, y), r, (0, 0, 255), 2)

    # Set up the SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 255

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(mask)

    # Draw detected blobs as red circles
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    cv2.drawKeypoints(image, keypoints, image, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def define_wall(contours):
    # Define the minimum area for a filled contour
    max_area = 100

    # Filter contours based on the area
    wall_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area]

    # Create an empty mask to draw the filled contours
    mask = np.zeros_like(edges)

    # Draw the filled contours on the mask
    cv2.drawContours(mask, wall_contours, -1, (255), thickness=cv2.FILLED)

    # Draw the filled contours on the original image
    cv2.drawContours(image, wall_contours, -1, (255, 155, 0), 1)


def click_event(event,x_start, y_start, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        # Create a copy of the path coordinates
        x_copy, y_copy = define_path(contours)

        # Create a copy of the image
        image_copy = image.copy()

        # Define the step size
        step_size = 50

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

if __name__ == "__main__":
    # Set the mouse callback function for the window
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', click_event)


    # Display the image
    define_path(contours)
    define_wall(contours)
    define_hole(contours)

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
