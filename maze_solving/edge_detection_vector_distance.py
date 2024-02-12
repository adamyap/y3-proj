import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from image_rectification import rectify
import time

def define_path(contours):
    # Define the minimum area for a filled contour
    min_area = 20000

    # Filter contours based on the area
    filled_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create an empty mask to draw the filled contours
    mask = np.zeros_like(edges)

    # Draw the filled contours on the mask
    cv2.drawContours(mask, filled_contours, -1, (255), thickness=cv2.FILLED)

    cv2.imshow('mask',mask)

    # Skeletonize the mask
    skeleton = skeletonize(mask // 255)
    skeleton = img_as_ubyte(skeleton)  # Convert the image back to 8-bit

    # Get the coordinates of the skeleton path
    y, x = np.where(skeleton == 255)

    # Find contours in the skeleton
    contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return x,y,contours

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

    return keypoints

def define_wall(contours):
    # Define the minimum area for a filled contour
    max_area = 10000

    # Filter contours based on the area
    wall_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area]

    return contours


def click_event(event,x_start,y_start,flags,param):
    global x,y
    if event == cv2.EVENT_LBUTTONDOWN:

        # Use processed image as a base.
        image_copy = processed_image.copy()

        # Calculate the Euclidean distance from the starting point to each point on the path
        distances = np.sqrt((x - x_start)**2 + (y - y_start)**2)

        # Find the index of the minimum distance
        idx = np.argmin(distances)

        # The nearest point on the path is then
        x_nearest, y_nearest = x[idx], y[idx]

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
        cv2.imshow('Working Image', image_copy)
        cv2.waitKey(1)

if __name__ == "__main__":
    # Connect to webcam (0 = default cam)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # Set the resolution to 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Set the frame rate to 60fps
    cap.set(cv2.CAP_PROP_FPS, 60)
    # Start the timer
    start_time = time.time()
    # Flag to indicate if the image has been captured
    image_captured = False

    # Backup image
    backup_image = cv2.imread('green.jpg')
    cv2.resize(backup_image,(1280,720))

    # Set the mouse callback function for the window
    cv2.namedWindow('Working Image')
    cv2.setMouseCallback('Working Image', click_event)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Resize the frame to desired size
        cv2.imshow('Video Feed', frame)
        try:
            frame = rectify(frame)
        except:
            frame = rectify(backup_image)

        if not image_captured and time.time() - start_time >= 3:
            image = frame
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to the image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Perform edge detection
            edges = cv2.Canny(blurred, 10, 50)
            # Find contours in the edges
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Define walls
            wall_contours = define_wall(contours)
            cv2.drawContours(image, wall_contours, -1, (255, 155, 0), 1)
            # Define hole
            keypoints = define_hole(contours)
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            cv2.drawKeypoints(image, keypoints, image, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # Define path
            x,y,path_contours = define_path(contours)
            for contour in path_contours:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
            cv2.imshow('Working Image', image)
            processed_image = image.copy() # copy final image
            image_captured = True  # update the flag
        if cv2.waitKey(1) & 0xFF == 27: #ESC key
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

