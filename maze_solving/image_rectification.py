import cv2
import numpy as np

# Defining green colour range
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 160])

#lower_color = np.array([0, 0, 150])
#upper_color = np.array([255, 50, 255])
width = 720
height = 540

def rectify(image):

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a binary mask where the color is within the range
    mask = cv2.inRange(hsv, lower_green, upper_green)
   

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the quadrilateral
    contour = max(contours, key=cv2.contourArea)

    # Get the convex hull of the contour
    hull = cv2.convexHull(contour)

    # Approximate the convex hull to a polygon with four vertices
    epsilon = 0.02*cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    #print(len(approx))

    # Now you can use these points in the perspective transform as before
    src_pts = np.float32(approx)
    src_pts = src_pts[src_pts[:, :, 1].argsort(axis=0)[:, 0]]
    src_pts[:2] = src_pts[:2][src_pts[:2, :, 0].argsort(axis=0)[:, 0]]
    src_pts[2:] = src_pts[2:][src_pts[2:, :, 0].argsort(axis=0)[:, 0][::-1]]
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    #dst_pts = np.float32([[width, 0], [width, height], [0, height], [0, 0]])
    # Compute the homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(image, H, (width, height))

    return warped_image

if __name__ == "__main__":
    # Connect to webcam (0 = default cam)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # Set the resolution to 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set the frame rate to 60fps
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the image
        cv2.imshow('Video', rectify(frame))

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()