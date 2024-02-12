import cv2
import numpy as np
import time

# Global variables for the frame read from the webcam and the color range
frame = None
lower_color = None
upper_color = None

def get_color(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frame[y,x]
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)
        print("HSV value:", hsv_pixel[0][0])

def main():
    global frame, lower_color, upper_color
    # Connect to webcam (0 = default cam)
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # Set the resolution to 720p
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set the frame rate to 60fps
    cam.set(cv2.CAP_PROP_FPS, 60)

    # Check if the webcam has been opened successfully
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_color)

    while True:
        # Start the timer
        start_time = time.time()

        # Read a frame from the webcam
        ret, frame = cam.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only red colors
        lower_red1 = np.array([0, 70, 130])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 130])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Merge all the contours into one
        if contours:
            merged_contour = np.concatenate([contour.reshape(-1, 2) for contour in contours])

            # Calculate the convex hull of the merged contour
            hull = cv2.convexHull(merged_contour)

            # Draw the bounding contour on the frame
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), thickness=cv2.FILLED)

            # Calculate the center of the hull
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Draw the center of the shape on the image
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, f"Center: ({cX}, {cY})", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw crosshair at the center
            cv2.line(frame, (cX - 15, cY), (cX + 15, cY), (255, 255, 255), 1)
            cv2.line(frame, (cX, cY - 15), (cX, cY + 15), (255, 255, 255), 1)



        # Display the frame in a window
        frame = cv2.resize(frame,(640,360))
        cv2.imshow('image', frame)

        # Break the loop if space key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
