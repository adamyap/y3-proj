import cv2
import numpy as np
import time

# Global variables for the frame read from the webcam
frame = None

def get_color(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frame[y, x]
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)
        print("HSV value:", hsv_pixel[0][0])

def process_contours(contours, color, color_name):
    global frame
    if contours:
        merged_contour = np.concatenate([contour.reshape(-1, 2) for contour in contours])
        hull = cv2.convexHull(merged_contour)
        cv2.drawContours(frame, [hull], -1, color, thickness=cv2.FILLED)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
        textX = cX + 20  # Offset the text to the right of the center
        textY = cY
        cv2.putText(frame, f"{color_name} Center: ({cX}, {cY})", (textX, textY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    global frame
    # Connect to webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
        # Read a frame from the webcam
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the HSV ranges for red and blue
        lower_red1 = np.array([0, 70, 130])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 130])
        upper_red2 = np.array([180, 255, 255])
        
        lower_blue = np.array([100, 150, 150])
        upper_blue = np.array([140, 255, 255])

        # Create masks for red and blue
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
