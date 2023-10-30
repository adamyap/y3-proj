import cv2
import numpy as np

def nothing(x):
    pass

# Function to create trackbars for color calibration
def create_color_trackbars(window_name):
    cv2.namedWindow(window_name)
    cv2.createTrackbar('H_low', window_name, 0, 179, nothing)
    cv2.createTrackbar('S_low', window_name, 0, 255, nothing)
    cv2.createTrackbar('V_low', window_name, 0, 255, nothing)
    cv2.createTrackbar('H_high', window_name, 179, 179, nothing)
    cv2.createTrackbar('S_high', window_name, 255, 255, nothing)
    cv2.createTrackbar('V_high', window_name, 255, 255, nothing)

def get_trackbar_values(window_name):
    H_low = cv2.getTrackbarPos('H_low', window_name)
    S_low = cv2.getTrackbarPos('S_low', window_name)
    V_low = cv2.getTrackbarPos('V_low', window_name)
    H_high = cv2.getTrackbarPos('H_high', window_name)
    S_high = cv2.getTrackbarPos('S_high', window_name)
    V_high = cv2.getTrackbarPos('V_high', window_name)
    return np.array([H_low, S_low, V_low]), np.array([H_high, S_high, V_high])

def main():
    # Create trackbars for seven colors
    colors = ['White', 'Black', 'Red', 'Yellow', 'Green', 'Magenta', 'Blue']
    for color in colors:
        create_color_trackbars(color)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Process each color
        for color in colors:
            # Get trackbar positions for this color
            lower_color, upper_color = get_trackbar_values(color)

            # Create a mask using the HSV frame
            mask = cv2.inRange(hsv, lower_color, upper_color)
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding box for each detected contour
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Optionally display the mask for each color in a separate window
            # cv2.imshow(f'Mask {color}', mask)

        # Display the original frame with bounding boxes
        cv2.imshow('Webcam Feed + Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):  # If spacebar is pressed
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()