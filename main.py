# initial python file just to fuck about in & learn this shit

import cv2

def main():
    # Connect to webcam (0 = defautl cam)
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
    while True:
        # Read a frame from the webcam
        ret, frame = cam.read()
        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        # Display the frame in a window
        frame = cv2.resize(frame,(640,360))
        cv2.imshow('Webcam Feed', frame)
        # Break the loop if space key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()