import cv2
import time

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

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('Video Feed', frame)
        # Break the loop on 'q' key press
        if not image_captured and time.time() - start_time >= 3:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Display the image
            cv2.imshow('Image', frame)
            image_captured = True  # update the flag
        if cv2.waitKey(1) & 0xFF == 27: #ESC key
            break
            

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()