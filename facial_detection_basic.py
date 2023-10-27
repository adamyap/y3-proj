# initial python file(s) just to fuck about in & learn this shit

import cv2

def main():
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Connect to webcam (0 = defautl cam)
    cam = cv2.VideoCapture(0)

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
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the frame in a window
        cv2.imshow('Webcam Feed + Facial Recognition', frame)
        # Break the loop if space key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    # Release the webcam and close the window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()