import cv2
import numpy as np

# Load an image
image = cv2.imread('maze_solving\maze_edited.png')

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def print_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('HSV:', hsv_image[y, x])

# Create a window
cv2.namedWindow('image')

# Set mouse callback function to print HSV value of pixel
cv2.setMouseCallback('image', print_hsv)

# Display the image
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
