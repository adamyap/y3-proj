import cv2
import numpy as np

# Defining green colour range
lower_green = np.array([50, 200, 200])
upper_green = np.array([60, 255, 255])

# Defining red colour range
lower_red1 = np.array([0, 70, 130])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 130])
upper_red2 = np.array([180, 255, 255])
#lower_color = np.array([0, 0, 150])
#upper_color = np.array([255, 50, 255])
width = 720
height = 540

def rectify(image):

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a binary mask where the color is within the range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
   

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    #src_pts = src_pts[src_pts[:, :, 1].argsort(axis=0)[:, 0]]
    #src_pts[:2] = src_pts[:2][src_pts[:2, :, 0].argsort(axis=0)[:, 0]]
    #src_pts[2:] = src_pts[2:][src_pts[2:, :, 0].argsort(axis=0)[:, 0][::-1]]
    #dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    dst_pts = np.float32([[width, 0], [width, height], [0, height], [0, 0]])
    # Compute the homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(image, H, (width, height))

    return warped_image

if __name__ == "__main__":
    # Load the image
    image = cv2.imread('maze_edited_noborder_warp.png')
    image = cv2.resize(image,(720,540))

    # Display the image
    cv2.imshow('Image', rectify(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()