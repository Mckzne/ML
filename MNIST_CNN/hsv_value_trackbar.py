import cv2
import numpy as np

# Callback function for trackbars (not used but needed)
def nothing(x):
    pass

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Create a window for trackbars
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 250)

# Create trackbars for Lower HSV (Default: Black)
cv2.createTrackbar("Lower H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 0, 255, nothing)

# Create trackbars for Upper HSV (Default: Black)
cv2.createTrackbar("Upper H", "Trackbars", 180, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 50, 255, nothing)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for easier drawing
    frame = cv2.flip(frame, 1)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read trackbar positions
    lh = cv2.getTrackbarPos("Lower H", "Trackbars")
    ls = cv2.getTrackbarPos("Lower S", "Trackbars")
    lv = cv2.getTrackbarPos("Lower V", "Trackbars")

    uh = cv2.getTrackbarPos("Upper H", "Trackbars")
    us = cv2.getTrackbarPos("Upper S", "Trackbars")
    uv = cv2.getTrackbarPos("Upper V", "Trackbars")

    # Define lower and upper HSV bounds
    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([uh, us, uv])

    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply mask on original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show all windows
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save HSV values to a .npy file
    if key == ord('s'):
        hsv_values = np.array([[lh, ls, lv], [uh, us, uv]])
        np.save('hsv_value.npy', hsv_values)
        print("\nHSV values saved successfully to hsv_value.npy!")

    # Press 'q' to exit
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
