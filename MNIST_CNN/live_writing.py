import cv2
import numpy as np

loadFromSys = True

if loadFromSys:
    hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

x1, y1 = 0, 0
noise_thresh = 800

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is not None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if loadFromSys:
        lower_range = np.array(hsv_value[0])
        upper_range = np.array(hsv_value[1])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            cv2.line(canvas, (x1, y1), (x2, y2), [0, 255, 0], 4)

        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0

    frame = cv2.add(canvas, frame)

    stacked = np.hstack((canvas, frame))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)
    if key == 10:  # Enter key
        break

    # Clear the canvas when 'c' is pressed
    if key & 0xFF == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
