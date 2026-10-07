# sources: https://docs.opencv.org/3.4.20/d4/d73/tutorial_py_contours_begin.html
# https://docs.opencv.org/3.4.20/df/d9d/tutorial_py_colorspaces.html 


import cv2 as cv
import numpy as np
import imutils


USE_WEBCAM : bool = False
video_file_path = ".temp/test.h264"



def draw_circle_contours(cnts, frame):
    # only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv.circle(frame, center, 5, (0, 0, 255), -1)


if USE_WEBCAM:
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture(video_file_path)


while True:
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of color in HSV
    lower = np.array( [0, 153, 0] )
    upper = np.array( [179, 255, 255] )

    # Threshold the HSV image to get only specific color
    black_and_white = cv.inRange(hsv, lower, upper)
    mask = black_and_white


    one_color = cv.bitwise_and(frame,frame, mask= black_and_white)


    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)


    # show different images / videos
    cv.imshow('unchanged',frame)
    cv.imshow('black and white', black_and_white)
    cv.imshow('one color',one_color)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break


cv.destroyAllWindows()

