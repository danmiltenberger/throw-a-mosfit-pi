# www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/ 

# imports


from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import time
import imutils


video_path = r".temp/test.h264"

prev_frames_to_draw = 64		# aka the buffer of points (default from website is 64)

pts = deque(maxlen=prev_frames_to_draw)
color_lower_HSV = (0,0,0)
color_upper_HSV = (0,0,0)

video_ref = cv2.VideoCapture(video_path)

frame_width_pix = 600

gauss_blur_size = (11,11)
sigma_x_for_blur = 0

min_radius = 10		# example from the website


# if --video is False, the program is supposed to use the webcam


# wait a moment for the video to warm up
time.sleep(2)

while True:

	# get a frame from the video
	frame = video_ref.read()

	# get the actual frame from the .read return (zeroth is a bool)
	frame = frame[1]

	# if the video has no more frames, that means it's over
	if frame is None:
		break


	# process the frame
	frame = imutils.resize(frame, width=frame_width_pix)
	blurred = cv2.GaussianBlur(frame, gauss_blur_size, 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color then remove any misc blobs left over
	mask = cv2.inRange(hsv, color_lower_HSV, color_upper_HSV)
	#mask = cv2.erode(mask,None, iterations=2)
	#mask = cv2.dilate(mask,None, iterations=2)
	
	# find contours of the mask and determine the center of the ball
	contours = cv2.findContours(
		mask.copy(),
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
		)

	contours = imutils.grab_contours(contours)
	center = None

	# only continue if at least one contour was found
	if len(contours) > 0:
		# find the largest contour then use it to find the circle
		largest_contour = max(contours, key=cv2.contourArea)
		((x,y), radius) = cv2.minEnclosingCircle(largest_contour)

		M = cv2.moments(largest_contour)
		center = (int(M["m10"] / M["m00"])), (int(M["m01"] / M["m00"]))

		# check if the radius is a reasonable size 
		if radius > min_radius:
			# draw the circle 
			circle_color = (0,255,255)
			circle_thickness = 2

			center_circle_rad = 5
			center_circle_color = (0,0,255)
			center_circle_thickness = -1

			# draw a circle around the ball
			cv2.circle(frame, (int(x), int(y)), int(radius), circle_color, circle_thickness)

			# draw a dot (circle) at the center of the ball
			cv2.circle(
				frame, 
				center, 
				center_circle_rad, 
				center_circle_color, 
				center_circle_thickness)

		# update the points queue
		pts.appendleft(center)

		# loop over the set of tracked points
		for i in range(1,len(pts)):

			# if either of the x or y are None, skip
			if pts[i-1] is None or pts[i] is None:
				continue

			line_color = (0,0,255)

			# compute the thickness of the line and draw a connecting line
			thickness = int(np.sqrt(prev_frames_to_draw / float(i+1)) * 2.5)
			cv2.line(frame, pts[i-1], pts[i], line_color, thickness)

		# show the frame to the screen
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the q key is pressed, quit
		if key == ord("q"):
			break

	video_ref.release()

	cv2.destroyAllWindows()
	
