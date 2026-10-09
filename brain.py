# the main thought process file for our soccer playing robot

import numpy as np
import cv2 as cv
from opencv_stack import evaluate_frame, draw_to_frame

from vars import red_ball


video_path = '.temp/test.h264'

USE_WEBCAM : bool = True

if USE_WEBCAM:
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture(video_path)





if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    unchanged_frame = frame.copy()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    verdic_dict = evaluate_frame(frame, red_ball)
    frame_annotated = draw_to_frame(unchanged_frame, [verdic_dict])

    
    cv.imshow('frame', frame_annotated)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
