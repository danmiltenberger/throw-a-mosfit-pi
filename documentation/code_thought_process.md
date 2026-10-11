The principle idea is that analysis can be from a live video feed or from a video (which can speed up development)

Programs to set up for video processing:
- Take video -> mp4 of training data
- Take photo -> cv2 frame and also png of training data

- Select hsv values -> load a photo and let the user click on the color to focus on. Then show a potential mask of that color. Finally, save to a file


Video Processing:
Input frame -> frame verdict (a file of what was seen in the frame, size, distance, etc)
Input video -> video verdict (same as aobve, includes timestamps)

Field Processing:
Input frame verdict -> draw the field with locations of other robot(s), ball, goals, etc
Input video verdict -> continually update the field with expected positions

Further Field Proessing:
- include some sort of certainty multiple where the robot will discard obviously wrong data (sees an EXIT sign as a red ball 100ft away, for example)

Decision Making:
- at the same time that the brain tells the arduino "move forward" also have a digital twin that moves forward at the expected rate