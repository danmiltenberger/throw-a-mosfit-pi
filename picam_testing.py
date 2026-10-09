import cv2
from picamera2 import Picamera2 # type: ignore
import time
import sys
from opencv_stack import evaluate_frame, draw_to_frame
from vars import red_ball

# https://electricalflux.com/mcu-raspberry/opencv-raspberry-pi-camera-setup-debugging

def main():
    # Initialize the PiCamera2 object
    picam2 = Picamera2()
    
    # Configure for OpenCV: RGB888 format prevents BGR/RGB confusion
    # 640x480 keeps CPU load low for real-time edge detection on Pi 5
    config = picam2.create_preview_configuration(
        main={'format': 'RGB888', 'size': (640, 480)}
    )
    picam2.configure(config)
    
    try:
        picam2.start()
        # Allow the AGC (Automatic Gain Control) to settle
        time.sleep(1.0) 
        
        print("Camera started. Press 'q' in the OpenCV window to quit.")
        
        while True:
            # capture_array() returns a NumPy array directly
            frame = picam2.capture_array()
            
            if frame is None:
                raise ValueError('Received empty frame buffer from libcamera.')

            unchanged_frame = frame.copy()
    
            verdict_dict : dict = evaluate_frame(frame, red_ball)
            
            frame_annotated = draw_to_frame(unchanged_frame, [verdict_dict])

            cv2.imshow('Annotated', frame_annotated)
            cv2.imshow('Original', unchanged_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f'Fatal runtime error: {e}', file=sys.stderr)
        sys.exit(1)
        
    finally:
        # Crucial: Always stop the camera to release libcamera buffers
        print('Stopping camera and releasing resources...')
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
