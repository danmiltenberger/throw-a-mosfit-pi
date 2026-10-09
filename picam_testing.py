import cv2
from picamera2 import Picamera2
import time
import sys

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
        
        print('Camera started. Press 'q' in the OpenCV window to quit.')
        
        while True:
            # capture_array() returns a NumPy array directly
            frame = picam2.capture_array()
            
            if frame is None:
                raise ValueError('Received empty frame buffer from libcamera.')
                
            # Convert to grayscale for the Canny algorithm
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection (thresholds: 50, 150)
            edges = cv2.Canny(gray, threshold1=50, threshold2=150)
            
            # Display the processed frame
            cv2.imshow('Pi5 OpenCV - Edge Detection', edges)
            
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
