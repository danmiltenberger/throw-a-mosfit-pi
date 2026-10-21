import cv2
import time
import sys
from datetime import datetime

try:
    from picamera2 import Picamera2 #type: ignore
except ImportError:  # pragma: no cover - only used when the Pi camera library is unavailable.
    Picamera2 = None  # type: ignore[assignment]

from tam_library_save_and_load import save_frame_as_image, get_or_create_directory, get_save_file_path




def take_photo():
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
        
        print("Camera started. Press 'q' in the OpenCV window to take photo.")
        
        while True:
            # capture_array() returns a NumPy array directly
            frame = picam2.capture_array()
            
            if frame is None:
                raise ValueError('Received empty frame buffer from libcamera.')

            cv2.imshow("press q to take photo", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):

                
                output_path = get_save_file_path(directory="training_data", ending=".png")

                save_frame_as_image(frame, output_path)
                return frame

                
    except Exception as e:
        print(f'Fatal runtime error: {e}', file=sys.stderr)
        sys.exit(1)
        
    finally:
        # Crucial: Always stop the camera to release libcamera buffers
        print('Stopping camera and releasing resources...')
        picam2.stop()
        cv2.destroyAllWindows()


def main():
    take_photo()

if __name__ == '__main__':
    main()
