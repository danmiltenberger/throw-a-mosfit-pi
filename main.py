from picamera2 import Picamera2 # type: ignore
import time
import cv2
import sys
from settings_run import CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y, USE_DRAWN_FIELD
from tam_library_OOP import get_side_by_side_frame


def main():
    # Initialize the PiCamera2 object
    picam2 = Picamera2()
    
    # Configure for OpenCV: RGB888 format prevents BGR/RGB confusion
    # 640x480 keeps CPU load low for real-time edge detection on Pi 5
    config = picam2.create_preview_configuration(
        main={'format': 'RGB888', 'size': (CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y)}
    )
    picam2.configure(config)


    if USE_DRAWN_FIELD:
        field_frame = cv2.imread("black_field.png")
    else:
        field_frame = cv2.imread("empty_field.png")
        
    try:
        picam2.start()
        # Allow the AGC (Automatic Gain Control) to settle
        time.sleep(1.0) 
        
        print("Camera started. Press 'q' in the OpenCV window to quit.")
        
        while True:
            camera_frame = picam2.capture_array()
            
            if camera_frame is None:
                raise ValueError('Received empty frame buffer from libcamera.')

            side_by_side_frame = get_side_by_side_frame(camera_frame, field_frame)

            cv2.imshow(side_by_side_frame)
            

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
