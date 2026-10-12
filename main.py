from picamera2 import Picamera2 # type: ignore
import time
import cv2
from get_objects_from_frame import SCREEN_PX_X, SCREEN_PX_Y, get_object_list_from_frame, draw_object_list_to_frame
import numpy as np
import sys
from pathlib import Path
from display import show_frames_side_by_side, get_side_and_top_down
from convert_front_to_top_down import draw_object_list_to_field


def main():
    # Initialize the PiCamera2 object
    picam2 = Picamera2()
    
    # Configure for OpenCV: RGB888 format prevents BGR/RGB confusion
    # 640x480 keeps CPU load low for real-time edge detection on Pi 5
    config = picam2.create_preview_configuration(
        main={'format': 'RGB888', 'size': (SCREEN_PX_X, SCREEN_PX_Y)}
    )
    picam2.configure(config)


    frame_black_field = cv2.imread("black_field.png")
    frame_empty_field = cv2.imread("empty_field.png")
    
    try:
        picam2.start()
        # Allow the AGC (Automatic Gain Control) to settle
        time.sleep(1.0) 
        
        print("Camera started. Press 'q' in the OpenCV window to quit.")
        
        while True:
            # capture_array() returns a NumPy array directly
            frame_camera = picam2.capture_array()
            
            if frame_camera is None:
                raise ValueError('Received empty frame buffer from libcamera.')

            combined_frame, object_list = get_side_and_top_down(frame_camera, frame_black_field)
            cv2.imshow('frame and field', combined_frame)

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




def get_or_create_directory(directory: str | Path) -> Path:
    """
    Create a directory if needed and return its absolute path.

    Missing parent directories are also created.
    Existing directories and their contents are left unchanged.

    Args:
        directory: Directory path as a string or Path object.

    Returns:
        Absolute Path object for the directory.

    Raises:
        FileExistsError: If the target path exists as a file.
        OSError: If creation fails, such as due to insufficient permissions.
    """
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

if __name__ == '__main__':
    main()
