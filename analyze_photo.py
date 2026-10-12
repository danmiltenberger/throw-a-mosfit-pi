import cv2
from display import get_side_and_top_down
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """Load an image as a BGR NumPy array, or raise an error."""
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError(f"Could not load image: {image_path}")

    return frame


def analyze_photo(path_to_frame):

    frame_black_field : np.ndarray = load_image("black_field.png")
    frame_empty_field : np.ndarray = load_image("empty_field.png")

    frame = load_image(path_to_frame)

    name = str(path_to_frame)
    


    combined_frame, object_list = get_side_and_top_down(
        frame, 
        frame_empty_field,)
    cv2.imshow('frame and field', combined_frame)

    while True:
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":

    path_to_frame = "training_photos/tennis_center.png"
    path_to_frame = "training_photos/tennis_right.png"


    analyze_photo(path_to_frame)