import cv2
from collections.abc import Iterator
import numpy as np

from display import get_side_and_top_down



def read_video_frames(video_path: str) -> Iterator[np.ndarray]:
    """Yield each video frame as a BGR NumPy array."""
    capture = cv2.VideoCapture(video_path)

    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        while True:
            success, frame = capture.read()

            if not success:
                break

            yield frame
    finally:
        capture.release()


if __name__ == "__main__":
    vid_path = 'training_videos/tennis_ball_blank_wall.mp4'

    frame_black_field = cv2.imread("black_field.png")
    frame_empty_field = cv2.imread("empty_field.png")

    frames = read_video_frames(vid_path)

    try:
        for frame_index, frame in enumerate(frames):
            # Process the current frame here.
            cv2.imshow("Video", frame)

            combined_frame, object_list = get_side_and_top_down(frame, frame_black_field)
            cv2.imshow('frame and field', combined_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Press 'q' to stop.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        frames.close()
        cv2.destroyAllWindows()