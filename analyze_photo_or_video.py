import cv2
from icecream import ic
from tam_library import get_side_and_top_down, load_image, FIELD_PNGS, read_video_frames
import numpy as np

from run_settings import USE_DRAWN_FIELD

#file_path = "training_data/2026-10-11T20_53_06.567733.png"
file_path = "training_data/tennis_ball_with_corners.mp4"


def get_frame_and_field(frame: np.ndarray) -> np.ndarray:

    if USE_DRAWN_FIELD:
        frame_field: np.ndarray = load_image(f"{FIELD_PNGS}/empty_field.png")
    else:
        frame_field: np.ndarray = load_image(f"{FIELD_PNGS}/black_field.png")

    combined_frame, object_list = get_side_and_top_down(
        frame,
        frame_field,
    )

    ic(object_list)

    return combined_frame


def analyze_photo_or_video(file_path: str) -> None:
    if file_path.endswith(".mp4"):
        frames = read_video_frames(file_path)
    elif file_path.endswith(".png") or file_path.endswith(".jpg"):
        frame = load_image(file_path)
        combined_frame = get_frame_and_field(frame)
        cv2.imshow("frame and field", combined_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .mp4, .png, or .jpg file."
        )

    try:
        for frame_index, frame in enumerate(frames):

            combined_frame = get_frame_and_field(frame)

            cv2.imshow("frame and field", combined_frame)

            # Press 'q' to stop.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        frames.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_photo_or_video(file_path)
