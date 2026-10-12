import cv2
import numpy as np


def show_frames_side_by_side(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    target_height: int | None = None,
    window_name: str = "Side-by-Side Frames",
) -> np.ndarray:
    """Display two BGR frames side by side at the same height.

    If target_height is None, use the right frame's original height.
    Each frame's aspect ratio is preserved.
    """
    for name, frame in (("frame_left", frame_left), ("frame_right", frame_right)):
        if frame is None or frame.size == 0:
            raise ValueError(f"{name} must be a non-empty image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{name} must be a three-channel BGR image.")

    if frame_left.dtype != frame_right.dtype:
        raise ValueError("Both frames must have the same data type.")

    if target_height is None:
        target_height = frame_right.shape[0]
    if target_height <= 0:
        raise ValueError("target_height must be positive.")

    def resize_to_height(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == target_height:
            return frame

        scale = target_height / height
        new_width = max(1, round(width * scale))
        return cv2.resize(
            frame,
            (new_width, target_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

    combined = np.hstack(
        (resize_to_height(frame_left), resize_to_height(frame_right))
    )
    cv2.imshow(window_name, combined)
    return combined

if __name__ == "__main__":
    frame_left = cv2.imread("training_photos/2026-10-11T20_53_44.168011.png")
    frame_right = cv2.imread("black_field.png")

    combined = show_frames_side_by_side(frame_left, frame_right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()