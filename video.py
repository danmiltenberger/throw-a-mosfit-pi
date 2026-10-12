import math
import time
from pathlib import Path
from main import SCREEN_PX_X, SCREEN_PX_Y

import cv2


def preview_video(
    source: str | Path | int = 0,
    window_name: str = "Video Preview",
    picam_size: tuple[int, int] = (SCREEN_PX_X, SCREEN_PX_Y),
) -> None:
    """
    Preview a video file, webcam, or Raspberry Pi camera.

    Args:
        source:
            File path: Play a video with a seconds-left overlay.
            Integer: Open a webcam by index (0 = default camera).
            "picam": Capture using Picamera2.
        window_name: Preview window title.
        picam_size: Pi camera resolution (width, height).

    Controls:
        Press 'q' or Escape, or close the window, to stop.

    Notes:
        Modifies only displayed frames; does not record video.
        Requires a graphical desktop for the OpenCV preview.
    """
    cap = None
    picam = None
    window_created = False

    is_picam = isinstance(source, str) and source == "picam"
    is_live = is_picam or isinstance(source, int)

    try:
        if is_picam:
            # Only import Picamera2 when using a Raspberry Pi camera.
            from picamera2 import Picamera2

            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": picam_size, "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()

            fps = None
            total_frames = None
        else:
            capture_source = source if isinstance(source, int) else str(source)
            cap = cv2.VideoCapture(capture_source)

            if not cap.isOpened():
                raise OSError(f"Could not open video source: {source}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if not is_live:
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Video has no valid FPS metadata.")
                if not math.isfinite(total_frames) or total_frames <= 0:
                    raise ValueError("Video has no valid frame-count metadata.")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True

        frames_read = 0
        start_time = time.perf_counter()

        while True:
            if is_picam:
                frame = picam.capture_array("main")
            else:
                success, frame = cap.read()
                if not success:
                    break

            frames_read += 1

            if is_live:
                elapsed = time.perf_counter() - start_time
                text = f"Live | Elapsed: {elapsed:.1f} s"
            else:
                seconds_left = max(0.0, (total_frames - frames_read) / fps)
                text = f"Time left: {seconds_left:.1f} s"

            # Draw a black outline and white text for readability.
            for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 2)]:
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)

            if is_live:
                delay_ms = 1
            else:
                next_frame_time = start_time + frames_read / fps
                delay_ms = max(
                    1,
                    int(round(
                        (next_frame_time - time.perf_counter()) * 1000
                    )),
                )

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        try:
            if cap is not None:
                cap.release()
            if picam is not None:
                picam.close()
        finally:
            if window_created:
                cv2.destroyWindow(window_name)