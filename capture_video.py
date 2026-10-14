import math
import time

import cv2
from picamera2 import Picamera2
from main import get_or_create_directory, SCREEN_PX_X, SCREEN_PX_Y
from get_objects_from_frame import get_current_time

def capture_with_countdown(
    picam: Picamera2,
    duration_seconds: float,
    output_path: str | None = None,
    fps: float = 30.0,
):
    """
    Capture from an already-started Picamera2 and print time remaining.

    Optionally save color frames to an MP4.
    Configure the camera's main stream as RGB888 (BGR array order).
    """
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and greater than zero.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and greater than zero.")
    if output_path is not None and not str(output_path).lower().endswith(".mp4"):
        raise ValueError("output_path must end with .mp4.")

    writer = None
    last_displayed = None
    end_time = time.monotonic() + duration_seconds

    try:
        while True:
            remaining = end_time - time.monotonic()
            if remaining <= 0:
                break

            seconds_left = math.ceil(remaining)
            if seconds_left != last_displayed:
                hours, remainder = divmod(seconds_left, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(
                    f"\rTime left: {hours:02d}:{minutes:02d}:{seconds:02d}",
                    end="",
                    flush=True,
                )
                last_displayed = seconds_left

            frame = picam.capture_array("main")

            # Add your OpenCV processing here.

            if output_path is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open MP4 writer for {output_path!r}."
                        )

                writer.write(frame)

        print("\rTime left: 00:00:00", end="", flush=True)

    finally:
        if writer is not None:
            writer.release()
        print()

fps = 30.0
picam = Picamera2()

config = picam.create_video_configuration(
    main={"size": (SCREEN_PX_X, SCREEN_PX_Y), "format": "RGB888"},
    controls={"FrameRate": fps},
)
picam.configure(config)

try:
    picam.start()

    # Capture for 30 seconds and save an MP4.
    directory = str(get_or_create_directory("training_videos"))
    file_name = get_current_time()
    output_path = f"{directory}/{file_name}.mp4"

    capture_with_countdown(
        picam,
        duration_seconds=30,
        output_path= output_path,
        fps=fps,
    )

    # To capture without saving, instead use:
    # capture_with_countdown(picam, duration_seconds=30)

finally:
    picam.stop()
    picam.close()