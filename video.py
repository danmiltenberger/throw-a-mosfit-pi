import math
import time
from pathlib import Path
import os
from get_objects_from_frame import get_current_time

import cv2


def preview_video(
    video_path: str | Path,
    window_name: str = "Video Preview",
) -> None:
    """
    Play a video with a seconds-remaining overlay.

    Press 'q' or Escape, or close the window, to stop.

    Args:
        video_path: Path to the video file.
        window_name: Preview window title.

    Raises:
        OSError: If the video cannot be opened.
        ValueError: If valid FPS or frame-count metadata is unavailable.
    """
    cap = cv2.VideoCapture(str(video_path))
    window_created = False

    try:
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS metadata.")
        if not math.isfinite(total_frames) or total_frames <= 0:
            raise ValueError("Video has no valid frame-count metadata.")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True

        frames_read = 0
        playback_start = time.perf_counter()

        while True:
            success, frame = cap.read()
            if not success:
                break

            frames_read += 1
            seconds_left = max(0.0, (total_frames - frames_read) / fps)
            text = f"Time left: {seconds_left:.1f} s"

            # Black outline improves readability over different backgrounds.
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

            # Account for decoding and drawing time when pacing playback.
            next_frame_time = playback_start + frames_read / fps
            delay_ms = max(
                1, int(round((next_frame_time - time.perf_counter()) * 1000))
            )

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        if window_created:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":

    file_name = input("enter a file name, hit enter to default to an iso date")
    if file_name == "":
        file_name = get_current_time()


    directory = "training_videos"
    os.mkdir(directory)

    file_path = directory + "/" + file_name + ".mp4"

    preview_video(file_path)