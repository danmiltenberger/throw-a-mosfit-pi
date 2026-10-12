import math
import time
from pathlib import Path

import cv2

from main import get_or_create_directory, SCREEN_PX_X, SCREEN_PX_Y


def preview_video(
    source: str | Path | int = 0,
    window_name: str = "Video Preview",
    picam_size: tuple[int, int] = (SCREEN_PX_X, SCREEN_PX_Y),
    save_path: str | Path | None = None,
    save_fps: float | None = None,
) -> None:
    """
    Preview a video file, webcam, or Pi camera, optionally saving to MP4.

    The preview includes a time overlay. The saved video does not.

    Args:
        source:
            File path: Play a video with a seconds-left overlay.
            Integer: Webcam index (0 = default camera).
            "picam": Use Picamera2.
        window_name: Preview window title.
        picam_size: Pi camera resolution (width, height).
        save_path: Output .mp4 path, or None to disable recording.
            Missing parent directories are created.
            Existing output files are overwritten.
        save_fps: Output video FPS. Defaults to the reported source FPS
            when available, otherwise 30 FPS. Does not change capture speed.

    Controls:
        Press 'q' or Escape, or close the window, to stop.

    Notes:
        Saves original frames without the preview overlay or audio.
        Requires a graphical desktop for the preview.
        File countdowns assume accurate, constant-frame-rate metadata.
        Live recording uses fixed output FPS, not capture timestamps.
    """
    cap = None
    picam = None
    writer = None
    window_created = False

    is_picam = isinstance(source, str) and source == "picam"
    is_live = is_picam or isinstance(source, int)

    output_path = None

    if save_path is not None:
        output_path = Path(save_path).expanduser().resolve()

        if output_path.suffix.lower() != ".mp4":
            raise ValueError("save_path must have a .mp4 extension.")

        if not is_live:
            input_path = Path(source).expanduser().resolve()
            if output_path == input_path:
                raise ValueError(
                    "save_path must differ from the input video path."
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_fps is not None:
        if not math.isfinite(save_fps) or save_fps <= 0:
            raise ValueError("save_fps must be a positive, finite number.")

    try:
        # Open the capture source.
        if is_picam:
            from picamera2 import Picamera2 # type: ignore 

            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": picam_size, "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()

            fps = None
            total_frames = None

        else:
            capture_source = (
                source if isinstance(source, int) else str(source)
            )
            cap = cv2.VideoCapture(capture_source)

            if not cap.isOpened():
                raise OSError(f"Could not open video source: {source}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if not is_live:
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Video has no valid FPS metadata.")

                if not math.isfinite(total_frames) or total_frames <= 0:
                    raise ValueError(
                        "Video has no valid frame-count metadata."
                    )

        source_fps_valid = (
            fps is not None and math.isfinite(fps) and fps > 0
        )

        if save_fps is not None:
            output_fps = save_fps
        elif source_fps_valid:
            output_fps = fps
        else:
            output_fps = 30.0

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True

        frames_read = 0
        start_time = time.perf_counter()

        while True:
            # Capture an original frame.
            if is_picam:
                frame = picam.capture_array("main")
            else:
                success, frame = cap.read()
                if not success:
                    break

            frames_read += 1

            # Save BEFORE adding any preview overlays.
            if output_path is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )

                    if not writer.isOpened():
                        raise OSError(
                            f"Could not open MP4 writer: {output_path}"
                        )

                writer.write(frame)

            # Use a separate copy for the annotated preview.
            preview_frame = frame.copy()

            if is_live:
                elapsed = time.perf_counter() - start_time
                text = f"Live | Elapsed: {elapsed:.1f} s"
            else:
                seconds_left = max(
                    0.0, (total_frames - frames_read) / fps
                )
                text = f"Time left: {seconds_left:.1f} s"

            # Black outline with white text.
            for color, thickness in [
                ((0, 0, 0), 4),
                ((255, 255, 255), 2),
            ]:
                cv2.putText(
                    preview_frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, preview_frame)

            # Pace file playback; keep live capture responsive.
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

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        try:
            if writer is not None:
                writer.release()

            if cap is not None:
                cap.release()

            if picam is not None:
                picam.close()

        finally:
            if window_created:
                cv2.destroyWindow(window_name)


if __name__ == "__main__":
    # Webcam preview with overlay; saved video without overlay.
    directory = str(get_or_create_directory("training_videos"))
    preview_video(0, save_path= (directory + "output/webcam.mp4"))

    # Raspberry Pi camera.
    # preview_video(
    #     "picam",
    #     picam_size=(1280, 720),
    #     save_path="output/picam.mp4",
    #     save_fps=30,
    # )

    # # Video file.
    # preview_video(
    #     "videos/input.mp4",
    #     save_path="output/clean_video.mp4",
    # )

    # # Preview without saving.
    # preview_video(0)