import time

import cv2
import numpy as np


ball: dict = {
    "lower": np.array([0, 153, 0]),
    "upper": np.array([179, 255, 255]),
}


def detect_ball_live(
    ball_hsv: dict,
    source: int | str = 0,
    *,
    use_picamera2: bool = False,
    capture_size: tuple[int, int] = (640, 480),
    processing_width: int = 320,
    requested_fps: int = 30,
    min_area: float = 50.0,
    min_circularity: float = 0.6,
    kernel_size: int = 3,
    show_fps: bool = True,
) -> None:
    """
    Display a live feed with the detected ball circled.

    Press Q or Escape in the display window to exit.

    Parameters
    ----------
    ball_hsv:
        Dictionary containing "lower" and "upper" HSV bounds.
    source:
        OpenCV camera index, video filename, or stream URL.
        Ignored when use_picamera2=True.
    use_picamera2:
        Use a Raspberry Pi CSI camera instead of OpenCV VideoCapture.
    capture_size:
        Requested capture dimensions as (width, height).
    processing_width:
        Maximum frame width used for detection and display.
        Frames are downscaled if needed, but never upscaled.
    requested_fps:
        Requested camera frame rate; hardware may not honor it.
    min_area:
        Minimum contour area in pixels squared AFTER downscaling.
    min_circularity:
        Minimum 4*pi*area/perimeter**2. Set to 0 to skip this filter.
    kernel_size:
        Odd morphology kernel size. Use 1 to skip mask cleanup.
    show_fps:
        Display measured loop throughput, including capture and display.

    Notes
    -----
    Selects the largest qualifying contour independently each frame.
    Requires an OpenCV installation with GUI support and a display session.
    """
    if processing_width <= 0:
        raise ValueError("processing_width must be positive.")
    if requested_fps <= 0:
        raise ValueError("requested_fps must be positive.")
    if len(capture_size) != 2 or any(v <= 0 for v in capture_size):
        raise ValueError("capture_size must contain two positive dimensions.")
    if min_area < 0 or not 0 <= min_circularity <= 1:
        raise ValueError("Invalid area or circularity threshold.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    lower = np.asarray(ball_hsv["lower"], dtype=float)
    upper = np.asarray(ball_hsv["upper"], dtype=float)
    limits = np.array([179, 255, 255])


    def validate_bounds(lower_key: str, upper_key: str):
        lower = np.asarray(ball_hsv[lower_key], dtype=float)
        upper = np.asarray(ball_hsv[upper_key], dtype=float)

        for bound in (lower, upper):
            if (
                bound.shape != (3,)
                or not np.all(np.isfinite(bound))
                or np.any(bound < 0)
                or np.any(bound > limits)
                or np.any(bound != np.floor(bound))
            ):
                raise ValueError(f"Invalid HSV bounds: {lower_key}, {upper_key}")

        if np.any(lower > upper):
            raise ValueError(
                f"Each {lower_key} value must be <= its {upper_key} value."
            )

        return lower.astype(np.uint8), upper.astype(np.uint8)


    # Validate and convert once, outside the frame-processing loop.
    lower, upper = validate_bounds("lower", "upper")
    lower2, upper2 = validate_bounds("lower2", "upper2")

    # Allocate the morphology kernel once, not on every frame.
    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        if kernel_size > 1
        else None
    )

    capture = None
    camera = None
    camera_started = False
    window_name = "Ball detection"

    cv2.setUseOptimized(True)

    try:
        if use_picamera2:
            from picamera2 import Picamera2

            camera = Picamera2()

            # Picamera2's RGB888 format gives BGR-ordered array bytes,
            # matching OpenCV's expected channel order.
            config = camera.create_video_configuration(
                main={"size": capture_size, "format": "RGB888"},
                controls={"FrameRate": float(requested_fps)},
                buffer_count=4,
            )
            camera.configure(config)
            camera.start()
            camera_started = True

        else:
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video source: {source}")

            if isinstance(source, int):
                # These are requests; some cameras/backends ignore them.
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
                capture.set(cv2.CAP_PROP_FPS, requested_fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        resize_shape = None
        input_shape = None
        measured_fps = 0.0
        fps_frames = 0
        fps_start = time.perf_counter()

        while True:
            if use_picamera2:
                frame = camera.capture_array("main")
            else:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

            # Cache resize dimensions until the source dimensions change.
            if frame.shape[:2] != input_shape:
                input_shape = frame.shape[:2]
                height, width = input_shape
                resize_shape = (
                    (
                        processing_width,
                        max(1, round(height * processing_width / width)),
                    )
                    if width > processing_width
                    else None
                )

            if resize_shape is not None:
                frame = cv2.resize(
                    frame, resize_shape, interpolation=cv2.INTER_AREA
                )

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask2 = cv2.inRange(hsv, lower2, upper2)

            # Accept pixels in either red range, reusing mask for the result.
            cv2.bitwise_or(mask, mask2, dst=mask)

            if kernel is not None:
                # One cleanup operation rather than opening + closing.
                cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, kernel, dst=mask
                )

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            best_contour = None
            best_area = min_area

            for contour in contours:
                area = cv2.contourArea(contour)

                # Do not calculate circularity for candidates that cannot
                # replace the current largest accepted candidate.
                if area < min_area or area <= best_area:
                    continue

                if min_circularity > 0:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter <= 0:
                        continue

                    circularity = 4.0 * np.pi * area / perimeter**2
                    if circularity < min_circularity:
                        continue

                best_contour = contour
                best_area = area

            # Draw directly on the frame; no annotation-frame copy.
            if best_contour is not None:
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                center = (round(x), round(y))

                cv2.circle(
                    frame, center, max(1, int(np.ceil(radius))),
                    (0, 255, 0), 2,
                )
                label = "Ball"
            else:
                label = "Ball not detected"

            if show_fps:
                fps_frames += 1
                now = time.perf_counter()
                elapsed = now - fps_start

                # Update the FPS value twice per second.
                if elapsed >= 0.5:
                    measured_fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_start = now

                label += f" | {measured_fps:.1f} FPS"

            cv2.putText(
                frame, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        if capture is not None:
            capture.release()

        if camera is not None:
            try:
                if camera_started:
                    camera.stop()
            finally:
                camera.close()

        cv2.destroyAllWindows()