from pathlib import Path

import cv2
import numpy as np


ball: dict = {
    "lower": np.array([0, 153, 0]),
    "upper": np.array([179, 255, 255]),
}


def detect_ball(
    input_path: str | Path,
    output_dir: str | Path,
    ball_hsv: dict,
    *,
    min_area: float = 100.0,
    min_circularity: float = 0.65,
    kernel_size: int = 5,
    fallback_fps: float = 30.0,
    overwrite: bool = False,
) -> dict:
    """
    Detect one ball per frame and save intermediate processing stages.

    Parameters
    ----------
    input_path:
        PNG image or video file.
    output_dir:
        Directory in which to save results.
    ball_hsv:
        Dictionary containing "lower" and "upper" HSV bounds.
    min_area:
        Minimum candidate contour area in pixels squared.
    min_circularity:
        Minimum 4*pi*area/perimeter**2. An ideal circle has value 1.
        Lower this for partially occluded or motion-blurred balls.
    kernel_size:
        Odd-sized morphology kernel. Use 1 to disable mask cleanup.
    fallback_fps:
        Output frame rate if the video does not report a valid FPS.
    overwrite:
        Allow replacement of existing output files.

    Returns
    -------
    dict:
        Output paths, processed frame count, and detection count.

    Notes
    -----
    Video outputs use MJPG in AVI containers and do not retain audio.
    This is frame-by-frame detection, not temporal tracking.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if min_area < 0:
        raise ValueError("min_area must be nonnegative.")
    if not 0 <= min_circularity <= 1:
        raise ValueError("min_circularity must be between 0 and 1.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if not np.isfinite(fallback_fps) or fallback_fps <= 0:
        raise ValueError("fallback_fps must be finite and positive.")

    # Validate before casting: casting invalid values to uint8 can wrap them.
    lower = np.asarray(ball_hsv["lower"], dtype=float)
    upper = np.asarray(ball_hsv["upper"], dtype=float)
    hsv_max = np.array([179, 255, 255])

    for bound in (lower, upper):
        if (
            bound.shape != (3,)
            or not np.all(np.isfinite(bound))
            or np.any(bound < 0)
            or np.any(bound > hsv_max)
            or np.any(bound != np.floor(bound))
        ):
            raise ValueError(
                "HSV bounds must contain three integers within "
                "H: 0–179, S: 0–255, V: 0–255."
            )

    if np.any(lower > upper):
        raise ValueError("Each lower HSV bound must be <= its upper bound.")

    lower = lower.astype(np.uint8)
    upper = upper.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    def process_frame(frame: np.ndarray) -> tuple[dict, bool]:
        """Apply the same processing pipeline to images and video frames."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, lower, upper)

        # Opening removes small specks; closing fills small gaps.
        clean_mask = raw_mask.copy()
        if kernel_size > 1:
            clean_mask = cv2.morphologyEx(
                clean_mask, cv2.MORPH_OPEN, kernel
            )
            clean_mask = cv2.morphologyEx(
                clean_mask, cv2.MORPH_CLOSE, kernel
            )

        # Retain original colors wherever the cleaned mask is white.
        color_mask = cv2.bitwise_and(frame, frame, mask=clean_mask)
        annotated = frame.copy()

        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_contour = None
        best_area = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / perimeter**2
            if circularity < min_circularity:
                continue

            if area > best_area:
                best_area = area
                best_contour = contour

        detected = best_contour is not None

        if detected:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(round(x)), int(round(y)))
            draw_radius = max(1, int(np.ceil(radius)))

            cv2.circle(annotated, center, draw_radius, (0, 255, 0), 2)
            cv2.circle(annotated, center, 3, (0, 0, 255), -1)

            label = f"Ball: ({center[0]}, {center[1]}), r={radius:.1f}px"
        else:
            label = "Ball not detected"

        # Fixed label location avoids clipping labels near the ball.
        label_position = (10, min(30, frame.shape[0] - 1))
        cv2.putText(
            annotated, label, label_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated, label, label_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
            cv2.LINE_AA,
        )

        return {
            "01_original": frame,
            "02_raw_mask": raw_mask,
            "03_clean_mask": clean_mask,
            "04_color_mask": color_mask,
            "05_annotated": annotated,
        }, detected

    stage_names = (
        "01_original",
        "02_raw_mask",
        "03_clean_mask",
        "04_color_mask",
        "05_annotated",
    )

    is_image = input_path.suffix.lower() == ".png"
    extension = ".png" if is_image else ".avi"

    paths = {
        name: output_dir / f"{name}{extension}"
        for name in stage_names
    }

    # Never overwrite the source itself.
    for path in paths.values():
        if path.resolve() == input_path.resolve():
            raise ValueError("An output path would overwrite the input file.")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use overwrite=True to replace it."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_image:
        frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read PNG: {input_path}")

        stages, detected = process_frame(frame)

        for name, image in stages.items():
            if not cv2.imwrite(str(paths[name]), image):
                raise OSError(f"Could not save image: {paths[name]}")

        frames_processed = 1
        frames_with_detection = int(detected)

    else:
        capture = cv2.VideoCapture(str(input_path))
        writers = {}
        frames_processed = 0
        frames_with_detection = 0

        try:
            if not capture.isOpened():
                raise ValueError(f"Could not open video: {input_path}")

            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("Video contains no readable frames.")

            height, width = frame.shape[:2]
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = fallback_fps

            fourcc = cv2.VideoWriter.fourcc(*"MJPG")

            for name, path in paths.items():
                writer = cv2.VideoWriter(
                    str(path), fourcc, fps, (width, height), True
                )
                writers[name] = writer
                if not writer.isOpened():
                    raise OSError(f"Could not create video: {path}")

            while ok:
                if frame.shape[:2] != (height, width):
                    raise ValueError("Video frame dimensions changed.")

                stages, detected = process_frame(frame)

                for name, image in stages.items():
                    # Write masks as three-channel black-and-white frames
                    # for consistent video encoder compatibility.
                    if image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    writers[name].write(image)

                frames_processed += 1
                frames_with_detection += int(detected)
                ok, frame = capture.read()

        finally:
            capture.release()
            for writer in writers.values():
                writer.release()

    return {
        "output_paths": paths,
        "frames_processed": frames_processed,
        "frames_with_detection": frames_with_detection,
    }


path = r".temp/vlcsnap-2026-10-05-16h43m03s033.png"


result = detect_ball(
    input_path=path,
    output_dir="ball_image_results",
    ball_hsv=ball,
)

print(result)
