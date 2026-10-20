import csv
import cv2
from pathlib import Path
from datetime import datetime
from tam_library_classes import Detection, OurRobot
import numpy as np

import csv
import json
from pathlib import Path
from typing import Sequence

# folder consts
HSV_BOUNDS = "hsv_bounds"
FIELD_PNGS = "field_pngs"
TRAINING_DATA = "training_data"
DETECIONS = "detections"


# Edit this list to select the Detection attributes to save/load.
# vid_frame is always included automatically as the first column.
DETECTION_HEADERS = [
    "name",
    "width_px",
    "screen_center_px.x",
    "screen_center_px.y",
    "area",
    "range",
    "azimuth_rad",
    "field_pos_in.x",
    "field_pos_in.y",
    "field_pos_px.x",
    "field_pos_px.y",
]

# Supported columns and their types when loading.
_COLUMN_TYPES = {
    "name": str,
    "width_px": int,
    "screen_center_px.x": float,
    "screen_center_px.y": float,
    "area": int,
    "range": float,
    "azimuth_rad": float,
    "field_pos_in.x": float,
    "field_pos_in.y": float,
    "field_pos_px.x": float,
    "field_pos_px.y": float,
}


def _validate_headers(headers: Sequence[str]) -> list[str]:
    headers = list(headers)

    if not headers:
        raise ValueError("Select at least one Detection column.")

    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate Detection headers are not allowed.")

    unknown = set(headers) - _COLUMN_TYPES.keys()
    if unknown:
        raise ValueError(f"Unsupported Detection headers: {sorted(unknown)}")

    return headers


def _get_attribute(detection: Detection, header: str):
    value = detection
    for attribute in header.split("."):
        value = getattr(value, attribute)
    return value


def _set_attribute(detection: Detection, header: str, value) -> None:
    attributes = header.split(".")
    target = detection

    for attribute in attributes[:-1]:
        target = getattr(target, attribute)

    setattr(target, attributes[-1], value)


def save_detections_csv(
    filename: str | Path,
    vid_frame: int,
    detections: Sequence[Detection],
    headers: Sequence[str] | None = None,
) -> None:
    """Append one frame's detections as one CSV row."""
    if not isinstance(vid_frame, int) or isinstance(vid_frame, bool):
        raise TypeError("vid_frame must be an integer.")

    headers = _validate_headers(
        DETECTION_HEADERS if headers is None else headers
    )
    fieldnames = ["vid_frame", *headers]
    path = Path(filename)
    has_content = path.exists() and path.stat().st_size > 0

    if has_content:
        with path.open("r", newline="", encoding="utf-8") as file:
            existing_headers = next(csv.reader(file), None)

        if existing_headers != fieldnames:
            raise ValueError(
                "Existing CSV headers do not match the selected headers.\n"
                f"Existing: {existing_headers}\n"
                f"Selected: {fieldnames}"
            )

    row = {"vid_frame": vid_frame}

    for header in headers:
        row[header] = json.dumps(
            [_get_attribute(detection, header) for detection in detections],
            allow_nan=False,
        )

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not has_content:
            writer.writeheader()

        writer.writerow(row)

    


def load_detections_csv(
    filename: str | Path,
    headers: Sequence[str] | None = None,
) -> list[tuple[int, list[Detection]]]:
    """
    Load CSV rows as (vid_frame, detections) tuples.

    Only selected attributes are loaded. Other attributes keep their
    Detection defaults.
    """
    headers = _validate_headers(
        DETECTION_HEADERS if headers is None else headers
    )
    frames = []

    with Path(filename).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        file_headers = reader.fieldnames

        if not file_headers or file_headers[0] != "vid_frame":
            raise ValueError("The first CSV column must be 'vid_frame'.")

        if len(file_headers) != len(set(file_headers)):
            raise ValueError("The CSV contains duplicate headers.")

        missing = set(headers) - set(file_headers)
        if missing:
            raise ValueError(
                f"Selected columns missing from CSV: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                if None in row:
                    raise ValueError("Row contains extra CSV cells.")

                vid_frame = int(row["vid_frame"])
                columns = {
                    header: json.loads(row[header])
                    for header in headers
                }

                if not all(isinstance(values, list)
                           for values in columns.values()):
                    raise ValueError(
                        "Each Detection cell must contain a JSON list."
                    )

                lengths = {len(values) for values in columns.values()}
                if len(lengths) != 1:
                    raise ValueError(
                        "Detection columns contain different list lengths."
                    )

                detection_count = lengths.pop()
                detections = []

                for index in range(detection_count):
                    detection = Detection()

                    for header in headers:
                        value = _COLUMN_TYPES[header](columns[header][index])
                        _set_attribute(detection, header, value)

                    detections.append(detection)

                frames.append((vid_frame, detections))

            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid CSV row {row_number}: {exc}"
                ) from exc

    return frames


def clear_detections_csv(filename: str | Path) -> None:
    """Clear all CSV contents, including the header."""
    with Path(filename).open("w", newline="", encoding="utf-8"):
        pass


def get_save_file_path(directory: str = "detections", file_name : str = datetime.now().strftime("%Y_%m_%dT%H_%M_%S"), ending : str = ".csv") -> str:
    file_path = directory + "/" + file_name + ending
    return file_path


def get_or_create_directory(directory: str | Path) -> Path:
    """
    Create a directory if needed and return its absolute path.

    Missing parent directories are also created.
    Existing directories and their contents are left unchanged.

    Args:
        directory: Directory path as a string or Path object.

    Returns:
        Absolute Path object for the directory.

    Raises:
        FileExistsError: If the target path exists as a file.
        OSError: If creation fails, such as due to insufficient permissions.
    """
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image_as_frame(image_path: str) -> np.ndarray:
    """Load an image as a BGR NumPy array, or raise an error."""
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError(f"Could not load image: {image_path}")

    return frame


def save_frame_as_image(output_path, frame):
    if not cv2.imwrite(output_path, frame):
        raise OSError(f"Could not save to path {output_path}")
    print(f"saved frame to path {output_path}")


def select_from_directory(directory, ending= ".png"):
    """
    Let the user select a PNG file from a directory.

    Returns:
        Path: the selected PNG file
    Raises:
        ValueError: if the directory is invalid or contains no PNG files
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"Not a valid directory: {directory}")

    # Only include regular files ending in .png (case-insensitive)
    sorted_files = sorted(
        p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ending
    )

    if not sorted_files:
        raise ValueError(f"No {ending} files found in: {dir_path}")

    print(f"Select a {ending} file:")

    for i, png_file in enumerate(sorted_files, start=1):
        print(f"{i}. {png_file.name}")

    while True:
        choice = input("Enter a number: ").strip()

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(sorted_files):
                selected = sorted_files[index - 1]
                print(f"Selected: {selected}")
                return selected

        print("Invalid selection. Please enter a number from the list.")


def load_video_as_list_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    return frames


def save_frames_as_video(
    frames,
    output_path,
    fps=30,
    codec="mp4v",
    frame_size=None,
):
    """
    Save a list of BGR frames as a video.

    Args:
        frames: List/array of frames, each shaped (H, W, 3) in BGR format.
        output_path: Output video file path, e.g. "output.mp4".
        fps: Frames per second.
        codec: FourCC codec. Use "mp4v" for .mp4 or "XVID" for .avi.
        frame_size: Optional (width, height). If None, inferred from the first frame.

    Returns:
        Path to the saved video.

    Raises:
        ValueError: If frames is empty or frames have inconsistent sizes.
    """
    if not frames:
        raise ValueError("frames must contain at least one frame")

    if frame_size is None:
        height, width = frames[0].shape[:2]
        frame_size = (width, height)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for {output_path}. "
            "Try codec='XVID' with an .avi output file."
        )

    try:
        for i, frame in enumerate(frames):
            if frame.shape[1::-1] != frame_size:
                raise ValueError(
                    f"Frame {i} has size {frame.shape[1::-1]}, expected {frame_size}"
                )
            writer.write(frame)
    finally:
        writer.release()

    return output_path




