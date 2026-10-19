import csv
from pathlib import Path
from datetime import datetime
from tam_library_classes import Detection, OurRobot

import csv
import json
from pathlib import Path
from typing import Sequence
from tam_library_OOP import get_detections_from_frame, load_image
from icecream import ic


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


def example_save_detections():
    # For example, save only these attributes.
    DETECTION_HEADERS = [
        "name",
        "range",
        "azimuth_rad",
        "field_pos_in.x",
        "field_pos_in.y",
    ]

    file_name = "detections/example_save_detection.csv"
    our_bot = OurRobot()

    camera_frame = load_image("training_data/tennis_ball_48in.png")
    list_detections = get_detections_from_frame(frame=camera_frame, bot=our_bot)

    # for clearer development
    clear_detections_csv(file_name)

    # Call once per camera frame.
    save_detections_csv(file_name, vid_frame=0, detections=list_detections)

    # Reload all frames.
    frames = load_detections_csv(file_name)

    ic(len(frames))

    for vid_frame, frame_detections in frames:
        print(f"Frame {vid_frame}:")
        for detection in frame_detections:
            print(detection)


if __name__ == "__main__":
    example_save_detections()