import csv
import json
from pathlib import Path

import cv2
import pandas as pd


def save_detections_to_csv(
    object_list,
    csv_path,
    video_timestamp_ms,
    frame_number=None,
    append=True,
):
    """
    Save one frame's object detections to a CSV file.

    Parameters
    ----------
    object_list : list[dict]
        Output from your image-recognition code.
    csv_path : str or Path
        Destination CSV file.
    video_timestamp_ms : float
        Timestamp of the frame, preferably from:
        cap.get(cv2.CAP_PROP_POS_MSEC)
    frame_number : int, optional
        Frame index, useful for joining detections back to a saved video.
    append : bool
        If True, append to an existing CSV. If False, overwrite it.

    Returns
    -------
    list[dict]
        The rows actually written to the CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed columns; add or remove fields here as your detector evolves.
    fieldnames = [
        "video_timestamp_ms",
        "frame_number",
        "name",
        "label",
        "shape",
        "color_bgr",
        "area",
        "width",
        "radius",
        "center_px_x",
        "center_px_y",
        "center_in_x",
        "center_in_y",
        "distance_in",
        "known_width_in",
        "theta_rad",
        "raw_object",
    ]



    rows = []
    for obj in object_list or []:
        center_px = obj.get("center_px", (None, None))
        center_in = obj.get("center_in", (None, None))
        color = obj.get("color", (None, None, None))

        rows.append({
            "video_timestamp_ms": video_timestamp_ms,
            "frame_number": frame_number,
            "name": obj.get("name", ""),
            "label": obj.get("label", ""),
            "shape": obj.get("shape", ""),
            "color_bgr": json.dumps(list(color)),
            "area": obj.get("area"),
            "width": obj.get("width"),
            "radius": obj.get("radius"),
            "center_px_x": center_px[0],
            "center_px_y": center_px[1],
            "center_in_x": center_in[0],
            "center_in_y": center_in[1],
            "distance_in": obj.get("distance_in"),
            "known_width_in": obj.get("known_width_in"),
            "theta_rad": obj.get("theta_rad"),
            
            # Keeps any future/unrecognized fields without changing the schema.
            "raw_object": json.dumps(obj, default=str),
        })

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    mode = "a" if append else "w"

    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

        # Write headers only for a new/overwritten file.
        if not append or not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    return rows


def load_detections_from_csv(csv_path):
    """
    Load detections into a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
        One row per detected object.
    """
    df = pd.read_csv(csv_path)

    # Restore tuple-like fields if needed.
    df["color_bgr"] = df["color_bgr"].apply(
        lambda x: tuple(json.loads(x)) if pd.notna(x) else None
    )
    df["raw_object"] = df["raw_object"].apply(
        lambda x: json.loads(x) if pd.notna(x) else None
    )

    return df