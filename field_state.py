import csv
from pathlib import Path
from typing import Sequence, Union
from tam_library_classes import OurRobot, Detection, Point


def save_frame_to_csv(
    file_path: Union[str, Path],
    frame_number: int,
    robots: Sequence["OurRobot"],
    detections: Sequence["Detection"],
) -> None:
    """
    Append one frame to a CSV file.

    - Supports up to 2 robots and 4 detections.
    - Expands Point attributes into separate x/y columns.
    - Leaves missing objects' columns blank.
    - Creates the file and header if needed.
    - Validates an existing file's header before appending.
    - Uses list order to assign robot1, robot2, detection1, etc.
    """
    max_robots = 2
    max_detections = 4

    if len(robots) > max_robots:
        raise ValueError(
            f"Expected at most {max_robots} robots; received {len(robots)}."
        )

    if len(detections) > max_detections:
        raise ValueError(
            f"Expected at most {max_detections} detections; "
            f"received {len(detections)}."
        )

    # Each entry is: (CSV column suffix, object attribute path).
    robot_fields = (
        ("label", "label"),
        ("field_pos_in_x", "field_pos_in.x"),
        ("field_pos_in_y", "field_pos_in.y"),
        ("azimuth_rad", "azimuth_rad"),
        ("field_pos_px_x", "field_pos_px.x"),
        ("field_pos_px_y", "field_pos_px.y"),
    )

    detection_fields = (
        ("name", "name"),
        ("width_px", "width_px"),
        ("screen_center_px_x", "screen_center_px.x"),
        ("screen_center_px_y", "screen_center_px.y"),
        ("area", "area"),
        ("range", "range"),
        ("azimuth_rad", "azimuth_rad"),
        ("field_pos_in_x", "field_pos_in.x"),
        ("field_pos_in_y", "field_pos_in.y"),
        ("field_pos_px_x", "field_pos_px.x"),
        ("field_pos_px_y", "field_pos_px.y"),
    )

    def get_value(obj, attribute_path):
        for attribute in attribute_path.split("."):
            obj = getattr(obj, attribute)
        return obj

    headers = ["frame_number"]
    row = {"frame_number": frame_number}

    groups = (
        ("robot", robots, max_robots, robot_fields),
        ("detection", detections, max_detections, detection_fields),
    )

    for prefix, objects, max_count, field_specs in groups:
        for index in range(max_count):
            obj = objects[index] if index < len(objects) else None

            for suffix, attribute_path in field_specs:
                column = f"{prefix}{index + 1}_{suffix}"
                headers.append(column)
                row[column] = (
                    get_value(obj, attribute_path) if obj is not None else ""
                )

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a+", newline="", encoding="utf-8") as csv_file:
        csv_file.seek(0)
        existing_header = next(csv.reader(csv_file), None)

        if existing_header is not None and existing_header != headers:
            raise ValueError(
                f"The existing CSV header does not match the expected "
                f"frame layout: {file_path}"
            )

        csv_file.seek(0, 2)
        writer = csv.DictWriter(csv_file, fieldnames=headers)

        if existing_header is None:
            writer.writeheader()

        writer.writerow(row)



def example():
    file_path_example = "field_states/tracking.csv"

    robots = [
        OurRobot(
            label="our bot",
            field_pos_in=Point(12.0, -30.0),
            azimuth_rad=0.5,
        ),
        OurRobot(
            label="second bot",
            field_pos_in=Point(24.0, 10.0),
            azimuth_rad=1.2,
        ),
    ]

    detections = [
        Detection(
            name="ball",
            width_px=25,
            screen_center_px=Point(320.0, 240.0),
            area=450,
            range=36.0,
            azimuth_rad=0.25,
            field_pos_in=Point(18.0, 5.0),
        ),
    ]

    save_frame_to_csv(
        file_path=file_path_example,
        frame_number=1,
        robots=robots,
        detections=detections,
    )

    # Append another frame with fewer objects.
    # robot2 and all detection columns will be blank in this row.
    save_frame_to_csv(
        file_path=file_path_example,
        frame_number=2,
        robots=robots[:1],
        detections=[],
    )

    print(f"saved to {file_path_example}")

if __name__ == "__main__":
    example()