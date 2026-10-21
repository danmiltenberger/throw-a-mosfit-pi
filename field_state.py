import csv
import cv2
import numpy as np
import math
from pathlib import Path
from typing import Sequence, Union
from tam_library_classes import (
    OurRobot,
    Detection,
    Point,
    FieldState,
    RADIANS_NORTH,
    RADIANS_SOUTH,
    POINT_BOTTOM_GOALBOX,
    POINT_TOP_GOALBOX,
    POINT_CENTER_FIELD,
)
from tam_library_save_and_load import clear_csv, save_field_state, load_field_state, get_field_frame
from tam_library_draw_and_show import draw_notched_rectangle, convert_to_field_pixels_global, draw_outlined_circle, draw_field
from tam_library_field import RADIUS_ROBOT, RADIUS_BALL
from settings_run import OUR_TEAM_COLOR, THEIR_TEAM_COLOR
from tam_library_colors import RED

# header consts
TIMESTEP = "time_step"
OUR_BOT_POSITION_IN = "our_bot_pos_in"
OUR_BOT_BEARING_RAD = "our_bot_bearing_rad"
THEIR_BOT_POSITION_IN = "their_bot_pos_in"
THEIR_BOT_BEARING_RAD = "their_bot_bearing_rad"
BALL_POS_IN = "ball_pos_in"


HEADERS_LIST = [
    TIMESTEP,
    OUR_BOT_POSITION_IN,
    OUR_BOT_BEARING_RAD,
    THEIR_BOT_POSITION_IN,
    THEIR_BOT_BEARING_RAD,
    BALL_POS_IN,
]


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
                row[column] = get_value(obj, attribute_path) if obj is not None else ""

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





def test_draw_field():

    state = FieldState(
            timestep=0.1,
            our_bot_pos=POINT_BOTTOM_GOALBOX,
            their_bot_pos=POINT_TOP_GOALBOX,
            our_bot_bearing=RADIANS_NORTH,
            their_bot_bearing=RADIANS_SOUTH,
            ball_position=POINT_CENTER_FIELD,
        )

    state.ball_position = Point(3,20)
    state.our_bot_pos = Point(8,-30)
    state.our_bot_bearing += np.pi/4

    field_frame = get_field_frame()

    field_frame = draw_field(field_frame= field_frame, field_state=state)
    
    cv2.imshow("Field", field_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_save_frame_to_csv():
    state = FieldState(
            timestep=0.1,
            our_bot_pos=POINT_BOTTOM_GOALBOX,
            their_bot_pos=POINT_TOP_GOALBOX,
            our_bot_bearing=RADIANS_NORTH,
            their_bot_bearing=RADIANS_SOUTH,
            ball_position=POINT_CENTER_FIELD,
        )
    
    state.ball_position = Point(3,-7)

    clear_csv("field_states/testing.csv")
    save_field_state(filename="field_states/testing.csv", state= state)
    loaded_state : FieldState = load_field_state("field_states/testing.csv")

    assert loaded_state == state


def test_draw_notched_rectangle():
    image = np.full((600, 700, 3), 240, dtype=np.uint8)
    
    draw_notched_rectangle(
        image,
        center_x=350,
        top_y=180,
        width=RADIUS_ROBOT * 2,
        height=RADIUS_ROBOT * 2,
        base_length=RADIUS_BALL * 2,
        angle_deg=90,
        color=(180, 100, 30),
        rotation_deg=30,
        border_thickness=1,
    )

    cv2.imshow("Notched rectangle with border", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_draw_field()

    

   