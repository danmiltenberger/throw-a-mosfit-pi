from pathlib import Path
import cv2  # pip install opencv-python
from icecream import ic  # pip install icecream
import math
import numpy as np  # pip install numpy
from datetime import datetime
import csv
import ast
import json
from dataclasses import dataclass
from typing import Iterator, Optional


from run_settings import (
    kernel_size,
    FOCAL_LENGTH,
    SCREEN_PX_X,
    SCREEN_PX_Y,
    CIRCLE,
    RECT,
    search_for_list,
    PIXELS_PER_INCH,
    USE_DRAWN_FIELD,
)
from tam_library_colors import *

# NOTE - collapse all with ctrl+K, ctrl+0


# folder consts
HSV_BOUNDS = "hsv_bounds"
FIELD_PNGS = "field_pngs"
TRAINING_DATA = "training_data"


# object consts
RADIUS_ROBOT = int(14 * PIXELS_PER_INCH / 2)
RADIUS_BALL = int(7 * PIXELS_PER_INCH / 2)
RADIUS_TENNIS_BALL = int(2.7 * PIXELS_PER_INCH / 2)


field_center_px_x = int(SCREEN_PX_X / 2)
field_center_px_y = int(SCREEN_PX_Y / 2)

# goalbox center is 60 inches up / down from center field
goalbox_center_top_in = (0, 60)
goalbox_center_bot_in = (0, -60)

# field consts
# in = inches
# px = pixels
playing_field_in_x = 8 * 12
playing_field_in_y = 12 * 12

goal_net_in_x = 48
goal_net_in_y = 12  # deep

goalbox_in_x = 48
goalbox_in_y = 24  # infront of the goal

perim_tape_thickness_in = 2
perim_black_beyond_tape_in = 2
total_buffer_in = perim_tape_thickness_in + perim_black_beyond_tape_in

total_window_in_y = 2 * goal_net_in_y + playing_field_in_y
total_window_in_x = 2 * total_buffer_in + playing_field_in_x

# total window
total_window_px_x = int(total_window_in_x * PIXELS_PER_INCH)
total_window_px_y = int(total_window_in_y * PIXELS_PER_INCH)


@dataclass
class Point:
    """A point in 2D space, in inches or pixels."""

    x: float
    y: float

  


@dataclass
class Bearing:
    """distance in inches and angle in radians with respect to bot"""

    dis_in: float
    theta_rad: float


@dataclass
class Detection:
    """structure that explains what the vision system detected in the frame"""

    # possibly later add goal post info

    detected_soccer_ball: bool
    detected_cyan_bot: bool
    detected_yellow_bot: bool

    # keep the original pixel position and width from the camera
    px_center_soccer_ball: Point
    px_center_cyan_bot: Point
    px_center_yellow_bot: Point

    px_width_soccer_ball: int
    px_width_cyan_bot: int
    px_width_yellow_bot: int

    # also keep inches and distance (if we later improve the top down conversion)
    bearing_soccer_ball: Bearing
    bearing_cyan_bot: Bearing
    bearing_yellow_bot: Bearing

    # finally, and most usefully, convert the robot's bearing to a field position in inches, with the origin at the center of the field
    in_global_soccer_ball: Point
    in_global_cyan_bot: Point
    in_global_yellow_bot: Point


@dataclass
class Rectangle:
    """Rectangle geometry following OpenCV's x, y, width, height convention."""

    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 100


def load_image(image_path: str) -> np.ndarray:
    """Load an image as a BGR NumPy array, or raise an error."""
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError(f"Could not load image: {image_path}")

    return frame


def select_png(directory):
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
    png_files = sorted(
        p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"
    )

    if not png_files:
        raise ValueError(f"No PNG files found in: {dir_path}")

    print("Select a PNG file:")

    for i, png_file in enumerate(png_files, start=1):
        print(f"{i}. {png_file.name}")

    while True:
        choice = input("Enter a number: ").strip()

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(png_files):
                selected = png_files[index - 1]
                print(f"Selected: {selected}")
                return selected

        print("Invalid selection. Please enter a number from the list.")


def pick_from_list(input_list: list, info_str: str = ""):

    print(info_str)

    index = 0
    for item in input_list:
        print(f"{index}: {item}")
        index += 1

    user_choice = input("select a number from above list: ")
    user_choice = int(user_choice)

    list_selection = input_list[user_choice]

    return list_selection


def save_cv2_frame_as_png(frame, output_path: str):
    if not cv2.imwrite(output_path, frame):
        raise OSError(f"Could not save to path {output_path}")

    print(f"saved frame to path {output_path}")


def load_cv2_frame_from_png(input_path: str):
    frame = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read PNG: {input_path}")

    print(f"loaded frame from path {input_path}")
    return frame


def calculate_focal_length():
    # in this example, a tennis ball
    # D = known diameter / length / height etc
    # d = percieved diameter / etc
    # Z = known distance to object
    D = 2.7
    d = 47
    Z = 6 * 12

    focal_length = (d * Z) / D
    ic(focal_length)

    return focal_length


def calc_distance(known: float, percieved: float) -> float:
    D = known
    d = percieved
    f = FOCAL_LENGTH
    Z = (D * f) / d

    return Z


def calc_angle(
    pixel_x: float,
    focal_length_x: float = FOCAL_LENGTH,
    center_x: float = SCREEN_PX_X / 2,
) -> float:
    """Return the horizontal angle from straight ahead, in radians.

    Args:
        pixel_x: Object's horizontal coordinate in the image.
        focal_length_x: Camera's horizontal focal length in pixels.
        center_x: Camera's horizontal principal point in pixels.

    Returns:
        Signed angle: negative to the left, positive to the right.
    """
    if focal_length_x <= 0:
        raise ValueError("focal_length_x must be positive.")

    return math.atan2(pixel_x - center_x, focal_length_x) * -1


def get_current_time() -> str:

    return datetime.now().isoformat()


def save_dict_of_dicts_csv(
    data: dict,
    file_path: str,
    key_column: str = "id",
) -> None:
    """Save a dictionary of dictionaries as a CSV.

    Outer keys become values in key_column.
    Inner keys become column headers.
    Columns follow the order in which keys first appear.
    Existing files are overwritten.
    """
    if any(not isinstance(row, dict) for row in data.values()):
        raise TypeError("Each value in data must be a dictionary.")

    # Collect all inner keys, preserving first-seen order.
    columns = list(dict.fromkeys(column for row in data.values() for column in row))

    if key_column in columns:
        raise ValueError(
            f"'{key_column}' already exists as an inner key. "
            "Choose a different key_column."
        )

    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[key_column] + columns,
            restval="",
        )
        writer.writeheader()

        for outer_key, row in data.items():
            writer.writerow({key_column: outer_key, **row})


def load_hsv_bounds(file_path: str) -> tuple:
    """Load lower_hsv and upper_hsv from a text file as NumPy arrays."""
    bounds = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            # Ignore blank lines and comments, including inline comments.
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            name, separator, value = line.partition("=")
            name = name.strip()

            if not separator or name not in ("lower_hsv", "upper_hsv"):
                raise ValueError(
                    f"Line {line_number}: expected lower_hsv or upper_hsv assignment."
                )

            if name in bounds:
                raise ValueError(f"Line {line_number}: duplicate {name}.")

            try:
                values = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError) as error:
                raise ValueError(
                    f"Line {line_number}: invalid list for {name}."
                ) from error

            if (
                not isinstance(values, (list, tuple))
                or len(values) != 3
                or any(type(v) is not int for v in values)
            ):
                raise ValueError(f"{name} must contain exactly three integers.")

            if any(not 0 <= v <= 255 for v in values):
                raise ValueError(f"{name} values must be between 0 and 255.")

            bounds[name] = np.array(values, dtype=np.uint8)

    missing = {"lower_hsv", "upper_hsv"} - bounds.keys()
    if missing:
        raise ValueError(f"Missing assignment(s): {', '.join(sorted(missing))}")

    return bounds["lower_hsv"], bounds["upper_hsv"]


def get_mask(hsv, object_dict: dict):

    mask = hsv.copy()

    if ("lower" in object_dict.keys()) and ("upper" in object_dict.keys()):
        lower = object_dict["lower"]
        upper = object_dict["upper"]
    else:
        raise ValueError("object does not have attribute lower and/or upper")

    mask = cv2.inRange(hsv, lower, upper)

    # if two hsv brackets, include it in the mask
    if ("lower1" in object_dict.keys()) and ("upper1" in object_dict.keys()):
        lower2 = object_dict["lower"]
        upper2 = object_dict["upper"]

        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2, dst=mask)

    return mask


def get_clean_mask(mask: np.ndarray):
    clean_mask = mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Opening removes small specks; closing fills small gaps.
    if kernel_size > 1:
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return clean_mask


def get_best_contour(contours, object_dict: dict):

    check_min_area: bool = False
    check_min_circularity: bool = False
    min_area: int = 0
    min_circularity: int = 0

    if "min_area" in object_dict.keys():
        min_area = object_dict["min_area"]
        check_min_area = True

    if "min_circularity" in object_dict.keys():
        min_area = object_dict["min_circularity"]
        min_circularity = True

    best_contour = None
    best_area = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if check_min_area:
            if area < min_area:
                continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        if check_min_circularity:
            circularity = 4.0 * np.pi * area / perimeter**2
            if circularity < min_circularity:
                continue

        if area > best_area:
            best_area = area
            best_contour = contour

    return best_contour, best_area


def save_dicts_to_json_txt(data_list, filename=""):
    """
    Saves a list of dictionaries to a text file using JSON formatting.
    """

    if filename == "":
        filename = "obj_lists/" + get_current_time() + ".txt"

    try:
        with open(filename, "w", encoding="utf-8") as file:
            # indent=4 makes the text file pretty-printed and highly readable
            json.dump(data_list, file, indent=4)
        print(f" Successfully saved data to {filename}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")


def create_label_line(verdict: dict) -> str:
    label = ""

    name = verdict["name"]

    area_px = round(verdict["area"], 2)
    width_px = round(verdict["width"], 2)
    width_in = round(verdict["known_width_in"], 2)
    distance_in = round(verdict["distance_in"], 2)
    angle_deg = round(math.degrees(verdict["theta_rad"]))

    label = f"{name} width_px {width_px}, width_in actual: {width_in}, distance_in: {distance_in}, deg {angle_deg}"

    return label


def get_object_list_from_frame(frame: np.ndarray, save_as_file: bool = False) -> list:

    obj_list = []

    for object_dict in search_for_list:
        object_dict: dict
        path_to_hsv = object_dict["path_to_hsv"]
        obj_name = object_dict["name"]
        lower, upper = load_hsv_bounds(path_to_hsv)

        # convert to hsv for processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create a mask using the hsv values
        mask = cv2.inRange(hsv, lower, upper)

        # get the contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # evaluate the best contour
        best_contour, best_area = get_best_contour(contours, object_dict)

        # if there is no contour, the object has not been detected
        # so just go to the next object
        if best_contour is None:
            continue

        # if there IS a contour, we want to know how large it is and where it is
        verdict = {}

        # copy over the useful stuff
        verdict["name"] = obj_name
        verdict["shape"] = object_dict["shape"]

        verdict["color"] = object_dict["color"]
        verdict["known_width_in"] = object_dict["known_width_in"]

        # analyze either as a circle or rectangle
        if object_dict["shape"] == CIRCLE:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            verdict["center_x"] = x
            verdict["center_y"] = y

            verdict["radius"] = radius
            verdict["width"] = radius * 2

            verdict["area"] = np.pi * radius * radius

        elif object_dict["shape"] == RECT:
            x, y, width, height = cv2.boundingRect(best_contour)
            verdict["center_x"] = x
            verdict["center_y"] = y

            verdict["width"] = width
            verdict["height"] = height

            verdict["radius"] = int(width / 2)

            verdict["area"] = width * height

        # convert from pixel width to inches
        verdict["distance_in"] = calc_distance(
            known=object_dict["known_width_in"],
            percieved=verdict["width"],
        )

        # convert the horizontal pixel distance from center to angle
        verdict["theta_rad"] = calc_angle(pixel_x=verdict["center_x"])

        verdict["label"] = create_label_line(verdict)

        obj_list.append(verdict)

    if save_as_file:
        save_dicts_to_json_txt(obj_list)

    return obj_list


def draw_object_list_to_frame(
    frame: np.ndarray, obj_list: list, draw_arrows: bool = False
):
    WHITE = (255, 255, 255)
    THICKNESS = 1

    write_list = []

    for object_dict in obj_list:
        object_dict: dict

        shape_center_x = int(object_dict["center_x"])
        shape_center_y = int(object_dict["center_y"])
        width = int(object_dict["width"])
        color = object_dict["color"]

        label = object_dict["label"]

        write_list.append(label)

        shape_center = (shape_center_x, shape_center_y)

        if object_dict["shape"] == CIRCLE:
            radius = object_dict["radius"]
            frame = cv2.circle(
                img=frame,
                radius=int(radius),
                center=shape_center,
                color=color,
                thickness=THICKNESS,
            )

        elif object_dict["shape"] == RECT:

            height = int(object_dict["height"])

            frame = cv2.rectangle(
                img=frame,
                pt1=shape_center,
                pt2=(shape_center_x + width, shape_center_y + height),
                color=color,
                thickness=THICKNESS,
            )

        if draw_arrows:
            # draw an arrow from the center of the object to the center of the frame
            screen_center_x = int(SCREEN_PX_X / 2)
            screen_center_y = int(SCREEN_PX_Y / 2)
            screen_center = (screen_center_x, screen_center_y)

            delta_x_px = abs(screen_center_x - shape_center_x)
            delta_y_px = abs(screen_center_y - shape_center_y)

            arrow_length = math.sqrt(pow(delta_x_px, 2) + pow(delta_y_px, 2))
            arrow_length = round(arrow_length)

            # # draw to center
            # frame = draw_labeled_arrow(
            #     frame= frame,
            #     color= BLACK,
            #     point1=shape_center,
            #     point2= screen_center,
            #     label=f"px: {arrow_length}"
            # )

            # draw along x axis
            frame = draw_labeled_arrow(
                frame=frame,
                color=BLUE,
                point1=(shape_center_x, shape_center_y),
                point2=(screen_center_x, shape_center_y),
                label=f"px: {delta_x_px}",
            )

            # # draw along y axis
            # frame = draw_labeled_arrow(
            #     frame= frame,
            #     color= RED,
            #     point1=(screen_center_x, shape_center_y),
            #     point2= (screen_center_x, screen_center_y),
            #     label=f"px: {delta_y_px}"
            # )

            # draw vertical arrow showing screen center
            frame = cv2.line(
                frame,
                pt1=(screen_center_x, 0),
                pt2=(screen_center_x, SCREEN_PX_Y),
                color=WHITE,
            )

    frame = draw_list_on_frame(frame, write_list)

    return frame


def draw_list_on_frame(
    frame: np.ndarray,
    items: list,
    position: tuple = (10, 10),
    color: tuple = (255, 255, 255),
    font_scale: float = 0.6,
    thickness: int = 1,
    line_gap: int = 6,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
    """Draw list items on separate lines near the top-left of a frame.

    position: (x, y) of the text block's top-left corner, in pixels.
    line_gap: Extra vertical spacing between lines.

    Modifies frame in place and returns it.
    Lines that do not fit vertically are omitted.
    """
    if font_scale <= 0 or thickness <= 0:
        raise ValueError("Font scale and thickness must be positive.")
    if line_gap < 0:
        raise ValueError("Line gap must be nonnegative.")

    if not items:
        return frame

    texts = [str(item) for item in items]
    x, top = map(int, position)

    # Use uniform spacing based on the tallest text in the list.
    sizes = [
        cv2.getTextSize(text or "Ag", font, font_scale, thickness) for text in texts
    ]
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes) + thickness
    line_height = text_height + baseline + line_gap

    text_y = top + text_height

    for text in texts:
        if text_y + baseline >= frame.shape[0]:
            break

        cv2.putText(
            frame,
            text,
            (x, text_y),
            font,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        text_y += line_height

    return frame


def draw_labeled_arrow(
    frame: np.ndarray,
    point1: tuple[float, float],
    point2: tuple[float, float],
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    label_offset: tuple[int, int] = (0, -10),
    tip_length: float = 0.15,
) -> np.ndarray:
    """
    Draw an arrow from point1 to point2 with a label near its midpoint.

    Args:
        frame: OpenCV BGR image. Modified in place.
        point1: Arrow start (pixel_x, pixel_y).
        point2: Arrow tip (pixel_x, pixel_y).
        label: Text displayed near the arrow's midpoint.
        color: Arrow and text color in BGR order.
        thickness: Arrow stroke width in pixels.
        font_scale: Text size multiplier.
        label_offset: (dx, dy) offset of the label's center from the
            arrow midpoint. Negative dy moves the label upward.
        tip_length: Arrowhead length as a fraction of the arrow length.

    Returns:
        The modified frame.
    """
    if thickness < 1 or font_scale <= 0:
        raise ValueError("thickness and font_scale must be positive")
    if not 0 < tip_length <= 1:
        raise ValueError("tip_length must be in (0, 1]")

    p1 = tuple(int(round(v)) for v in point1)
    p2 = tuple(int(round(v)) for v in point2)

    cv2.arrowedLine(
        frame,
        p1,
        p2,
        color,
        thickness=thickness,
        line_type=cv2.LINE_AA,
        tipLength=tip_length,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    text = str(label)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, text_thickness
    )

    center_x = (p1[0] + p2[0]) / 2 + label_offset[0]
    center_y = (p1[1] + p2[1]) / 2 + label_offset[1]

    text_origin = (
        int(round(center_x - text_width / 2)),
        int(round(center_y + (text_height - baseline) / 2)),
    )

    cv2.putText(
        frame,
        text,
        text_origin,
        font,
        font_scale,
        color,
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )

    return frame


def draw_centered_rectangle(
    frame: np.ndarray,
    rectangle: Rectangle,
    color: tuple = WHITE,
    thickness: int = -1,
) -> np.ndarray:
    """Draw a centered rectangle in place and return the frame.

    The supplied rectangle's x and y are ignored, without modifying the object.
    Use thickness=-1 for a filled rectangle or a positive value for an outline.
    """
    frame_height, frame_width = frame.shape[:2]

    if not (0 < rectangle.width <= frame_width):
        raise ValueError("Rectangle width must be positive and fit in the frame.")
    if not (0 < rectangle.height <= frame_height):
        raise ValueError("Rectangle height must be positive and fit in the frame.")

    x = (frame_width - rectangle.width) // 2
    y = (frame_height - rectangle.height) // 2

    cv2.rectangle(
        frame,
        (x, y, rectangle.width, rectangle.height),
        color,
        thickness,
    )

    return frame


def draw_vert_centered_rectangle(
    frame: np.ndarray,
    rectangle: Rectangle,
    y: int = 0,
    color: tuple = WHITE,
    thickness: int = -1,
) -> np.ndarray:
    """Draw a horizontally centered rectangle with a vertical offset.

    y:
        0  = vertically centered
        <0 = above center
        >0 = below center

    Modifies frame in place and returns it.
    The rectangle object's x and y fields are ignored.
    """
    frame_height, frame_width = frame.shape[:2]

    if not (0 < rectangle.width <= frame_width):
        raise ValueError("Rectangle width must be positive and fit in the frame.")
    if not (0 < rectangle.height <= frame_height):
        raise ValueError("Rectangle height must be positive and fit in the frame.")

    x_position = (frame_width - rectangle.width) // 2
    y_position = (frame_height - rectangle.height) // 2 + y

    if y_position < 0 or y_position + rectangle.height > frame_height:
        raise ValueError("Vertical offset places the rectangle outside the frame.")

    cv2.rectangle(
        frame,
        (x_position, y_position, rectangle.width, rectangle.height),
        color,
        thickness,
    )

    return frame


def draw_centered_octagon(
    frame: np.ndarray,
    rectangle: Rectangle,
    chamfer: int,
    y: int = 0,
    color: tuple = WHITE,
    thickness: int = -1,
) -> np.ndarray:
    """Draw a rectangle with four 45-degree chamfered corners.

    chamfer: Distance along each edge from the original corner, in pixels.
    y: Vertical offset from frame center; negative moves up, positive moves down.
    thickness: -1 for filled, or a positive integer for an outline.

    Modifies frame in place and returns it.
    The rectangle object's x and y fields are ignored.
    """
    frame_height, frame_width = frame.shape[:2]
    width = rectangle.width
    height = rectangle.height

    if not (0 < width <= frame_width):
        raise ValueError("Rectangle width must be positive and fit in the frame.")
    if not (0 < height <= frame_height):
        raise ValueError("Rectangle height must be positive and fit in the frame.")

    if not isinstance(chamfer, (int, np.integer)):
        raise TypeError("Chamfer must be an integer number of pixels.")

    # Keep all eight sides nonzero in length.
    if not (0 < 2 * chamfer < min(width - 1, height - 1)):
        raise ValueError(
            "Chamfer must be positive and less than half the shortest "
            "corner-to-corner span."
        )

    if thickness != -1 and thickness <= 0:
        raise ValueError("Thickness must be -1 (filled) or a positive integer.")

    left = (frame_width - width) // 2
    top = (frame_height - height) // 2 + y
    right = left + width - 1
    bottom = top + height - 1

    if top < 0 or bottom >= frame_height:
        raise ValueError("Vertical offset places the octagon outside the frame.")

    # Eight vertices ordered clockwise.
    points = np.array(
        [
            (left + chamfer, top),
            (right - chamfer, top),
            (right, top + chamfer),
            (right, bottom - chamfer),
            (right - chamfer, bottom),
            (left + chamfer, bottom),
            (left, bottom - chamfer),
            (left, top + chamfer),
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    if thickness == -1:
        cv2.fillPoly(frame, [points], color)
    else:
        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=color,
            thickness=thickness,
        )

    return frame


def draw_empty_field(
    is_show_image: bool = False,
    draw_centerpoints: bool = False,
    just_black: bool = False,
) -> np.ndarray:

    frame: np.ndarray = np.zeros(
        (total_window_px_y, total_window_px_x, 3), dtype=np.uint8
    )

    # start with a background
    background = Rectangle(width=total_window_px_x, height=total_window_px_y)

    # then the 2'' perim 2'' from the edges

    # this has been updated to have a chamber 1' from the edges
    white_px_x = convert_to_px(perim_tape_thickness_in * 2 + playing_field_in_x)
    white_px_y = convert_to_px(perim_tape_thickness_in * 2 + playing_field_in_y)
    white_perimiter = Rectangle(width=white_px_x, height=white_px_y)

    # then the playing field
    playing_field_px_x = convert_to_px(playing_field_in_x)
    playing_field_px_y = convert_to_px(playing_field_in_y)
    playing_field = Rectangle(
        width=playing_field_px_x,
        height=playing_field_px_y,
    )

    goal_net_px_x = convert_to_px(goal_net_in_x)
    goal_net_px_y = convert_to_px(goal_net_in_y)

    goal_net = Rectangle(width=goal_net_px_x, height=goal_net_px_y)

    goalbox_px_x = convert_to_px(goalbox_in_x)
    goalbox_px_y = convert_to_px(goalbox_in_y)
    goalbox = Rectangle(width=goalbox_px_x, height=goalbox_px_y)

    # BEGIN DRAWING TO SCREEN
    draw_centered_rectangle(frame, background, BLACK)

    if just_black:
        cv2.imwrite("black_field.png", frame)
        return frame

    draw_centered_octagon(
        frame,
        white_perimiter,
        convert_to_px(12),
        color=WHITE,
    )

    draw_centered_octagon(
        frame,
        playing_field,
        convert_to_px(12),
        color=GRAY,
    )

    # draw the goal nets
    goal_net_px_y_half = goal_net_px_y / 2
    goal_net_offset = int(total_window_px_y / 2 - goal_net_px_y_half)
    draw_vert_centered_rectangle(
        frame,
        goal_net,
        goal_net_offset,
        YELLOW,
    )

    draw_vert_centered_rectangle(
        frame,
        goal_net,
        -goal_net_offset,
        CYAN,
    )

    # draw the goal boxes
    goalbox_offset = int(goal_net_offset - goal_net_px_y - goal_net_px_y_half)
    draw_vert_centered_rectangle(
        frame,
        goalbox,
        goalbox_offset,
        LIGHT_GRAY,
    )

    draw_vert_centered_rectangle(
        frame,
        goalbox,
        -goalbox_offset,
        LIGHT_GRAY,
    )

    if draw_centerpoints:
        centerpoint = Rectangle(
            width=30,
            height=30,
        )

        # draw a box right at the center of the field
        draw_centered_rectangle(frame, centerpoint)

        # and then at the center of the light gray goalboxes
        draw_vert_centered_rectangle(
            frame,
            centerpoint,
            goalbox_offset,
        )

        draw_vert_centered_rectangle(
            frame,
            centerpoint,
            -goalbox_offset,
        )

    if is_show_image:
        cv2.imshow("empty_field", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite("empty_field.png", frame)
    return frame


def show_frames_side_by_side(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    target_height: int | None = None,
    window_name: str = "Side-by-Side Frames",
) -> np.ndarray:
    """Display two BGR frames side by side at the same height.

    If target_height is None, use the right frame's original height.
    Each frame's aspect ratio is preserved.
    """
    for name, frame in (("frame_left", frame_left), ("frame_right", frame_right)):
        if frame is None or frame.size == 0:
            raise ValueError(f"{name} must be a non-empty image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{name} must be a three-channel BGR image.")

    if frame_left.dtype != frame_right.dtype:
        raise ValueError("Both frames must have the same data type.")

    if target_height is None:
        target_height = frame_right.shape[0]
    if target_height <= 0:
        raise ValueError("target_height must be positive.")

    def resize_to_height(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == target_height:
            return frame

        scale = target_height / height
        new_width = max(1, round(width * scale))
        return cv2.resize(
            frame,
            (new_width, target_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

    combined = np.hstack((resize_to_height(frame_left), resize_to_height(frame_right)))
    # cv2.imshow(window_name, combined)
    return combined


def get_side_and_top_down(frame_camera: np.ndarray, frame_field: np.ndarray):

    frame_unchanged = frame_camera.copy()
    field_unchanged = frame_field.copy()

    obj_list: list = get_object_list_from_frame(frame_camera, save_as_file=False)
    frame_annotated: np.ndarray = draw_object_list_to_frame(
        frame_unchanged, obj_list, draw_arrows=True
    )

    field_annotated: np.ndarray = draw_object_list_to_field(
        at_goalbox_center=True,
        field_frame=field_unchanged,
        object_list=obj_list,
        draw_arrows=False,
    )

    combined_frame = show_frames_side_by_side(frame_annotated, field_annotated)

    return combined_frame, obj_list


def to_base_frame(
    ref_x: float,
    ref_y: float,
    ref_theta: float,
    distance: float,
    angle: float,
) -> tuple[float, float, float]:
    """
    Convert a distance and bearing in a reference frame to the base frame.

    Args:
        ref_x, ref_y: Reference-frame origin expressed in the base frame.
        ref_theta: Reference-frame orientation in the base frame, in radians.
        distance: Distance from the reference-frame origin to the point.
        angle: Bearing to the point from the reference frame's positive x-axis,
            in radians. Positive angles are counterclockwise.

    Returns:
        (x, y, theta) in the base frame. theta is the bearing of the
        displacement expressed in the base frame, wrapped to [-pi, pi).

    Note:
        theta is not an independently measured orientation of the point.
    """
    theta = ref_theta + angle

    x = ref_x + distance * math.cos(theta)
    y = ref_y + distance * math.sin(theta)

    theta = (theta + math.pi) % (2 * math.pi) - math.pi

    return x, y, theta


def convert_in_point_to_screen(point) -> tuple[int, int]:

    # the screen measures down and right from the top left,
    # but we measure the field +up and +right from the center

    # an input of 0,0 should result in window_x/2, window_y /2
    inch_x, inch_y = point

    inch_x = int(inch_x)
    inch_y = int(inch_y)

    px_x = convert_to_px(inch_x)
    px_y = convert_to_px(inch_y)

    # left means center minus the input
    rel_px_x = field_center_px_x + px_x
    rel_px_x = int(rel_px_x)

    # up means center MINUS input
    rel_px_y = field_center_px_y - px_y
    rel_px_y = int(rel_px_y)

    return (rel_px_x, rel_px_y)


def convert_to_px(inches: float) -> int:
    """Convert inches to pixels using the global scale factor."""
    return int(round(inches * PIXELS_PER_INCH))


goalbox_center_top_px = convert_in_point_to_screen(point=goalbox_center_top_in)
goalbox_center_bot_px = convert_in_point_to_screen(point=goalbox_center_bot_in)
center_field_px = convert_in_point_to_screen(point=(0, 0))


def load_dicts_from_txt(filename="output.txt"):
    """
    Loads a text file formatted in JSON and returns it as a list of dictionaries.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data_list = json.load(file)
        print(f"Successfully loaded data from {filename}")
        return data_list
    except FileNotFoundError:
        print(f"❌ Error: The file '{filename}' does not exist.")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{filename}' is not valid JSON text.")
        return []
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []


def convert_to_top_down(verdict: dict, robot: dict):
    """
    from the camera's understanding of pixel center and apparent width, convert to angle and distance

    example of a verdict:
    {
        "name": "ball_yellow",
        "shape": "RECT",
        "center_x": 136,
        "center_y": 253,
        "width": 47,
        "height": 33,
        "area": 1551
        "known_width_in" : 2.7,
    }
    """

    # read the status of the robot
    robot_in_x, robot_in_y = robot["center_in"]

    # change the coordinate system from the robot reference to the global reference
    verdict["center_in"] = convert_robot_dis_and_angle_to_global(
        angle_rad_robot=robot["theta_rad"],
        angle_rad_to_obj=verdict["theta_rad"],
        robot_x=robot_in_x,
        robot_y=robot_in_y,
        distance=verdict["distance_in"],
    )

    # convert the inches measurement to screen pixels
    verdict["center_px"] = convert_in_point_to_screen(verdict["center_in"])

    return verdict


def convert_robot_dis_and_angle_to_global(
    distance: float,
    angle_rad_to_obj: float,
    robot_x: float,
    robot_y: float,
    angle_rad_robot,
):

    object_x = robot_x + distance * math.cos(angle_rad_to_obj + angle_rad_robot)
    object_y = robot_y + distance * math.sin(angle_rad_to_obj + angle_rad_robot)

    return object_x, object_y


def draw_labeled_circle(
    frame: np.ndarray,
    center: tuple,
    radius: int,
    label: str,
    circle_color: tuple = (255, 255, 255),
    text_color: tuple = (255, 255, 255),
    circle_thickness: int = -1,
    font_scale: float = 0.6,
    text_thickness: int = 1,
    label_gap: int = 10,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
    """Draw a circle with a horizontally centered label above it.

    center: Circle center as (x, y), in pixels.
    radius: Circle radius in pixels.
    circle_thickness: -1 for filled, or a positive integer for an outline.
    label_gap: Spacing between the label's bottom and the circle's top.

    Modifies frame in place and returns it.
    """
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    if label_gap < 0:
        raise ValueError("Label gap must be nonnegative.")
    if font_scale <= 0 or text_thickness <= 0:
        raise ValueError("Font scale and text thickness must be positive.")
    if circle_thickness != -1 and circle_thickness <= 0:
        raise ValueError("Circle thickness must be -1 or a positive integer.")

    cx, cy = (int(round(value)) for value in center)

    cv2.circle(
        frame,
        (cx, cy),
        radius,
        circle_color,
        thickness=circle_thickness,
        lineType=cv2.LINE_AA,
    )

    if label:
        (text_width, _), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )

        # Allow for an outline extending beyond the nominal circle radius.
        outline_padding = (circle_thickness + 1) // 2 if circle_thickness > 0 else 0

        text_x = int(round(cx - text_width / 2))
        text_y = cy - radius - outline_padding - label_gap - baseline - text_thickness

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return frame


def draw_transparent_circle(
    frame: np.ndarray,
    center: tuple,
    radius: int,
    color: tuple = (255, 255, 255),
    opacity: float = 0.8,
    border_thickness: int = 2,
) -> np.ndarray:
    """Draw a translucent filled circle with an opaque black border.

    center: Circle center as (x, y), in pixels.
    radius: Circle radius in pixels.
    color: Fill color in BGR order.
    opacity: Fill opacity from 0.0 to 1.0.
    border_thickness: Black outline thickness in pixels.

    Modifies frame in place and returns it.
    """
    if not isinstance(radius, (int, np.integer)) or radius <= 0:
        raise ValueError("Radius must be a positive integer.")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Opacity must be between 0.0 and 1.0.")
    if not isinstance(border_thickness, (int, np.integer)) or border_thickness <= 0:
        raise ValueError("Border thickness must be a positive integer.")

    center = tuple(int(round(value)) for value in center)

    # Copy the existing frame so pixels outside the circle remain unchanged.
    overlay = frame.copy()

    cv2.circle(
        overlay,
        center,
        radius,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        opacity,
        frame,
        1.0 - opacity,
        0,
        dst=frame,
    )

    # Draw the border after blending so it remains opaque.
    cv2.circle(
        frame,
        center,
        radius,
        (0, 0, 0),
        thickness=border_thickness,
        lineType=cv2.LINE_AA,
    )

    return frame


def draw_to_field(frame: Optional[np.ndarray], list_of_objects: list) -> np.ndarray:
    # from the generated field, draw a ball, robots, etc
    # if frame is None (e.g. cv2.imread failed), create an empty field
    if frame is None:
        frame = draw_empty_field()

    for obj in list_of_objects:
        if isinstance(obj, dict) == False:
            ValueError("object is not a dict!")
        else:
            obj: dict
            ic("drawing to field", obj)

        frame = draw_transparent_circle(
            frame,
            center=obj["center_px"],
            radius=obj["radius"],
            color=obj["color"],
            opacity=1,
        )

    return frame


def get_most_recent_file(directory: str | Path) -> Path | None:
    """
    Return the absolute path to the most recently modified file.

    Args:
        directory: Directory to search (non-recursively).

    Returns:
        A Path object, or None if the directory contains no files.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    directory = Path(directory).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    return max(
        (path for path in directory.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        default=None,
    )


def draw_line(
    frame: np.ndarray,
    point1: tuple[float, float],
    point2: tuple[float, float],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a line between two (pixel_x, pixel_y) points.

    Args:
        frame: OpenCV BGR image. Modified in place.
        point1: Starting point (pixel_x, pixel_y).
        point2: Ending point (pixel_x, pixel_y).
        color: Line color in BGR order. Defaults to green.
        thickness: Line width in pixels; must be positive.

    Returns:
        The modified frame.
    """
    if thickness < 1:
        raise ValueError("thickness must be at least 1")

    p1 = (int(round(point1[0])), int(round(point1[1])))
    p2 = (int(round(point2[0])), int(round(point2[1])))

    cv2.line(frame, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    return frame


def draw_circle_with_arrow(
    frame: np.ndarray,
    point: tuple[float, float],
    radius: int = 30,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    angle: float = 9999999,
) -> np.ndarray:
    """
    Draw a circle and an arrow extending from its center.

    Args:
        frame: OpenCV BGR image. Modified in place.
        point: Circle center (pixel_x, pixel_y).
        angle: Direction in radians, counterclockwise from the right:
            0 = right, pi/2 = up, pi = left, -pi/2 = down.
        radius: Circle radius in pixels.
        arrow_length: Distance from the center to the arrow tip in pixels.
        color: Circle and arrow color in BGR order.
        thickness: Stroke width in pixels.

    Returns:
        The modified frame.
    """

    arrow_length = int(radius * 2)
    radius = int(radius)

    if radius <= 0 or arrow_length <= 0 or thickness < 1:
        radius = 1
        arrow_length = 1
        thickness = 1
        # FIXME handle this more elegantly
        # raise ValueError("radius, arrow_length, and thickness must be positive")

    center = (int(round(point[0])), int(round(point[1])))

    # Subtract the y component so positive angles point upward.
    endpoint = (
        int(round(center[0] + arrow_length * math.cos(angle))),
        int(round(center[1] - arrow_length * math.sin(angle))),
    )

    cv2.circle(
        frame,
        center,
        radius,
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # only draw the arrow if the angle is different than the default
    if angle != 9999999:
        cv2.arrowedLine(
            frame,
            center,
            endpoint,
            color,
            thickness=thickness,
            line_type=cv2.LINE_AA,
            tipLength=0.25,
        )

    return frame


def draw_arrows_from_object_to_robot(frame, robot: dict, obj: dict):

    shape_center = obj["center_px"]
    shape_center_x, shape_center_y = shape_center
    robot_center = robot["center_px"]

    # draw an arrow from the center of the object to the center of the frame
    screen_center_x = int(SCREEN_PX_X / 2)
    screen_center_y = int(SCREEN_PX_Y / 2)

    delta_x_px = abs(screen_center_x - shape_center_x)
    delta_y_px = abs(screen_center_y - shape_center_y)

    arrow_length = math.sqrt(pow(delta_x_px, 2) + pow(delta_y_px, 2))
    arrow_length = round(arrow_length)

    # draw to center
    frame = draw_labeled_arrow(
        frame=frame,
        color=BLACK,
        point1=shape_center,
        point2=robot_center,
        label=f"px: {arrow_length}",
    )

    # draw along x axis
    frame = draw_labeled_arrow(
        frame=frame,
        color=BLUE,
        point1=(shape_center_x, shape_center_y),
        point2=(screen_center_x, shape_center_y),
        label=f"px: {delta_x_px}",
    )

    # draw along y axis
    frame = draw_labeled_arrow(
        frame=frame,
        color=RED,
        point1=(screen_center_x, shape_center_y),
        point2=(screen_center_x, screen_center_y),
        label=f"px: {delta_y_px}",
    )

    # draw vertical arrow showing screen center
    frame = cv2.line(
        frame,
        pt1=(screen_center_x, 0),
        pt2=(screen_center_x, SCREEN_PX_Y),
        color=WHITE,
    )


def draw_object_list_to_field(
    field_frame: np.ndarray,
    object_list: list,
    at_goalbox_center: bool = True,
    draw_arrows: bool = False,
) -> np.ndarray:

    robot = {}

    if at_goalbox_center:
        robot_field_goal = {
            "name": "our_bot",
            "center_in": goalbox_center_bot_in,
            "center_px": convert_in_point_to_screen((goalbox_center_bot_in)),
            "radius": RADIUS_ROBOT,
            "color": CYAN,
            "theta_rad": math.radians(90),  # up screen
        }

        robot = robot_field_goal

    # draw robot
    field_frame = draw_circle_with_arrow(
        angle=robot["theta_rad"],
        color=robot["color"],
        frame=field_frame,
        point=robot["center_px"],
        radius=robot["radius"],
    )

    # draw objects
    for obj in object_list:
        ojb = convert_to_top_down(obj, robot)

        # draw object
        field_frame = draw_circle_with_arrow(
            angle=ojb["theta_rad"],
            color=ojb["color"],
            frame=field_frame,
            point=ojb["center_px"],
            radius=obj["radius"],
        )

        if draw_arrows:
            draw_arrows_from_object_to_robot(frame=field_frame, robot=robot, obj=obj)

    return field_frame


def read_video_frames(video_path: str) -> Iterator[np.ndarray]:
    """Yield each video frame as a BGR NumPy array."""
    capture = cv2.VideoCapture(video_path)

    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        while True:
            success, frame = capture.read()

            if not success:
                break

            yield frame
    finally:
        capture.release()


def get_frame_and_field(frame: np.ndarray) -> np.ndarray:

    if USE_DRAWN_FIELD:
        frame_field: np.ndarray = load_image(f"{FIELD_PNGS}/empty_field.png")
    else:
        frame_field: np.ndarray = load_image(f"{FIELD_PNGS}/black_field.png")

    combined_frame, object_list = get_side_and_top_down(
        frame,
        frame_field,
    )

    ic(object_list)

    return combined_frame


def analyze_photo_or_video(file_path: str) -> None:
    if file_path.endswith(".mp4"):
        frames = read_video_frames(file_path)
    elif file_path.endswith(".png") or file_path.endswith(".jpg"):
        frame = load_image(file_path)
        combined_frame = get_frame_and_field(frame)
        cv2.imshow("frame and field", combined_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .mp4, .png, or .jpg file."
        )

    try:
        for frame_index, frame in enumerate(frames):

            combined_frame = get_frame_and_field(frame)

            cv2.imshow("frame and field", combined_frame)

            # Press 'q' to stop.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        frames.close()
        cv2.destroyAllWindows()