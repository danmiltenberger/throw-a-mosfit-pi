### all the functions and constants that determine what is pulled from each frame
from settings_run import CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y, kernel_size, MIN_AREA, MIN_CIRCLARITY, search_for_list, CIRCLE, KNOWN_WIDTH_IN, RECT, search_for_dict, FOCAL_LENGTH
from tam_library_classes import OurRobot, Detection, Point
from tam_library_field import convert_to_px, field_center_px_y, field_center_px_x
import ast
import numpy as np
import cv2
import math

# screen consts
SCREEN_CENTER_X = int(CAMERA_CAPTURE_X /2)
SCREEN_CENTER_Y = int(CAMERA_CAPTURE_Y /2)


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


def is_valid_contour(contour, area, object_dict: dict) -> bool:
    """
    check if the contour is valid based on area, circularity, etc
    """

    if MIN_AREA in object_dict.keys():
        if area < object_dict[MIN_AREA]:
            return False

    elif MIN_CIRCLARITY in object_dict.keys():
        # Find contour properties
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter**2)
        else:
            circularity = 0

        if circularity < object_dict[MIN_CIRCLARITY]:
            return False
    else:
        return True


def get_detections_from_frame(frame: np.ndarray, bot: OurRobot) -> list[Detection]:
    "from the input frame, search for the objects as specified in the run_settings.py search_list"

    detect_list = []
    frame = frame.copy()

    for object_dict in search_for_list:

        # load the necessary info from the reference list
        path_to_hsv = object_dict["path_to_hsv"]
        obj_name = object_dict["name"]
        lower, upper = load_hsv_bounds(path_to_hsv)

        # convert to hsv for processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create a mask using the hsv values
        mask = cv2.inRange(hsv, lower, upper)

        # get the contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get the best contour
        best_contour, best_area = get_best_contour(contours, object_dict=object_dict)

        # if there is no contour, this object has not been detected, go to next object
        if best_contour is None:
            continue

        # if there IS a contour, we want to know how large it is and where it is
        best_contour, best_area = get_best_contour(contours, object_dict)

        # if the contour is no good, skip it
        contour_verdict = is_valid_contour(
            contour=best_contour, area=best_area, object_dict=object_dict
        )
        if contour_verdict == False:
            continue

        # okay, so at this point we have decided that the contour is useful. Let's package and ship it!
        this_detect = Detection()
        this_detect.name = object_dict["name"]

        # analyze either as a circle or rectangle
        if object_dict["shape"] == CIRCLE:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center_point = Point(x, y)
            this_detect.screen_center_px = center_point
            this_detect.width_px = int(radius * 2)

            this_detect.area = int(np.pi * radius * radius)

        elif object_dict["shape"] == RECT:
            left_x, top_y, width, height = cv2.boundingRect(best_contour)

            x_center = int(left_x + width / 2)
            y_center = int(top_y + height / 2)
            center_point = Point(x=x_center, y=y_center)
            this_detect.screen_center_px = center_point
            this_detect.width_px = width
            this_detect.area = int(width * height)

        # determine the distance and angle relative to the camera
        this_detect.range = calc_distance_from_bot(this_detect)
        this_detect.azimuth_rad = calc_angle_from_screen_center(
            this_detect.screen_center_px
        )

        # convert to the field xy system in inches
        this_detect.field_pos_in = convert_pos_and_ang_to_robot_in(this_detect, bot)

        # convert the field inches to the window pixels
        this_detect.field_pos_px = convert_to_field_pixels_global(
            this_detect.field_pos_in
        )

        detect_list.append(this_detect)

    return detect_list


def convert_pos_and_ang_to_robot_in(this_detect: Detection, robot: OurRobot) -> Point:

    # get the detection's global position in inches
    object_x = robot.field_pos_in.x + this_detect.range * math.cos(
        this_detect.azimuth_rad + robot.azimuth_rad
    )

    object_y = robot.field_pos_in.y + this_detect.range * math.sin(
        this_detect.azimuth_rad + robot.azimuth_rad
    )

    return Point(object_x, object_y)


def calc_distance_from_bot(this_detect: Detection) -> float:
    known_width = search_for_dict[this_detect.name][KNOWN_WIDTH_IN]
    percieved_width = this_detect.width_px
    Z = (known_width * FOCAL_LENGTH) / percieved_width
    return Z


def calc_angle_from_screen_center(point: Point) -> float:
    """Return the horizontal angle from straight ahead, in radians.

    Args:
        pixel_x: Object's horizontal coordinate in the image.
        focal_length_x: Camera's horizontal focal length in pixels.
        center_x: Camera's horizontal principal point in pixels.

    Returns:
        Signed angle: negative to the left, positive to the right.
    """
    pixel_x = point.x
    delta_x = pixel_x - SCREEN_CENTER_X

    return -math.atan2(delta_x, FOCAL_LENGTH)


def convert_to_field_pixels_global(field_point_inches: Point) -> Point:
    # the screen measures down and right from the top left,
    # but we measure the field +up and +right from the center

    # an input of 0,0 should result in window_x/2, window_y /2

    px_x = convert_to_px(field_point_inches.x)
    px_y = convert_to_px(field_point_inches.y)

    # left means center minus the input
    rel_px_x = field_center_px_x + px_x
    rel_px_x = int(rel_px_x)

    # up means center MINUS input
    rel_px_y = field_center_px_y - px_y
    rel_px_y = int(rel_px_y)

    pixel_point = Point(rel_px_x, rel_px_y)

    return pixel_point





