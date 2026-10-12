import numpy as np
import cv2
import ast
import numpy as np
import csv
from datetime import datetime
from icecream import ic
import json
from opencv_colors import *
import math

kernel_size = 3

CIRCLE = "CIRCLE"
RECT = "RECT"

ball_red: dict = {
    "name" : "red_ball",
    "shape" : CIRCLE,
    "path_to_hsv" : "hsv_bounds/ball_red.txt",
    "known_width_in" : 7,
    "color" : RED,
}

ball_yellow : dict = {
    "name" : "ball_yellow",
    "shape" : CIRCLE,
    "path_to_hsv" : "hsv_bounds/ball_yellow.txt",
    "known_width_in" : 2.7,
    "color" : YELLOW,
}

hat_cyan : dict = {
    "name" : "hat_cyan",
    "shape" : RECT,
    "path_to_hsv" : "hsv_bounds/hat_cyan.txt",
    "known_width_in" : 8,     # FIXME
    "color" : CYAN,
}

hat_yellow : dict = {
    "name" : "hat_yellow",
    "shape" : RECT,
    "path_to_hsv" : "hsv_bounds/hat_yellow.txt", 
    "known_width_in" : 8,     # FIXME
    "color" : YELLOW,
}

search_for_list = [
    ball_red,
    ball_yellow,
    hat_cyan,
    #hat_yellow,
]


FOCAL_LENGTH = 1253.333     # recalc this when able
SCREEN_PX_X = 640
SCREEN_PX_Y = 480



def calculate_focal_length():
    # in this example, a tennis ball
    # D = known diameter / length / height etc
    # d = percieved diameter / etc
    # Z = known distance to object
    D = 2.7
    d = 47
    Z = 6*12

    focal_length = (d*Z)/D
    ic(focal_length)


    return focal_length


def calc_distance(known: float, percieved: float) -> float:
    D = known
    d = percieved
    f = FOCAL_LENGTH
    Z = (D*f) / d

    return Z


def calc_angle(center_px_x, screen_px_x)-> float:
    pixels_per_degree = 27.64
    degrees_per_pixel = 1/ pixels_per_degree

    delta_x_px = screen_px_x - center_px_x
    degrees = delta_x_px * degrees_per_pixel

    return degrees


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
    columns = list(
        dict.fromkeys(
            column
            for row in data.values()
            for column in row
        )
    )

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

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        ))

    # Opening removes small specks; closing fills small gaps.
    if kernel_size > 1:
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return clean_mask


def get_best_contour(contours, object_dict: dict):

    check_min_area : bool = False
    check_min_circularity: bool = False
    min_area : int = 0
    min_circularity : int = 0

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

    area_px = round(verdict["area"])
    width_px = round(verdict["width"])
    width_in = round(verdict["known_width_in"])
    distance_in = round(verdict["distance_in"])
    angle_deg = round(math.degrees(verdict["theta_rad"]))

    label = f"{name} width_px {width_px}, width_in actual: {width_in}, distance_in: {distance_in}, deg {angle_deg}"

    return label


def get_object_list_from_frame(frame: np.ndarray, save_as_file: bool = False) -> list:

    obj_list = []

    for object_dict in search_for_list:
        object_dict : dict
        path_to_hsv = object_dict["path_to_hsv"]
        obj_name = object_dict["name"]
        lower, upper = load_hsv_bounds(path_to_hsv)
        

        # convert to hsv for processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create a mask using the hsv values
        mask = cv2.inRange(hsv, lower, upper)

        # get the contours from the mask
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)

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
            (x,y), radius = cv2.minEnclosingCircle(best_contour)
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
            known= object_dict["known_width_in"],
            percieved= verdict["width"],
        )

        # convert the horizontal pixel distance from center to angle
        verdict["theta_rad"] = calc_angle(
            center_px_x=verdict["center_x"],
            screen_px_x= SCREEN_PX_X)

        verdict["label"] = create_label_line(verdict)

        obj_list.append(verdict)

    if save_as_file:
        save_dicts_to_json_txt(obj_list)


    return obj_list


def draw_object_list_to_frame(frame: np.ndarray, obj_list: list, draw_arrows: bool = False):
    WHITE = (255,255,255)
    THICKNESS = 1

    write_list = []

    for object_dict in obj_list:
        object_dict : dict


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
                img= frame,
                radius=int(radius),
                center=shape_center,
                color= color,
                thickness=THICKNESS
            )

        elif object_dict["shape"] == RECT:
            
            height = int(object_dict["height"])


            frame = cv2.rectangle(
                img=frame,
                pt1=shape_center,
                pt2=(shape_center_x + width, shape_center_y + height),
                color=color,
                thickness=THICKNESS
            )

        if draw_arrows:
            # draw an arrow from the center of the object to the center of the frame
            screen_center_x = int(SCREEN_PX_X / 2)
            screen_center_y = int(SCREEN_PX_Y / 2)
            screen_center = (screen_center_x, screen_center_y)


            delta_x_px = abs(screen_center_x-shape_center_x)
            delta_y_px = abs(screen_center_y-shape_center_y)

            arrow_length = math.sqrt(pow(delta_x_px, 2) + pow(delta_y_px,2))
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
                frame= frame,
                color= BLUE,
                point1=(shape_center_x, shape_center_y),
                point2= (screen_center_x, shape_center_y),
                label=f"px: {delta_x_px}"
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
        cv2.getTextSize(text or "Ag", font, font_scale, thickness)
        for text in texts
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
        frame, p1, p2, color,
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
        frame, text, text_origin, font, font_scale, color,
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )

    return frame



def main():
    path = "training_photos/2026-10-11T20_53_56.553302.png"
    frame : np.ndarray = cv2.imread(path) # type: ignore
    frame_unchanged : np.ndarray = frame.copy()

    obj_list : list = get_object_list_from_frame(frame, save_as_file=True)
    frame_annotated : np.ndarray = draw_object_list_to_frame(frame_unchanged, obj_list, draw_arrows=True)

    cv2.imshow("annotated", frame_annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()