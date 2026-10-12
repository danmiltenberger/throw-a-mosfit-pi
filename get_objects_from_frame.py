import numpy as np
import cv2
import ast
import numpy as np
import csv
from datetime import datetime
from icecream import ic
import json
from opencv_colors import *


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
    "shape" : RECT,
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
        verdict["known_width_in"] = object_dict["known_width_in"]
        verdict["color"] = object_dict["color"]

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

        obj_list.append(verdict)

    if save_as_file:
        save_dicts_to_json_txt(obj_list)


    return obj_list


def draw_object_list_to_frame(frame: np.ndarray, obj_list: list):
    WHITE = (255,255,255)
    THICKNESS = 1

    write_list = []

    for object_dict in obj_list:
        object_dict : dict
        name = object_dict["name"]

        center_x = int(object_dict["center_x"])
        center_y = int(object_dict["center_y"])
        width = int(object_dict["width"])

        label = f'{name} area: {object_dict["area"]}, width_px: {width}'

        write_list.append(label)


        shape_center = (center_x, center_y)

        if object_dict["shape"] == CIRCLE:
            radius = object_dict["radius"]
            frame = cv2.circle(
                img= frame,
                radius=int(radius),
                center=shape_center,
                color= WHITE,
                thickness=THICKNESS
            )

        elif object_dict["shape"] == RECT:
            
            height = int(object_dict["height"])


            frame = cv2.rectangle(
                img=frame,
                pt1=shape_center,
                pt2=(center_x + width, center_y + height),
                color=WHITE,
                thickness=THICKNESS
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


def main():
    path = "training_photos/2026-10-11T20_53_56.553302.png"
    frame : np.ndarray = cv2.imread(path) # type: ignore
    frame_unchanged : np.ndarray = frame.copy()

    obj_list : list = get_object_list_from_frame(frame, save_as_file=True)


    frame_annotated : np.ndarray = draw_object_list_to_frame(frame_unchanged, obj_list)

    cv2.imshow("annotated", frame_annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()