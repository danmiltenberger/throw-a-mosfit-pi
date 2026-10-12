import cv2
from opencv_colors import *
import numpy as np
from pathlib import Path
from typing import Optional
from draw_empty_field import draw_empty_field
from convert_front_to_top_down import *

RADIUS_ROBOT = int(14*PIXELS_PER_INCH/2)
RADIUS_BALL = int(7*PIXELS_PER_INCH/2)
RADIUS_TENNIS_BALL = int(2.7*PIXELS_PER_INCH/2)


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
        outline_padding = (
            (circle_thickness + 1) // 2 if circle_thickness > 0 else 0
        )

        text_x = int(round(cx - text_width / 2))
        text_y = (
            cy - radius - outline_padding
            - label_gap - baseline - text_thickness
        )

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
    if (
        not isinstance(border_thickness, (int, np.integer))
        or border_thickness <= 0
    ):
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
        overlay, opacity,
        frame, 1.0 - opacity,
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
            obj : dict

        draw_transparent_circle(
            frame,
            center=obj["center"],
            radius=obj["radius"],
            color=obj["color"],
            opacity=0.5,
            )

    return frame

ball_red = {
    "label" : "ball",
    "center" : center_field_px,
    "radius" : RADIUS_BALL,
    "color" : RED,
}

ball_yellow = {
    "label" : "ball",
    "center" : center_field_px,
    "radius" : RADIUS_TENNIS_BALL,
    "color" : YELLOW,
}


robot_us = {
    "label" : "US",
    "center" : goalbox_center_bot_px,
    "radius" : RADIUS_ROBOT,
    "color" : YELLOW,
}

robot_them = {
    "label" : "OPPONENT",
    "center" : goalbox_center_top_px,
    "radius" : RADIUS_ROBOT,
    "color" : CYAN,
}

obj_list = [ball_yellow, robot_us, robot_them]

if __name__ == "__main__":

    # load the png or create it if necessary
    file_path = Path("1_analysis/empty_field.png")
    if file_path.is_file():
        frame = cv2.imread(file_path)
    else:
        frame = draw_empty_field()
    
    frame = draw_to_field(frame, obj_list)
    cv2.imshow("updated field", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()