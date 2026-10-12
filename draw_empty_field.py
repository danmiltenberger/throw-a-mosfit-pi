from dataclasses import dataclass

import cv2
import numpy as np
from opencv_colors import *



@dataclass
class Rectangle:
    """Rectangle geometry following OpenCV's x, y, width, height convention."""
    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 100


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


def convert_to_px(inches: int) -> int:
    return int(inches * PIXELS_PER_INCH)



# Values in inches
playing_field_in_x = 8 * 12
playing_field_in_y = 12 * 12

goal_net_in_x = 48
goal_net_in_y = 12     # deep

goalbox_in_x = 48
goalbox_in_y = 24   # infront of the goal

perim_tape_thickness_in = 2
perim_black_beyond_tape_in = 2
total_buffer_in = perim_tape_thickness_in + perim_black_beyond_tape_in

total_window_in_y = 2*goal_net_in_y + playing_field_in_y
total_window_in_x = 2*total_buffer_in + playing_field_in_x

# total window
total_window_px_x = convert_to_px(total_window_in_x)
total_window_px_y = convert_to_px(total_window_in_y)


def draw_empty_field(is_show_image : bool = False, draw_centerpoints: bool = False) -> np.ndarray:

    frame : np.ndarray = np.zeros((total_window_px_y, total_window_px_x, 3), dtype=np.uint8)

    # start with a background
    background = Rectangle(
        width= total_window_px_x,
        height= total_window_px_y
    )

    # then the 2'' perim 2'' from the edges

    # this has been updated to have a chamber 1' from the edges
    white_px_x = convert_to_px(perim_tape_thickness_in *2 + playing_field_in_x)
    white_px_y = convert_to_px(perim_tape_thickness_in *2 + playing_field_in_y)
    white_perimiter = Rectangle(
        width= white_px_x,
        height= white_px_y
    )

    # then the playing field
    playing_field_px_x = convert_to_px(playing_field_in_x)
    playing_field_px_y = convert_to_px(playing_field_in_y)
    playing_field = Rectangle(
        width= playing_field_px_x,
        height= playing_field_px_y,
    )

    goal_net_px_x = convert_to_px(goal_net_in_x)
    goal_net_px_y = convert_to_px(goal_net_in_y)

    goal_net = Rectangle(
        width= goal_net_px_x,
        height=goal_net_px_y
    )


    goalbox_px_x = convert_to_px(goalbox_in_x)
    goalbox_px_y = convert_to_px(goalbox_in_y)
    goalbox = Rectangle(
        width= goalbox_px_x,
        height=goalbox_px_y
    )


    # BEGIN DRAWING TO SCREEN
    draw_centered_rectangle(frame, background, BLACK)
    draw_centered_octagon(
        frame, 
        white_perimiter, 
        convert_to_px(12),
        color=WHITE,)

    draw_centered_octagon(
        frame, 
        playing_field, 
        convert_to_px(12),
        color=GRAY,)

    

    # draw the goal nets
    goal_net_px_y_half = goal_net_px_y/2
    goal_net_offset = int(total_window_px_y/2 - goal_net_px_y_half)
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
                width= 30,
                height=30,
            )

        # draw a box right at the center of the field
        draw_centered_rectangle(
            frame,
            centerpoint
        )

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

    cv2.imwrite('1_analysis/empty_field.png', frame)
    return frame

    


if __name__ == "__main__":
    draw_empty_field()