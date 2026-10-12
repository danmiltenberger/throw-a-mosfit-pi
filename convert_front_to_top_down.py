# given a reading of width and pixel position,
# convert to a top down view

# 0,0 is field center
# from the screen, our bot will always be at the bottom
# up screen is psotive, north
# down screen is negative, south
# left screen is negative, west
# right screen is positive, east
# angles are measured as a deviation from north, ex: northwest is -45, northeast is 45
from opencv_colors import *
from draw_empty_field import *
from icecream import ic
import json
import math
from typing import Optional
from pathlib import Path



field_center_in = (0,0)



# window width = 4 + 4 + 8*12 = 104, half = 52
# window height = 12 + 12 + 12*12 = 168, half = 84
field_center_px_x = convert_to_px(52)
field_center_px_y = convert_to_px(84)

field_center_to_goalbox_center_in = 60
goalbox_center_bot_in = (0, -60)
goalbox_center_top_in = (0, +60)





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

    return (rel_px_x,rel_px_y)


goalbox_center_top_px = convert_in_point_to_screen(point=goalbox_center_top_in)
goalbox_center_bot_px = convert_in_point_to_screen(point=goalbox_center_bot_in)
center_field_px = convert_in_point_to_screen(point=(0,0))


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


def convert_to_top_down(verdict: dict, robot: dict ):
    '''
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
    '''

    # read the status of the robot
    robot_in_x, robot_in_y = robot["center_in"]


    # change the coordinate system from the robot reference to the global reference
    verdict["center_in"] = convert_robot_dis_and_angle_to_global(
        angle_rad_robot= robot["theta_rad"],
        angle_rad_to_obj=verdict["theta_rad"],
        robot_x= robot_in_x,
        robot_y= robot_in_y,
        distance= verdict["distance_in"]
    )


    # convert the inches measurement to screen pixels
    verdict["center_px"] = convert_in_point_to_screen(verdict["center_in"])

    return verdict



def convert_robot_dis_and_angle_to_global(distance: float, angle_rad_to_obj: float, robot_x: float, robot_y: float, angle_rad_robot):

    object_x = robot_x + distance * math.cos(angle_rad_to_obj + angle_rad_robot)
    object_y = robot_y + distance * math.sin(angle_rad_to_obj + angle_rad_robot)

    return object_x, object_y

    




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



#obj_list = [ball_yellow, robot_us, robot_them]


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


    arrow_length = int(radius*2)
    radius = int(radius)

    if radius <= 0 or arrow_length <= 0 or thickness < 1:
        raise ValueError("radius, arrow_length, and thickness must be positive")

    center = (int(round(point[0])), int(round(point[1])))

    # Subtract the y component so positive angles point upward.
    endpoint = (
        int(round(center[0] + arrow_length * math.cos(angle))),
        int(round(center[1] - arrow_length * math.sin(angle))),
    )

    cv2.circle(
        frame, center, radius, color,
        thickness=thickness, lineType=cv2.LINE_AA,
    )

    # only draw the arrow if the angle is different than the default
    if angle != 9999999:
        cv2.arrowedLine(
            frame, center, endpoint, color,
            thickness=thickness, line_type=cv2.LINE_AA, tipLength=0.25,
        )

    return frame




if __name__ == "__main__":


    # load the png or create it if necessary
    file_path = Path("empty_field.png")
    if file_path.is_file():
        frame : np.ndarray = cv2.imread(file_path) # type: ignore
    else:
        frame = draw_empty_field()


    directory = "obj_lists"
    most_recent_obj_list = get_most_recent_file(directory)
    obj_list : list = load_dicts_from_txt(str(most_recent_obj_list))


    robot_field_goal = {
        "name" : "our_bot",
        "center_in" : goalbox_center_bot_in, 
        "center_px" : convert_in_point_to_screen((goalbox_center_bot_in)),
        "radius" : RADIUS_ROBOT,
        "color" : CYAN,
        "theta_rad" : math.radians(90)      # up screen
    }

    # FIXME manually set the robot centered in our goal box
    robot = robot_field_goal

    frame = draw_circle_with_arrow(
        angle= robot["theta_rad"],
        color=robot["color"],
        frame= frame,
        point= robot["center_px"],
        radius= robot["radius"]
    )


    for obj in obj_list:
        ojb = convert_to_top_down(obj, robot_field_goal)

        ic(obj)

        # draw object
        frame = draw_circle_with_arrow(
            angle= ojb["theta_rad"],
            color=ojb["color"],
            frame= frame,
            point= ojb["center_px"],
            radius= obj["radius"]
        )


    cv2.imshow("updated field", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    


