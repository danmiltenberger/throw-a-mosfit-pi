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


screen_px_x = 640

field_center_in = (0,0)

field_center_px_x = convert_to_px(52)
field_center_px_y = convert_to_px(84)

field_center_to_goalbox_center_in = 60
goalbox_center_bot_in = (0, -60)
goalbox_center_top_in = (0, +60)


FOCAL_LENGTH = 1253.333     # recalc this when able


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


def convert_in_point_to_screen(point : tuple[int,int]) -> tuple[int, int]:

    # the screen measures down and right from the top left,
    # but we measure the field +up and +right from the center

    # an input of 0,0 should result in window_x/2, window_y /2
    inch_x, inch_y = point

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


def convert_to_top_down(verdict: dict, robot_x, robot_y, robot_theta ):
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

    px_x = verdict["center_x"]

    angle = calc_angle(px_x, screen_px_x)

    obj_dis_in = calc_new_distance(
        known= verdict["known_width_in"],
        percieved= verdict["width"])

    x, y, theta = to_base_frame(
        ref_x= robot_x,
        ref_y= robot_y,
        ref_theta= robot_theta,
        distance= obj_dis_in,
        angle= angle)
    
    verdict["center"] = int(x), int(y)
    return verdict




def calc_angle(center_px_x, screen_px_x)-> float:
    pixels_per_degree = 27.64
    degrees_per_pixel = 1/ pixels_per_degree

    delta_x_px = screen_px_x - center_px_x
    degrees = delta_x_px * degrees_per_pixel

    return degrees


def convert_robot_dis_and_angle_to_global(distance: float, angle_degrees: float, robot_x: float, robot_y: float, robot_angle: float):

    object_x = robot_x + distance * math.cos(math.radians(angle_degrees))
    object_y = robot_y + distance * math.cos(math.radians(angle_degrees))

    return object_x, object_y

    
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


def calc_new_distance(known: float, percieved: float) -> float:
    D = known
    d = percieved
    f = FOCAL_LENGTH
    Z = (D*f) / d

    return Z



if __name__ == "__main__":

    verd_path = r"0_vision\obj_lists\22_25_46.txt"

    obj_list : list = load_dicts_from_txt(verd_path)

    robot_x, robot_y = goalbox_center_bot_in
    robot_theta = 0

    ic(obj_list)
    for obj in obj_list:
        ojb = convert_to_top_down(obj, robot_x, robot_y, robot_theta)

    ic(obj_list)

