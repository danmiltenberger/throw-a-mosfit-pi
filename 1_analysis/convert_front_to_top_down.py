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

field_center_in = (0,0)

field_center_px_x = convert_to_px(52)
field_center_px_y = convert_to_px(84)

field_center_to_goalbox_center_in = 60
goalbox_center_bot_in = (0, -60)
goalbox_center_top_in = (0, +60)


FOCAL_LENGTH = 1253.333     # recalc this when able


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


def convert_to_top_down(verdict: dict):
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
    px_y = verdict["center_y"]


    obj_dis_in = calc_new_distance(
        known= verdict["known_width_in"],
        percieved= verdict["width"])

    ic(obj_dis_in)




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

    for obj in obj_list:
        convert_to_top_down(obj)

