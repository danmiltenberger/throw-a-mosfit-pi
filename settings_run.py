from tam_library_colors import RED, YELLOW, CYAN 

kernel_size = 3
FOCAL_LENGTH = 1253.333     # recalc this when able
CAMERA_CAPTURE_X = 640
CAMERA_CAPTURE_Y = 480

FILLED = -1
PIXELS_PER_INCH = 5

CIRCLE = "CIRCLE"
RECT = "RECT"

USE_DRAWN_FIELD : bool = True

DRAW_THICKNESS : int = 3

# names for the objects. No magic strings!
HAT_YELLOW = "hat_yellow"
HAT_CYAN = "hat_cyan"
BALL_RED = "ball_red"
BALL_YELLOW = "ball_yellow"

# keys
KNOWN_WIDTH_IN = "known_width_in"
SHAPE_TO_DRAW = "shape"
DRAW_COLOR = "color"
MIN_CIRCLARITY = "min_circularity"
MIN_AREA = "min_area"
#MIN_WIDTH = "min_width"


ball_red: dict = {
    "name" : BALL_RED,
    "shape" : RECT,
    "path_to_hsv" : "hsv_bounds/ball_red.txt",
    KNOWN_WIDTH_IN : 7,
    "color" : RED,
    MIN_AREA : 10E3
}

ball_yellow : dict = {
    "name" : BALL_YELLOW,
    "shape" : CIRCLE,
    "path_to_hsv" : "hsv_bounds/ball_yellow.txt",
    KNOWN_WIDTH_IN : 2.7,
    "color" : YELLOW,
    MIN_AREA : 10E3
}

hat_cyan : dict = {
    "name" : HAT_CYAN,
    "shape" : RECT,
    "path_to_hsv" : "hsv_bounds/hat_cyan.txt",
    KNOWN_WIDTH_IN : 8,     # FIXME
    "color" : CYAN,
    MIN_AREA : 10E3
}

hat_yellow : dict = {
    "name" : HAT_YELLOW,
    "shape" : RECT,
    "path_to_hsv" : "hsv_bounds/hat_yellow.txt", 
    KNOWN_WIDTH_IN : 8,     # FIXME
    "color" : YELLOW,
    MIN_AREA : 10E3
}

search_for_list = [
    ball_red,
    ball_yellow,
    hat_cyan,
    #hat_yellow,
]

search_for_dict = {
    BALL_RED : ball_red,
    BALL_YELLOW : ball_yellow,
    HAT_CYAN : hat_cyan,
    HAT_YELLOW : hat_yellow,
}