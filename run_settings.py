from tam_library import RED, YELLOW, CYAN 

kernel_size = 3
FOCAL_LENGTH = 1253.333     # recalc this when able
SCREEN_PX_X = 640
SCREEN_PX_Y = 480

FILLED = -1
PIXELS_PER_INCH = 5

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
    #ball_yellow,
    hat_cyan,
    hat_yellow,
]