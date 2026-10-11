import numpy as np


YELLOW = (0,255,255)
CYAN = (255,255,0)
RED = (0,0,255)


draw_thickness = 5


ball_red: dict = {

    "lower" : np.array( [177, 65, 0] ),
    "upper" : np.array( [179, 255, 255] ),

    # evaluating contours
    "min_area" : 250,
    "min_circularity" : 0.6,

    "name" : "ball_red",
    "draw_shape" : "circle",
    "draw_color" : RED,
}


hat_cyan : dict = {
    "lower" : np.array( [81, 133, 0] ),
    "upper" : np.array( [106, 255, 255] ),

    "name" : "hat_cyan",
    "draw_shape" : "rectangle",
    "draw_color" : CYAN,

    "min_area" : 250,
}


hat_yellow : dict = {
    "name" : "hat_yellow",
    "draw_shape" : "rectangle",
    "draw_color" : YELLOW,


    "lower" : np.array( [21, 167, 0] ),
    "upper" : np.array( [38, 255, 255] ),

    # evaluating contours
    "min_area" : 250,


}


object_hsv_list = [
    ball_red,
    hat_cyan,
    hat_yellow,
]

kernel_size = 3