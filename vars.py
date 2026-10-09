import numpy as np


ball_red: dict = {
    # # Red near the beginning of the hue scale.
    # "lower": np.array([0, 153, 50], dtype=np.uint8),
    # "upper": np.array([10, 255, 255], dtype=np.uint8),

    # # Red near the end of the hue scale.
    # "lower2": np.array([170, 153, 50], dtype=np.uint8),
    # "upper2": np.array([179, 255, 255], dtype=np.uint8),

    "lower" : np.array( [177, 65, 0] ),
    "upper" : np.array( [179, 255, 255] ),


    # evaluating contours
    "min_area" : 50,
    "min_circularity" : 0.6,

    "detection_frame_color" : (0, 255, 0),
    "name" : "ball_red",
    "draw_shape" : "circle",
    "draw_color" : (0,255,0),
    "draw_thickness" : 5,
}

hat_cyan : dict = {
    "lower" : np.array( [81, 133, 0] ),
    "upper" : np.array( [106, 255, 255] ),



    # evaluating contours
    "min_area" : 50,
    "min_circularity" : 0.6,

    "detection_frame_color" : (0, 255, 0),
    "name" : "hat_cyan",
    "draw_shape" : "rectangle",
    "draw_color" : (0,255,0),
    "draw_thickness" : 5,
}

hat_yellow : dict = {
    "lower" : np.array( [21, 167, 0] ),
    "upper" : np.array( [38, 255, 255] ),



    # evaluating contours
    "min_area" : 50,
    "min_circularity" : 0.6,

    "detection_frame_color" : (0, 255, 0),
    "name" : "hat_yellow",
    "draw_shape" : "rectangle",
    "draw_color" : (0,255,0),
    "draw_thickness" : 5,
}


object_hsv_list = [
    ball_red,
    hat_cyan,
    hat_yellow,
]

kernel_size = 3