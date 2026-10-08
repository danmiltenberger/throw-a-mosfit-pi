import numpy as np


red_ball: dict = {
    # Red near the beginning of the hue scale.
    "lower": np.array([0, 153, 50], dtype=np.uint8),
    "upper": np.array([10, 255, 255], dtype=np.uint8),

    # Red near the end of the hue scale.
    "lower2": np.array([170, 153, 50], dtype=np.uint8),
    "upper2": np.array([179, 255, 255], dtype=np.uint8),

    # evaluating contours
    "min_area" : 50,
    "min_circularity" : 0.6,

    "detection_frame_color" : (0, 255, 0),
    "detection_label" : "red_ball",
}


object_hsv_list = [
    red_ball
]

kernel_size = 3