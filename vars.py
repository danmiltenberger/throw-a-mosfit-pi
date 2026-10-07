import numpy as np


ball: dict = {
    # Red near the beginning of the hue scale.
    "lower": np.array([0, 153, 50], dtype=np.uint8),
    "upper": np.array([10, 255, 255], dtype=np.uint8),

    # Red near the end of the hue scale.
    "lower2": np.array([170, 153, 50], dtype=np.uint8),
    "upper2": np.array([179, 255, 255], dtype=np.uint8),
}


camera : dict = {
    "focal_length_mm" : 10
}