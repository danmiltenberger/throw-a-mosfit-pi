import cv2
import numpy as np


def get_hsv_bounds(object_dict: dict):
    # Validate before casting: casting invalid values to uint8 can wrap them.
    lower = np.asarray(object_dict["lower"], dtype=float)
    upper = np.asarray(object_dict["upper"], dtype=float)
    hsv_max = np.array([179, 255, 255])

    for bound in (lower, upper):
        if (
            bound.shape != (3,)
            or not np.all(np.isfinite(bound))
            or np.any(bound < 0)
            or np.any(bound > hsv_max)
            or np.any(bound != np.floor(bound))
        ):
            raise ValueError(
                "HSV bounds must contain three integers within "
                "H: 0–179, S: 0–255, V: 0–255."
            )

    if np.any(lower > upper):
        raise ValueError("Each lower HSV bound must be <= its upper bound.")

    lower = lower.astype(np.uint8)
    upper = upper.astype(np.uint8)

    return lower, upper


def range_finder(object_dict: dict, frame):
    # determine the range and angle of the object from a given frame

