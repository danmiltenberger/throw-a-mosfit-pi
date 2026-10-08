# given a photo / frame of a video, determine
# 1. what is in the photo (if anything)
# 2. what distance and angle from the center is each object?

import numpy as np
import cv2
from vars import red_ball, kernel_size


def save_cv2_frame_as_png(frame, output_path: str):
    if not cv2.imwrite(output_path, frame):
        raise OSError(f"Could not save to path {output_path}")

    print(f"saved frame to path {output_path}")


def load_cv2_frame_from_png(input_path: str):
    frame = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read PNG: {input_path}")

    print(f"loaded frame from path {input_path}")
    return frame


def validate_hsv_bounds(object_dict: dict, lower_key: str, upper_key: str):
    lower = np.asarray(object_dict[lower_key], dtype=float)
    upper = np.asarray(object_dict[upper_key], dtype=float)
    limits = np.array([179, 255, 255])

    for bound in (lower, upper):
        if (
            bound.shape != (3,)
            or not np.all(np.isfinite(bound))
            or np.any(bound < 0)
            or np.any(bound > limits)
            or np.any(bound != np.floor(bound))
        ):
            raise ValueError(f"Invalid HSV bounds: {lower_key}, {upper_key}")

    if np.any(lower > upper):
        raise ValueError(
            f"Each {lower_key} value must be <= its {upper_key} value."
        )

    return lower.astype(np.uint8), upper.astype(np.uint8)


def get_mask(hsv, object_dict: dict):

    mask = hsv.copy()

    if ("lower" in object_dict.keys()) and ("upper" in object_dict.keys()):
        lower = object_dict["lower"]
        upper = object_dict["upper"]
    else:
        raise ValueError("object does not have attribute lower and/or upper")


    mask = cv2.inRange(hsv, lower, upper)

    # if two hsv brackets, include it in the mask
    if ("lower1" in object_dict.keys()) and ("upper1" in object_dict.keys()):
        lower2 = object_dict["lower"]
        upper2 = object_dict["upper"]

        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2, dst=mask)

    return mask


def get_clean_mask(mask: np.ndarray):
    clean_mask = mask.copy()

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        ))

    # Opening removes small specks; closing fills small gaps.
    if kernel_size > 1:
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return clean_mask


def annotate_frame(frame, best_contour):
    frame_annotated = frame.copy()

    (x_px, y_px), radius = cv2.minEnclosingCircle(best_contour)
    center = (int(round(x_px)), int(round(y_px)))

    draw_radius = max(1, int(np.ceil(radius)))
    draw_color = (0, 255, 0)
    draw_thickness = 2

    # draw circle bounding the object
    frame_annotated = cv2.circle(
        frame_annotated,
        center,
        draw_radius,
        draw_color,
        draw_thickness)

    # draw small dot at the object center
    frame_annotated = cv2.circle(
        frame_annotated, 
        center, 
        3, 
        draw_color, 
        -1)

    label = f"object at: ({center[0]}, {center[1]}), r={radius:.1f}px"


    return frame_annotated, label


def add_text_to_frame(frame, label):
    # Fixed label location avoids clipping labels near the ball.
    label_position = (10, min(30, frame.shape[0] - 1))

    frame = cv2.putText(
        frame, label, label_position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
        cv2.LINE_AA,
    )

    return frame


def get_best_contour(contours, object_dict: dict):

    check_min_area : bool = False
    check_min_circularity: bool = False
    min_area : int = 0
    min_circularity : int = 0

    if "min_area" in object_dict.keys():
        min_area = object_dict["min_area"]
        check_min_area = True

    if "min_circularity" in object_dict.keys():
        min_area = object_dict["min_circularity"]
        min_circularity = True
    
    best_contour = None
    best_area = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if check_min_area:
            if area < min_area:
                continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        if check_min_circularity:
            circularity = 4.0 * np.pi * area / perimeter**2
            if circularity < min_circularity:
                continue

        if area > best_area:
            best_area = area
            best_contour = contour

    return best_contour, best_area


def evaluate_frame(frame, object_dict: dict):

    unchanged_frame = frame.copy()


    # convert to hsv for processing
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create a mask using the hsv values
    mask = get_mask(hsv, object_dict)

    # clean the mask by blurring and then re-evaluating
    clean_mask = get_clean_mask(mask)


    # get the contours from the mask
    contours, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)


    frame_with_contours = cv2.drawContours(frame, contours, -1, 0, -1)

    # evaluate the best contour
    best_contour, best_area = get_best_contour(contours, object_dict)

    label = "object not detected"
    frame_annotated = unchanged_frame.copy()
    if best_contour is not None:
        frame_annotated, label = annotate_frame(frame_annotated, best_contour)

    frame_annotated = add_text_to_frame(frame_annotated, label)


    # save the images as pngs
    save_cv2_frame_as_png(unchanged_frame, "open_cv_stack/0_original.png")
    save_cv2_frame_as_png(hsv, "open_cv_stack/1_hsv.png")
    save_cv2_frame_as_png(mask, "open_cv_stack/2_mask.png")
    save_cv2_frame_as_png(clean_mask, "open_cv_stack/3_clean_mask.png")
    save_cv2_frame_as_png(frame_with_contours, "open_cv_stack/4_frame_with_contours.png")
    save_cv2_frame_as_png(frame_annotated, "open_cv_stack/5_frame_annotated.png")

    pass



def main():
    #input_path = r".temp/vlcsnap-2026-10-05-16h43m03s033.png"

    input_path = r".temp/vlcsnap-2026-10-07-15h58m42s688.png"

    frame = load_cv2_frame_from_png(input_path)

    evaluate_frame(frame, red_ball)




if __name__ == "__main__":
    main()