# given a photo / frame of a video, determine
# 1. what is in the photo (if anything)
# 2. what distance and angle from the center is each object?

import numpy as np
import cv2
from vars import red_ball, kernel_size


import os


# https://www.geeksforgeeks.org/python/python-loop-through-folders-and-files-in-directory/



def print_directory(given_dir):
    for e in os.scandir(given_dir):
        if e.is_file():
            print(e)


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


def annotate_frame(frame, x_px, y_px, radius):
    frame_annotated = frame.copy()

    
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


def show_images_side_by_side(frame1, frame2, title : str ="side by side"):
    # Source - https://stackoverflow.com/a/57792197
    # Posted by Khan, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-10-08, License - CC BY-SA 4.0

    numpy_horizontal_concat = np.concatenate((frame1, frame2), axis=1)
    cv2.imshow(title, numpy_horizontal_concat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def evaluate_frame(frame, object_dict: dict, save_as_pngs: bool = False,):
    '''
    From a given frame, and a dictionary of information to look for in that frame, 
    return a "verdict" dict with information on objects and positions
    Optionally save the inbetween steps as pngs for debugging
    '''

    unchanged_frame = frame.copy()

    object_name = object_dict["name"]


    verdict_dict: dict = {
        "object_name" : object_name,
        "is_detected" : False,
        "draw_shape" : object_dict["draw_shape"],
        "draw_color" : object_dict["draw_color"],
        "draw_thickness" : object_dict["draw_thickness"]
    }


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

    # get the contours (lines that describe the different shapes)
    frame_with_contours = cv2.drawContours(frame, contours, -1, 0, -1)

    # evaluate the best contour
    best_contour, best_area = get_best_contour(contours, object_dict)

    frame_annotated = unchanged_frame.copy()

    # write useful values to the verdict dictionary
    if best_contour is not None:
        verdict_dict["is_detected"] = True


        if verdict_dict["draw_shape"] == "circle":
            (x_px, y_px), radius = cv2.minEnclosingCircle(best_contour)
            verdict_dict["radius"] = radius
            verdict_dict["width"] = radius*2

        elif verdict_dict["draw_shape"] == "rectangle":
            x_px, y_px, width_px, height_px = cv2.boundingRect(best_contour)
            verdict_dict["width"] = width_px
            verdict_dict["height"] = height_px

        else:
            raise ValueError("draw_shape not 'circle' or 'rectangle'")
        
        verdict_dict["x_px"] = x_px
        verdict_dict["y_px"] = y_px


    if save_as_pngs:
        folder = "open_cv_stack"
        save_cv2_frame_as_png(unchanged_frame,          f"{folder}/0_original.png")
        save_cv2_frame_as_png(hsv,                      f"{folder}/1_hsv.png")
        save_cv2_frame_as_png(mask,                     f"{folder}/2_mask.png")
        save_cv2_frame_as_png(clean_mask,               f"{folder}/3_clean_mask.png")
        save_cv2_frame_as_png(frame_with_contours,      f"{folder}/4_frame_with_contours.png")

    return verdict_dict

    

def draw_to_frame(frame, verdict_list, verbose_printout: bool = False):
    # read the verdict list and draw the relevant objects to frame

    frame_annotated = frame.copy()

    for verdict_dict in verdict_list:
        verdict_dict : dict
        object_name = verdict_dict["object_name"]

        if verdict_dict["is_detected"] == False:
            label = f"{object_name}: not detected"
            frame_annotated = add_text_to_frame(frame_annotated, label)
            return frame_annotated


        else:
            x_px = verdict_dict["x_px"]
            y_px = verdict_dict["y_px"]

            draw_color = verdict_dict["draw_color"]
            draw_thickness = verdict_dict["draw_thickness"]

            center = (int(round(x_px)), int(round(y_px)))

            label = f"{object_name}: at {center}"
            frame_annotated = add_text_to_frame(frame_annotated, label)


            # draw small dot at the object center
            if verbose_printout:
                print(f"drew center dot for {object_name}")

            frame_annotated = cv2.circle(
                frame_annotated, 
                center, 
                3, 
                draw_color, 
                draw_thickness)

            # draw circle bounding the object
            if verdict_dict["draw_shape"] == "circle":
                radius_px = verdict_dict["radius"]

                # the radius will either be 1 or the measured radius
                draw_radius = max(1, int(np.ceil(radius_px)))

                if verbose_printout:
                    print(f"drew bounding circle for {object_name}")

                frame_annotated = cv2.circle(
                    frame_annotated,
                    center,
                    draw_radius,
                    draw_color,
                    draw_thickness)

            # draw rectangle bounding the object
            if verdict_dict["draw_shape"] == "rectangle":
                width_px = verdict_dict["width"]
                height_px = verdict_dict["height"]



                frame_annotated = cv2.rectangle(
                    frame_annotated,
                    (x_px, y_px),
                    (x_px + width_px, y_px + height_px),
                    draw_color,
                    draw_thickness
                )

            
                

    return frame_annotated



def eval_and_show(frame):

    
    unchanged_frame = frame.copy()
    
    verdict_dict : dict = evaluate_frame(frame, red_ball)
    
    frame_annotated = draw_to_frame(unchanged_frame, [verdict_dict])

    return frame_annotated



def main():


    input_path = r".temp/vlcsnap-2026-10-07-15h58m42s688.png"
    show_to_screen : bool = True



    frame = load_cv2_frame_from_png(input_path)

    unchanged_frame = frame.copy()

    verdict_dict : dict = evaluate_frame(frame, red_ball)

    print(verdict_dict)


    if show_to_screen:
        frame_annotated = draw_to_frame(unchanged_frame, [verdict_dict])
        cv2.imshow("annotated", frame_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





if __name__ == "__main__":
    main()