import cv2
import numpy as np
import math
import csv
import json
from icecream import ic
from settings_run import search_for_dict, MIN_AREA, MIN_CIRCLARITY
from pathlib import Path
from datetime import datetime
from tam_library_classes import Detection, Point, OurRobot
from tam_library import (
    search_for_list,
    load_hsv_bounds,
    get_best_contour,
    CIRCLE,
    RECT,
    SCREEN_CENTER_X,
    FOCAL_LENGTH,
    convert_to_px,
    KNOWN_WIDTH_IN,
    field_center_px_x,
    field_center_px_y,
    load_image_as_frame,
    show_frames_side_by_side,
    playing_field_in_x,
    playing_field_in_y,
)

from settings_run import SHAPE_TO_DRAW, DRAW_COLOR, DRAW_THICKNESS, USE_DRAWN_FIELD


def is_valid_contour(contour, area, object_dict: dict) -> bool:
    """
    check if the contour is valid based on area, circularity, etc
    """

    if MIN_AREA in object_dict.keys():
        if area < object_dict[MIN_AREA]:
            return False

    elif MIN_CIRCLARITY in object_dict.keys():
        # Find contour properties
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter**2)
        else:
            circularity = 0

        if circularity < object_dict[MIN_CIRCLARITY]:
            return False
    else:
        return True


def get_detections_from_frame(frame: np.ndarray, bot: OurRobot) -> list[Detection]:
    "from the input frame, search for the objects as specified in the run_settings.py search_list"

    detect_list = []
    frame = frame.copy()

    for object_dict in search_for_list:

        # load the necessary info from the reference list
        path_to_hsv = object_dict["path_to_hsv"]
        obj_name = object_dict["name"]
        lower, upper = load_hsv_bounds(path_to_hsv)

        # convert to hsv for processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create a mask using the hsv values
        mask = cv2.inRange(hsv, lower, upper)

        # get the contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get the best contour
        best_contour, best_area = get_best_contour(contours, object_dict=object_dict)

        # if there is no contour, this object has not been detected, go to next object
        if best_contour is None:
            continue

        # if there IS a contour, we want to know how large it is and where it is
        best_contour, best_area = get_best_contour(contours, object_dict)

        # if the contour is no good, skip it
        contour_verdict = is_valid_contour(
            contour=best_contour, area=best_area, object_dict=object_dict
        )
        if contour_verdict == False:
            continue

        # okay, so at this point we have decided that the contour is useful. Let's package and ship it!
        this_detect = Detection()
        this_detect.name = object_dict["name"]

        # analyze either as a circle or rectangle
        if object_dict["shape"] == CIRCLE:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center_point = Point(x, y)
            this_detect.screen_center_px = center_point
            this_detect.width_px = int(radius * 2)

            this_detect.area = int(np.pi * radius * radius)

        elif object_dict["shape"] == RECT:
            left_x, top_y, width, height = cv2.boundingRect(best_contour)

            x_center = int(left_x + width / 2)
            y_center = int(top_y + height / 2)
            center_point = Point(x=x_center, y=y_center)
            this_detect.screen_center_px = center_point
            this_detect.width_px = width
            this_detect.area = int(width * height)

        # determine the distance and angle relative to the camera
        this_detect.range = calc_distance_from_bot(this_detect)
        this_detect.azimuth_rad = calc_angle_from_screen_center(
            this_detect.screen_center_px
        )

        # convert to the field xy system in inches
        this_detect.field_pos_in = convert_pos_and_ang_to_robot_in(this_detect, bot)

        # convert the field inches to the window pixels
        this_detect.field_pos_px = convert_to_field_pixels_global(
            this_detect.field_pos_in
        )

        detect_list.append(this_detect)

    return detect_list


def convert_pos_and_ang_to_robot_in(this_detect: Detection, robot: OurRobot) -> Point:

    # get the detection's global position in inches
    object_x = robot.field_pos_in.x + this_detect.range * math.cos(
        this_detect.azimuth_rad + robot.azimuth_rad
    )

    object_y = robot.field_pos_in.y + this_detect.range * math.sin(
        this_detect.azimuth_rad + robot.azimuth_rad
    )

    return Point(object_x, object_y)


def calc_distance_from_bot(this_detect: Detection) -> float:
    known_width = search_for_dict[this_detect.name][KNOWN_WIDTH_IN]
    percieved_width = this_detect.width_px
    Z = (known_width * FOCAL_LENGTH) / percieved_width
    return Z


def calc_angle_from_screen_center(point: Point) -> float:
    """Return the horizontal angle from straight ahead, in radians.

    Args:
        pixel_x: Object's horizontal coordinate in the image.
        focal_length_x: Camera's horizontal focal length in pixels.
        center_x: Camera's horizontal principal point in pixels.

    Returns:
        Signed angle: negative to the left, positive to the right.
    """
    pixel_x = point.x
    delta_x = pixel_x - SCREEN_CENTER_X

    return -math.atan2(delta_x, FOCAL_LENGTH)


def draw_detection_list_to_frame(
    frame: np.ndarray, detect_list: list[Detection]
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    label_gap_px = 8

    frame_height, frame_width = frame.shape[:2]

    for detection in detect_list:
        # Quick lookup values
        shape = search_for_dict[detection.name][SHAPE_TO_DRAW]
        color = search_for_dict[detection.name][DRAW_COLOR]

        # Values specific to this detection
        shape_center_x = detection.screen_center_px.x
        shape_center_y = detection.screen_center_px.y
        shape_center = (int(shape_center_x), int(shape_center_y))

        range_to_object = detection.range
        azimuth_deg = np.degrees(detection.azimuth_rad)

        # Default top edge for label placement
        shape_top_y = shape_center[1] - int(detection.width_px / 2)

        # Draw a small dot at the center
        frame = cv2.circle(
            img=frame,
            center=shape_center,
            radius=3,
            color=color,
            thickness=-1,
        )

        # Draw a circle around the object
        if shape == CIRCLE:
            radius = int(detection.width_px / 2)
            frame = cv2.circle(
                img=frame,
                radius=radius,
                center=shape_center,
                color=color,
                thickness=DRAW_THICKNESS,
            )

        # Or draw a square
        elif shape == RECT:
            frame = draw_rectangle(
                frame=frame,
                color=color,
                width=detection.width_px,
                height=detection.width_px,
                center=shape_center,
                label=detection.name,
            )

        # Draw a label above the object
        # Use "deg" rather than a degree symbol for OpenCV font compatibility.
        label = (
            f"Range: {range_to_object:.2f} in | "
            f"Azim: {azimuth_deg:.1f} deg | Point: {detection.field_pos_in}"
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness,
        )

        # putText uses the bottom-left corner of the text as its origin.
        label_x = shape_center[0] - text_width // 2
        label_y = shape_top_y - label_gap_px - baseline

        # Keep the label within the frame where possible.
        label_x = max(0, min(label_x, frame_width - text_width - 1))
        label_y = max(
            text_height,
            min(label_y, frame_height - baseline - 1),
        )

        frame = cv2.putText(
            img=frame,
            text=label,
            org=(label_x, label_y),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return frame


def draw_detection_list_to_field(
    field_frame: np.ndarray,
    detect_list: list[Detection],
    bot: OurRobot,
    is_keep_on_field: bool = False,
) -> np.ndarray:

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    label_gap_px = 8

    def keep_on_field():
        playing_field_px_x = convert_to_px(playing_field_in_x)
        playing_field_px_y = convert_to_px(playing_field_in_y)
        if detection.field_pos_px.x > playing_field_px_x:
            detection.field_pos_px.x = playing_field_px_x

        if detection.field_pos_px.y > playing_field_px_y:
            detection.field_pos_px.y = playing_field_px_y

    for detection in detect_list:
        draw_shape = search_for_dict[detection.name][SHAPE_TO_DRAW]
        draw_color = search_for_dict[detection.name][DRAW_COLOR]
        known_width_in = search_for_dict[detection.name][KNOWN_WIDTH_IN]
        known_width_px = convert_to_px(known_width_in)

        if is_keep_on_field:
            keep_on_field()

        #ic(draw_shape, draw_color, detection.field_pos_px)

        cv_format_center = (
            int(detection.field_pos_px.x),
            int(detection.field_pos_px.y),
        )

        if draw_shape == CIRCLE:
            cv2.circle(
                img=field_frame,
                center=cv_format_center,
                color=draw_color,
                radius=int(known_width_px / 2),
                thickness=DRAW_THICKNESS,
            )

        elif draw_shape == RECT:
            field_frame = draw_rectangle(
                frame=field_frame,
                color=draw_color,
                width=detection.width_px,
                height=detection.width_px,
                center=detection.field_pos_px,
            )

        # Draw a label
        # Use "deg" rather than a degree symbol for OpenCV font compatibility.
        range_to_object = round(detection.range)
        azimuth_deg = math.degrees(detection.azimuth_rad)

        label = (
            f"Range: {range_to_object:.2f} in |"
            + f"Azim: {azimuth_deg:.1f} deg |"
            + f"Point:{round(detection.field_pos_in.x,2)}, {round(detection.field_pos_in.y,2)}"
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness,
        )

        # putText uses the bottom-left corner of the text as its origin.
        shape_top_y = cv_format_center[1]
        label_x = cv_format_center[0] - text_width // 2
        label_y = shape_top_y - label_gap_px - baseline

        frame_width = convert_to_px(playing_field_in_x)
        frame_height = convert_to_px(playing_field_in_y)

        # Keep the label within the frame where possible.
        label_x = max(0, min(label_x, frame_width - text_width - 1))
        label_y = max(
            text_height,
            min(label_y, frame_height - baseline - 1),
        )

        field_frame = cv2.putText(
            img=field_frame,
            text=label,
            org=(label_x, label_y),
            fontFace=font,
            fontScale=font_scale,
            color=draw_color,
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

        # draw an arrow from the bot camera to the object center
        draw_arrow(
            frame=field_frame,
            start=bot.field_pos_px,
            end=detection.field_pos_px,
            color=draw_color,
        )

    return field_frame


def convert_to_field_pixels_global(field_point_inches: Point) -> Point:
    # the screen measures down and right from the top left,
    # but we measure the field +up and +right from the center

    # an input of 0,0 should result in window_x/2, window_y /2

    px_x = convert_to_px(field_point_inches.x)
    px_y = convert_to_px(field_point_inches.y)

    # left means center minus the input
    rel_px_x = field_center_px_x + px_x
    rel_px_x = int(rel_px_x)

    # up means center MINUS input
    rel_px_y = field_center_px_y - px_y
    rel_px_y = int(rel_px_y)

    pixel_point = Point(rel_px_x, rel_px_y)

    return pixel_point


def draw_robot_to_field(field_frame: np.ndarray, bot: OurRobot) -> np.ndarray:

    bot_width_px = convert_to_px(bot.width_in)

    # draw a bounding rectangle
    field_frame = draw_rectangle(
        frame=field_frame,
        angle_radians=bot.azimuth_rad,
        center=bot.field_pos_px,
        color=bot.color,
        height=bot_width_px,
        width=bot_width_px,
    )

    return field_frame


def draw_arrow(
    frame,
    start: Point,
    distance: float | None = None,
    angle_rad: float | None = None,
    end: Point | None = None,
    color=(0, 255, 0),
    thickness: int = 2,
    tip_size: int = 10,
):
    """Draw an arrow from `start`, optionally with a rotated label.

    Requires either:
      - `distance` AND `angle_rad`, or
      - `end`

    Args:
        frame: BGR image to draw on.
        start: Point with `.x` and `.y` attributes.
        distance: Arrow length in pixels.
        angle_rad: Direction angle in radians, using image coordinates.
        end: Optional Point defining the arrow tip.
        color: BGR color.
        thickness: Line thickness.
        tip_size: Arrowhead size in pixels.


    Returns:
        The modified frame.
    """
    has_vector = distance is not None and angle_rad is not None
    has_end = end is not None

    if has_vector == has_end:
        raise ValueError(
            "Provide either (distance AND angle_rad) OR end, but not both."
        )

    sx, sy = int(start.x), int(start.y)

    if has_end:
        ex, ey = int(end.x), int(end.y)
        angle_rad = math.atan2(ey - sy, ex - sx)
    else:
        ex = sx + int(round(distance * math.cos(angle_rad)))
        ey = sy + int(round(distance * math.sin(angle_rad)))

    cv2.arrowedLine(
        frame,
        (sx, sy),
        (ex, ey),
        color,
        thickness=thickness,
        tipLength=max(0.03, tip_size / max(1.0, math.hypot(ex - sx, ey - sy))),
    )

    return frame


def draw_rectangle(
    frame: np.ndarray,
    center: Point,
    width: int,
    height: int,
    angle_radians=0.0,
    label=None,
    color=(0, 255, 0),
    thickness=2,
    label_gap=10,
    font_scale=0.6,
    text_thickness=1,
    show_arrow: bool = True,
):
    """
    Draw a rotated rectangle, optional horizontal label, and direction arrow.

    Parameters
    ----------
    image : np.ndarray
        OpenCV image, modified in place.
    centerpoint : tuple[float, float]
        Rectangle center (x, y), in image pixels.
    width, height : float
        Rectangle dimensions in pixels. At angle=0, width is horizontal
        and height is vertical.
    angle_radians : float, optional
        Counterclockwise rotation from the positive x-axis.
        0 points right; pi/2 points up.
    label : str or None, optional
        Unrotated text placed to the right of the rotated rectangle.
    color : tuple, optional
        BGR color used for the rectangle, arrow, and label.
    thickness : int, optional
        Rectangle and arrow line thickness.
    label_gap : int, optional
        Horizontal spacing between the rectangle and label, in pixels.
    font_scale : float, optional
        Label font scale.
    text_thickness : int, optional
        Label stroke thickness.

    Returns
    -------
    np.ndarray
        The same image, with the drawing applied.

    Notes
    -----
    The arrow starts at the center and has a center-to-tip length equal
    to height, subject to pixel rounding. Its direction is angle_radians.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if thickness < 1 or text_thickness < 1:
        raise ValueError("Line thicknesses must be positive integers.")

    cx = center.x
    cy = center.y
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)

    # Local corners before rotation.
    corners = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ],
        dtype=np.float64,
    )

    # Image y increases downward, so positive CCW angles need this matrix.
    rotation = np.array(
        [
            [c, s],
            [-s, c],
        ]
    )

    rotated_corners = corners @ rotation.T + np.array([cx, cy])
    points = np.rint(rotated_corners).astype(np.int32)

    cv2.polylines(
        frame,
        [points.reshape(-1, 1, 2)],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Direction vector in image coordinates: (cos(angle), -sin(angle)).
    arrow_start = (int(round(cx)), int(round(cy)))
    arrow_end = (
        int(round(cx + height * c)),
        int(round(cy - height * s)),
    )

    if show_arrow:
        cv2.arrowedLine(
            frame,
            arrow_start,
            arrow_end,
            color,
            thickness=thickness,
            line_type=cv2.LINE_AA,
            tipLength=0.2,
        )

    if label is not None and str(label):
        text = str(label)
        font = cv2.FONT_HERSHEY_SIMPLEX

        (_, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, text_thickness
        )

        # Position outside the rotated rectangle's rightmost extent.
        text_x = int(np.ceil(rotated_corners[:, 0].max())) + label_gap
        text_y = int(round(cy + (text_height - baseline) / 2))

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return frame


def show_field(camera_frame: np.ndarray, bot: OurRobot) -> list[np.ndarray]:

    output_frames_list = []

    ### Exame the camera feed and get the list of detections
    detect_list = get_detections_from_frame(camera_frame, bot)
    camera_frame = draw_detection_list_to_frame(camera_frame, detect_list)

    ### Show to the user with the top down GUI
    if USE_DRAWN_FIELD:
        field_frame = "field_pngs/empty_field.png"
    else:
        field_frame = "field_pngs/black_field.png"
    field_frame = load_image_as_frame(field_frame)

    field_frame = draw_detection_list_to_field(
        field_frame=field_frame, detect_list=detect_list, robot=bot
    )

    field_frame = draw_robot_to_field(field_frame=field_frame, bot=bot)

    combined_frame = show_frames_side_by_side(camera_frame, field_frame)

    output_frames_list.append(combined_frame)
    return output_frames_list


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    return frames


def save_frames_as_video(
    frames,
    output_path,
    fps=30,
    codec="mp4v",
    frame_size=None,
):
    """
    Save a list of BGR frames as a video.

    Args:
        frames: List/array of frames, each shaped (H, W, 3) in BGR format.
        output_path: Output video file path, e.g. "output.mp4".
        fps: Frames per second.
        codec: FourCC codec. Use "mp4v" for .mp4 or "XVID" for .avi.
        frame_size: Optional (width, height). If None, inferred from the first frame.

    Returns:
        Path to the saved video.

    Raises:
        ValueError: If frames is empty or frames have inconsistent sizes.
    """
    if not frames:
        raise ValueError("frames must contain at least one frame")

    if frame_size is None:
        height, width = frames[0].shape[:2]
        frame_size = (width, height)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for {output_path}. "
            "Try codec='XVID' with an .avi output file."
        )

    try:
        for i, frame in enumerate(frames):
            if frame.shape[1::-1] != frame_size:
                raise ValueError(
                    f"Frame {i} has size {frame.shape[1::-1]}, expected {frame_size}"
                )
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def show_video_from_list_of_frames(frames, window_name="Video", fps=30, wait_key=1):
    if not frames:
        raise ValueError("No frames to display.")

    delay = max(1, int(1000 / fps))

    for frame in frames:
        cv2.imshow(window_name, frame)

        # Press 'q' to stop early
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def show_photo_from_frame(frame: np.ndarray, window_name="Default"):

    cv2.imshow(window_name, mat=frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_and_show_video(video_path: str, save_to_csv: bool = True):
    our_bot = OurRobot()
    our_bot.field_pos_px = convert_to_field_pixels_global(our_bot.field_pos_in)

    print("loading video frames")
    video_frames = load_video_frames(video_path)

    print("loading field frame")
    field_frame = load_image_as_frame("field_pngs/empty_field.png")

    print("calculating and showing output to user")

    field_list_of_frames = []

    for camera_frame in video_frames:
        camera_frame : np.ndarray
        camera_frame_cop = camera_frame.copy()
        field_frame_copy = field_frame.copy()

        side_by_side_frame = get_side_by_side_frame(camera_frame_cop, field_frame_copy)
        field_list_of_frames.append(side_by_side_frame)

    print("preparing to show video")
    show_video_from_list_of_frames(field_list_of_frames)



def get_side_by_side_frame(camera_frame: np.ndarray, field_frame : np.ndarray):
    our_bot = OurRobot()
    our_bot.field_pos_px = convert_to_field_pixels_global(our_bot.field_pos_in)

    detection_list = get_detections_from_frame(
        bot=our_bot,
        frame=camera_frame,
    )

    updated_cam_frame = draw_detection_list_to_frame(
        frame=camera_frame, detect_list=detection_list
    )

    updated_field_frame = draw_robot_to_field(field_frame=field_frame, bot=our_bot)
    updated_field_frame = draw_detection_list_to_field(
        field_frame=field_frame, detect_list=detection_list, bot=our_bot
    )

    return show_frames_side_by_side(updated_cam_frame, updated_field_frame)
    

if __name__ == "__main__":
    # example_photo = "training_data/tennis_right.png"
    # example_frame = load_image_as_frame(example_photo)
    # frame = get_side_by_side_frame(example_frame)
    # show_photo_from_frame(frame)

    example_video = "training_data/tennis_ball_with_corners.mp4"
    load_and_show_video(example_video)



