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

    object_y = float(robot.field_pos_in.y) + float(this_detect.range) * math.sin(
        float(this_detect.azimuth_rad) + float(robot.azimuth_rad)
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

    return math.atan2(delta_x, FOCAL_LENGTH)


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
            frame = draw_centered_rectangle(
                frame=frame,
                color=color,
                width_px=detection.width_px,
                height_px=detection.width_px,
                center=shape_center,
            )

        # Draw a label above the object
        # Use "deg" rather than a degree symbol for OpenCV font compatibility.
        label = f"Range: {range_to_object:.2f} in | " f"Azim: {azimuth_deg:.1f} deg"

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
) -> np.ndarray:

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

        keep_on_field()

        ic(draw_shape, draw_color, detection.field_pos_px)

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
            field_frame = draw_centered_rectangle(
                frame=field_frame,
                color=draw_color,
                width_px=detection.width_px,
                height_px=detection.width_px,
                center=detection.field_pos_px,
            )

    return field_frame


def draw_centered_rectangle(
    frame,
    center,
    width_px: int,
    height_px: int,
    angle_rad: float = 0.0,
    color=(0, 255, 0),
    thickness: int = 2,
):
    """Draw a rectangle centered at `center`, optionally rotated.

    Args:
        frame: BGR image to draw on.
        center: Point with `.x` and `.y`, or an `(x, y)` tuple/list.
        width_px: Rectangle width in pixels.
        height_px: Rectangle height in pixels.
        angle_rad: Rotation angle in radians. Positive angles rotate
            clockwise in OpenCV image coordinates.
        color: BGR color.
        thickness: Line thickness; use -1 for a filled rectangle.

    Returns:
        The modified frame.
    """
    if hasattr(center, "x") and hasattr(center, "y"):
        cx, cy = float(center.x), float(center.y)
    else:
        cx, cy = float(center[0]), float(center[1])

    # Unrotated corners relative to the center
    half_w = width_px / 2.0
    half_h = height_px / 2.0

    corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = []
    for dx, dy in corners:
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        rotated.append((int(round(x)), int(round(y))))

    pts = np.array(rotated, dtype=np.int32)

    if thickness == -1:
        cv2.fillPoly(frame, [pts], color)
    else:
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    return frame


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
    field_frame = draw_centered_rectangle(
        width_px=bot_width_px,
        height_px=bot_width_px,
        frame=field_frame,
        color=bot.color,
        center=bot.field_pos_px,
        angle_rad=bot.azimuth_rad,
    )

    # draw an arrow showing which direction the bot is facing
    field_frame = draw_arrow(
        frame=field_frame,
        start=bot.field_pos_px,
        color=bot.color,
        angle_rad=bot.azimuth_rad,
        distance=bot_width_px * 2,
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


def load_and_show_video(example_video: str):
    our_bot = OurRobot()
    our_bot.field_pos_px = convert_to_field_pixels_global(our_bot.field_pos_in)

    print("loading video frames")
    video_frames = load_video_frames(example_video)

    print("loading field frame")
    field_frame = load_image_as_frame("field_pngs/empty_field.png")

    print("calculating and showing output to user")

    field_list_of_frames = []

    for camera_frame in video_frames:

        detection_list = get_detections_from_frame(
            bot=our_bot,
            frame=camera_frame,
        )
        # ic(detection_list)

        updated_cam_frame = draw_detection_list_to_frame(
            frame=camera_frame, detect_list=detection_list
        )

        updated_field_frame = draw_robot_to_field(field_frame=field_frame, bot=our_bot)
        updated_field_frame = draw_detection_list_to_field(
            field_frame=field_frame, detect_list=detection_list, robot=our_bot
        )

        side_by_side_frame = show_frames_side_by_side(
            updated_cam_frame, updated_field_frame
        )

        field_list_of_frames.append(side_by_side_frame)

    print("preparing to show video")
    show_video_from_list_of_frames(field_list_of_frames)


def load_and_show_photo(example_photo: str, is_side_by_side: bool = True):

    camera_frame = load_image_as_frame(example_photo)
    our_bot = OurRobot()
    our_bot.field_pos_px = convert_to_field_pixels_global(our_bot.field_pos_in)

    field_frame = load_image_as_frame("field_pngs/empty_field.png")

    detection_list = get_detections_from_frame(
        bot=our_bot,
        frame=camera_frame,
    )


    updated_cam_frame = draw_detection_list_to_frame(
        frame=camera_frame, detect_list=detection_list
    )

    updated_field_frame = draw_robot_to_field(field_frame=field_frame, bot=our_bot)
    updated_field_frame = draw_detection_list_to_field(
        field_frame=field_frame, detect_list=detection_list
    )

    if is_side_by_side:
        side_by_side_frame = show_frames_side_by_side(
            updated_cam_frame, updated_field_frame
        )

        show_photo_from_frame(side_by_side_frame)

    else:
        show_photo_from_frame(updated_field_frame)





if __name__ == "__main__":
    example_photo = "training_data/tennis_ball_48in.png"


    load_and_show_photo(example_photo, is_side_by_side=False)
