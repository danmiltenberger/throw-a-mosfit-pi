import cv2
import numpy as np
import math
from icecream import ic
from settings_view import SHOW_ARROWS
from settings_run import search_for_dict, MIN_AREA, MIN_CIRCLARITY, BALL_RED
from tam_library_classes import Detection, Point, OurRobot
from tam_library_colors import BLUE, WHITE
from tam_library import (
    search_for_list,
    load_hsv_bounds,
    get_best_contour,
    CIRCLE,
    RECT,
    SCREEN_CENTER_X,
    SCREEN_CENTER_Y,
    FOCAL_LENGTH,
    convert_to_px,
    KNOWN_WIDTH_IN,
    field_center_px_x,
    field_center_px_y,
    draw_labeled_arrow,
    load_image,
    show_frames_side_by_side,
    
)

from settings_run import SHAPE_TO_DRAW, DRAW_COLOR, DRAW_THICKNESS, USE_DRAWN_FIELD

# let's update the codebase with rigorous object oriented programming


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


def get_detections_from_frame(frame: np.ndarray) -> list[Detection]:
    "from the input frame, search for the objects as specified in the run_settings.py search_list"
    "this file DOES NOT convert the camera view to the global positions, just returns a list of what it sees in pixels"

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
            this_detect.center_px = center_point
            this_detect.width_px = int(radius * 2)

            this_detect.area = int(np.pi * radius * radius)

        elif object_dict["shape"] == RECT:
            left_x, top_y, width, height = cv2.boundingRect(best_contour)

            x_center = int(left_x + width / 2)
            y_center = int(top_y + height / 2)
            center_point = Point(x=x_center, y=y_center)
            this_detect.center_px = center_point
            this_detect.width_px = width
            this_detect.area = int(width * height)

        detect_list.append(this_detect)

    return detect_list


def get_detection_global_pos(this_detect: Detection, robot: OurRobot) -> Detection:
    """
    Convert what the camera sees to a range and azimuth. Then, convert that from the robot's frame to the global frame
    """

    def calc_angle_rad_from_point(point: Point) -> float:
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

        return math.atan2(delta_x, FOCAL_LENGTH) + np.pi

    def calc_distance_from_bot() -> float:
        known_width = search_for_dict[this_detect.name][KNOWN_WIDTH_IN]
        percieved_width = this_detect.width_px
        Z = (known_width * FOCAL_LENGTH) / percieved_width
        return Z

    def convert_pos_and_ang_to_robot_in() -> Point:

        # get the detection's global position in inches
        object_x = robot.field_pos_in.x + this_detect.range * math.cos(
            this_detect.azimuth_rad + robot.azimuth_rad
        )

        object_y = float(robot.field_pos_in.y) + float(this_detect.range) * math.sin(
            float(this_detect.azimuth_rad) + float(robot.azimuth_rad)
        )

        return Point(object_x, object_y)

    # get the range and azimuth from the robot
    this_detect.range = calc_distance_from_bot()
    this_detect.azimuth_rad = calc_angle_rad_from_point(this_detect.screen_center_px)

    # convert to the global position in inches
    this_detect.field_pos_in = convert_pos_and_ang_to_robot_in()

    # convert to the global position in pixels
    this_detect.field_pos_px = convert_to_field_pixels_global(this_detect.field_pos_in)

    return this_detect


def draw_detection_list_to_frame(
    frame: np.ndarray, detect_list: list[Detection]
) -> np.ndarray:
    for detection in detect_list:

        shape = search_for_dict[detection.name][SHAPE_TO_DRAW]
        color = search_for_dict[detection.name][DRAW_COLOR]

        shape_center_x = detection.center_px.x
        shape_center_y = detection.center_px.y
        shape_center = (int(shape_center_x), int(shape_center_y))

        # draw a small dot at the center
        frame = cv2.circle(
            img=frame,
            center=shape_center,
            radius=3,
            color=color,
            thickness=-1,
        )

        # draw a circle around the object
        if shape == CIRCLE:
            radius = int(detection.width_px / 2)
            frame = cv2.circle(
                img=frame,
                radius=int(radius),
                center=shape_center,
                color=color,
                thickness=DRAW_THICKNESS,
            )

        # or draw a square
        elif shape == RECT:

            frame = draw_centered_rectangle(
                frame=frame,
                color=color,
                width_px=detection.width_px,
                height_px=detection.width_px,
                center=detection.center_px,
            )

    return frame


def draw_detection_list_to_field(
    field_frame: np.ndarray, detect_list: list[Detection], robot: OurRobot
) -> np.ndarray:
    for detection in detect_list:
        draw_shape = search_for_dict[detection.name][SHAPE_TO_DRAW]
        draw_color = search_for_dict[detection.name][DRAW_COLOR]
        known_width_in = search_for_dict[detection.name][KNOWN_WIDTH_IN]
        known_width_px = convert_to_px(known_width_in)

        detection = get_detection_global_pos(detection, robot)

        if draw_shape == CIRCLE:
            cv2.circle(
                img=field_frame,
                center=(int(detection.field_pos_px.x), int(detection.field_pos_px.y)),
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


def _draw_rotated_label(
    frame,
    text: str,
    center,
    angle_rad: float,
    color=(0, 255, 0),
    font_scale: float = 0.5,
    thickness: int = 1,
    padding: int = 4,
):
    """Draw text rotated to align with `angle_rad`.

    For an arrow pointing straight up, the text reads top-to-bottom,
    like the spine of a book.
    """
    if not text:
        return frame

    text = str(text)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Render the text once on a temporary image to measure it.
    # This avoids OpenCV 5's changed getTextSize() API.
    temp = np.zeros((200, 600, 3), dtype=np.uint8)
    temp_org = (20, 100)

    cv2.putText(
        temp,
        text,
        temp_org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    ys, xs = np.nonzero(temp[:, :, 0])
    if len(xs) == 0 or len(ys) == 0:
        return frame

    text_w = int(xs.max() - xs.min() + 1)
    text_h = int(ys.max() - ys.min() + 1)

    # Create a tight transparent canvas containing only the text.
    canvas_w = text_w + 2 * padding
    canvas_h = text_h + 2 * padding

    text_img = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    cv2.putText(
        text_img,
        text,
        (padding, padding + text_h),
        font,
        font_scale,
        (255, 255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Rotate the text canvas.
    # A straight-up arrow has angle_rad = -pi/2 in image coordinates.
    # Adding 90 degrees makes the label read top-to-bottom, like a book spine.
    text_angle_deg = math.degrees(angle_rad) + 90.0

    rotation_matrix = cv2.getRotationMatrix2D(
        (canvas_w / 2.0, canvas_h / 2.0),
        text_angle_deg,
        1.0,
    )

    # Make the output canvas large enough to contain the rotated text.
    diag = int(math.ceil(math.hypot(canvas_w, canvas_h)))
    rotated = cv2.warpAffine(
        text_img,
        rotation_matrix,
        (diag, diag),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Position the rotated label centered at the requested point.
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0 = cx - diag // 2
    y0 = cy - diag // 2
    x1 = x0 + diag
    y1 = y0 + diag

    # Clip the label to the frame boundaries.
    frame_h, frame_w = frame.shape[:2]

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    src_x1 = diag - max(0, x1 - frame_w)
    src_y1 = diag - max(0, y1 - frame_h)

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame_w, x1)
    y1 = min(frame_h, y1)

    if x1 <= x0 or y1 <= y0:
        return frame

    roi = frame[y0:y1, x0:x1]
    label_roi = rotated[src_y0:src_y1, src_x0:src_x1]

    # Alpha-blend the rotated label onto the frame.
    alpha = label_roi[:, :, 3:4].astype(float) / 255.0
    label_bgr = label_roi[:, :, :3].astype(float)

    # Convert the white rendered text to the requested BGR color.
    label_bgr[..., 0] *= color[0] / 255.0
    label_bgr[..., 1] *= color[1] / 255.0
    label_bgr[..., 2] *= color[2] / 255.0

    frame[y0:y1, x0:x1] = (
        alpha * label_bgr + (1.0 - alpha) * roi.astype(float)
    ).astype(frame.dtype)

    return frame


def draw_arrow(
    frame,
    start: Point,
    distance: float | None = None,
    angle_rad: float | None = None,
    end: Point | None = None,
    label: str | None = None,
    color=(0, 255, 0),
    thickness: int = 2,
    tip_size: int = 10,
    label_offset_px: int = 12,
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
        label: Optional text drawn beside the arrow.
        color: BGR color.
        thickness: Line thickness.
        tip_size: Arrowhead size in pixels.
        label_offset_px: Perpendicular distance from the arrow spine
            to the center of the label.

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

    if label:
        # Midpoint of the arrow
        mx = (sx + ex) / 2.0
        my = (sy + ey) / 2.0

        # Perpendicular offset: rotate arrow direction by +90 degrees
        offset_x = -math.sin(angle_rad) * label_offset_px
        offset_y = math.cos(angle_rad) * label_offset_px

        label_center = (mx + offset_x, my + offset_y)

        frame = _draw_rotated_label(
            frame=frame,
            text=label,
            center=label_center,
            angle_rad=angle_rad,
            color=color,
        )

    return frame


def show_field(camera_frame: np.ndarray, show_field : bool = True, show_cam : bool = True, show_one_frame: bool = True) -> list[np.ndarray]:

    output_frames_list = []

    ### Exame the camera feed and get the list of detections
    detect_list = get_detections_from_frame(camera_frame)
    camera_frame = draw_detection_list_to_frame(camera_frame, detect_list)

    ### Show to the user with the top down GUI
    if USE_DRAWN_FIELD:
        field_frame = "field_pngs/empty_field.png"
    else:
        field_frame = "field_pngs/black_field.png"
    field_frame = load_image(field_frame)

    our_bot = OurRobot()
    our_bot.field_pos_px = convert_to_field_pixels_global(our_bot.field_pos_in)

    field_frame = draw_detection_list_to_field(
        field_frame=field_frame, detect_list=detect_list, robot=our_bot
    )

    field_frame = draw_robot_to_field(field_frame=field_frame, bot=our_bot)


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


def show_video(frames, window_name="Video", fps=30, wait_key=1):
    if not frames:
        raise ValueError("No frames to display.")

    delay = max(1, int(1000 / fps))

    for frame in frames:
        cv2.imshow(window_name, frame)

        # Press 'q' to stop early
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()




if __name__ == "__main__":
    example_photo = "training_data/2026-10-11T20_53_06.567733.png"
    example_video = "training_data/tennis_ball_with_corners.mp4"

    print("loading video frames")
    video_frames = load_video_frames(example_video)

    print("calculating and showing output to user")

    field_list_of_frames = []

    for frame in video_frames:
        output_list = show_field(frame)
        field_list_of_frames.append(output_list[0])
        

    show_video(field_list_of_frames)
    