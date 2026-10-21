import cv2
import numpy as np
import math
from settings_run import search_for_dict, search_for_list, MIN_AREA, MIN_CIRCLARITY, KNOWN_WIDTH_IN, FOCAL_LENGTH
from pathlib import Path
from tam_library_classes import Detection, Point, OurRobot, FieldState
from tam_library_field import field_center_px_x, field_center_px_y, playing_field_in_x, playing_field_in_y, convert_to_px, RADIUS_BALL, RADIUS_ROBOT
from tam_library_camera_utils import SCREEN_CENTER_X, convert_to_field_pixels_global, get_detections_from_frame
from tam_library_colors import *
from settings_run import SHAPE_TO_DRAW, DRAW_COLOR, DRAW_THICKNESS, RECT, CIRCLE, OUR_TEAM_COLOR, THEIR_TEAM_COLOR


### Draw things to the frames
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


def show_frames_side_by_side(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    target_height: int | None = None,
    window_name: str = "Side-by-Side Frames",
) -> np.ndarray:
    """Display two BGR frames side by side at the same height.

    If target_height is None, use the right frame's original height.
    Each frame's aspect ratio is preserved.
    """
    for name, frame in (("frame_left", frame_left), ("frame_right", frame_right)):
        if frame is None or frame.size == 0:
            raise ValueError(f"{name} must be a non-empty image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{name} must be a three-channel BGR image.")

    if frame_left.dtype != frame_right.dtype:
        raise ValueError("Both frames must have the same data type.")

    if target_height is None:
        target_height = frame_right.shape[0]
    if target_height <= 0:
        raise ValueError("target_height must be positive.")

    def resize_to_height(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == target_height:
            return frame

        scale = target_height / height
        new_width = max(1, round(width * scale))
        return cv2.resize(
            frame,
            (new_width, target_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

    combined = np.hstack((resize_to_height(frame_left), resize_to_height(frame_right)))
    # cv2.imshow(window_name, combined)
    return combined


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


def draw_notched_rectangle(
    image : np.ndarray,
    center_x : int,
    top_y : int,
    width : int,
    height : int,
    base_length : int,
    angle_deg : int,
    color=(255, 255, 255),
    rotation_deg=0.0,
    border_thickness=1,
) -> np.ndarray:
    """
    Draw a filled, rotated rectangle with a V-notch and black border.

    Parameters
    ----------
    image : np.ndarray
        Destination image; modified in place.
    center_x, top_y : float
        Horizontal center and top edge BEFORE rotation, in pixels.
    width, height : float
        Rectangle dimensions before rotation, in pixels.
    base_length : float
        Width of the triangular opening along the unrotated top edge.
    angle_deg : float
        Interior angle at the bottom tip of the V, in degrees.
    color : tuple or int
        OpenCV BGR fill color, or a scalar for a grayscale image.
    rotation_deg : float
        Rotation about the rectangle's center, in degrees.
        Positive = counterclockwise as viewed on the image.
    border_thickness : int
        Black border width in pixels; must be at least 1.

    Returns
    -------
    np.ndarray
        The modified input image.
    """
    values = (
        center_x, top_y, width, height,
        base_length, angle_deg, rotation_deg,
    )
    if not all(np.isfinite(v) for v in values):
        raise ValueError("All geometry parameters must be finite.")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if not 0 < base_length < width:
        raise ValueError("base_length must be between 0 and width.")
    if not 0 < angle_deg < 180:
        raise ValueError("angle_deg must be between 0 and 180.")
    if (
        not isinstance(border_thickness, (int, np.integer))
        or border_thickness < 1
    ):
        raise ValueError("border_thickness must be a positive integer.")

    notch_depth = (base_length / 2) / np.tan(
        np.deg2rad(angle_deg / 2)
    )
    if notch_depth >= height:
        raise ValueError("The notch must be shallower than the rectangle.")

    left = center_x - width / 2
    right = center_x + width / 2
    bottom = top_y + height

    vertices = np.array([
        [left,                       top_y],
        [center_x - base_length / 2,  top_y],
        [center_x,                   top_y + notch_depth],
        [center_x + base_length / 2,  top_y],
        [right,                      top_y],
        [right,                      bottom],
        [left,                       bottom],
    ], dtype=np.float64)

    pivot = (float(center_x), float(top_y + height / 2))
    matrix = cv2.getRotationMatrix2D(
        pivot, float(rotation_deg), 1.0
    )
    rotated_vertices = vertices @ matrix[:, :2].T + matrix[:, 2]

    # Retain subpixel geometry until rasterization.
    shift = 8
    points = np.rint(
        rotated_vertices * (1 << shift)
    ).astype(np.int32)

    cv2.fillPoly(
        image,
        [points],
        color=color,
        lineType=cv2.LINE_AA,
        shift=shift,
    )

    black = 0 if image.ndim == 2 else (0,) * image.shape[2]

    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=black,
        thickness=int(border_thickness),
        lineType=cv2.LINE_AA,
        shift=shift,
    )

    return image


import cv2
import numpy as np


def draw_outlined_circle(
    image,
    center_x,
    center_y,
    radius,
    color=(255, 255, 255),
    border_thickness=1,
):
    """
    Draw a filled circle with a thin black outline.

    Parameters
    ----------
    image : np.ndarray
        Destination image; modified in place.
    center_x, center_y : float
        Circle center in image coordinates, in pixels.
    radius : float
        Circle radius in pixels; must be positive.
    color : tuple or int
        OpenCV BGR fill color, or a scalar for a grayscale image.
    border_thickness : int
        Black outline width in pixels; must be at least 1.

    Returns
    -------
    np.ndarray
        The modified input image.
    """
    if not all(np.isfinite(v) for v in (center_x, center_y, radius)):
        raise ValueError("Center coordinates and radius must be finite.")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if (
        not isinstance(border_thickness, (int, np.integer))
        or border_thickness < 1
    ):
        raise ValueError("border_thickness must be a positive integer.")

    # Preserve subpixel coordinates until rasterization.
    shift = 8
    scale = 1 << shift
    center = (
        int(round(center_x * scale)),
        int(round(center_y * scale)),
    )
    scaled_radius = int(round(radius * scale))

    cv2.circle(
        image,
        center,
        scaled_radius,
        color=color,
        thickness=cv2.FILLED,
        lineType=cv2.LINE_AA,
        shift=shift,
    )

    black = 0 if image.ndim == 2 else (0,) * image.shape[2]

    cv2.circle(
        image,
        center,
        scaled_radius,
        color=black,
        thickness=int(border_thickness),
        lineType=cv2.LINE_AA,
        shift=shift,
    )

    return image


def draw_field(field_state: FieldState, field_frame: np.ndarray) -> np.ndarray:

    # draw our robot
    if field_state.draw_our_bot:
        field_frame = draw_notched_rectangle(
                field_frame,
                center_x=convert_to_field_pixels_global(field_state.our_bot_pos).x,
                top_y=convert_to_field_pixels_global(field_state.our_bot_pos).y - RADIUS_ROBOT,
                width=RADIUS_ROBOT * 2,
                height=RADIUS_ROBOT * 2,
                base_length=RADIUS_BALL * 2,
                angle_deg=90,
                color=OUR_TEAM_COLOR,
                rotation_deg=math.degrees(field_state.our_bot_bearing)-90,
                border_thickness=1,
            )

    # draw their robot
    if field_state.draw_their_bot:
        field_frame = draw_notched_rectangle(
            field_frame,
            center_x=convert_to_field_pixels_global(field_state.their_bot_pos).x,
            top_y=convert_to_field_pixels_global(field_state.their_bot_pos).y - RADIUS_ROBOT,
            width=RADIUS_ROBOT * 2,
            height=RADIUS_ROBOT * 2,
            base_length=RADIUS_BALL * 2,
            angle_deg=90,
            color=THEIR_TEAM_COLOR,
            rotation_deg=math.degrees(field_state.their_bot_bearing)-90,
            border_thickness=1,
        )

    # draw ball
    if field_state.draw_ball:
        field_frame = draw_outlined_circle(
            image= field_frame,
            center_x= convert_to_field_pixels_global(field_state.ball_position).x,
            center_y= convert_to_field_pixels_global(field_state.ball_position).y,
            color= RED,
            radius= RADIUS_BALL,
        )
    
    
    return field_frame
