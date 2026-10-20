import time

import cv2
import numpy as np


def create_black_background(width, height):
    """Return a black BGR image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def update_ball_position(position, target, speed, dt):
    """Move toward target at speed pixels/second without overshooting."""
    displacement = target - position
    distance = np.linalg.norm(displacement)
    step = speed * dt

    if distance <= step:
        return target.copy()

    return position + displacement / distance * step


def draw_blue_x(image, position, size=12, thickness=2):
    """Draw a blue X centered on position, modifying image in place."""
    x, y = map(lambda value: int(round(value)), position)

    cv2.line(
        image,
        (x - size, y - size),
        (x + size, y + size),
        (255, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (x - size, y + size),
        (x + size, y - size),
        (255, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def run_ball_demo(width=800, height=600, ball_radius=20, speed=250.0):
    """Display a red ball that moves toward each left-click position."""
    window_name = "Click to move the ball"
    background = create_black_background(width, height)

    position = np.array([width / 2, height / 2], dtype=float)
    target = position.copy()
    clicked_point = None

    def on_mouse(event, x, y, flags, param):
        nonlocal target, clicked_point

        if event == cv2.EVENT_LBUTTONDOWN:
            target = np.array([x, y], dtype=float)
            clicked_point = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    previous_time = time.perf_counter()

    try:
        while True:
            current_time = time.perf_counter()
            dt = current_time - previous_time
            previous_time = current_time

            position = update_ball_position(position, target, speed, dt)

            # Start with a fresh background so movement leaves no trail.
            frame = background.copy()

            center = tuple(np.rint(position).astype(int))
            cv2.circle(
                frame,
                center,
                ball_radius,
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            # Draw last so the X stays visible when the ball reaches it.
            if clicked_point is not None:
                draw_blue_x(frame, clicked_point)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_ball_demo()