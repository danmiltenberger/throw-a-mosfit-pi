import time

import cv2
import numpy as np
from draw_field import Rectangle
from opencv_colors import *


def bounce_ball(
    rectangle: Rectangle,
    chamfer: int,
    frame_width: int = 640,
    frame_height: int = 480,
    y: int = 0,
    radius: int = 12,
    velocity: tuple = (180.0, 130.0),
    ball_color: tuple = (0, 0, 255),
    wall_color: tuple = (255, 255, 255),
    fps: int = 60,
):
    """Animate a ball bouncing inside a centered chamfered octagon.

    rectangle: Bounding rectangle; its x and y fields are ignored.
    chamfer: Corner-cut distance along each adjoining edge, in pixels.
    y: Vertical offset of the octagon from the frame center.
    radius: Ball radius in pixels.
    velocity: Initial (vx, vy) in pixels per second.
    fps: Target display rate.

    Press Q or Escape, or close the window, to stop.
    """
    width, height = rectangle.width, rectangle.height

    if not (0 < width <= frame_width and 0 < height <= frame_height):
        raise ValueError("Rectangle dimensions must be positive and fit the frame.")

    if not isinstance(chamfer, (int, np.integer)):
        raise TypeError("Chamfer must be an integer number of pixels.")

    if not (0 < 2 * chamfer < min(width - 1, height - 1)):
        raise ValueError("Chamfer must leave all eight sides nonzero in length.")

    if not isinstance(radius, (int, np.integer)) or radius <= 0:
        raise ValueError("Radius must be a positive integer.")

    if fps <= 0:
        raise ValueError("FPS must be positive.")

    left = (frame_width - width) // 2
    top = (frame_height - height) // 2 + y
    right = left + width - 1
    bottom = top + height - 1

    if top < 0 or bottom >= frame_height:
        raise ValueError("Vertical offset places the octagon outside the frame.")

    vertices = np.array(
        [
            (left + chamfer, top),
            (right - chamfer, top),
            (right, top + chamfer),
            (right, bottom - chamfer),
            (right - chamfer, bottom),
            (left + chamfer, bottom),
            (left, bottom - chamfer),
            (left, top + chamfer),
        ],
        dtype=np.float64,
    )

    # Build inward-facing unit normals for all eight walls.
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    position = vertices.mean(axis=0)
    velocity = np.asarray(velocity, dtype=np.float64).copy()

    if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
        raise ValueError("Velocity must contain two finite numbers.")

    # Each wall requires the ball center to be at least radius pixels inside.
    initial_distances = np.sum((position - vertices) * normals, axis=1)
    if np.any(initial_distances <= radius):
        raise ValueError("The ball is too large to fit inside the octagon.")

    background = np.zeros(
        (frame_height, frame_width, 3), dtype=np.uint8
    )
    cv2.polylines(
        background,
        [vertices.astype(np.int32).reshape(-1, 1, 2)],
        isClosed=True,
        color=wall_color,
        thickness=1,
    )

    window_name = "Bouncing Ball"
    cv2.namedWindow(window_name)

    epsilon = 1e-7
    previous_time = time.perf_counter()

    try:
        while True:
            frame_start = time.perf_counter()
            remaining = min(frame_start - previous_time, 0.05)
            previous_time = frame_start

            # Resolve collisions in chronological order within this frame.
            while remaining > 1e-10:
                clearances = (
                    np.sum((position - vertices) * normals, axis=1) - radius
                )
                normal_speeds = normals @ velocity

                hit_times = np.full(8, np.inf)
                approaching = normal_speeds < -1e-10
                hit_times[approaching] = (
                    np.maximum(clearances[approaching], 0.0)
                    / -normal_speeds[approaching]
                )

                first_hit = np.min(hit_times)

                if first_hit > remaining:
                    position += velocity * remaining
                    break

                position += velocity * first_hit
                remaining -= first_hit

                # At a corner, multiple walls may be contacted simultaneously.
                hit_walls = np.flatnonzero(
                    np.abs(hit_times - first_hit) <= 1e-9
                )

                # Reflect against contacted walls until motion points inward.
                for _ in range(16):
                    outward = normals[hit_walls] @ velocity
                    worst = np.argmin(outward)

                    if outward[worst] >= -1e-10:
                        break

                    normal = normals[hit_walls[worst]]
                    velocity -= 2.0 * np.dot(velocity, normal) * normal

                # Move slightly inward to avoid repeated zero-time collisions.
                position += epsilon * normals[hit_walls].sum(axis=0)

            frame = background.copy()
            center = tuple(np.rint(position).astype(int))

            cv2.circle(
                frame,
                center,
                radius,
                ball_color,
                thickness=-1,
            )
            cv2.imshow(window_name, frame)

            processing_time = time.perf_counter() - frame_start
            delay_ms = max(
                1, int(1000 * (1.0 / fps - processing_time))
            )
            key = cv2.waitKey(delay_ms) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    bounce_ball(
        rectangle=Rectangle(width=500, height=360),
        chamfer=70,
        radius=15,
        velocity=(220.0, 160.0),
        ball_color=RED,
        wall_color=WHITE,
    )