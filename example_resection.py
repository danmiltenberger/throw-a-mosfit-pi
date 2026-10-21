import math

import cv2
import numpy as np


WIDTH, HEIGHT = 1000, 700

# Corner pixels are (0, 0) and (WIDTH - 1, 0).
LANDMARK_A = np.array([10.0, 10.0])
LANDMARK_B = np.array([WIDTH - 11.0, 10.0])


def azimuth_to_point(observer, target):
    """Return observer-to-target azimuth in degrees, clockwise from up."""
    dx, dy = np.asarray(target, dtype=float) - np.asarray(
        observer, dtype=float
    )

    if math.hypot(dx, dy) < 1e-12:
        raise ValueError("Azimuth is undefined at the landmark itself.")

    # North is negative screen y.
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def observer_from_azimuths(
    azimuth_a,
    azimuth_b,
    landmark_a=LANDMARK_A,
    landmark_b=LANDMARK_B,
):
    """
    Recover observer (x, y) from two observer-to-landmark azimuths.

    Angles are degrees clockwise from screen north.
    Returns floating-point coordinates, not rounded pixel coordinates.

    Raises ValueError for parallel/near-parallel bearings or bearings
    whose forward directions are inconsistent with the intersection.
    """
    angles = np.asarray([azimuth_a, azimuth_b], dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Azimuths must be finite numbers.")

    a = np.asarray(landmark_a, dtype=float)
    b = np.asarray(landmark_b, dtype=float)

    # Unit vectors pointing FROM observer TO each landmark.
    theta_a, theta_b = np.radians(angles % 360.0)
    direction_a = np.array([np.sin(theta_a), -np.cos(theta_a)])
    direction_b = np.array([np.sin(theta_b), -np.cos(theta_b)])

    # A = observer + distance_a * direction_a
    # B = observer + distance_b * direction_b
    #
    # Therefore:
    # distance_a * direction_a - distance_b * direction_b = A - B
    matrix = np.column_stack((direction_a, -direction_b))

    if abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError(
            "Bearings are parallel or nearly parallel; "
            "no stable unique position."
        )

    distance_a, distance_b = np.linalg.solve(matrix, a - b)

    if distance_a < -1e-7 or distance_b < -1e-7:
        raise ValueError(
            "The bearings intersect behind an observer-to-landmark ray."
        )

    observer = a - distance_a * direction_a
    return float(observer[0]), float(observer[1])


def pixel(point):
    """Convert a floating-point point to an OpenCV pixel tuple."""
    return tuple(int(round(value)) for value in point)


def draw_text(image, text, position, color=(230, 230, 230), scale=0.6):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def main():
    rng = np.random.default_rng()
    window = "Two-bearing observer resection"

    # BGR colors.
    color_a = (255, 190, 60)
    color_b = (80, 220, 255)
    color_true = (80, 230, 80)
    color_estimate = (220, 80, 240)

    def random_observer():
        # Keep away from the landmark baseline and the text panel.
        return np.array([
            rng.uniform(40, WIDTH - 41),
            rng.uniform(230, HEIGHT - 41),
        ])

    true_observer = random_observer()
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Simulated measurements: only these angles go to the solver.
            azimuth_a = azimuth_to_point(true_observer, LANDMARK_A)
            azimuth_b = azimuth_to_point(true_observer, LANDMARK_B)

            estimated_observer = np.array(
                observer_from_azimuths(azimuth_a, azimuth_b)
            )
            error = np.linalg.norm(estimated_observer - true_observer)

            image = np.full((HEIGHT, WIDTH, 3), 25, dtype=np.uint8)

            # Faint coordinate grid.
            for x in range(0, WIDTH, 50):
                cv2.line(image, (x, 0), (x, HEIGHT - 1), (40, 40, 40))
            for y in range(0, HEIGHT, 50):
                cv2.line(image, (0, y), (WIDTH - 1, y), (40, 40, 40))

            # Reverse-bearing rays originate at the known landmarks and
            # pass through the independently recovered observer position.
            for landmark, color in (
                (LANDMARK_A, color_a),
                (LANDMARK_B, color_b),
            ):
                cv2.line(
                    image,
                    pixel(landmark),
                    pixel(estimated_observer),
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(image, pixel(landmark), 6, color, -1)

            draw_text(image, "A (10, 10)", (22, 31), color_a)
            draw_text(
                image,
                f"B ({int(LANDMARK_B[0])}, 10)",
                (WIDTH - 180, 31),
                color_b,
            )

            # Green filled dot = ground truth.
            # Magenta ring = recovered position; they should overlap.
            cv2.circle(
                image, pixel(true_observer), 5, color_true, -1,
                cv2.LINE_AA
            )
            cv2.circle(
                image, pixel(estimated_observer), 11, color_estimate, 2,
                cv2.LINE_AA
            )

            # Information panel.
            cv2.rectangle(image, (20, 48), (WIDTH - 21, 206), (15, 15, 15), -1)

            rows = [
                (
                    f"Azimuth to A: {azimuth_a:.4f} deg    "
                    f"Azimuth to B: {azimuth_b:.4f} deg",
                    (230, 230, 230),
                ),
                (
                    f"Actual:    x={true_observer[0]:.3f}, "
                    f"y={true_observer[1]:.3f}  (green dot)",
                    color_true,
                ),
                (
                    f"Recovered: x={estimated_observer[0]:.3f}, "
                    f"y={estimated_observer[1]:.3f}  (magenta ring)",
                    color_estimate,
                ),
                (
                    f"Position error: {error:.3e} pixels",
                    (230, 230, 230),
                ),
                (
                    "SPACE: randomize observer    Q / ESC: quit",
                    (190, 190, 190),
                ),
            ]

            for i, (text, color) in enumerate(rows):
                draw_text(image, text, (32, 75 + i * 28), color)

            cv2.imshow(window, image)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                true_observer = random_observer()

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()