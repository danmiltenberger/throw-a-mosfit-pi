import math

# VALUES
ball_dia_in = 7             # Inches
hw = 1920                   # Pixels
hfov = math.radians(62.2)   # Degrees to radians
ball_dia_px = 0             # INPUT: Ball diameter in pixels

# FUNCTION
def calculate_distance(ball_dia_in, horizontal_width_px,
                       hfov, ball_dia_px):

    if ball_dia_px <= 0:
        return None

    distance_in = (
        horizontal_width_px * ball_dia_in
        / (2 * math.tan(hfov / 2) * ball_dia_px)
    )

    return distance_in

# CALCULATIONS
distance_in = calculate_distance(
    ball_dia_in,
    hw,
    hfov,
    ball_dia_px
)

# OUTPUT
if distance_in is None:
    print("NO BALL DETECTED")
else:
    print(f"Distance: {distance_in:.2f} in")
    print(f"Distance: {distance_in / 12:.2f} ft")

