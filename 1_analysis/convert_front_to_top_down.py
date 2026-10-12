# given a reading of width and pixel position,
# convert to a top down view

# 0,0 is field center
# from the screen, our bot will always be at the bottom
# up screen is psotive, north
# down screen is negative, south
# left screen is negative, west
# right screen is positive, east
# angles are measured as a deviation from north, ex: northwest is -45, northeast is 45
from opencv_colors import *
from draw_field import *

field_center_in = (0,0)

# field in inches including goal depth and extension around tape is 
# (12+12+12*12) = 168 long
# (4+4+8*12) = 104 wide
# measured from the top left, the field center is 
# 4+4*12 = 52 in from the left
# (goalnet + playing area) /2 = 24 + 6*12 = 96 from the top

field_center_px_x = convert_to_px(52)
field_center_px_y = convert_to_px(84)

field_center_to_goalbox_center_in = 60
goalbox_center_bot_in = (0, -60)
goalbox_center_top_in = (0, +60)


def convert_in_point_to_screen(point : tuple[int,int]) -> tuple[int, int]:

    # the screen measures down and right from the top left,
    # but we measure the field +up and +right from the center

    # an input of 0,0 should result in window_x/2, window_y /2
    inch_x, inch_y = point

    px_x = convert_to_px(inch_x)
    px_y = convert_to_px(inch_y)

    # left means center minus the input
    rel_px_x = field_center_px_x + px_x
    rel_px_x = int(rel_px_x)

    # up means center MINUS input
    rel_px_y = field_center_px_y - px_y
    rel_px_y = int(rel_px_y)

    return (rel_px_x,rel_px_y)


goalbox_center_top_px = convert_in_point_to_screen(point=goalbox_center_top_in)
goalbox_center_bot_px = convert_in_point_to_screen(point=goalbox_center_bot_in)
center_field_px = convert_in_point_to_screen(point=(0,0))


def convert_to_top_down(verdict: dict):
    '''
    from the camera's understanding of pixel center and apparent width, convert to angle and distance
    '''

    pixel_position = verdict["center"]
    object_width = verdict["width"]


def get_distance(seen_width, known_width):

    pass
