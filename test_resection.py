### using the generated resection programs, test with our actual field and robot
from tam_library_save_and_load import get_field_frame
from tam_library_classes import FieldState, Point, RADIANS_NORTH_EAST
from tam_library_draw_and_show import draw_field, show_frame, draw_arrow, convert_to_field_pixels_global
from tam_library_field import GOALPOST_LEFT_TOP, GOALPOST_RIGHT_TOP
from settings_run import GOALPOST_COLOR_EAST, GOALPOST_COLOR_WEST
import cv2




def test_resection():
    field_frame = get_field_frame()

    field_state = FieldState(
        draw_ball= False,
        draw_their_bot= False,
        draw_our_bot= True,
        our_bot_pos= Point(-8,-10),
        our_bot_bearing=RADIANS_NORTH_EAST,
    )

    # draw the frame with the bot
    field_frame = draw_field(field_frame= field_frame, field_state= field_state)

    # draw arrows from our bot center to goalposts

    ourbot_pos = convert_to_field_pixels_global(field_state.our_bot_pos)

    field_frame = cv2.arrowedLine(
        img=field_frame,
        pt1=(int(ourbot_pos.x), int(ourbot_pos.y)),
        pt2=(int(GOALPOST_LEFT_TOP.x), int(GOALPOST_LEFT_TOP.y)),
        color= GOALPOST_COLOR_EAST
    )

    field_frame = cv2.arrowedLine(
        img=field_frame,
        pt1=(int(ourbot_pos.x), int(ourbot_pos.y)),
        pt2=(int(GOALPOST_RIGHT_TOP.x), int(GOALPOST_RIGHT_TOP.y)),
        color= GOALPOST_COLOR_WEST,
    )

    angle_to_top_right_from_3oclock = 0
    

    show_frame(field_frame)


test_resection()