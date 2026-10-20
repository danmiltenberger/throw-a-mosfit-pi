import numpy as np

from tam_library_classes import OurRobot
from tam_library_save_and_load import load_video_frames, load_image_as_frame
from tam_library_draw_and_show import convert_to_field_pixels_global, get_side_by_side_frame, show_video_from_list_of_frames



def main():
    example_video = "training_data/tennis_ball_with_corners.mp4"
    load_and_show_video(example_video)



def load_and_show_video(video_path: str):
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





if __name__ == "__main__":
    main()