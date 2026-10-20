from tam_library_save_and_load import save_detections_csv, get_save_file_path, load_video_as_list_of_frames, clear_detections_csv, load_detections_csv, load_image_as_frame
from tam_library_camera_utils import get_detections_from_frame
from tam_library_draw_and_show import convert_to_field_pixels_global, draw_detection_list_to_field, show_video_from_list_of_frames
from tam_library_classes import OurRobot




def main():



    example_video = "training_data/tennis_ball_with_corners.mp4"
    csv_path = "detections/2026_10_19T21_55_52.csv"
    #create_detection_csv_from_video(example_video)

    load_and_animate_detection_csv(detection_csv_path=csv_path)


def create_detection_csv_from_video(vid_file_path: str):
    #### instead of going right from frame analysis to showing the user, let's save and then load from a csv

    save_file_path = get_save_file_path()

    clear_detections_csv(save_file_path)

    our_bot = OurRobot()
    our_bot.field_pos_px = convert_to_field_pixels_global(our_bot.field_pos_in)

    print(f"loading frames from video {vid_file_path}")
    list_frames = load_video_as_list_of_frames(vid_file_path)

    frame_int : int = 0

    print("writing detections to csv")
    for frame in list_frames:
        detect_list = get_detections_from_frame(frame=frame, bot= our_bot)

        save_detections_csv(detections= detect_list, filename= save_file_path,vid_frame= frame_int)

        frame_int +=1

    print(f"saved to: {save_file_path}")


def load_and_animate_detection_csv(detection_csv_path: str):
    frame_int_detect_list_tuple = load_detections_csv(filename=detection_csv_path)

    field_frame = load_image_as_frame("field_pngs/empty_field.png")

    our_bot = OurRobot()
    our_bot.field_pos_px = convert_to_field_pixels_global(our_bot.field_pos_in)

    output_frames_list = []

    for frame_int, detection_list in frame_int_detect_list_tuple:
        field_frame_copy = field_frame.copy()
        annotated_field = draw_detection_list_to_field(detect_list= detection_list, bot= our_bot, field_frame=field_frame_copy)
        output_frames_list.append(annotated_field)
        
    show_video_from_list_of_frames(output_frames_list, "field")



if __name__ == "__main__":

    main()