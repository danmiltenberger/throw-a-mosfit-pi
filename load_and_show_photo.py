from tam_library_save_and_load import load_image_as_frame
from tam_library_draw_and_show import get_side_by_side_frame, show_photo_from_frame

def main():
    example_photo = "training_data/tennis_ball_99in.png"
    example_frame = load_image_as_frame(example_photo)
    field_frame = load_image_as_frame("field_pngs/empty_field.png")
    frame = get_side_by_side_frame(example_frame, field_frame=field_frame)
    show_photo_from_frame(frame)


if __name__ == "__main__":
    main()