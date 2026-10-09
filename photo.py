from picamera2 import Picamera2, Preview
import time




def take_photo(save_path: str):
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    picam2.start_preview(Preview.QTGL)
    picam2.start()
    time.sleep(5)
    picam2.capture_file(save_path)


if __name__ == "__main__":
    take_photo("training_photos/red_ball_1.png")