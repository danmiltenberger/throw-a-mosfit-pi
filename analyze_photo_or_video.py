import cv2
from icecream import ic
from tam_library import analyze_photo_or_video
import numpy as np

from run_settings import USE_DRAWN_FIELD

#file_path = "training_data/2026-10-11T20_53_06.567733.png"
file_path = "training_data/tennis_ball_with_corners.mp4"


if __name__ == "__main__":
    analyze_photo_or_video(file_path)
