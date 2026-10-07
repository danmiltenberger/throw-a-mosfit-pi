from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
import time


print("starting video.py")
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

encoder = H264Encoder(10000000)


picam2.start_recording(encoder, 'test.h264')
time.sleep(60)
picam2.stop_recording()
