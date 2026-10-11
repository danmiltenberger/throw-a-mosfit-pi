import cv2
from picamera2 import Picamera2  # type: ignore
import time
from datetime import datetime


def get_iso_string_name():
    return datetime.now().isoformat()


def take_video(filename=None, fps=30.0, size=(640, 480)):
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")

    if filename is None:
        filename = get_iso_string_name() + ".mp4"

    picam2 = Picamera2()
    video_writer = None
    camera_started = False

    try:
        config = picam2.create_video_configuration(
            main={"format": "RGB888", "size": size},
            controls={"FrameRate": fps},
        )
        picam2.configure(config)

        picam2.start()
        camera_started = True

        # Allow automatic exposure and gain to settle.
        time.sleep(1.0)

        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            raise ValueError("Received empty frame buffer from libcamera.")

        # Use the actual captured dimensions for the video writer.
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            filename,
            fourcc,
            fps,
            (width, height),
        )

        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {filename}")

        print(f"Recording to: {filename}")
        print("Press 'q' in the OpenCV window to stop recording.")

        while True:
            video_writer.write(frame)
            cv2.imshow("Press q to stop recording", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                raise ValueError("Received empty frame buffer from libcamera.")

    finally:
        print("Stopping camera and releasing resources...")

        # Finalize the video file even if capture fails.
        try:
            if video_writer is not None:
                video_writer.release()
        finally:
            try:
                if camera_started:
                    picam2.stop()
            finally:
                try:
                    picam2.close()
                finally:
                    cv2.destroyAllWindows()

    print(f"Video saved: {filename}")
    return filename


def main():
    take_video()


if __name__ == "__main__":
    main()