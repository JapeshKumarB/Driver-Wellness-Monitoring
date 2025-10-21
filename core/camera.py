import cv2

class VideoSource:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        # Light tuning (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self):
        if self.cap:
            self.cap.release()