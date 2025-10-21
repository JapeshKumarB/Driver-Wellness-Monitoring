import cv2
import dlib
from imutils import face_utils

class LandmarkDetector:
    def __init__(self, model_path: str):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(model_path)
            self.ready = True
        except Exception:
            self.predictor = None
            self.ready = False
            print("Warning: dlib landmark model missing; using face rects only.")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        rects_cv = [self._dlib_to_cv(r) for r in rects]
        landmarks = []
        if self.ready:
            for r in rects:
                shape = self.predictor(gray, r)
                shape_np = face_utils.shape_to_np(shape)
                landmarks.append(shape_np)
        return rects_cv, landmarks

    @staticmethod
    def _dlib_to_cv(rect):
        return (rect.left(), rect.top(), rect.width(), rect.height())