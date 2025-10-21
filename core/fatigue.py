import numpy as np
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye_pts):
    # eye_pts: 6 points
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_opening(mouth_pts):
    # 20 points (outer+inner); use top-bottom distance
    top = np.mean(mouth_pts[2:4], axis=0)  # approximate
    bottom = np.mean(mouth_pts[8:10], axis=0)
    return dist.euclidean(top, bottom)

class FatigueAnalyzer:
    def __init__(self, ear_thresh=0.21, perclos_thresh=0.4, yawn_thresh=28.0, window_sec=60, fps=30):
        self.ear_thresh = ear_thresh
        self.perclos_thresh = perclos_thresh
        self.yawn_thresh = yawn_thresh
        self.window_len = max(1, int(window_sec * fps))
        self.ear_hist = []
        self.blink_count = 0

    def update(self, frame, face_rects, landmarks):
        metrics = {"ear_avg": 0.0, "perclos": 0.0, "yawn": 0.0}
        if not landmarks:
            return metrics

        # Using first face
        shape = landmarks[0]
        # indices per 68-point model
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        mouth = shape[48:68]

        ear_l = eye_aspect_ratio(left_eye)
        ear_r = eye_aspect_ratio(right_eye)
        ear = (ear_l + ear_r) / 2.0

        self.ear_hist.append(ear)
        if len(self.ear_hist) > self.window_len:
            self.ear_hist = self.ear_hist[-self.window_len:]

        closed = [1 if e < self.ear_thresh else 0 for e in self.ear_hist]
        perclos = sum(closed) / max(1, len(closed))

        yawn = mouth_opening(mouth)

        metrics["ear_avg"] = float(np.mean(self.ear_hist)) if self.ear_hist else 0.0
        metrics["perclos"] = float(perclos)
        metrics["yawn"] = float(yawn)
        return metrics