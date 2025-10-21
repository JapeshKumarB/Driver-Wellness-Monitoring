import os
import numpy as np

class DriverIdentifier:
    def __init__(self, drivers_dir: str):
        self.drivers_dir = drivers_dir
        try:
            import face_recognition
            self.fr = face_recognition
            self.available = True
            self.known_encodings, self.known_names = self._load_known(drivers_dir)
        except Exception:
            self.fr = None
            self.available = False
            self.known_encodings, self.known_names = [], []
            print("Info: face_recognition not available; identity will be Unknown.")

    def _load_known(self, dir_path):
        import face_recognition
        encodings = []
        names = []
        for fn in os.listdir(dir_path):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(dir_path, fn)
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    encodings.append(encs[0])
                    names.append(os.path.splitext(fn)[0])
        return encodings, names

    def identify(self, frame, face_rects):
     if not self.available or not face_rects or not self.known_encodings:
        return None
     x, y, w, h = face_rects[0]
     rgb = frame[:, :, ::-1]
     # Correct conversion from (x,y,w,h) to (top,right,bottom,left)
     box = (y, x + w, y + h, x)
     encs = self.fr.face_encodings(rgb, [box])
     if not encs:
        return None
     matches = self.fr.compare_faces(self.known_encodings, encs[0], tolerance=0.5)
     if True in matches:
        idx = matches.index(True)
        return self.known_names[idx]
     return None
