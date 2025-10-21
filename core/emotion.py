import numpy as np

class EmotionAnalyzer:
    def __init__(self):
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.available = True
        except Exception:
            self.deepface = None
            self.available = False
            print("Info: DeepFace not available; using neutral fallback.")

    def estimate(self, frame, face_rects):
        result = {"dominant_emotion": "neutral", "stress_score": 0.2}
        if not self.available or not face_rects:
            return result
        x, y, w, h = face_rects[0]
        crop = frame[max(0,y):y+h, max(0,x):x+w]
        try:
            analysis = self.deepface.analyze(crop, actions=["emotion"], enforce_detection=False)
            emo = analysis[0]["dominant_emotion"] if isinstance(analysis, list) else analysis["dominant_emotion"]
            # Simple mapping to a stress score
            stress_map = {
                "angry": 0.8, "fear": 0.8, "sad": 0.6, "disgust": 0.7,
                "surprise": 0.5, "happy": 0.2, "neutral": 0.3
            }
            score = stress_map.get(emo, 0.3)
            return {"dominant_emotion": emo, "stress_score": float(np.clip(score, 0.0, 1.0))}
        except Exception:
            return result