from collections import deque
import time
import numpy as np

class TrendBuffer:
    def __init__(self, window_minutes=30):
        self.window_sec = window_minutes * 60
        self.store = {}

    def update(self, identity, metrics, stress):
        key = identity or "unknown"
        now = time.time()
        buf = self.store.get(key)
        if buf is None:
            buf = deque()
            self.store[key] = buf
        buf.append((now, metrics, stress))
        # prune
        while buf and now - buf[0][0] > self.window_sec:
            buf.popleft()

    def summary(self, identity):
        key = identity or "unknown"
        buf = self.store.get(key, [])
        if not buf:
            return {}
        ears = [b[1].get("ear_avg", 0.0) for b in buf]
        percloses = [b[1].get("perclos", 0.0) for b in buf]
        yawns = [b[1].get("yawn", 0.0) for b in buf]
        stresses = [b[2].get("stress_score", 0.0) for b in buf]
        return {
            "ear_mean": float(np.mean(ears)),
            "perclos_mean": float(np.mean(percloses)),
            "yawn_mean": float(np.mean(yawns)),
            "stress_mean": float(np.mean(stresses)),
            "samples": len(buf)
        }