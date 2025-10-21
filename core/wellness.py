import time
import csv
import os
import pyttsx3

class WellnessOrchestrator:
    def __init__(self, config, thresholds_manager):
        self.cfg = config
        self.tman = thresholds_manager
        self.voice = None
        if self.cfg.enable_voice:
            try:
                self.voice = pyttsx3.init()
                self.voice.setProperty('rate', 170)
                self.voice.setProperty('volume', 0.85)
            except Exception:
                self.voice = None
        os.makedirs(os.path.dirname(self.cfg.events_log_path), exist_ok=True)
        if not os.path.exists(self.cfg.events_log_path):
            with open(self.cfg.events_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ts", "driver", "alert_level", "reason", "ear", "perclos", "yawn", "stress"])

    def evaluate(self, identity, metrics, stress, per_driver):
        ear = metrics.get("ear_avg", 0.0)
        perclos = metrics.get("perclos", 0.0)
        yawn = metrics.get("yawn", 0.0)
        stress_score = stress.get("stress_score", 0.0)

        ear_thr = per_driver.get("ear_thresh", self.cfg.ear_drowsy_thresh)
        perclos_thr = per_driver.get("perclos_thresh", self.cfg.perclos_drowsy_thresh)
        yawn_thr = per_driver.get("yawn_thresh", self.cfg.yawn_thresh)

        reasons = []
        if perclos > perclos_thr:
            reasons.append("High PERCLOS")
        if ear < ear_thr:
            reasons.append("Low EAR")
        if yawn > yawn_thr:
            reasons.append("Yawn")
        if stress_score > 0.7:
            reasons.append("High stress")

        alert_level = "none"
        if len(reasons) >= 2 or perclos > perclos_thr + 0.1:
            alert_level = "high"
        elif reasons:
            alert_level = "medium"

        needs_intervention = alert_level in ("medium", "high")
        if needs_intervention:
            self._log(identity, alert_level, ",".join(reasons), ear, perclos, yawn, stress_score)
            self.tman.adapt(identity, metrics)
        return {"alert_level": alert_level, "needs_intervention": needs_intervention, "reasons": reasons}

    def nudge(self, identity, status, metrics, stress):
        # Consistent, brief suggestions
        msg = None
        reasons = status.get("reasons", [])
        if "High PERCLOS" in reasons or "Low EAR" in reasons:
            msg = "You seem drowsy. If safe, take a short break or drink water."
        elif "Yawn" in reasons:
            msg = "Yawning detected. Consider a quick pause when safe."
        elif "High stress" in reasons:
            msg = "Stress is elevated. Breathe steadily and loosen grip."
        else:
            msg = "Maintain safe driving. You're doing fine."

        # Voice (brief, single sentence), no repeated nags
        if self.voice:
            try:
                self.voice.say(msg)
                self.voice.runAndWait()
            except Exception:
                pass

    def _log(self, identity, alert_level, reason, ear, perclos, yawn, stress):
        ts = int(time.time())
        driver_label = identity if not self.cfg.privacy_anonymize_logs else "driver"
        try:
            with open(self.cfg.events_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, driver_label, alert_level, reason, f"{ear:.3f}", f"{perclos:.3f}", f"{yawn:.1f}", f"{stress:.2f}"])
        except Exception:
            pass