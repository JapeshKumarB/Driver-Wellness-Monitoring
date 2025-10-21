import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    camera_source: int | str = 0
    dlib_landmarks_path: str = os.path.join("assets", "shape_predictor_68_face_landmarks.dat")
    drivers_dir: str = os.path.join("assets", "drivers")
    thresholds_path: str = os.path.join("data", "thresholds.json")
    events_log_path: str = os.path.join("data", "events.log")
    enable_voice: bool = True
    enable_privacy_blur: bool = False
    privacy_anonymize_logs: bool = True
    streamlit_port: int = 8501
    # Baseline thresholds (can be adapted per-driver)
    ear_drowsy_thresh: float = 0.21
    perclos_drowsy_thresh: float = 0.4
    yawn_thresh: float = 28.0  # lip distance
    fatigue_window_sec: int = 60
    trend_window_minutes: int = 30
    intervention_min_interval_sec: int = 90  # avoid frequent nudges

CONFIG = AppConfig()