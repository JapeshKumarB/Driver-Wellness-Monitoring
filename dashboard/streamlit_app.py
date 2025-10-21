import time
import cv2
import streamlit as st

import os, sys
# Add parent directory (DriveMind/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from core.camera import VideoSource
from core.landmarks import LandmarkDetector
from core.fatigue import FatigueAnalyzer
from core.emotion import EmotionAnalyzer
from core.driver_id import DriverIdentifier
from core.profiling import ThresholdManager
from core.privacy import apply_privacy
from core.trend import TrendBuffer
from core.wellness import WellnessOrchestrator

st.set_page_config(page_title="DriveMind Dashboard", layout="wide")

st.title("🚗 DriveMind — Driver Wellness Monitoring")

col_left, col_right = st.columns([2, 1])

source = st.sidebar.text_input("Camera index or video path", str(CONFIG.camera_source))
enable_blur = st.sidebar.checkbox("Privacy blur", value=CONFIG.enable_privacy_blur)
run = st.sidebar.checkbox("Run")

video_placeholder = col_left.empty()
metrics_placeholder = col_right.empty()
trend_placeholder = col_right.empty()
alert_placeholder = st.empty()

if run:
    cam = VideoSource(int(source) if source.isdigit() else source)
    lmk = LandmarkDetector(CONFIG.dlib_landmarks_path)
    fatigue = FatigueAnalyzer(
        ear_thresh=CONFIG.ear_drowsy_thresh,
        perclos_thresh=CONFIG.perclos_drowsy_thresh,
        yawn_thresh=CONFIG.yawn_thresh,
        window_sec=CONFIG.fatigue_window_sec
    )
    emotion = EmotionAnalyzer()
    driver_id = DriverIdentifier(CONFIG.drivers_dir)
    thresholds = ThresholdManager(CONFIG.thresholds_path)
    trends = TrendBuffer(window_minutes=CONFIG.trend_window_minutes)
    orchestrator = WellnessOrchestrator(CONFIG, thresholds)

    last_intervention_ts = 0
    for frame in cam.frames():
        face_rects, landmarks = lmk.detect(frame)
        identity = driver_id.identify(frame, face_rects)
        per_driver = thresholds.get_for_driver(identity)

        metrics = fatigue.update(frame, face_rects, landmarks)
        stress = emotion.estimate(frame, face_rects)
        trends.update(identity, metrics, stress)

        status = orchestrator.evaluate(identity, metrics, stress, per_driver)

        display_frame = apply_privacy(frame.copy(), face_rects, enable_blur)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        hud = f"{identity or 'Unknown'} | EAR:{metrics.get('ear_avg',0):.2f} PERCLOS:{metrics.get('perclos',0):.2f} Yawn:{metrics.get('yawn',0):.1f} Stress:{stress.get('stress_score',0):.2f}"
        cv2.putText(display_frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if status.get("needs_intervention"):
            cv2.putText(display_frame, "⚠️ ALERT: " + status["alert_level"].upper(), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            now = time.time()
            if now - last_intervention_ts > CONFIG.intervention_min_interval_sec:
                orchestrator.nudge(identity, status, metrics, stress)
                last_intervention_ts = now

            alert_placeholder.error(f"⚠️ {identity or 'Driver'}: {status['alert_level'].upper()} alert — {', '.join(status['reasons'])}")
        else:
            alert_placeholder.success("🟢 Status: Normal")

        video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        metrics_placeholder.markdown(f"""
**Driver:** {identity or 'Unknown'}  
**EAR avg:** {metrics.get('ear_avg',0):.2f}  
**PERCLOS:** {metrics.get('perclos',0):.2f}  
**Yawn:** {metrics.get('yawn',0):.1f}  
**Stress score:** {stress.get('stress_score',0):.2f}  
""")

        summ = trends.summary(identity)
        trend_placeholder.markdown(f"""
**Trend samples:** {summ.get('samples',0)}  
**EAR mean:** {summ.get('ear_mean',0):.2f}  
**PERCLOS mean:** {summ.get('perclos_mean',0):.2f}  
**Yawn mean:** {summ.get('yawn_mean',0):.1f}  
**Stress mean:** {summ.get('stress_mean',0):.2f}  
""")

        time.sleep(0.01)
    cam.release()