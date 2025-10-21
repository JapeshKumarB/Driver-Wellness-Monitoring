import argparse
import time
import cv2
from config import CONFIG
from core.camera import VideoSource
from core.landmarks import LandmarkDetector
from core.fatigue import FatigueAnalyzer
from core.emotion import EmotionAnalyzer
from core.driver_id import DriverIdentifier
from core.profiling import ThresholdManager
from core.privacy import apply_privacy
from core.wellness import WellnessOrchestrator
from core.trend import TrendBuffer

def main():
    parser = argparse.ArgumentParser(description="DriveMind runtime")
    parser.add_argument("--source", default=CONFIG.camera_source, help="Camera index or video path")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice suggestions")
    parser.add_argument("--blur", action="store_true", help="Enable privacy blur for faces")
    args = parser.parse_args()

    CONFIG.enable_voice = not args.no_voice
    CONFIG.enable_privacy_blur = args.blur

    cam = VideoSource(args.source)
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
    print("DriveMind started. Press 'q' to quit.")
    try:
        for frame in cam.frames():
            face_rects, landmarks = lmk.detect(frame)
            identity = driver_id.identify(frame, face_rects)
            per_driver = thresholds.get_for_driver(identity)

            metrics = fatigue.update(frame, face_rects, landmarks)
            stress = emotion.estimate(frame, face_rects)
            trends.update(identity, metrics, stress)

            status = orchestrator.evaluate(identity, metrics, stress, per_driver)

            display_frame = apply_privacy(frame.copy(), face_rects, enable_blur=CONFIG.enable_privacy_blur)

            # Color-coded HUD
            color = (0, 255, 0)
            if status["alert_level"] == "medium":
                color = (0, 165, 255)
            elif status["alert_level"] == "high":
                color = (0, 0, 255)

            hud = f"{identity or 'Unknown'} | EAR:{metrics.get('ear_avg',0):.2f} PERCLOS:{metrics.get('perclos',0):.2f} Yawn:{metrics.get('yawn',0):.1f} Stress:{stress.get('stress_score',0):.2f} Alert:{status.get('alert_level','none')}"
            cv2.putText(display_frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Alert banner
            if status.get("needs_intervention"):
                cv2.putText(display_frame, "⚠️ ALERT: " + status["alert_level"].upper(), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                now = time.time()
                if now - last_intervention_ts > CONFIG.intervention_min_interval_sec:
                    orchestrator.nudge(identity, status, metrics, stress)
                    last_intervention_ts = now

            cv2.imshow("DriveMind", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("Exiting DriveMind...")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()