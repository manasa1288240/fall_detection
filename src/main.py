# ─────────────────────────────────────────────────────────
# main.py  —  Full fall detection pipeline
# Press Q to quit.
# ─────────────────────────────────────────────────────────

import cv2
import datetime
import os

from preprocess     import preprocess_frame
from pose_estimator import PoseEstimator
from fall_detector  import FallDetector

# ── Config ────────────────────────────────────────────────
CAMERA_INDEX = 0         # change to 1 if webcam not found
LOG_DIR      = "../logs"
SHOW_DEBUG   = True       # show angle/ratio info on screen

os.makedirs(LOG_DIR, exist_ok=True)

def log_fall(info):
    """Saves fall event with timestamp to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path  = os.path.join(LOG_DIR, "fall_log.txt")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] FALL DETECTED | {info}\n")
    print(f"[ALERT] Fall logged at {timestamp}")

def draw_debug(frame, info, display_fall):
    """Overlays debug info on the frame."""
    colour = (0, 0, 255) if display_fall else (0, 255, 100)
    label  = "!! FALL DETECTED !!" if display_fall else "Normal"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

    if SHOW_DEBUG:
        lines = [
            f"Torso angle : {info['torso_angle']} deg",
            f"Aspect ratio: {info['aspect_ratio']}",
            f"Hip velocity: {info['hip_velocity']}",
            f"Fallen frames: {info['fallen_frames']}/{info['threshold']}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 90 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
    return frame

def main():
    cap      = cv2.VideoCapture(CAMERA_INDEX)
    estimator = PoseEstimator()
    detector  = FallDetector()
    display_fall_frames = 0  # frames to display fall alert on screen

    if not cap.isOpened():
        print("[ERROR] Camera not found. Run camera_test.py first.")
        return

    print("[INFO] Fall detection running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]

        # Step 1: Fix lighting
        frame = preprocess_frame(frame)

        # Step 2: Get keypoints
        frame, landmarks = estimator.process(frame)

        if landmarks:
            # Step 3: Extract needed keypoints
            L_SH  = estimator.get_keypoint(landmarks, estimator.LEFT_SHOULDER,  w, h)
            R_SH  = estimator.get_keypoint(landmarks, estimator.RIGHT_SHOULDER, w, h)
            L_HIP = estimator.get_keypoint(landmarks, estimator.LEFT_HIP,       w, h)
            R_HIP = estimator.get_keypoint(landmarks, estimator.RIGHT_HIP,      w, h)
            L_ANK = estimator.get_keypoint(landmarks, estimator.LEFT_ANKLE,     w, h)
            R_ANK = estimator.get_keypoint(landmarks, estimator.RIGHT_ANKLE,    w, h)

            # Step 4: Check for fall
            is_fall, info = detector.analyse(
                L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK, h
            )

            if is_fall:
                log_fall(info)
                display_fall_frames = 120  # display for 4 seconds at 30fps

            # Update display timer
            if display_fall_frames > 0:
                display_fall_frames -= 1

            display_fall = display_fall_frames > 0

            frame = draw_debug(frame, info, display_fall)

        cv2.imshow("Fall Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    estimator.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()