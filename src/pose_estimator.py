# ─────────────────────────────────────────────────────────
# pose_estimator.py
# Uses MediaPipe Pose to extract 33 body keypoints.
# This IS the deep learning component.
# ─────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    """
    Wraps MediaPipe Pose.
    Call process(frame) on each frame to get keypoints.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw  = mp.solutions.drawing_utils

        # min_detection_confidence: how sure it needs to be
        # min_tracking_confidence : how sure to keep tracking
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Useful landmark indices (memorise these)
        self.LEFT_SHOULDER  = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP       = 23
        self.RIGHT_HIP      = 24
        self.LEFT_KNEE      = 25
        self.RIGHT_KNEE     = 26
        self.LEFT_ANKLE     = 27
        self.RIGHT_ANKLE    = 28

    def process(self, frame):
        """
        Takes a BGR frame, returns (annotated_frame, landmarks).
        landmarks is None if no person detected.
        """
        # MediaPipe needs RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            # Draw skeleton on frame
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame, results.pose_landmarks

    def get_keypoint(self, landmarks, index, frame_w, frame_h):
        """
        Returns (x, y) pixel coordinates of a keypoint.
        MediaPipe gives normalised 0–1 values; we convert to pixels.
        """
        lm = landmarks.landmark[index]
        x  = int(lm.x * frame_w)
        y  = int(lm.y * frame_h)
        return np.array([x, y])

    def close(self):
        self.pose.close()