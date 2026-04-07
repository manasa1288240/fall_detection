# ─────────────────────────────────────────────────────────
# pose_estimator.py
# Uses MediaPipe Pose to extract 33 body keypoints.
# This IS the deep learning component.
# ─────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np
import os

class PoseEstimator:
    """
    Wraps MediaPipe Pose.
    Call process(frame) on each frame to get keypoints.
    """

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'pose_landmarker_full.task')
        self.base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp_tasks.vision.RunningMode.IMAGE
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)

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
        # MediaPipe needs RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.landmarker.detect(mp_image)

        annotated_frame = frame.copy()

        if results.pose_landmarks:
            # Draw landmarks
            # For simplicity, skip drawing for now
            pass

        return annotated_frame, results.pose_landmarks[0] if results.pose_landmarks else None

    def get_keypoint(self, landmarks, index, frame_w, frame_h):
        """
        Returns (x, y) pixel coordinates of a keypoint.
        MediaPipe gives normalised 0–1 values; we convert to pixels.
        """
        lm = landmarks[index]
        x  = int(lm.x * frame_w)
        y  = int(lm.y * frame_h)
        # return a lightweight tuple (avoid allocating numpy arrays per keypoint)
        return (x, y)

    def close(self):
        self.landmarker.close()