# ─────────────────────────────────────────────────────────
# fall_detector.py
# Calculates features from keypoints and decides if a
# fall occurred. Uses rule-based logic as a baseline.
# ─────────────────────────────────────────────────────────

import math
from collections import deque

class FallDetector:
    """
    Analyses body keypoints frame-by-frame.
    Returns True if a fall is detected.
    """

    def __init__(self, history_len=15, fallen_threshold=3):
        # Store last N hip positions to detect rapid drops
        self.hip_y_history = deque(maxlen=history_len)
        self.fall_cooldown = 0   # prevents repeated alerts
        self.fallen_frames = 0   # consecutive frames in fallen state
        self.fallen_threshold = fallen_threshold  # frames to confirm fall

        # Default thresholds for a balanced fall detector
        self.angle_threshold = 65.0
        self.aspect_threshold = 1.2
        self.leg_angle_threshold = 42.0
        self.drop_threshold = 0.16
        self.low_hip_threshold = 0.60

    def set_sensitivity(self, level):
        """Adjust thresholds for the sensitivity slider (1 = strict, 10 = sensitive)."""
        level = max(1, min(10, level))
        self.fallen_threshold = 5 if level <= 3 else 4 if level <= 7 else 3
        sensitivity_factor = 1.0 - (level - 5) * 0.04

        self.angle_threshold = max(55.0, 68.0 * sensitivity_factor)
        self.aspect_threshold = max(1.0, 1.25 * sensitivity_factor)
        self.leg_angle_threshold = max(38.0, 52.0 * sensitivity_factor)
        self.drop_threshold = max(0.12, 0.16 * sensitivity_factor)
        self.low_hip_threshold = max(0.56, 0.62 * sensitivity_factor)

    def _calculate_angle(self, p1, p2):
        """
        Calculates the angle (degrees) of the line p1→p2
        relative to vertical (0° = standing, 90° = lying flat).
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # use math.atan2 on scalars (faster than numpy for tiny vectors)
        angle_rad = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_rad)
        return abs(angle_deg)

    def _get_mid(self, p1, p2):
        """Returns midpoint between two keypoints."""
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    def analyse(self,
                  left_shoulder, right_shoulder,
                  left_hip,      right_hip,
                  left_ankle,    right_ankle,
                  frame_h):
        """
        Main analysis function.
        All inputs are numpy arrays of (x, y) pixel coords.
        Returns (is_fall: bool, debug_info: dict).
        """
        # ── 1. Mid-points ───────────────────────────
        mid_shoulder = self._get_mid(left_shoulder, right_shoulder)
        mid_hip      = self._get_mid(left_hip, right_hip)
        mid_ankle    = self._get_mid(left_ankle, right_ankle)

        # ── 2. Torso angle ──────────────────────────
        # 0° = perfectly vertical, 90° = lying horizontal
        torso_angle = self._calculate_angle(mid_shoulder, mid_hip)

        # ── 3. Body bounding box aspect ratio ───────
        # tall person → ratio < 1. Fallen person → ratio > 1
        # Compute a simple, robust body width: distance between shoulders
        body_width = abs(left_shoulder[0] - right_shoulder[0])
        body_height = abs(mid_shoulder[1] - mid_ankle[1])
        aspect_ratio = body_width / (body_height + 1e-6)

        # ── 4. Hip drop velocity ─────────────────────
        # Normalise hip Y to 0–1 so it's camera-independent
        hip_y_norm = mid_hip[1] / frame_h
        self.hip_y_history.append(hip_y_norm)

        hip_drop_velocity = 0.0
        if len(self.hip_y_history) >= 5:
            hip_drop_velocity = self.hip_y_history[-1] - self.hip_y_history[-5]

        leg_angle = self._calculate_angle(mid_hip, mid_ankle)

        rule_posture = (
            torso_angle > self.angle_threshold and
            aspect_ratio > self.aspect_threshold and
            leg_angle > self.leg_angle_threshold and
            hip_y_norm > self.low_hip_threshold
        )

        rule_rapid_drop = (
            hip_drop_velocity > self.drop_threshold and
            torso_angle > 45 and
            hip_y_norm > 0.52
        )

        rule_low_body = (
            hip_y_norm > 0.72 and
            torso_angle > 55 and
            aspect_ratio > 1.05
        )

        fallen = rule_posture or rule_rapid_drop or rule_low_body

        if fallen:
            self.fallen_frames += 1
            if self.fallen_frames >= self.fallen_threshold and self.fall_cooldown == 0:
                is_fall = True
                self.fall_cooldown = 60
                self.fallen_frames = 0  # reset after alert
            else:
                is_fall = False
        else:
            self.fallen_frames = 0
            is_fall = False

        if self.fall_cooldown > 0:
            self.fall_cooldown -= 1

        debug_info = {
            "torso_angle": round(torso_angle, 1),
            "leg_angle": round(leg_angle, 1),
            "aspect_ratio": round(aspect_ratio, 2),
            "hip_velocity": round(hip_drop_velocity, 3),
            "hip_y_norm": round(hip_y_norm, 3),
            "fallen_frames": self.fallen_frames,
            "threshold": self.fallen_threshold,
            "is_fall": is_fall,
            "rule_posture": rule_posture,
            "rule_rapid_drop": rule_rapid_drop,
            "rule_low_body": rule_low_body,
        }
        return is_fall, debug_info