# ─────────────────────────────────────────────────────────
# preprocess.py
# Illumination normalization — makes detection work in
# low light, bright light, and changing conditions.
# ─────────────────────────────────────────────────────────

import cv2
import numpy as np

def apply_clahe(frame):
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast without blowing out bright areas.
    Works great for dim rooms and uneven lighting.
    Returns a 3-channel BGR frame (same format as input).
    """
    # Convert BGR → LAB colour space
    # LAB separates brightness (L) from colour (A, B)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split into 3 channels
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE only to the L (brightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0,
                              tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Merge back and convert to BGR
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def apply_gamma(frame, gamma=1.5):
    """
    Gamma correction — brightens dark frames.
    gamma > 1.0 brightens, gamma < 1.0 darkens.
    Use this as a backup if CLAHE isn't enough.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        (i / 255.0) ** inv_gamma * 255
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(frame, table)


def preprocess_frame(frame):
    """
    Main preprocessing pipeline.
    Call this on every frame before pose estimation.
    """
    frame = apply_clahe(frame)
    return frame