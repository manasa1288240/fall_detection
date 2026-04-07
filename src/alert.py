# ─────────────────────────────────────────────────────────
# alert.py  —  Audio alert system for fall detection
# ─────────────────────────────────────────────────────────

import pygame
import os

# Initialize pygame mixer
pygame.mixer.init()

def play_alert_sound():
    """
    Plays the alert.wav sound file.
    Assumes alert.wav is in the assets directory.
    """
    alert_path = os.path.join(os.path.dirname(__file__), "..", "assets", "alert.wav")
    if os.path.exists(alert_path):
        pygame.mixer.music.load(alert_path)
        pygame.mixer.music.play()
    else:
        print("[ERROR] alert.wav not found in assets directory.")
