import sys
import os
import threading
import datetime
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QSlider, QMenuBar, QMenu, QSystemTrayIcon, QHBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon

from pose_estimator import PoseEstimator
from fall_detector import FallDetector
from alert import play_alert_sound

# Constants
CAMERA_INDEX = 0
SCREENSHOTS_DIR = '../screenshots'
LOGS_DIR = '../logs'

# Create directories
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def preprocess_frame(frame):
    """Improve lighting and contrast for robust detection in varying lighting conditions."""
    # Apply noise reduction first
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # Convert to LAB color space for better lighting adjustment
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply stronger CLAHE for dramatic lighting difference handling
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    l = clahe.apply(l)
    
    # Normalize brightness to handle very dark/bright scenes
    brightness = cv2.mean(l)[0]
    if brightness < 80:
        # Scene is too dark, boost it
        l = cv2.convertScaleAbs(l, alpha=1.3, beta=30)
    elif brightness > 200:
        # Scene is too bright, reduce it slightly
        l = cv2.convertScaleAbs(l, alpha=0.9, beta=0)
    
    # Ensure values stay in valid range
    l = np.clip(l, 0, 255).astype(np.uint8)
    
    # Merge and convert back to BGR
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def log_fall(info):
    """Log fall detection to file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] FALL DETECTED - {info}\n"
    log_path = os.path.join(LOGS_DIR, 'fall_log.txt')
    with open(log_path, 'a') as f:
        f.write(log_entry)
    print(f"[ALERT] {log_entry.strip()}")


def draw_debug(frame, info, display_fall):
    """Overlay detection metrics on the video frame."""
    colour = (0, 0, 255) if display_fall else (0, 255, 100)
    label = "!! FALL DETECTED !!" if display_fall else "Normal"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

    if info is not None:
        lines = [
            f"Torso angle : {info.get('torso_angle', 0)} deg",
            f"Leg angle   : {info.get('leg_angle', 0)} deg",
            f"Aspect ratio: {info.get('aspect_ratio', 0)}",
            f"Hip vel     : {info.get('hip_velocity', 0)}",
            f"Fallen frms : {info.get('fallen_frames', 0)}/{info.get('threshold', 0)}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 80 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
    return frame

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fall_detected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.estimator = None
        self.detector = None
        self.display_fall_frames = 0
        self.alert_cooldown = 0
        self.screenshot_cooldown = 0
        self.current_frame = None

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.estimator = PoseEstimator()
        self.detector = FallDetector()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            self.current_frame = frame.copy()
            
            # Preprocess
            frame = preprocess_frame(frame)
            
            # Pose estimation
            frame, landmarks = self.estimator.process(frame)
            
            if landmarks:
                # Extract keypoints
                L_SH = self.estimator.get_keypoint(landmarks, self.estimator.LEFT_SHOULDER, w, h)
                R_SH = self.estimator.get_keypoint(landmarks, self.estimator.RIGHT_SHOULDER, w, h)
                L_HIP = self.estimator.get_keypoint(landmarks, self.estimator.LEFT_HIP, w, h)
                R_HIP = self.estimator.get_keypoint(landmarks, self.estimator.RIGHT_HIP, w, h)
                L_ANK = self.estimator.get_keypoint(landmarks, self.estimator.LEFT_ANKLE, w, h)
                R_ANK = self.estimator.get_keypoint(landmarks, self.estimator.RIGHT_ANKLE, w, h)
                
                # Fall detection
                is_fall, info = self.detector.analyse(L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK, h)
                
                if is_fall:
                    self.fall_detected.emit(info)
                    self.display_fall_frames = 150  # Show fall for 5 seconds at 30fps
                    
                    # Screenshot
                    if self.screenshot_cooldown == 0:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"fall_{timestamp}.jpg"
                        filepath = os.path.join(SCREENSHOTS_DIR, filename)
                        cv2.imwrite(filepath, self.current_frame)
                        print(f"[SCREENSHOT] Saved {filepath}")
                        self.screenshot_cooldown = 120  # 4 seconds at 30fps
                
                # Determine if we should show fall indicator
                display_fall = self.display_fall_frames > 0
                frame = draw_debug(frame, info, display_fall)
            
            self.frame_ready.emit(frame)
            
            # Update display timer
            if self.display_fall_frames > 0:
                self.display_fall_frames -= 1
            
            # Update cooldowns
            if self.alert_cooldown > 0:
                self.alert_cooldown -= 1
            if self.screenshot_cooldown > 0:
                self.screenshot_cooldown -= 1
        
        cap.release()
        if self.estimator:
            self.estimator.close()

    def start_detection(self):
        self.running = True
        self.start()

    def stop_detection(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fall Detection System")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        
        # Controls
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        
        # Sensitivity slider
        self.sensitivity_label = QLabel("Sensitivity: 5")
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        layout.addLayout(controls_layout)
        
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(self.sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        layout.addLayout(sensitivity_layout)
        
        central_widget.setLayout(layout)
        
        # Menu bar
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Settings')
        
        camera_action = QAction('Camera Settings', self)
        camera_action.triggered.connect(self.show_camera_settings)
        settings_menu.addAction(camera_action)
        
        # System tray
        self.tray_icon = QSystemTrayIcon(self)
        # For now, use default icon
        self.tray_icon.setToolTip("Fall Detection System")
        
        tray_menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        
        hide_action = QAction("Hide to Tray", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)
        
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_activated)
        self.tray_icon.show()
        
        # Camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.fall_detected.connect(self.on_fall_detected)
        
        # Status
        self.statusBar().showMessage("Ready")
    
    def start_detection(self):
        self.camera_thread.start_detection()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Detection running...")
    
    def stop_detection(self):
        self.camera_thread.stop_detection()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Detection stopped")
    
    def update_sensitivity(self, value):
        self.sensitivity_label.setText(f"Sensitivity: {value}")
        # Update detector sensitivity if possible
        if self.camera_thread.detector:
            # Adjust thresholds based on sensitivity
            pass
    
    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)
    
    def on_fall_detected(self, info):
        log_fall(info)
        if self.camera_thread.alert_cooldown == 0:
            threading.Thread(target=play_alert_sound).start()
            self.camera_thread.alert_cooldown = 120
        self.statusBar().showMessage("FALL DETECTED!", 5000)

    def show_camera_settings(self):
        # Placeholder for camera settings dialog
        pass
    
    def tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()
    
    def closeEvent(self, event):
        self.camera_thread.stop_detection()
        self.tray_icon.hide()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())