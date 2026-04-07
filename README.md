# Fall Detection System

A real-time fall detection application using computer vision and pose estimation. This system leverages MediaPipe and OpenCV to monitor human movement, applying specific kinematic rules to detect falls and trigger immediate alerts.

## Features

* **Real-Time Pose Estimation**: Uses MediaPipe to track 33 body keypoints with high fidelity.
* **Dual-Rule Detection Logic**:
    * **Posture-Based (Rule A)**: Detects falls based on the torso angle (relative to the vertical) and the bounding box aspect ratio.
    * **Velocity-Based (Rule B)**: Monitors sudden vertical drops by calculating the velocity of the hip keypoints.
* **Audio Alerts**: Integrated Pygame-based audio system that plays an alert sound upon detection.
* **Automatic Screenshots**: Logic to capture and save timestamped images to a `screenshots/` folder whenever a fall is triggered.
* **Persistent Logging**: Records fall events with technical data (angle, velocity, timestamp) in `logs/fall_log.txt`.
* **Image Preprocessing**: Utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve detection accuracy in varied lighting conditions.

## Project Structure

```text
fall_detection/
├── assets/
│   └── alert.wav          # Audio file for alerts
├── logs/
│   └── fall_log.txt       # History of detected falls
├── screenshots/           # Saved images of fall events
├── src/
│   ├── main.py            # Main entry point and UI loop
│   ├── fall_detector.py   # Detection logic and thresholds
│   ├── pose_estimator.py  # MediaPipe pose initialization
│   ├── preprocess.py      # Image enhancement filters
│   ├── alert.py           # Threaded audio playback logic
│   └── camera_test.py     # Utility to verify camera feed
├── requirements.txt       # Project dependencies
└── README.md
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd fall_detection
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main application from the `src` directory:

```bash
cd src
python main.py
```
* **'Q' Key**: Quit the application.
* **'D' Key**: Toggle the debug overlay (shows angles and velocity).
* **'P' Key**: Toggle Privacy Mode (displays skeleton on a black background).

## Technical Details

### Detection Thresholds
The system identifies a fall if one of the following conditions is met:
1.  **Rule A**: The torso angle exceeds 60 degrees AND the person's bounding box aspect ratio is greater than 1.2.
2.  **Rule B**: The hip drop velocity exceeds 0.15.

### Audio & Visual Feedback
To prevent UI freezing, audio alerts run on a separate thread. A cooldown mechanism is implemented to ensure that a single fall event triggers only one audio alert and one screenshot, preventing redundant files and noise.

## Contributors

* **Diya Satish Kumar**
* **Manasa Mohan**
* **Zahra Dalal**

## License
This project is for educational and research purposes.
