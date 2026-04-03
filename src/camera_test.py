# ─────────────────────────────────────────────────────────
# camera_test.py
# Run this FIRST to confirm your camera is working.
# Press Q to quit the window.
# ─────────────────────────────────────────────────────────

import cv2
import sys

def test_camera(camera_index=0):
    """
    Opens the camera and shows a live feed.
    camera_index=0 is usually the built-in or first webcam.
    If it fails, try camera_index=1 or 2.
    """
    print(f"[INFO] Trying camera index {camera_index}...")

    cap = cv2.VideoCapture(camera_index)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        print("       Try changing camera_index to 1 or 2")
        sys.exit(1)

    # Get camera properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print(f"[OK]   Camera opened successfully!")
    print(f"       Resolution : {width} x {height}")
    print(f"       FPS        : {fps}")
    print(f"[INFO] Press Q in the window to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Can't receive frame. Exiting...")
            break

        # Flip so it acts like a mirror (more natural)
        frame = cv2.flip(frame, 1)

        # Show resolution info on screen
        h, w = frame.shape[:2]
        cv2.putText(frame,
                    f"Camera OK  |  {w}x{h}  |  Press Q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 100),   # green text
                    2)

        cv2.imshow("Camera Test — Fall Detection Project", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up properly
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. All good!")


if __name__ == "__main__":
    # Change this to 1 or 2 if your webcam isn't found
    test_camera(camera_index=0)