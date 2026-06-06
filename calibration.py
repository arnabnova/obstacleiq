"""
calibration.py — Measure your camera's focal length using a known-size object.

HOW TO USE:
  1. Hold a standard A4 sheet (21 cm wide) at exactly 50 cm from your camera.
  2. Run:  python calibration.py
  3. A window opens. The script measures the sheet's pixel width → computes focal length.
  4. Enter the reported focal_length_px into the Streamlit sidebar.

Alternatively: use a ruler on screen and note the cm/pixel ratio.
"""

import cv2
import numpy as np

KNOWN_OBJECT_WIDTH_CM = 21.0   # A4 sheet width
KNOWN_DISTANCE_CM     = 50.0   # hold object at this distance

def calibrate(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    print("\n📐 CAMERA FOCAL LENGTH CALIBRATION")
    print("=" * 45)
    print(f"  Hold an A4 sheet ({KNOWN_OBJECT_WIDTH_CM}cm wide) exactly {KNOWN_DISTANCE_CM}cm from the camera.")
    print("  Press [SPACE] to capture — [Q] to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot open camera.")
            break

        cv2.putText(frame, "Hold A4 paper at 50cm. Press SPACE to capture.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.imshow("Calibration — press SPACE", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            focal_px = _measure_focal(frame)
            if focal_px:
                print(f"\n✅  Estimated focal length: {focal_px:.1f} px")
                print(f"    → Enter this value in the Streamlit sidebar.\n")
            else:
                print("\n⚠️  Could not detect the sheet automatically.")
                print("    Try manual method: measure the sheet's pixel width with the ruler tool,")
                print(f"    then compute: focal_px = pixel_width × {KNOWN_DISTANCE_CM} / {KNOWN_OBJECT_WIDTH_CM}\n")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _measure_focal(frame: np.ndarray):
    """Attempt to find a large rectangular region (the A4 sheet) and measure pixel width."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None

    pts = best.reshape(4, 2).astype(float)
    # Width = max of horizontal distances
    widths = [
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3]),
    ]
    pixel_width = max(widths)
    focal_px = (pixel_width * KNOWN_DISTANCE_CM) / KNOWN_OBJECT_WIDTH_CM
    return focal_px


if __name__ == "__main__":
    calibrate(camera_index=0)