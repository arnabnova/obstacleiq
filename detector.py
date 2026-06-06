"""
detector.py — Core object detection, distance estimation, and car decision logic.
Uses YOLOv8 (ultralytics) for real-time detection.
"""

import cv2
import numpy as np

import math
from ultralytics import YOLO

# ─── Object physical database ─────────────────────────────────────────────────
# Maps COCO class names → (typical_height_cm, typical_weight_g, density_class)
# density_class: "soft", "rigid_light", "rigid_heavy"
OBJECT_DB = {
    # Nature/ground objects
    "stone":         (4,    200,  "rigid_heavy"),
    "rock":          (8,    800,  "rigid_heavy"),
    "twig":          (1.5,  15,   "rigid_light"),
    "stick":         (2,    30,   "rigid_light"),
    "leaf":          (0.3,  3,    "soft"),
    "paper":         (0.3,  5,    "soft"),
    "book":          (4,    400,  "rigid_light"),
    "bottle":        (25,   500,  "rigid_light"),
    "cup":           (10,   150,  "rigid_light"),
    "can":           (12,   200,  "rigid_light"),
    "box":           (20,   800,  "rigid_light"),
    "ball":          (10,   150,  "rigid_light"),
    "sports ball":   (20,   430,  "rigid_light"),
    "backpack":      (40,  1500,  "soft"),
    "handbag":       (25,   600,  "soft"),
    "suitcase":      (60,  3000,  "rigid_heavy"),
    "chair":         (90,  5000,  "rigid_heavy"),
    "couch":         (80, 30000,  "rigid_heavy"),
    "potted plant":  (35,  2000,  "rigid_light"),
    "vase":          (25,   500,  "rigid_light"),
    "teddy bear":    (30,   400,  "soft"),
    "cat":           (25,  4500,  "soft"),
    "dog":           (40, 15000,  "soft"),
    "bird":          (15,   300,  "soft"),
    "person":        (170, 70000, "rigid_heavy"),   # human — always AVOID
    "bicycle":       (100, 10000, "rigid_heavy"),
    "motorcycle":    (120, 90000, "rigid_heavy"),
    "car":           (150, 1400000, "rigid_heavy"),
    "truck":         (250, 3500000, "rigid_heavy"),
    "bench":         (80, 20000,  "rigid_heavy"),
    "umbrella":      (90,   500,  "soft"),
    "tie":           (5,    50,   "soft"),
    "shoe":          (10,   500,  "rigid_light"),
    "skateboard":    (10,  1000,  "rigid_light"),
    "laptop":        (3,   2000,  "rigid_heavy"),
    "cell phone":    (1.5,  180,  "rigid_light"),
    "keyboard":      (3,    800,  "rigid_light"),
    "mouse":         (4,    100,  "rigid_light"),
    "remote":        (3,     90,  "rigid_light"),
    "scissors":      (10,   100,  "rigid_light"),
    "toothbrush":    (4,     20,  "rigid_light"),
    "fork":          (2,     50,  "rigid_light"),
    "spoon":         (2,     30,  "rigid_light"),
    "knife":         (2,     60,  "rigid_light"),
    "banana":        (8,    120,  "soft"),
    "apple":         (7,    182,  "soft"),
    "orange":        (8,    131,  "soft"),
    "carrot":        (3,     61,  "soft"),
    "broccoli":      (15,   272,  "soft"),
    "pizza":         (3,    300,  "soft"),
    "sandwich":      (5,    200,  "soft"),
    "hot dog":       (4,    150,  "soft"),
    "donut":         (4,     57,  "soft"),
    "cake":          (10,   500,  "soft"),
    "clock":         (30,  1000,  "rigid_heavy"),
    "scissors":      (10,   100,  "rigid_light"),
    "fire hydrant":  (60,  7000,  "rigid_heavy"),
    "stop sign":     (75, 10000,  "rigid_heavy"),
    "traffic light": (100, 15000, "rigid_heavy"),
    "parking meter": (120, 18000, "rigid_heavy"),
}

# Default for unknown objects (estimate from bounding box size)
DEFAULT_PROPS = (10, 200, "rigid_light")

# COCO class names (YOLOv8 default)
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Known average real-world heights for distance estimation (cm)
KNOWN_HEIGHTS_CM = {
    "person": 170, "car": 150, "bicycle": 100, "motorcycle": 120,
    "truck": 250, "bus": 300, "chair": 90, "bottle": 25, "cup": 10,
    "laptop": 3, "cell phone": 15, "dog": 40, "cat": 25, "sports ball": 20,
    "backpack": 40, "suitcase": 60, "book": 25, "clock": 30, "vase": 25,
    "potted plant": 35, "fire hydrant": 60, "stop sign": 75,
}
DEFAULT_KNOWN_HEIGHT = 15  # cm fallback


class ObjectDetector:
    """YOLOv8-based object detector with distance + weight estimation."""

    def __init__(self, model_name="yolov8n", conf_threshold=0.45, focal_length_px=600):
        self.model = YOLO(f"{model_name}.pt")   # downloads on first run
        self.conf_threshold = conf_threshold
        self.focal_length_px = focal_length_px

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a BGR frame. Returns list of result dicts."""
        h_frame, w_frame = frame.shape[:2]
        yolo_results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"obj_{cls_id}"
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Pixel dimensions of bounding box
                box_h_px = y2 - y1
                box_w_px = x2 - x1

                # Distance estimation (pinhole camera model)
                known_h = KNOWN_HEIGHTS_CM.get(label, DEFAULT_KNOWN_HEIGHT)
                if box_h_px > 0:
                    distance_cm = (known_h * self.focal_length_px) / box_h_px
                else:
                    distance_cm = 999.0

                # Estimate real-world dimensions at that distance
                est_height_cm = known_h  # we used known height
                est_width_cm  = (box_w_px * distance_cm) / self.focal_length_px

                # Weight & physical props
                db_entry = OBJECT_DB.get(label, None)
                if db_entry:
                    ref_height, ref_weight, density_class = db_entry
                    # Scale weight by ratio of visible size vs reference
                    scale = max(0.3, min(3.0, est_height_cm / max(ref_height, 1)))
                    est_weight_g = ref_weight * (scale ** 2)
                else:
                    # Unknown: estimate from bounding box volume proxy
                    vol_proxy = est_height_cm * est_width_cm
                    est_weight_g = vol_proxy * 2.5   # rough density 2.5 g/cm²
                    density_class = "rigid_light"

                detections.append({
                    "label":         label,
                    "confidence":    conf,
                    "bbox":          (x1, y1, x2, y2),
                    "distance_cm":   round(distance_cm, 1),
                    "est_height_cm": round(est_height_cm, 1),
                    "est_width_cm":  round(est_width_cm, 1),
                    "est_weight_g":  round(est_weight_g, 1),
                    "density_class": density_class,
                })

        # Sort by confidence desc
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections

    def draw_boxes(self, frame: np.ndarray, results: list[dict],
                   car_clearance: float, car_push_force: float, car_width: float) -> np.ndarray:
        """Draw annotated bounding boxes on the frame."""
        for det in results:
            label = det["label"]
            x1, y1, x2, y2 = det["bbox"]
            decision, _ = get_decision(det, car_clearance, car_push_force)

            # Color by decision
            color_map = {
                "ROLL_OVER": (110, 255, 110),
                "PUSH":      (102, 229, 255),
                "AVOID":     (80,  80,  255),
                "NO_OBJECT": (120, 120, 120),
            }
            color = color_map.get(decision, (200, 200, 200))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents (more polished look)
            corner_len = 12
            thickness = 3
            for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                     (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

            # Label background
            tag = f"{label.upper()} | {det['distance_cm']:.0f}cm | {decision.replace('_',' ')}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pad = 5
            by = max(y1 - th - pad * 2, 0)
            cv2.rectangle(frame, (x1, by), (x1 + tw + pad * 2, y1), color, -1)
            cv2.putText(frame, tag, (x1 + pad, y1 - pad),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Weight sub-label
            sub = f"~{det['est_weight_g']:.0f}g  conf:{det['confidence']:.0%}"
            cv2.putText(frame, sub, (x1 + 4, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # HUD overlay
        _draw_hud(frame, results, car_clearance)
        return frame


def get_decision(det: dict, car_clearance: float, car_push_force: float) -> tuple[str, str]:
    """
    Decide what the car should do with the detected obstacle.

    Returns (decision_key, reason_string)
    decision_key: "ROLL_OVER" | "PUSH" | "AVOID"
    """
    label  = det["label"].lower()
    height = det["est_height_cm"]
    weight = det["est_weight_g"]
    density = det["density_class"]

    # ── Rule 1: Always avoid humans ──
    if label == "person":
        return "AVOID", "Human detected — must not engage"

    # ── Rule 2: Large/heavy animals ──
    if label in ("dog", "cat", "horse", "cow", "elephant", "bear"):
        if weight > 5000:
            return "AVOID", f"Animal too heavy ({weight:.0f}g)"
        return "AVOID", "Living creature — avoid to be safe"

    # ── Rule 3: Can roll over? ──
    if height <= car_clearance:
        if weight <= 200:
            return "ROLL_OVER", f"Height {height:.1f}cm ≤ clearance {car_clearance}cm; weight {weight:.0f}g ≤ 200g"
        elif density == "soft":
            return "ROLL_OVER", f"Soft & low ({height:.1f}cm); compressible under car"

    # ── Rule 4: Can push? ──
    # Estimate push force needed: F ≈ μ × m × g (μ=0.4 for typical floor friction)
    gravity = 9.81
    mu = 0.4
    push_needed_N = (mu * weight / 1000) * gravity   # weight in kg → N

    if height <= car_clearance * 3 and push_needed_N <= car_push_force:
        if density != "rigid_heavy":
            return "PUSH", f"Pushable: needs ~{push_needed_N:.1f}N ≤ car force {car_push_force}N"

    # ── Rule 5: Avoid ──
    if height > car_clearance * 3:
        return "AVOID", f"Too tall ({height:.1f}cm > {car_clearance*3:.1f}cm)"
    if push_needed_N > car_push_force:
        return "AVOID", f"Too heavy to push ({push_needed_N:.1f}N > {car_push_force}N)"
    if density == "rigid_heavy":
        return "AVOID", "Rigid/heavy — risk of damage"

    return "AVOID", "Could not determine safe action"


def _draw_hud(frame: np.ndarray, results: list, car_clearance: float):
    """Draw a small HUD in bottom-left corner."""
    h, w = frame.shape[:2]
    hud_x, hud_y = 10, h - 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + 200, h - 10), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"Objects: {len(results)}", (hud_x + 8, hud_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1)
    cv2.putText(frame, f"Clearance: {car_clearance}cm", (hud_x + 8, hud_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1)
    cv2.putText(frame, "ObstacleIQ v1.0", (hud_x + 8, hud_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 80, 100), 1)