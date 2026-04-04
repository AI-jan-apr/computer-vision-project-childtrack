# اكتشاف الأشخاص في الفريم
# رسم الـ Bounding Box على كل شخص
# تحديد منطقة الدخول والخروج (ROI)

# ============================================================
# detect.py — Person Detection using trained YOLOv8 model
# ============================================================
# Owner: Person 1 (CV Engineer - Detection)
#
# What this file does:
#   1. Load the trained YOLO model (best.pt)
#   2. Run detection on each video frame
#   3. Extract for each detected person:
#        - bbox      : bounding box [x1, y1, x2, y2]
#        - conf      : confidence score
#        - label     : 'adult' or 'child'
#        - centroid  : center point (cx, cy)
#
# Model classes (from best.pt):
#   0 = adult
#   1 = child
# ============================================================

import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE


# ──────────────────────────────────────
# Class mapping — from best.pt
# ──────────────────────────────────────
CLASS_NAMES = {
    0: 'adult',
    1: 'child'
}

#OR 
# label = model.names[cls]
# If you want to use the model's built-in class names instead of hardcoding them, you can simply access the `model.names` dictionary after loading the model. Here's how you can modify the `load_model` function to print the class names from the model:


# ──────────────────────────────────────
# 1. Load the YOLO model
# ──────────────────────────────────────
def load_model(model_path=YOLO_MODEL_PATH):
    """
    Loads the trained YOLOv8 model from disk.
    Called once at startup.

    Args:
        model_path — path to best.pt (set in config.py)

    Returns:
        loaded YOLO model
    """
    model = YOLO(model_path)
    print(f" Model loaded — Classes: {model.names}")
    return model


# ──────────────────────────────────────
# 2. Run detection on a single frame
# ──────────────────────────────────────
def detect_persons(model, frame):
    """
    Runs YOLO on a single frame and returns all detected persons.

    Args:
        model — loaded YOLO model
        frame — current video frame (numpy array from OpenCV)

    Returns:
        list of dicts, one per detected person:
        [
            {
                'bbox'     : [x1, y1, x2, y2],
                'conf'     : 0.91,
                'label'    : 'adult' or 'child',
                'centroid' : (cx, cy)
            },
            ...
        ]
    """
    results    = model(frame, conf=YOLO_CONFIDENCE, verbose=False)
    detections = parse_results(results)
    return detections


# ──────────────────────────────────────
# 3. Parse raw YOLO output
# ──────────────────────────────────────
def parse_results(results):
    """
    Converts raw YOLO output into our clean dict format.

    YOLO gives us:
        box.xyxy  → [x1, y1, x2, y2]
        box.conf  → confidence score
        box.cls   → class index (0=adult, 1=child)

    We convert that into a simple dict the rest of the
    pipeline (track.py, group_manager.py) can easily use.
    """
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf            = round(float(box.conf[0]), 2)
        cls             = int(box.cls[0])
        label           = CLASS_NAMES.get(cls, 'unknown')
        cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2

        detections.append({
            'bbox'     : [x1, y1, x2, y2],
            'conf'     : conf,
            'label'    : label,
            'centroid' : (cx, cy)
        })

    return detections


# ──────────────────────────────────────
# 4. Draw detections on frame (for visualization)
# ──────────────────────────────────────
def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on the frame.

    Colors:
        Blue  = adult
        Green = child

    Args:
        frame      — original video frame
        detections — list of dicts from detect_persons()

    Returns:
        annotated frame
    """
    colors = {
        'adult' : (255, 100, 0),
        'child' : (0, 200, 0)
    }

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        label           = d['label']
        conf            = d['conf']
        color           = colors.get(label, (200, 200, 200))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label + confidence above the box
        text = f"{label} {conf}"
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Centroid dot
        cx, cy = d['centroid']
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


# ──────────────────────────────────────
# 5. Quick test — run this file directly
#    to verify detection is working
# ──────────────────────────────────────
if __name__ == "__main__":
    """
    Quick real-time test:
        python src/detect.py

    Shows webcam feed with bounding boxes.
    Press Q to quit.

    What to check:
        - 'adult' and 'child' labels appear correctly
        - Confidence scores look reasonable (> 0.5)
        - Bounding boxes follow the person smoothly
    """
    model = load_model()
    cap   = cv2.VideoCapture(0)
    #OR
    #cap = cv2.VideoCapture("data/videos/test.mp4")
    #If you want to test on a video file instead of webcam, replace the argument to VideoCapture with the path to your video file.

    print("Running detection test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_persons(model, frame)
        frame      = draw_detections(frame, detections)
        print(detections)

        # Show detection count
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("detect.py — Test", frame)

        # Print to terminal so you can see raw output
        for d in detections:
            print(d)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()