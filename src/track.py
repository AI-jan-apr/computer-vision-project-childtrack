# To Try this Run "python src/track.py" in terminal


# # إعطاء كل شخص Track ID ثابت
# # تتبعه من فريم لفريم
# # الحفاظ على الـ ID حتى لو اختفى مؤقتاً

# # تحديد الاتجاه (دخول/خروج) بناءً على الحركة


#=======  The new code using Model.track() ===================

# ============================================================
# track.py — Person Tracking & Direction Detection
# ============================================================
# Owner: Person 1 (CV Engineer - Detection)
#
# Ideas in this file (in order):
#   1. Load YOLO model (same best.pt — no separate tracker needed)
#   2. Run model.track() to get Detection + Track ID in one step
#   3. Assign a stable Track ID to each person
#   4. Preserve ID even during temporary occlusion
#   5. Determine direction (entry/exit) based on horizontal movement
#   6. Return tracks ready for group_manager.py
#
# Note:
#   We use model.track() from Ultralytics — no ByteTrack install needed.
#   ByteTrack runs inside Ultralytics via bytetrack.yaml config.
#   Camera is side view — movement is horizontal.
#   Moving right = entry | Moving left = exit
#   Change ENTRY_DIRECTION in config.py if your setup is opposite.
# ============================================================

import cv2
import sys
import os

import reid

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE, MOVEMENT_THRESHOLD, ENTRY_DIRECTION
from detect import load_model
from reid import ReID
from group_manager import GroupManager
from alert_engine import AlertEngine
from matcher import Matcher

prev_centroids = {}
# Format: { track_id: (cx, cy) }
# Updated every frame to compute movement direction
 
static_counter = {}
# Format: { track_id: int }
# Counts consecutive frames without movement
# If it exceeds MAX_STATIC_FRAMES → skip (background object)

last_known_boxes = {}
# Format: { track_id: {bbox, centroid, label, conf, direction} }

reid = ReID()
gm = GroupManager()
ae = AlertEngine()
matcher = Matcher()
 
MAX_STATIC_FRAMES = 30


# ──────────────────────────────────────
# 1. Main function — runs every frame
# ──────────────────────────────────────
def track_persons(model, frame):
    """
    Runs YOLO detection + ByteTrack tracking in one call.
    Returns each detected person with a stable ID and direction.

    Args:
        model — loaded YOLO model (from detect.py load_model)
        frame — current video frame (numpy array from OpenCV)

    Returns:
        list of dicts, one per tracked person:
        [
            {
                'track_id'  : 3,
                'bbox'      : [x1, y1, x2, y2],
                'centroid'  : (cx, cy),
                'label'     : 'adult' or 'child',
                'conf'      : 0.91,
                'direction' : 'entry' / 'exit' / 'stationary'
            },
            ...
        ]

    How model.track() works:
        - persist=True  → keeps the same ID across frames
        - tracker=...   → uses ByteTrack config built into Ultralytics
        - No separate install needed — works with your existing best.pt
    """
    results = model.track(
        frame,
        persist = True,
        conf    = YOLO_CONFIDENCE, #or 0.3 for more detection 
        iou     = 0.5,
        tracker = "bytetrack.yaml",
        verbose = False
    )
    
    print(results[0].boxes)
    print(results[0].boxes.id)



    tracks = []

    # No detections this frame
    if results[0].boxes.id is None:
        return []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id        = int(box.id)
        label           = model.names[int(box.cls)]
        conf            = round(float(box.conf), 2)
        cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
        crop = frame[y1:y2, x1:x2]
        embedding = reid.extract(crop)
        person_id = reid.match(embedding)
        direction       = _get_direction(track_id, (cx, cy))

        tracks.append({
            'track_id'  : track_id,
            'person_id' : person_id,
            'bbox'      : [x1, y1, x2, y2],
            'centroid'  : (cx, cy),
            'label'     : label,
            'conf'      : conf,
            'direction' : direction
        })

            # update last known for all visible tracks
    ##for t in tracks:
        ##last_known_boxes[t['track_id']] = t

    # add back recently lost tracks
    #for tid, t in last_known_boxes.items():
       # if tid not in {x['track_id'] for x in tracks}:
           # if static_counter.get(tid, 0) <= MAX_STATIC_FRAMES:
              #  ghost = t.copy()
               # ghost['direction'] = 'stationary'
              #  tracks.append(ghost)

    return tracks


# ──────────────────────────────────────
# 2. Compute movement direction
# ──────────────────────────────────────
def _get_direction(track_id, curr_centroid):
    """
    Determines entry / exit / stationary based on horizontal movement (dx).

    Logic:
        dx = current_x - previous_x
        dx positive → moving right
        dx negative → moving left

        If ENTRY_DIRECTION = 'right':
            moving right = entry
            moving left  = exit

        If ENTRY_DIRECTION = 'left':
            moving left  = entry
            moving right = exit

    First frame for any ID is always 'stationary' (no previous position yet).
    Movements smaller than MOVEMENT_THRESHOLD pixels are ignored (noise).
    """
    global prev_centroids

    if track_id not in prev_centroids:
        prev_centroids[track_id] = curr_centroid
        return 'stationary'

    prev_cx, _ = prev_centroids[track_id]
    curr_cx, _ = curr_centroid
    dx         = curr_cx - prev_cx

    # Update stored position
    prev_centroids[track_id] = curr_centroid

    # Too small to count
    if abs(dx) < MOVEMENT_THRESHOLD:
        return 'stationary'

    moving_right = dx > 0

    if ENTRY_DIRECTION == 'right':
        return 'entry' if moving_right else 'exit'
    else:
        return 'exit' if moving_right else 'entry'

# ──────────────────────────────────────
# 3. Clean up stale IDs
# ──────────────────────────────────────
def cleanup_old_tracks(active_track_ids):
    """
    Removes IDs from prev_centroids when a person disappears permanently.
    Prevents the dict from growing forever.

    Call this every frame from main.py:
        active_ids = {t['track_id'] for t in tracks}
        cleanup_old_tracks(active_ids)

    Args:
        active_track_ids — set of IDs visible in the current frame
    """
    global prev_centroids
    to_delete = [tid for tid in prev_centroids if tid not in active_track_ids]
    for tid in to_delete:
        del prev_centroids[tid]


# ──────────────────────────────────────
# 4. Draw results on frame
# ──────────────────────────────────────
def draw_tracks(frame, tracks):
    """
    Draws above each person:
        ID:3 | adult | entry

    Colors:
        Green = entry
        Red   = exit
        Gray  = stationary
    """
    colors = {
        'entry'      : (0, 255, 0),
        'exit'       : (0, 0, 255),
        'stationary' : (150, 150, 150)
    }

    for t in tracks:
        x1, y1, x2, y2 = t['bbox']
        color           = colors.get(t['direction'], (150, 150, 150))
        text            = f"ID:{t['track_id']} | {t['label']} | {t['direction']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Centroid dot
        cv2.circle(frame, t['centroid'], 4, color, -1)

    return frame


# ──────────────────────────────────────
# 5. Quick test
# ──────────────────────────────────────
if __name__ == "__main__":
    """
    How to run:
        python src/track.py

    What you should see on screen above each person:
        ID:3 | adult | entry
        ID:7 | child | stationary

    What to verify:
        1. ID is stable  — same number stays on same person while walking
        2. Label is correct — adult on adults, child on children
        3. Direction makes sense — entry when walking toward store, exit when leaving

    Press Q to quit.
    """
    model = load_model()
    cap = cv2.VideoCapture(r"C:\Users\Admin\OneDrive\Desktop\child_track\computer-vision-project-childtrack\WhatsApp Video 2026-04-04 at 6.02.00 PM.mp4") 


    print("Track test running — press Q to quit")
    print("Look for: ID:N | adult/child | entry/exit/stationary")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection + tracking
        tracks = track_persons(model, frame)
        events = gm.update(tracks)
        alerts = ae.process(events)

        for e in events:
            print("EVENT:", e)

        for a in alerts:
            print("🚨 ALERT:", a)

        # Cleanup stale IDs
        active_ids = {t['track_id'] for t in tracks}
        cleanup_old_tracks(active_ids)

        # Draw on frame
        frame = draw_tracks(frame, tracks)

        # Show count
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("track.py test", frame)

        # Print to terminal
        for t in tracks:
            print(f"  ID:{t['track_id']} | {t['label']} | {t['direction']} | centroid:{t['centroid']}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()