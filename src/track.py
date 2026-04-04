# إعطاء كل شخص Track ID ثابت
# تتبعه من فريم لفريم
# الحفاظ على الـ ID حتى لو اختفى مؤقتاً
# تحديد الاتجاه (دخول/خروج) بناءً على الحركة

# ============================================================
# track.py — Person Tracking & Direction Detection
# ============================================================
# Owner: Person 1 (CV Engineer - Detection)
#
# Ideas in this file (in order):
#   1. Initialize ByteTrack
#   2. Pass detections to the tracker each frame
#   3. Assign a stable Track ID to each person
#   4. Preserve ID even during temporary occlusion
#   5. Determine direction (entry/exit) based on horizontal movement
#   6. Return tracks ready for reid.py and group_manager.py
#
# Note:
#   Camera is mounted at door angle (side view) — movement is horizontal
#   Moving right = entry (going into the store)
#   Moving left  = exit  (leaving the store)
#   Change ENTRY_DIRECTION in config.py if your setup is the opposite
# ============================================================

import cv2
import numpy as np
from config import TRACK_MAX_AGE, TRACK_MIN_HITS, TRACK_IOU_THRESH, MOVEMENT_THRESHOLD, ENTRY_DIRECTION


# ──────────────────────────────────────
# State: stores the last known centroid for each Track ID
# ──────────────────────────────────────
prev_centroids = {}
# Format: { track_id: (cx, cy) }
# Updated every frame to compute movement direction


# ──────────────────────────────────────
# 1. Initialize the Tracker
# ──────────────────────────────────────
def load_tracker():
    """
    Initializes ByteTrack with settings from config.py.
    Called once at startup.

    Why ByteTrack over SORT / DeepSORT:
        - Faster
        - Handles occlusion better
        - Keeps the same ID after a person temporarily disappears
    """
    try:
        from bytetracker import BYTETracker

        class Args:
            track_thresh  = 0.5
            track_buffer  = TRACK_MAX_AGE
            match_thresh  = TRACK_IOU_THRESH
            mot20         = False

        tracker = BYTETracker(Args())
        print("✅ ByteTrack loaded")
        return tracker

    except ImportError:
        print("⚠️  ByteTrack not installed — running simple fallback")
        print("   Install: pip install git+https://github.com/ifzhang/ByteTrack.git")
        return None


# ──────────────────────────────────────
# 2. Update the Tracker each frame
# ──────────────────────────────────────
def update_tracker(tracker, detections):
    """
    Passes detections from detect.py into the tracker and returns
    each person with a stable Track ID and movement direction.

    Args:
        tracker    — initialized tracker (or None if not installed)
        detections — list of detections from detect.py
                     each item has: bbox, conf, centroid, label

    Returns:
        list of dicts, one per tracked person:
        {
            'track_id'  : 5,
            'bbox'      : [x1, y1, x2, y2],
            'centroid'  : (cx, cy),
            'direction' : 'entry' / 'exit' / 'stationary',
            'label'     : 'child' / 'adult'
        }
    """
    if not detections:
        return []

    # Convert detections to ByteTrack format: [x1, y1, x2, y2, conf]
    boxes = np.array([
        [*d['bbox'], d['conf']] for d in detections
    ], dtype=np.float32)

    # -- ByteTrack available --
    if tracker is not None:
        try:
            img_shape  = (720, 1280)   # adjust to match your camera resolution
            raw_tracks = tracker.update(boxes, img_shape, img_shape)
            return _parse_bytetrack_output(raw_tracks, detections)
        except Exception as e:
            print(f"⚠️  ByteTrack error: {e} — switching to fallback")

    # -- Fallback: no ByteTrack --
    return _simple_fallback(detections)


# ──────────────────────────────────────
# 3. Parse raw ByteTrack output
# ──────────────────────────────────────
def _parse_bytetrack_output(raw_tracks, original_detections):
    """
    Converts raw ByteTrack output into our unified dict format.
    Adds:
        - movement direction (entry / exit / stationary)
        - label from the original detection (child / adult)
    """
    result = []

    for track in raw_tracks:
        x1  = int(track.tlbr[0])
        y1  = int(track.tlbr[1])
        x2  = int(track.tlbr[2])
        y2  = int(track.tlbr[3])
        tid = int(track.track_id)

        cx, cy    = (x1 + x2) // 2, (y1 + y2) // 2
        direction = _get_direction(tid, (cx, cy))
        label     = _match_label(cx, cy, original_detections)

        result.append({
            'track_id'  : tid,
            'bbox'      : [x1, y1, x2, y2],
            'centroid'  : (cx, cy),
            'direction' : direction,
            'label'     : label
        })

    return result


# ──────────────────────────────────────
# 4. Simple fallback (no ByteTrack)
# ──────────────────────────────────────
def _simple_fallback(detections):
    """
    Runs when ByteTrack is not installed.
    Assigns temporary IDs based on detection order.
    Not ideal for production — good enough for initial testing.
    """
    result = []
    for i, d in enumerate(detections):
        cx, cy    = d['centroid']
        direction = _get_direction(i, (cx, cy))

        result.append({
            'track_id'  : i,
            'bbox'      : d['bbox'],
            'centroid'  : (cx, cy),
            'direction' : direction,
            'label'     : d.get('label', 'unknown')
        })
    return result


# ──────────────────────────────────────
# 5. Compute movement direction
# ──────────────────────────────────────
def _get_direction(track_id, curr_centroid):
    """
    Determines whether a person is entering or exiting
    based on their horizontal movement (dx).

    Camera at door angle (side view):
        dx positive (moving right) = entry — going into the store
        dx negative (moving left)  = exit  — leaving the store

    If your store is on the opposite side, change ENTRY_DIRECTION in config.py.

    Note:
        We need at least 2 frames to determine direction.
        First frame is always 'stationary'.
    """
    global prev_centroids

    if track_id not in prev_centroids:
        # First time we see this person — no direction yet
        prev_centroids[track_id] = curr_centroid
        return 'stationary'

    prev_cx, prev_cy = prev_centroids[track_id]
    curr_cx, curr_cy = curr_centroid

    dx = curr_cx - prev_cx   # positive = right, negative = left

    # Update stored position
    prev_centroids[track_id] = curr_centroid

    # Ignore tiny movements (noise / person standing still)
    if abs(dx) < MOVEMENT_THRESHOLD:
        return 'stationary'

    moving_right = dx > 0

    if ENTRY_DIRECTION == 'right':
        return 'entry' if moving_right else 'exit'
    else:  # ENTRY_DIRECTION == 'left'
        return 'exit' if moving_right else 'entry'


# ──────────────────────────────────────
# 6. Match label from original detections
# ──────────────────────────────────────
def _match_label(cx, cy, detections, tolerance=30):
    """
    ByteTrack doesn't carry the child/adult label.
    We match the centroid against the nearest original detection
    and pull the label from it.
    """
    for d in detections:
        dx, dy = d['centroid']
        if abs(cx - dx) < tolerance and abs(cy - dy) < tolerance:
            return d.get('label', 'unknown')
    return 'unknown'


# ──────────────────────────────────────
# 7. Clean up stale Track IDs
# ──────────────────────────────────────
def cleanup_old_tracks(active_track_ids):
    """
    Removes IDs from prev_centroids once a person disappears permanently.
    Called periodically from main.py to prevent the dict from growing forever.

    Args:
        active_track_ids — set of IDs visible in the current frame
    """
    global prev_centroids
    to_delete = [tid for tid in prev_centroids if tid not in active_track_ids]
    for tid in to_delete:
        del prev_centroids[tid]


# ──────────────────────────────────────
# 8. Draw results on frame
# ──────────────────────────────────────
def draw_tracks(frame, tracks):
    """
    Draws above each person:
        - Track ID
        - Direction (entry / exit / stationary)
        - Label (child / adult)

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
        tid       = t['track_id']
        direction = t.get('direction', 'stationary')
        label     = t.get('label', '')
        color     = colors.get(direction, (150, 150, 150))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label above the box
        text = f"ID:{tid} | {label} | {direction}"
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


# ──────────────────────────────────────
# 9. Quick test — detect + track together
# ──────────────────────────────────────
if __name__ == "__main__":
    """
    Quick real-time test for detect + track combined.
        python src/track.py

    What to check:
        - Each person gets a stable ID (number doesn't change while walking)
        - ID stays the same even if person is briefly hidden
        - Direction changes correctly: entry / exit / stationary
        - Label (adult/child) matches what you see

    Press Q to quit.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from src.detect import load_model, detect_persons

    model   = load_model()
    tracker = load_tracker()
    cap     = cv2.VideoCapture(0)

    print("Running detect + track test — press Q to quit")
    print("Expected output per person: ID:N | adult/child | entry/exit/stationary")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: detect
        detections = detect_persons(model, frame)

        # Step 2: track
        tracks = update_tracker(tracker, detections)

        # Step 3: cleanup stale IDs
        active_ids = {t['track_id'] for t in tracks}
        cleanup_old_tracks(active_ids)

        # Step 4: draw
        frame = draw_tracks(frame, tracks)

        # Show track count
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("track.py — Test", frame)

        # Print raw output to terminal
        for t in tracks:
            print(f"  ID:{t['track_id']} | {t['label']} | {t['direction']} | centroid:{t['centroid']}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
