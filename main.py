import cv2
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import VIDEO_SOURCE, WINDOW_NAME
from detect        import load_model
from track         import track_persons, cleanup_old_tracks, draw_tracks
from reid          import ReID
from group_manager import GroupManager
from alert_engine  import AlertEngine
from matcher       import Matcher
from database      import Database


# ── Colors for alert overlay ─────────────────────────────────

ALERT_COLORS = {
    'ADULT_LEFT_CHILD'    : (0, 165, 255),   # orange
    'CHILD_EXITED_ALONE'  : (0, 0, 255),     # red
    'CHILD_WITH_STRANGER' : (0, 0, 180),     # dark red
}


# ── Draw group info on frame ──────────────────────────────────

def draw_groups(frame, gm):
    groups  = gm.get_groups()
    pending = gm._pending

    y = 30
    cv2.putText(frame, f"Groups:{len(groups)} Pending:{len(pending)}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25

    for g in groups:
        state = 'OPEN' if g.is_open else 'CLOSED'
        text  = f"{g.group_id}[{state}] A:{g.inside_adults} C:{g.inside_children} | {g.status}"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)
        y += 22

    return frame


def draw_alerts(frame, active_alerts):
    h = frame.shape[0]
    y = h - 20
    for alert, _ in reversed(active_alerts[-5:]):
        color = ALERT_COLORS.get(alert['type'], (0, 0, 255))
        text  = f"[{alert['type']}] Group:{alert['group_id']}"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y -= 25
    return frame


# ── Main loop ─────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  ChildTrack CCMS — Starting...")
    print("  Q = quit | S = status")
    print("=" * 50)

    # Init all modules
    model   = load_model()
    reid    = ReID()
    gm      = GroupManager()
    ae      = AlertEngine()
    db      = Database()
    matcher = Matcher(db, gm)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Cannot open: {VIDEO_SOURCE}")
        sys.exit(1)

    active_alerts = []
    ALERT_FRAMES  = 90   # ~3 seconds at 30fps
    frame_count   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        now = time.monotonic()

        # ── 1. Track ──────────────────────────────────────────
        tracks = track_persons(model, frame)

        # ── 2. Save members + embeddings on entry ─────────────
        for t in tracks:
            if t['direction'] == 'entry':
                gid = gm.get_group_of(t['track_id'])
                if gid and t.get('embedding') is not None:
                    db.save_member(
                        track_id   = t['track_id'],
                        group_id   = gid,
                        label      = t['label'],
                        embedding  = t['embedding'],
                        entry_time = now,
                    )
                    db.log_event(gid, t['track_id'], 'entry', now)

        # ── 3. Group Manager ──────────────────────────────────
        exit_events = gm.update(tracks)

        # ── 4. Save groups to database ────────────────────────
        for record in gm.get_db_records():
            db.save_group(record)

        # ── 5. Alert Engine ───────────────────────────────────
        new_alerts = ae.process(exit_events, gm)

        # ── 6. Handle exits in database ───────────────────────
        for e in exit_events:
            db.update_member_exit(e['exited_id'], e['timestamp'])
            db.log_event(e['group_id'], e['exited_id'], 'exit', e['timestamp'])

        # ── 7. Stranger detection (CASE 4) ────────────────────
        exiting_tracks = [t for t in tracks if t['direction'] == 'exit']
        children_exiting = [t for t in exiting_tracks if t['label'] == 'child']
        adults_exiting   = [t for t in exiting_tracks if t['label'] == 'adult']

        for child in children_exiting:
            for adult in adults_exiting:
                import math
                dist = math.dist(child['centroid'], adult['centroid'])
                if dist < 150:  # close together physically
                    stranger = ae.stranger_alert(child['track_id'], adult['track_id'], gm)
                    if stranger:
                        new_alerts.append(stranger)

        # ── 8. Save alerts + show ─────────────────────────────
        for alert in new_alerts:
            active_alerts.append((alert, ALERT_FRAMES))
            db.save_alert({
                'group_id'    : alert['group_id'],
                'alert_type'  : alert['type'],
                'who_exited'  : [alert.get('exited_id', alert.get('child_id', ''))],
                'who_remained': alert.get('child_ids', []),
                'timestamp'   : now,
            })
            print(f"\n🚨 {alert['type']} | Group:{alert['group_id']}")

        # Countdown alert display
        active_alerts = [(a, f - 1) for a, f in active_alerts if f > 0]

        # ── 9. Draw ───────────────────────────────────────────
        frame = draw_tracks(frame, tracks)
        frame = draw_groups(frame, gm)
        frame = draw_alerts(frame, active_alerts)

        cv2.putText(frame, f"Frame:{frame_count}",
                    (10, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow(WINDOW_NAME, frame)

        # ── 10. Cleanup ───────────────────────────────────────
        cleanup_old_tracks({t['track_id'] for t in tracks})

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            gm.print_status()

    cap.release()
    cv2.destroyAllWindows()
    db.close()
    print("\n✅ Done")
    gm.print_status()


if __name__ == "__main__":
    main()
