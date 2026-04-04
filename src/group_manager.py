# فكرة الـ 4 ثواني (Time Window)
# فكرة المسافة بين الأشخاص (Distance)
# ربط الطفل بالبالغ في Group ID
# إغلاق نافذة التسجيل بعد انتهاء الـ 4 ثواني
# منع إطلاق تنبيه لو اختفوا داخل المتجر
# فكرة الـ 4 ثواني (Time Window)
# فكرة المسافة بين الأشخاص (Distance)
# ربط الطفل بالبالغ في Group ID
# إغلاق نافذة التسجيل بعد انتهاء الـ 4 ثواني
# منع إطلاق تنبيه لو اختفوا داخل المتجر
# ============================================================
# group_manager.py — Group Formation & Exit Logic
# ============================================================
# Owner: Person 3 (Logic Engineer)
#
# Ideas in this file (in order):
#   1. Receive tracks from track.py every frame
#   2. Entry Logic  — link people who entered together into a Group
#   3. Time Window  — 4 second window to join a group
#   4. Distance     — max 150px between centroids to be grouped
#   5. Pending      — adult without child waits before forming a group
#   6. Group Formation — create Group ID, store adult/child IDs
#   7. Close window — after 4 seconds no one new can join
#   8. Exit Logic   — confirm exit after EXIT_CONFIRM_FRAMES frames
#   9. Return raw exit_event to main.py → alert_engine decides
#
# Input  (from track.py every frame):
#   track_id | label | direction | centroid | bbox | conf
#
# Output (returned to main.py — never calls other files directly):
#   → exit_events : List[dict]   for alert_engine
#   → db_records  : List[dict]   for database
#
# group_manager does NOT call alert_engine, database, or reid.
# All thresholds live in config.py only.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import math
import logging

from config import DISTANCE_THRESHOLD, ENTRY_TIME_WINDOW, EXIT_CONFIRM_FRAMES

logger = logging.getLogger(__name__)


# ── MemberInfo ────────────────────────────────────────────────

class MemberInfo:
    """One person inside a group. status: 'inside' | 'exited'."""

    def __init__(self, track_id, label, centroid, entry_time):
        self.track_id   = track_id
        self.label      = label       # 'adult' | 'child'
        self.centroid   = centroid    # (cx, cy)
        self.entry_time = entry_time
        self.exit_time  = None
        self.status     = 'inside'

    def __repr__(self):
        return f"Member(id={self.track_id}, {self.label}, {self.status})"


# ── Group ─────────────────────────────────────────────────────

class Group:
    """
    One group of people who entered together.
    status: 'forming' | 'active' | 'partial_exit' | 'fully_exited'
    """

    def __init__(self, group_id, entry_time):
        self.group_id   = group_id
        self.members    = {}       # track_id -> MemberInfo
        self.entry_time = entry_time
        self.is_open    = True
        self.status     = 'forming'

    @property
    def adult_ids(self):
        return [tid for tid, m in self.members.items() if m.label == 'adult']

    @property
    def child_ids(self):
        return [tid for tid, m in self.members.items() if m.label == 'child']

    @property
    def inside_adults(self):
        return [tid for tid, m in self.members.items()
                if m.label == 'adult' and m.status == 'inside']

    @property
    def inside_children(self):
        return [tid for tid, m in self.members.items()
                if m.label == 'child' and m.status == 'inside']

    @property
    def has_child(self):
        return bool(self.child_ids)

    @property
    def all_exited(self):
        """Empty group returns False — not considered complete."""
        return bool(self.members) and all(
            m.status == 'exited' for m in self.members.values()
        )

    def all_adults_exited(self):
        """No adults in group returns False — child alone is a risk."""
        adults = [m for m in self.members.values() if m.label == 'adult']
        return bool(adults) and all(m.status == 'exited' for m in adults)

    def centroid_avg(self):
        """
        Group centroid for distance checks.
        Open: all members. Closed: inside members only.
        """
        pool = (list(self.members.values()) if self.is_open
                else [m for m in self.members.values() if m.status == 'inside'])
        if not pool:
            return None
        return (
            sum(m.centroid[0] for m in pool) / len(pool),
            sum(m.centroid[1] for m in pool) / len(pool),
        )

    def refresh_status(self):
        """Updates status based on member states."""
        if not self.members:
            return
        has_inside = any(m.status == 'inside' for m in self.members.values())
        has_exited = any(m.status == 'exited' for m in self.members.values())
        if not self.is_open and has_inside and not has_exited:
            self.status = 'active'
        elif has_inside and has_exited:
            self.status = 'partial_exit'
        elif has_exited and not has_inside:
            self.status = 'fully_exited'

    def to_db_record(self):
        return {
            'group_id'   : self.group_id,
            'adult_ids'  : self.adult_ids,
            'child_ids'  : self.child_ids,
            'entry_time' : self.entry_time,
            'status'     : self.status,
        }

    def __repr__(self):
        return (f"Group({self.group_id} | {'OPEN' if self.is_open else 'CLOSED'}"
                f" | {self.status} | A={self.adult_ids} C={self.child_ids})")


# ── GroupManager ──────────────────────────────────────────────

class GroupManager:
    """
    Answers: who entered with whom? did they exit together?

    main.py usage:
        gm = GroupManager()
        exit_events = gm.update(tracks)   # every frame
        records     = gm.get_db_records() # for database
    """

    def __init__(self):
        self._groups        = {}   # group_id  -> Group
        self._member_index  = {}   # track_id  -> group_id
        self._pending       = {}   # track_id  -> MemberInfo
        self._exit_counter  = {}   # track_id  -> consecutive exit frames
        self._group_counter = 0

    # ── Public API ────────────────────────────────────────────

    def update(self, tracks):
        """Called every frame. Returns List[dict] exit_events."""
        self._close_expired_groups()
        self._cleanup_expired_pending()

        for t in tracks:
            if t['direction'] == 'entry':
                self._handle_entry(t)

        exiting_ids = {t['track_id'] for t in tracks if t['direction'] == 'exit'}
        confirmed   = self._confirm_exits(exiting_ids, tracks)

        exit_events = [e for tid in confirmed
                       for e in [self._handle_exit(tid)] if e]

        for g in self._groups.values():
            g.refresh_status()

        return exit_events

    def get_db_records(self):
        return [g.to_db_record() for g in self._groups.values()]

    def get_group_of(self, track_id):
        """Returns group_id for a track_id — used by alert_engine and matcher."""
        return self._member_index.get(track_id)

    def get_monitored_groups(self):
        """Groups with at least one child that have not fully exited."""
        return [g for g in self._groups.values()
                if g.has_child and not g.all_exited]

    def same_group(self, id_a, id_b):
        """True if both persons are in the same group — for stranger detection."""
        a = self._member_index.get(id_a)
        return a is not None and a == self._member_index.get(id_b)

    def is_pending(self, track_id):
        return track_id in self._pending

    def get_groups(self):
        return list(self._groups.values())

    # ── Entry Logic ───────────────────────────────────────────

    def _handle_entry(self, track):
        tid, label, centroid, now = (
            track['track_id'], track['label'],
            track['centroid'], time.monotonic()
        )

        if tid in self._member_index:
            gid = self._member_index[tid]
            if tid in self._groups.get(gid, Group('', 0)).members:
                self._groups[gid].members[tid].centroid = centroid
            return

        if tid in self._pending:
            self._pending[tid].centroid = centroid
            return

        member     = MemberInfo(tid, label, centroid, now)
        candidates = self._find_candidate_groups(centroid, now)

        if len(candidates) == 1:
            self._join_group(candidates[0], member)

        elif len(candidates) > 1:
            if label == 'child':
                self._create_group(member, now)
            else:
                self._pending[tid] = member

        else:
            nearby = self._find_nearest_pending(centroid, now)
            if nearby is not None:
                gid = self._create_group(self._pending.pop(nearby), now)
                self._join_group(gid, member)
            elif label == 'child':
                self._create_group(member, now)
            else:
                self._pending[tid] = member

    def _find_candidate_groups(self, centroid, now):
        return [
            gid for gid, g in self._groups.items()
            if g.is_open
            and (now - g.entry_time) < ENTRY_TIME_WINDOW
            and (avg := g.centroid_avg()) is not None
            and math.dist(centroid, avg) < DISTANCE_THRESHOLD
        ]

    def _find_nearest_pending(self, centroid, now):
        best_tid, best_dist = None, float('inf')
        for tid, m in self._pending.items():
            if (now - m.entry_time) >= ENTRY_TIME_WINDOW:
                continue
            d = math.dist(centroid, m.centroid)
            if d < DISTANCE_THRESHOLD and d < best_dist:
                best_tid, best_dist = tid, d
        return best_tid

    def _join_group(self, gid, member):
        self._groups[gid].members[member.track_id] = member
        self._member_index[member.track_id] = gid

    def _create_group(self, first_member, now):
        self._group_counter += 1
        gid   = f"G{self._group_counter}"
        group = Group(gid, now)
        group.members[first_member.track_id] = first_member
        self._groups[gid] = group
        self._member_index[first_member.track_id] = gid
        logger.info(f"[GROUP] {gid} created — id={first_member.track_id} ({first_member.label})")
        return gid

    def _close_expired_groups(self):
        now = time.monotonic()
        for g in self._groups.values():
            if g.is_open and (now - g.entry_time) >= ENTRY_TIME_WINDOW:
                g.is_open = False
                g.refresh_status()
                logger.info(f"[GROUP] {g.group_id} closed — A={g.adult_ids} C={g.child_ids}")

    def _cleanup_expired_pending(self):
        now     = time.monotonic()
        expired = [tid for tid, m in self._pending.items()
                   if (now - m.entry_time) >= ENTRY_TIME_WINDOW]
        for tid in expired:
            del self._pending[tid]

    # ── Exit Logic ────────────────────────────────────────────

    def _confirm_exits(self, exiting_ids, all_tracks):
        tracked = set(self._member_index) | set(self._pending)
        for tid in exiting_ids:
            if tid in tracked:
                self._exit_counter[tid] = self._exit_counter.get(tid, 0) + 1
        for tid in ({t['track_id'] for t in all_tracks} - exiting_ids):
            if tid in self._exit_counter:
                self._exit_counter[tid] = 0
        confirmed = [tid for tid, c in self._exit_counter.items()
                     if c >= EXIT_CONFIRM_FRAMES]
        for tid in confirmed:
            del self._exit_counter[tid]
        return confirmed

    def _handle_exit(self, track_id):
        """Returns raw exit_event dict — alert_engine decides the alert type."""
        if track_id in self._pending:
            del self._pending[track_id]
            return None

        if track_id not in self._member_index:
            return None

        gid   = self._member_index[track_id]
        group = self._groups.get(gid)
        if not group:
            return None

        m            = group.members[track_id]
        m.status     = 'exited'
        m.exit_time  = time.monotonic()

        return {
            'group_id'          : gid,
            'exited_id'         : track_id,
            'exited_label'      : m.label,
            'remained_ids'      : [x.track_id for x in group.members.values() if x.status == 'inside'],
            'remained_children' : [x.track_id for x in group.members.values() if x.status == 'inside' and x.label == 'child'],
            'remained_adults'   : [x.track_id for x in group.members.values() if x.status == 'inside' and x.label == 'adult'],
            'all_adults_exited' : group.all_adults_exited(),
            'group_has_child'   : group.has_child,
            'timestamp'         : m.exit_time,
        }

    # ── Debug ─────────────────────────────────────────────────

    def print_status(self):
        print("\n" + "=" * 50)
        print(f"  Groups: {len(self._groups)} | Pending: {len(self._pending)}")
        print("=" * 50)
        for g in self._groups.values():
            print(f"  {'OPEN  ' if g.is_open else 'CLOSED'} {g.group_id} | {g.status}")
            print(f"    Adults  : {g.inside_adults}")
            print(f"    Children: {g.inside_children}")
        if self._pending:
            print(f"  Pending : {list(self._pending.keys())}")
        print("=" * 50 + "\n")


# ── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    def fake(tid, label, direction, cx=100, cy=200):
        return {'track_id': tid, 'label': label, 'direction': direction,
                'centroid': (cx, cy), 'bbox': [0, 0, 10, 10], 'conf': 0.9}

    gm = GroupManager()

    print("--- Entry: adult + child ---")
    gm.update([fake(1, 'adult', 'entry', 100, 200),
               fake(2, 'child', 'entry', 120, 200)])
    gm.print_status()

    print("--- Close window ---")
    gm._groups['G1'].entry_time -= 4.1
    gm.update([fake(1, 'adult', 'stationary'), fake(2, 'child', 'stationary')])
    gm.print_status()

    print("--- Adult exits ---")
    for _ in range(8):
        events = gm.update([fake(1, 'adult', 'exit'), fake(2, 'child', 'stationary')])
    for e in events:
        print(f"  event: {e}")

    print("--- Child exits ---")
    for _ in range(8):
        events = gm.update([fake(2, 'child', 'exit')])
    for e in events:
        print(f"  event: {e}")

    gm.print_status()
