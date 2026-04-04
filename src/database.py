import sqlite3
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import DB_PATH


class Database:

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS groups (
                group_id   TEXT PRIMARY KEY,
                adult_ids  TEXT,
                child_ids  TEXT,
                entry_time REAL,
                status     TEXT
            );
            CREATE TABLE IF NOT EXISTS members (
                track_id   INTEGER PRIMARY KEY,
                group_id   TEXT,
                label      TEXT,
                embedding  BLOB,
                entry_time REAL,
                exit_time  REAL,
                status     TEXT DEFAULT 'inside'
            );
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id     TEXT,
                alert_type   TEXT,
                who_exited   TEXT,
                who_remained TEXT,
                timestamp    REAL
            );
            CREATE TABLE IF NOT EXISTS logs (
                log_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id  TEXT,
                track_id  INTEGER,
                event     TEXT,
                timestamp REAL
            );
        """)
        self.conn.commit()

    def save_group(self, record):
        self.conn.execute("""
            INSERT INTO groups (group_id, adult_ids, child_ids, entry_time, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(group_id) DO UPDATE SET
                adult_ids = excluded.adult_ids,
                child_ids = excluded.child_ids,
                status    = excluded.status
        """, (
            record['group_id'],
            json.dumps(record['adult_ids']),
            json.dumps(record['child_ids']),
            record['entry_time'],
            record['status'],
        ))
        self.conn.commit()

    def get_group(self, group_id):
        row = self.conn.execute(
            "SELECT * FROM groups WHERE group_id = ?", (group_id,)
        ).fetchone()
        if not row:
            return None
        return {
            'group_id'  : row['group_id'],
            'adult_ids' : json.loads(row['adult_ids']),
            'child_ids' : json.loads(row['child_ids']),
            'entry_time': row['entry_time'],
            'status'    : row['status'],
        }

    def save_member(self, track_id, group_id, label, embedding, entry_time):
        emb_bytes = embedding.tobytes() if embedding is not None else None
        self.conn.execute("""
            INSERT INTO members (track_id, group_id, label, embedding, entry_time)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
                group_id  = excluded.group_id,
                embedding = excluded.embedding
        """, (track_id, group_id, label, emb_bytes, entry_time))
        self.conn.commit()

    def update_member_exit(self, track_id, exit_time):
        self.conn.execute(
            "UPDATE members SET status = 'exited', exit_time = ? WHERE track_id = ?",
            (exit_time, track_id)
        )
        self.conn.commit()

    def get_embedding(self, track_id):
        row = self.conn.execute(
            "SELECT embedding FROM members WHERE track_id = ?", (track_id,)
        ).fetchone()
        if not row or not row['embedding']:
            return None
        return np.frombuffer(row['embedding'], dtype=np.float32)

    def save_alert(self, alert):
        self.conn.execute("""
            INSERT INTO alerts (group_id, alert_type, who_exited, who_remained, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            alert['group_id'],
            alert['alert_type'],
            json.dumps(alert['who_exited']),
            json.dumps(alert['who_remained']),
            alert['timestamp'],
        ))
        self.conn.commit()

    def log_event(self, group_id, track_id, event, timestamp):
        self.conn.execute(
            "INSERT INTO logs (group_id, track_id, event, timestamp) VALUES (?, ?, ?, ?)",
            (group_id, track_id, event, timestamp)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import time

    db = Database()

    db.save_group({'group_id': 'G1', 'adult_ids': [1], 'child_ids': [2],
                   'entry_time': time.monotonic(), 'status': 'active'})

    emb = np.random.rand(512).astype(np.float32)
    db.save_member(1, 'G1', 'adult', emb, time.monotonic())
    db.save_member(2, 'G1', 'child', emb, time.monotonic())

    assert db.get_embedding(1).shape == (512,)
    assert db.get_group('G1')['group_id'] == 'G1'

    db.update_member_exit(1, time.monotonic())

    db.save_alert({'group_id': 'G1', 'alert_type': 'adult_left_child',
                   'who_exited': [1], 'who_remained': [2], 'timestamp': time.monotonic()})

    db.log_event('G1', 1, 'entry', time.monotonic())
    db.log_event('G1', 1, 'exit',  time.monotonic())

    db.close()
    print("All tests passed.")
