"""Database access for surveillance events."""
import sqlite3
from datetime import datetime
from config import DB_PATH


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                event_type    TEXT NOT NULL,
                camera_id     TEXT NOT NULL,
                confidence    REAL,
                snapshot_path TEXT,
                details       TEXT,
                reviewed      INTEGER DEFAULT 0
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_time ON events(timestamp)")


def add_event(event_type, camera_id, confidence, snapshot_path, details=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO events (timestamp, event_type, camera_id, confidence,"
            " snapshot_path, details) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, event_type, camera_id, confidence, str(snapshot_path), details),
        )
        return cur.lastrowid


def get_events(event_type=None, camera_id=None, date_from=None, date_to=None,
               limit=200):
    query, params = "SELECT * FROM events WHERE 1=1", []
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if date_from:
        query += " AND timestamp >= ?"
        params.append(f"{date_from} 00:00:00")
    if date_to:
        query += " AND timestamp <= ?"
        params.append(f"{date_to} 23:59:59")
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def mark_reviewed(event_id):
    with _conn() as c:
        c.execute("UPDATE events SET reviewed = 1 WHERE id = ?", (event_id,))