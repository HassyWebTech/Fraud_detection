
import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    channel TEXT NOT NULL CHECK (channel IN ('app', 'ussd')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cached_baselines (
    user_id INTEGER PRIMARY KEY REFERENCES users(user_id),
    last_known_device TEXT,
    typical_amount_band_low REAL,
    typical_amount_band_high REAL,
    known_recipients TEXT,          -- JSON array, kept small deliberately
    typical_hour_start INTEGER,
    typical_hour_end INTEGER,
    synced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    user_id INTEGER NOT NULL,
    scored_at TEXT NOT NULL,
    engine TEXT NOT NULL CHECK (engine IN ('ml_model', 'offline_fallback')),
    risk_score REAL,                -- NULL for rule-based (uses tier instead)
    tier TEXT NOT NULL CHECK (tier IN ('allow', 'monitor', 'block')),
    reasons TEXT NOT NULL,          -- JSON array of human-readable reasons
    latency_ms REAL,
    true_label INTEGER              -- NULL until known (offline eval / later confirmation)
);

CREATE TABLE IF NOT EXISTS reconciliation_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL REFERENCES decisions(decision_id),
    queued_at TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT,
    final_tier TEXT,
    tier_changed INTEGER            -- 1 if the full model disagreed with the offline call
);

CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_decisions_tier ON decisions(tier);
CREATE INDEX IF NOT EXISTS idx_reconciliation_unresolved ON reconciliation_queue(resolved);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn(db_path: str = "../data/ato.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = "../data/ato.db"):
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_user_and_baseline(conn, user_id: int, channel: str, baseline: dict):
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, channel, created_at) VALUES (?, ?, ?)",
        (user_id, channel, now_iso()),
    )
    conn.execute(
        """INSERT INTO cached_baselines
           (user_id, last_known_device, typical_amount_band_low, typical_amount_band_high,
            known_recipients, typical_hour_start, typical_hour_end, synced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
              last_known_device=excluded.last_known_device,
              typical_amount_band_low=excluded.typical_amount_band_low,
              typical_amount_band_high=excluded.typical_amount_band_high,
              known_recipients=excluded.known_recipients,
              typical_hour_start=excluded.typical_hour_start,
              typical_hour_end=excluded.typical_hour_end,
              synced_at=excluded.synced_at""",
        (user_id, baseline["last_known_device"], baseline["typical_amount_band_low"],
         baseline["typical_amount_band_high"], json.dumps(list(baseline["known_recipients"])),
         baseline["typical_hour_range"][0], baseline["typical_hour_range"][1], now_iso()),
    )


def log_decision(conn, user_id: int, engine: str, tier: str, reasons: list,
                  session_id: int = None, risk_score: float = None,
                  latency_ms: float = None, true_label: int = None) -> int:
   
    cur = conn.execute(
        """INSERT INTO decisions
           (session_id, user_id, scored_at, engine, risk_score, tier, reasons, latency_ms, true_label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, user_id, now_iso(), engine, risk_score, tier,
         json.dumps(reasons), latency_ms, true_label),
    )
    decision_id = cur.lastrowid

    if engine == "offline_fallback":
        conn.execute(
            "INSERT INTO reconciliation_queue (decision_id, queued_at, resolved) VALUES (?, ?, 0)",
            (decision_id, now_iso()),
        )
    return decision_id


def get_pending_reconciliation(conn, limit: int = 100):
    return conn.execute(
        """SELECT rq.queue_id, d.* FROM reconciliation_queue rq
           JOIN decisions d ON rq.decision_id = d.decision_id
           WHERE rq.resolved = 0 LIMIT ?""",
        (limit,),
    ).fetchall()


def resolve_reconciliation(conn, queue_id: int, final_tier: str, original_tier: str):
    conn.execute(
        """UPDATE reconciliation_queue
           SET resolved = 1, resolved_at = ?, final_tier = ?, tier_changed = ?
           WHERE queue_id = ?""",
        (now_iso(), final_tier, int(final_tier != original_tier), queue_id),
    )


if __name__ == "__main__":
    init_db("../data/ato.db")
    print("Database initialized at ../data/ato.db")
    with get_conn("../data/ato.db") as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print("Tables:", [t["name"] for t in tables])