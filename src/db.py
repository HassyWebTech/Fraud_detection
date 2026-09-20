import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager


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
    
        (user_id, baseline["last_known_device"], baseline["typical_amount_band_low"],
         baseline["typical_amount_band_high"], json.dumps(list(baseline["known_recipients"])),
         baseline["typical_hour_range"][0], baseline["typical_hour_range"][1], now_iso()),
    )


def log_decision(conn, user_id: int, engine: str, tier: str, reasons: list,
                  session_id: int = None, risk_score: float = None,
                  latency_ms: float = None, true_label: int = None) -> int:
    """Every scored session goes through here — this is the "nothing goes
    untracked" guarantee. Returns the decision_id."""
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