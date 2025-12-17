#!/usr/bin/env python3
"""
Claude Code SessionEnd Hook for Real-time Metrics

Captures session cost metrics from Claude Code SessionEnd events
and stores them in SQLite for Grafana visualization.

Hook input (stdin JSON):
{
  "session_id": "uuid",
  "transcript_path": "/path/to/transcript.jsonl",
  "cost": {
    "total_cost_usd": 0.36932425,
    "total_duration_ms": 93979,
    "total_api_duration_ms": 18361,
    "total_lines_added": 0,
    "total_lines_removed": 0
  }
}

Hook output (stdout JSON): {} (empty object satisfies hook interface)
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_DB_PATH = Path.home() / '.claude' / 'claude_metrics.db'


def run_background(data_file: str, db_path: str) -> None:
    """Background worker - does actual metrics storage."""
    try:
        with open(data_file) as f:
            session_data = json.load(f)
        os.unlink(data_file)
        _do_metrics(session_data, Path(db_path))
    except Exception:
        pass


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with same schema as parser."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create sessions table matching parser schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            total_cost_usd REAL DEFAULT 0,
            total_duration_ms INTEGER DEFAULT 0,
            total_api_duration_ms INTEGER DEFAULT 0,
            lines_added INTEGER DEFAULT 0,
            lines_removed INTEGER DEFAULT 0,
            commits INTEGER DEFAULT 0,
            tool_calls INTEGER DEFAULT 0
        )
    """)

    # Create token_usage table for granular metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            model TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_creation_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # Create indexes for efficient querying
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp
        ON token_usage(timestamp)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_token_usage_session
        ON token_usage(session_id)
    """)

    conn.commit()
    return conn


def store_session_metrics(conn: sqlite3.Connection, session_data: dict) -> None:
    """Store session cost metrics in SQLite using INSERT OR REPLACE."""
    cursor = conn.cursor()

    session_id = session_data.get('session_id')
    if not session_id:
        return

    cost = session_data.get('cost', {})

    # Use INSERT OR REPLACE to upsert session data
    cursor.execute("""
        INSERT OR REPLACE INTO sessions (
            session_id,
            start_time,
            end_time,
            total_cost_usd,
            total_duration_ms,
            total_api_duration_ms,
            lines_added,
            lines_removed,
            commits,
            tool_calls
        ) VALUES (
            ?,
            COALESCE((SELECT start_time FROM sessions WHERE session_id = ?), datetime('now')),
            datetime('now'),
            ?,
            ?,
            ?,
            ?,
            ?,
            COALESCE((SELECT commits FROM sessions WHERE session_id = ?), 0),
            COALESCE((SELECT tool_calls FROM sessions WHERE session_id = ?), 0)
        )
    """, (
        session_id,
        session_id,
        cost.get('total_cost_usd', 0),
        cost.get('total_duration_ms', 0),
        cost.get('total_api_duration_ms', 0),
        cost.get('total_lines_added', 0),
        cost.get('total_lines_removed', 0),
        session_id,
        session_id,
    ))

    conn.commit()


def _do_metrics(session_data: dict, db_path: Path) -> None:
    """Actual metrics work - runs in background."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = init_db(db_path)
        try:
            store_session_metrics(conn, session_data)
        finally:
            conn.close()
    except Exception:
        pass


def main() -> None:
    """Main hook entry point - fire and forget."""
    parser = argparse.ArgumentParser(description='SessionEnd hook for metrics')
    parser.add_argument('--db-path', type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument('--background', type=str, help='Background mode with data file')
    args = parser.parse_args()

    # Handle background mode
    if args.background:
        run_background(args.background, args.db_path)
        return

    # Read stdin and spawn background worker
    try:
        session_data = json.load(sys.stdin)
        fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="session-metrics-")
        with os.fdopen(fd, "w") as f:
            json.dump(session_data, f)

        subprocess.Popen(
            [sys.executable, __file__, "--background", temp_path, "--db-path", args.db_path],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        pass

    print(json.dumps({}))


if __name__ == '__main__':
    main()
