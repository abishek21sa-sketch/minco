"""SQLite connection and schema management for MINCO audit records."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config.paths import AUDIT_DB_PATH

DB_PATH = AUDIT_DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_schema(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decision_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                pipeline_mode TEXT NOT NULL DEFAULT 'demo',
                scenario TEXT,
                overall_alert TEXT,
                runtime_seconds REAL
            );
            CREATE TABLE IF NOT EXISTS decision_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL REFERENCES decision_runs(run_id),
                reviewed_at TEXT NOT NULL,
                reviewed_by TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(
                    decision IN (
                        'ACCEPTED_FOR_OPERATIONS_REVIEW',
                        'DEFERRED',
                        'REJECTED'
                    )
                ),
                comment TEXT NOT NULL,
                autonomous_execution_permitted INTEGER NOT NULL DEFAULT 0 CHECK(
                    autonomous_execution_permitted = 0
                )
            );
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES decision_runs(run_id),
                selected_policy TEXT NOT NULL,
                decision_score REAL,
                predicted_regime TEXT,
                unsafe_risk_prob REAL,
                utilization_critical_prob REAL,
                predicted_blocked_arrivals REAL,
                predicted_unsafe_excess REAL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES decision_runs(run_id),
                level TEXT NOT NULL,
                trigger TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS whatif_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                icu_bed_delta INTEGER NOT NULL DEFAULT 0,
                transfer_multiplier REAL NOT NULL DEFAULT 1.0,
                demand_multiplier REAL NOT NULL DEFAULT 1.0,
                n_replications INTEGER,
                total_unsafe_excess REAL,
                max_utilization REAL,
                blocked_arrivals REAL,
                solve_seconds REAL,
                surface TEXT DEFAULT 'sandbox'
            );
            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                registered_at TEXT NOT NULL,
                task TEXT NOT NULL,
                model_name TEXT NOT NULL,
                metric_name TEXT,
                metric_value REAL,
                model_path TEXT,
                UNIQUE(task, model_name)
            );
            CREATE TABLE IF NOT EXISTS run_manifests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                manifest_path TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decision_runs_timestamp ON decision_runs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_decision_reviews_run_id ON decision_reviews(run_id);
            CREATE INDEX IF NOT EXISTS idx_decision_reviews_reviewed_at ON decision_reviews(reviewed_at);
            CREATE INDEX IF NOT EXISTS idx_alerts_run_id ON alerts(run_id);
            CREATE INDEX IF NOT EXISTS idx_whatif_runs_timestamp ON whatif_runs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_run_manifests_run_id ON run_manifests(run_id);
            """
        )
        existing_review_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(decision_reviews)").fetchall()
        }
        migrations = {
            "hash_algorithm": "TEXT NOT NULL DEFAULT 'SHA-256'",
            "previous_hash": "TEXT",
            "event_hash": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in existing_review_columns:
                conn.execute(f"ALTER TABLE decision_reviews ADD COLUMN {column} {definition}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decision_reviews_event_hash "
            "ON decision_reviews(event_hash)"
        )
    conn.close()


if __name__ == "__main__":
    initialize_schema()
    print(f"Schema initialized at {DB_PATH}")
