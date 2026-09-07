"""
audit_repository.py — Write operations for the audit database.

Every decision, recommendation, alert, and what-if run is persisted here.
Healthcare audit requirements: who decided what, with which model, on which
inputs, at what time. This module is the single write path to the DB.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.review import REVIEW_DECISIONS
from src.storage.db import DB_PATH, get_connection, initialize_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _make_review_id() -> str:
    return f"review_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"


HASH_ALGORITHM = "SHA-256"
GENESIS_HASH = "GENESIS"


def _review_event_hash(
    *,
    review_id: str,
    run_id: str,
    reviewed_at: str,
    reviewed_by: str,
    decision: str,
    comment: str,
    autonomous_execution_permitted: int,
    previous_hash: str,
) -> str:
    payload = {
        "autonomous_execution_permitted": int(autonomous_execution_permitted),
        "comment": comment,
        "decision": decision,
        "previous_hash": previous_hash,
        "review_id": review_id,
        "reviewed_at": reviewed_at,
        "reviewed_by": reviewed_by,
        "run_id": run_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _backfill_review_chain(conn: sqlite3.Connection) -> None:
    """Populate hashes for pre-chain rows without repairing non-null tampering."""
    rows = conn.execute("SELECT * FROM decision_reviews ORDER BY id ASC").fetchall()
    previous_hash = GENESIS_HASH
    for row in rows:
        computed = _review_event_hash(
            review_id=row["review_id"],
            run_id=row["run_id"],
            reviewed_at=row["reviewed_at"],
            reviewed_by=row["reviewed_by"],
            decision=row["decision"],
            comment=row["comment"],
            autonomous_execution_permitted=row["autonomous_execution_permitted"],
            previous_hash=previous_hash,
        )
        if row["previous_hash"] is None or row["event_hash"] is None:
            conn.execute(
                """
                UPDATE decision_reviews
                SET hash_algorithm = ?, previous_hash = ?, event_hash = ?
                WHERE id = ?
                """,
                (HASH_ALGORITHM, previous_hash, computed, row["id"]),
            )
        previous_hash = row["event_hash"] or computed


def _prepare_review_chain(db_path: Path) -> None:
    initialize_schema(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            _backfill_review_chain(conn)
    finally:
        conn.close()


def verify_decision_review_chain(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Verify the append-only SHA-256 chain without changing non-null records."""
    _prepare_review_chain(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM decision_reviews ORDER BY id ASC").fetchall()
    finally:
        conn.close()

    previous_hash = GENESIS_HASH
    violations: list[dict[str, Any]] = []
    for row in rows:
        expected = _review_event_hash(
            review_id=row["review_id"],
            run_id=row["run_id"],
            reviewed_at=row["reviewed_at"],
            reviewed_by=row["reviewed_by"],
            decision=row["decision"],
            comment=row["comment"],
            autonomous_execution_permitted=row["autonomous_execution_permitted"],
            previous_hash=previous_hash,
        )
        if row["hash_algorithm"] != HASH_ALGORITHM:
            violations.append({"review_id": row["review_id"], "code": "hash_algorithm_mismatch"})
        if row["previous_hash"] != previous_hash:
            violations.append({"review_id": row["review_id"], "code": "previous_hash_mismatch"})
        if row["event_hash"] != expected:
            violations.append({"review_id": row["review_id"], "code": "event_hash_mismatch"})
        previous_hash = row["event_hash"] or expected

    return {
        "status": "VALID" if not violations else "INVALID",
        "review_count": len(rows),
        "head_hash": previous_hash,
        "hash_algorithm": HASH_ALGORITHM,
        "genesis_hash": GENESIS_HASH,
        "violations": violations,
        "checked_at": _now(),
        "autonomous_execution_permitted": False,
    }


class AuditRunNotFoundError(ValueError):
    """Raised when a review references a decision run absent from the audit log."""


def log_decision_run(
    scenario: str,
    overall_alert: str,
    pipeline_mode: str = "demo",
    runtime_seconds: Optional[float] = None,
    db_path: Path = DB_PATH,
) -> str:
    """
    Insert a new decision run record. Returns the run_id, which callers
    use to associate recommendations, alerts, and what-if results.
    """
    initialize_schema(db_path)
    run_id = _make_run_id()
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO decision_runs (run_id, timestamp, pipeline_mode,
                                       scenario, overall_alert, runtime_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, _now(), pipeline_mode, scenario, overall_alert, runtime_seconds),
        )
    conn.close()
    return run_id


def get_decision_run(run_id: str, db_path: Path = DB_PATH) -> Optional[Dict]:
    """Return one persisted decision run, including its immutable manifest reference."""
    initialize_schema(db_path)
    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT d.*, m.manifest_path, m.manifest_sha256, m.created_at AS manifest_created_at
        FROM decision_runs d
        LEFT JOIN run_manifests m ON d.run_id = m.run_id
        WHERE d.run_id = ?
        """,
        (run_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def log_decision_review(
    run_id: str,
    reviewed_by: str,
    decision: str,
    comment: str,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """Append an immutable human disposition to a persisted decision run."""
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"Unsupported review decision: {decision}")
    _prepare_review_chain(db_path)
    review_id = _make_review_id()
    reviewed_at = _now()
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute(
            "SELECT run_id FROM decision_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise AuditRunNotFoundError(f"Unknown decision run: {run_id}")
        previous = conn.execute(
            "SELECT event_hash FROM decision_reviews ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous["event_hash"] if previous and previous["event_hash"] else GENESIS_HASH
        event_hash = _review_event_hash(
            review_id=review_id,
            run_id=run_id,
            reviewed_at=reviewed_at,
            reviewed_by=reviewed_by,
            decision=decision,
            comment=comment,
            autonomous_execution_permitted=0,
            previous_hash=previous_hash,
        )
        conn.execute(
            """
            INSERT INTO decision_reviews (
                review_id, run_id, reviewed_at, reviewed_by, decision,
                comment, autonomous_execution_permitted, hash_algorithm,
                previous_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (review_id, run_id, reviewed_at, reviewed_by, decision, comment,
             HASH_ALGORITHM, previous_hash, event_hash),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "review_id": review_id,
        "run_id": run_id,
        "reviewed_at": reviewed_at,
        "reviewed_by": reviewed_by,
        "decision": decision,
        "comment": comment,
        "autonomous_execution_permitted": False,
        "hash_algorithm": HASH_ALGORITHM,
        "previous_hash": previous_hash,
        "event_hash": event_hash,
    }


def _review_payload(row: Any) -> Dict[str, Any]:
    payload = dict(row)
    payload.pop("id", None)
    payload["autonomous_execution_permitted"] = bool(
        payload.get("autonomous_execution_permitted", 0)
    )
    return payload


def get_recent_decision_reviews(n: int = 20, db_path: Path = DB_PATH) -> List[Dict]:
    """Return the most recent human-review events in reverse chronological order."""
    _prepare_review_chain(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM decision_reviews ORDER BY reviewed_at DESC, id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [_review_payload(row) for row in rows]


def get_reviews_for_run(run_id: str, db_path: Path = DB_PATH) -> List[Dict]:
    """Return the immutable review history for one decision run."""
    _prepare_review_chain(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT * FROM decision_reviews
        WHERE run_id = ?
        ORDER BY reviewed_at ASC, id ASC
        """,
        (run_id,),
    ).fetchall()
    conn.close()
    return [_review_payload(row) for row in rows]


def read_recent_decision_reviews(n: int = 20, db_path: Path = DB_PATH) -> List[Dict]:
    """Read review history without creating or migrating an audit database."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM decision_reviews ORDER BY reviewed_at DESC, id DESC LIMIT ?", (n,)
        ).fetchall()
    except sqlite3.OperationalError:
        # Older installations may predate the review table; the read model
        # remains available while the governed write path can migrate safely.
        return []
    finally:
        conn.close()
    return [_review_payload(row) for row in rows]


def log_recommendation(
    run_id: str,
    selected_policy: str,
    decision_score: Optional[float] = None,
    predicted_regime: Optional[str] = None,
    unsafe_risk_prob: Optional[float] = None,
    utilization_critical_prob: Optional[float] = None,
    predicted_blocked_arrivals: Optional[float] = None,
    predicted_unsafe_excess: Optional[float] = None,
    db_path: Path = DB_PATH,
) -> None:
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO recommendations (
                run_id, selected_policy, decision_score, predicted_regime,
                unsafe_risk_prob, utilization_critical_prob,
                predicted_blocked_arrivals, predicted_unsafe_excess, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, selected_policy, decision_score, predicted_regime,
                unsafe_risk_prob, utilization_critical_prob,
                predicted_blocked_arrivals, predicted_unsafe_excess, _now(),
            ),
        )
    conn.close()


def log_alerts(
    run_id: str,
    alerts: List[Dict[str, str]],
    db_path: Path = DB_PATH,
) -> None:
    """
    Persist a list of alert dicts (each with 'level', 'trigger', 'message').
    Accepts either dataclass instances (with those attributes) or plain dicts.
    """
    if not alerts:
        return
    conn = get_connection(db_path)
    now = _now()
    rows = []
    for a in alerts:
        if hasattr(a, "level"):
            rows.append((run_id, a.level, a.trigger, a.message, now))
        else:
            rows.append((run_id, a["level"], a["trigger"], a["message"], now))
    with conn:
        conn.executemany(
            "INSERT INTO alerts (run_id, level, trigger, message, created_at) VALUES (?,?,?,?,?)",
            rows,
        )
    conn.close()


def log_whatif_run(
    icu_bed_delta: int,
    transfer_multiplier: float,
    demand_multiplier: float,
    n_replications: int,
    result: Dict[str, Any],
    surface: str = "sandbox",
    db_path: Path = DB_PATH,
) -> str:
    """
    Persist a live what-if solve result and make it reviewable through the
    governed decision-review workflow. Returns the run_id.
    surface: 'sandbox', 'copilot', or 'event_simulator'
    """
    initialize_schema(db_path)
    run_id = _make_run_id()
    conn = get_connection(db_path)
    with conn:
        # Scenario evaluations are what-if records for operational analytics,
        # but they must also be first-class governed runs if an operator is
        # going to review the recommendation. The shared run id lets the
        # existing immutable review chain and foreign key protect both views.
        conn.execute(
            """
            INSERT INTO decision_runs (
                run_id, timestamp, pipeline_mode, scenario,
                overall_alert, runtime_seconds
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _now(),
                "scenario_evaluation",
                f"whatif:{surface}",
                None,
                result.get("solve_runtime_seconds"),
            ),
        )
        conn.execute(
            """
            INSERT INTO whatif_runs (
                run_id, timestamp, icu_bed_delta, transfer_multiplier,
                demand_multiplier, n_replications, total_unsafe_excess,
                max_utilization, blocked_arrivals, solve_seconds, surface
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, _now(), icu_bed_delta, transfer_multiplier,
                demand_multiplier, n_replications,
                result.get("total_unsafe_excess"),
                result.get("max_utilization_ratio"),
                result.get("total_blocked_arrivals"),
                result.get("solve_runtime_seconds"),
                surface,
            ),
        )
    conn.close()
    return run_id


def log_run_manifest(
    run_id: str,
    manifest_path: Path | str,
    manifest_sha256: str,
    db_path: Path = DB_PATH,
) -> None:
    """Persist the immutable manifest reference for any audited run."""
    initialize_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO run_manifests (run_id, manifest_path, manifest_sha256, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                manifest_path = excluded.manifest_path,
                manifest_sha256 = excluded.manifest_sha256,
                created_at = excluded.created_at
            """,
            (run_id, str(manifest_path), manifest_sha256, _now()),
        )
    conn.close()


def get_run_manifest(run_id: str, db_path: Path = DB_PATH) -> Optional[Dict]:
    initialize_schema(db_path)
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM run_manifests WHERE run_id = ?", (run_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def get_whatif_run(run_id: str, db_path: Path = DB_PATH) -> Optional[Dict]:
    """Return one persisted scenario/what-if run for packet composition."""
    initialize_schema(db_path)
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM whatif_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row is not None else None


def log_model_registry(
    task: str,
    model_name: str,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
    model_path: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Upsert a model record (INSERT OR REPLACE on task+model_name unique key)."""
    initialize_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO model_registry
                (registered_at, task, model_name, metric_name, metric_value, model_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(task, model_name) DO UPDATE SET
                registered_at = excluded.registered_at,
                metric_name   = excluded.metric_name,
                metric_value  = excluded.metric_value,
                model_path    = excluded.model_path
            """,
            (_now(), task, model_name, metric_name, metric_value, model_path),
        )
    conn.close()


def get_recent_runs(n: int = 10, db_path: Path = DB_PATH) -> List[Dict]:
    """Return the n most recent decision runs as plain dicts."""
    initialize_schema(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT * FROM decision_runs
        WHERE pipeline_mode IS NULL OR pipeline_mode <> 'scenario_evaluation'
        ORDER BY timestamp DESC LIMIT ?
        """,
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]



def get_recent_audit_runs(n: int = 10, db_path: Path = DB_PATH) -> List[Dict]:
    """Return decision and what-if runs in one reverse-chronological audit trail."""
    initialize_schema(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT
                'decision' AS record_type,
                d.run_id,
                d.timestamp,
                d.pipeline_mode AS source,
                d.scenario,
                d.overall_alert,
                d.runtime_seconds,
                NULL AS icu_bed_delta,
                NULL AS transfer_multiplier,
                NULL AS demand_multiplier,
                NULL AS n_replications,
                NULL AS total_unsafe_excess,
                NULL AS max_utilization,
                NULL AS blocked_arrivals,
                m.manifest_path,
                m.manifest_sha256
            FROM decision_runs d
            LEFT JOIN run_manifests m ON d.run_id = m.run_id
            WHERE d.pipeline_mode IS NULL
               OR d.pipeline_mode <> 'scenario_evaluation'

            UNION ALL

            SELECT
                'what_if' AS record_type,
                w.run_id,
                w.timestamp,
                w.surface AS source,
                NULL AS scenario,
                NULL AS overall_alert,
                w.solve_seconds AS runtime_seconds,
                w.icu_bed_delta,
                w.transfer_multiplier,
                w.demand_multiplier,
                w.n_replications,
                w.total_unsafe_excess,
                w.max_utilization,
                w.blocked_arrivals,
                m.manifest_path,
                m.manifest_sha256
            FROM whatif_runs w
            LEFT JOIN run_manifests m ON w.run_id = m.run_id
        )
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_alerts(n: int = 20, db_path: Path = DB_PATH) -> List[Dict]:
    initialize_schema(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT a.*, d.timestamp as run_timestamp, d.scenario
        FROM alerts a
        JOIN decision_runs d ON a.run_id = d.run_id
        ORDER BY a.created_at DESC LIMIT ?
        """,
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_whatif_runs(n: int = 20, db_path: Path = DB_PATH) -> List[Dict]:
    initialize_schema(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM whatif_runs ORDER BY timestamp DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
