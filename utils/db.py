"""
db.py
-----------------------------------------
SQLite persistence layer. Auto-initializes on first run (no external
DB server required).

Tables:
  alerts              - every alert + AI recommendation + evidence + final decision
  decision_history     - full history of every decision/override/rollback event
  high_impact_actions   - simulated high-impact actions + human confirmation
  change_review         - proposed rule/threshold changes + approve/reject
  model_versions        - model + rule version registry (for versioning/rollback)
  feedback              - analyst decisions accumulated for controlled retraining

Nothing is ever deleted. Overrides and rollbacks add new history rows;
they never overwrite or erase the original AI recommendation.
-----------------------------------------
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

from utils import audit_csv

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database.db")


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT,
            alert_type TEXT,
            timestamp TEXT,
            source_ip TEXT,
            destination_ip TEXT,
            location TEXT,
            device TEXT,
            severity TEXT,
            alert_source TEXT,
            raw_json TEXT,

            ai_recommendation TEXT,
            simple_label TEXT,
            confidence REAL,
            ml_prediction TEXT,
            ml_confidence REAL,
            novelty_score REAL,
            novelty_gate_triggered INTEGER,
            insufficient_evidence INTEGER,
            high_risk_gate_triggered INTEGER DEFAULT 0,
            strong_fp_rule_triggered INTEGER DEFAULT 0,
            rule_triggered TEXT,
            risk_level TEXT,
            low_confidence INTEGER DEFAULT 0,
            evidence_json TEXT,
            recommended_action TEXT,
            is_high_impact_action INTEGER,

            analyst_decision TEXT DEFAULT 'Pending',
            override_status INTEGER DEFAULT 0,
            override_decision TEXT,
            override_reason TEXT,

            human_confirmation TEXT DEFAULT 'Not Required',

            model_version TEXT,
            rule_version TEXT,

            used_in_retrain INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS decision_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_row_id INTEGER,
            event_type TEXT,
            decision TEXT,
            override_reason TEXT,
            changed_at TEXT,
            FOREIGN KEY (alert_row_id) REFERENCES alerts (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS high_impact_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_row_id INTEGER,
            action_type TEXT,
            reason TEXT,
            human_confirmation TEXT DEFAULT 'Pending',
            confirmed_at TEXT,
            created_at TEXT,
            FOREIGN KEY (alert_row_id) REFERENCES alerts (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS change_review (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposed_change TEXT,
            reason TEXT,
            current_version TEXT,
            proposed_version TEXT,
            change_payload TEXT,
            status TEXT DEFAULT 'Pending Review',
            reviewer TEXT,
            created_at TEXT,
            decided_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_type TEXT,          -- 'model' or 'rule'
            version TEXT,
            description TEXT,
            reason TEXT,
            reviewer TEXT,
            status TEXT DEFAULT 'active',   -- active | inactive | rolled_back
            config_snapshot TEXT,
            metrics_snapshot TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS rollback_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_type TEXT,               -- 'model' or 'rule'
            source_version_row_id INTEGER,   -- unique model_versions.id restored FROM,
                                              -- when known precisely (rule rollback always
                                              -- knows this; model rollback matches archived
                                              -- files by label+timestamp, so this is NULL there)
            target_label TEXT,               -- display label of the version restored
            resulting_version_row_id INTEGER,-- model_versions.id of the new
                                              -- '<label>-restored' row this rollback created
            restored_version TEXT,           -- display label of the resulting active version
            reason TEXT,
            reviewer TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_row_id INTEGER,
            decision TEXT,
            override_decision TEXT,
            derived_label TEXT,
            used_in_retrain INTEGER DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY (alert_row_id) REFERENCES alerts (id)
        )
    """)

    conn.commit()
    _migrate_alerts_table(conn)
    conn.close()


def _migrate_alerts_table(conn):
    """
    Lightweight forward migration: add any new alerts columns introduced by
    later releases to a pre-existing database.db so upgrades don't require
    deleting the database. Safe to call on every startup.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(alerts)")
    existing = {row["name"] for row in cur.fetchall()}
    new_columns = {
        "simple_label": "TEXT",
        "high_risk_gate_triggered": "INTEGER DEFAULT 0",
        "strong_fp_rule_triggered": "INTEGER DEFAULT 0",
        "rule_triggered": "TEXT",
        "risk_level": "TEXT",
        "low_confidence": "INTEGER DEFAULT 0",
    }
    for col, col_type in new_columns.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE alerts ADD COLUMN {col} {col_type}")
    conn.commit()


# ---------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------

def insert_alert(alert: dict, result: dict, evidence: list, model_version: str, rule_version: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO alerts (
            alert_id, alert_type, timestamp, source_ip, destination_ip, location, device,
            severity, alert_source, raw_json,
            ai_recommendation, simple_label, confidence, ml_prediction, ml_confidence,
            novelty_score, novelty_gate_triggered, insufficient_evidence,
            high_risk_gate_triggered, strong_fp_rule_triggered, rule_triggered, risk_level, low_confidence,
            evidence_json, recommended_action, is_high_impact_action,
            analyst_decision, human_confirmation, model_version, rule_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert.get("alert_id", ""), alert.get("alert_type", ""), alert.get("timestamp", ""),
        alert.get("source_ip", ""), alert.get("destination_ip", ""), alert.get("location", ""),
        alert.get("device", ""), alert.get("severity", ""), alert.get("alert_source", ""),
        json.dumps(alert),
        result["recommendation"], result.get("simple_label"), result["confidence"],
        result["ml_prediction"], result["ml_confidence"],
        result["novelty_score"], int(result["novelty_gate_triggered"]), int(result["insufficient_evidence"]),
        int(result.get("high_risk_gate_triggered", False)), int(result.get("strong_fp_rule_triggered", False)),
        result.get("rule_triggered"), result.get("risk_level"),
        int(result.get("low_confidence", False)),
        json.dumps(evidence), result["recommended_action"], int(result["is_high_impact_action"]),
        "Pending",
        "Required" if result["is_high_impact_action"] else "Not Required",
        model_version, rule_version, _now(),
    ))
    row_id = cur.lastrowid

    cur.execute("""
        INSERT INTO decision_history (alert_row_id, event_type, decision, changed_at)
        VALUES (?, 'ai_recommendation', ?, ?)
    """, (row_id, result["recommendation"], _now()))

    if result["is_high_impact_action"]:
        cur.execute("""
            INSERT INTO high_impact_actions (alert_row_id, action_type, reason, human_confirmation, created_at)
            VALUES (?, ?, ?, 'Pending', ?)
        """, (row_id, result["recommended_action"], result["recommendation"], _now()))

    conn.commit()
    conn.close()

    # Durable, plain-text audit trail (audit_log.csv) in addition to the DB.
    try:
        audit_csv.append_alert_scored(alert, result)
    except Exception:
        pass  # never let audit-log I/O errors break scoring

    return row_id


def update_decision(row_id: int, decision: str, override_decision: str = None, override_reason: str = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE alerts
        SET analyst_decision = ?, override_status = ?, override_decision = ?, override_reason = ?
        WHERE id = ?
    """, (decision, 1 if decision == "Override" else 0, override_decision, override_reason, row_id))

    cur.execute("""
        INSERT INTO decision_history (alert_row_id, event_type, decision, override_reason, changed_at)
        VALUES (?, 'analyst_decision', ?, ?, ?)
    """, (row_id, decision if decision != "Override" else f"Override -> {override_decision}",
          override_reason, _now()))

    # Record feedback for the controlled retraining loop.
    alert = get_alert(row_id)
    derived_label = None
    if alert:
        if decision == "Override" and override_decision:
            if override_decision == "Likely True Incident":
                derived_label = "True Incident"
            elif override_decision == "Likely False Positive":
                derived_label = "False Positive"
            # "Novel / Human Investigation" override -> no binary label derived
        elif decision == "Accept":
            if alert["ai_recommendation"] == "LIKELY TRUE INCIDENT":
                derived_label = "True Incident"
            elif alert["ai_recommendation"] == "LIKELY FALSE POSITIVE":
                derived_label = "False Positive"

    cur.execute("""
        INSERT INTO feedback (alert_row_id, decision, override_decision, derived_label, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (row_id, decision, override_decision, derived_label, _now()))

    conn.commit()
    conn.close()

    # Durable, plain-text audit trail (audit_log.csv) in addition to the DB.
    # Overrides are ALWAYS logged here, with their reason, alongside every
    # accept/reject -- nothing is ever recorded silently.
    try:
        if alert:
            audit_csv.append_analyst_decision(alert, decision, override_decision, override_reason)
    except Exception:
        pass  # never let audit-log I/O errors break the analyst's decision


def rollback_decision(row_id: int):
    """Revert an alert's analyst_decision to the previous decision_history entry."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT decision, changed_at FROM decision_history
        WHERE alert_row_id = ? AND event_type = 'analyst_decision'
        ORDER BY id DESC LIMIT 2
    """, (row_id,))
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 2:
        return None

    previous_decision = rows[1]["decision"]
    simple_previous = previous_decision.split(" -> ")[0]
    update_decision(row_id, simple_previous)

    conn2 = get_connection()
    cur2 = conn2.cursor()
    cur2.execute("""
        INSERT INTO decision_history (alert_row_id, event_type, decision, changed_at)
        VALUES (?, 'rollback', ?, ?)
    """, (row_id, f"Rolled back to: {simple_previous}", _now()))
    conn2.commit()
    conn2.close()
    return simple_previous


def get_all_alerts(limit=500):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_alert(row_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM alerts WHERE id = ?", (row_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_decision_history(row_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM decision_history WHERE alert_row_id = ? ORDER BY id ASC", (row_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------
# High-impact actions
# ---------------------------------------------------------------------

def confirm_high_impact_action(row_id: int, confirmation: str):
    """confirmation must be 'Approved' or 'Rejected'."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE high_impact_actions SET human_confirmation = ?, confirmed_at = ?
        WHERE alert_row_id = ?
    """, (confirmation, _now(), row_id))
    cur.execute("UPDATE alerts SET human_confirmation = ? WHERE id = ?", (confirmation, row_id))
    cur.execute("""
        INSERT INTO decision_history (alert_row_id, event_type, decision, changed_at)
        VALUES (?, 'high_impact_confirmation', ?, ?)
    """, (row_id, confirmation, _now()))
    conn.commit()
    conn.close()


def get_high_impact_action(row_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM high_impact_actions WHERE alert_row_id = ?", (row_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_pending_high_impact_actions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM high_impact_actions WHERE human_confirmation = 'Pending' ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------
# Change review
# ---------------------------------------------------------------------

def propose_change(proposed_change: str, reason: str, current_version: str,
                    proposed_version: str, change_payload: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO change_review (proposed_change, reason, current_version, proposed_version,
                                    change_payload, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'Pending Review', ?)
    """, (proposed_change, reason, current_version, proposed_version, json.dumps(change_payload), _now()))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_change_reviews():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM change_review ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_change_review(row_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM change_review WHERE id = ?", (row_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def decide_change_review(row_id: int, status: str, reviewer: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE change_review SET status = ?, reviewer = ?, decided_at = ? WHERE id = ?
    """, (status, reviewer, _now(), row_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Model / rule versioning
# ---------------------------------------------------------------------

def record_version(version_type: str, version: str, description: str, reason: str,
                    reviewer: str, config_snapshot: dict, metrics_snapshot: dict = None,
                    status: str = "active") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE model_versions SET status = 'inactive' WHERE version_type = ? AND status = 'active'",
                (version_type,))
    cur.execute("""
        INSERT INTO model_versions (version_type, version, description, reason, reviewer,
                                     status, config_snapshot, metrics_snapshot, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (version_type, version, description, reason, reviewer, status,
          json.dumps(config_snapshot), json.dumps(metrics_snapshot) if metrics_snapshot else None, _now()))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_versions(version_type: str = None):
    conn = get_connection()
    cur = conn.cursor()
    if version_type:
        cur.execute("SELECT * FROM model_versions WHERE version_type = ? ORDER BY id DESC", (version_type,))
    else:
        cur.execute("SELECT * FROM model_versions ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_active_version(version_type: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM model_versions WHERE version_type = ? AND status = 'active'
        ORDER BY id DESC LIMIT 1
    """, (version_type,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def mark_version_rolled_back(row_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE model_versions SET status = 'rolled_back' WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


def get_version_by_id(row_id: int):
    """Look up a SINGLE model_versions row by its unique database id.

    Used by rule-rollback so it can restore the EXACT historical snapshot
    the analyst selected, even when two rows happen to share the same
    display label (version labels are just numeric display bumps and are
    not guaranteed unique -- see routes/change_review.py). Looking up by
    row id removes that ambiguity entirely.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_versions WHERE id = ?", (row_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------
# Rollback audit trail (surfaces model/rule rollbacks in the SAME audit
# views used for alerts: /api/audit, /audit, audit_log.csv -- in addition
# to, not instead of, the existing model_versions history table above.)
# ---------------------------------------------------------------------

def record_rollback_audit(version_type: str, target_label: str, restored_version: str,
                           reason: str, reviewer: str,
                           source_version_row_id: int = None,
                           resulting_version_row_id: int = None) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO rollback_audit (version_type, source_version_row_id, target_label,
                                     resulting_version_row_id, restored_version, reason,
                                     reviewer, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (version_type, source_version_row_id, target_label, resulting_version_row_id,
          restored_version, reason, reviewer, _now()))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_rollback_audit_events():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM rollback_audit ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------
# Feedback / retraining
# ---------------------------------------------------------------------

def get_unused_feedback():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT f.*, a.raw_json, a.alert_id as source_alert_id
        FROM feedback f JOIN alerts a ON f.alert_row_id = a.id
        WHERE f.used_in_retrain = 0 AND f.derived_label IS NOT NULL
        ORDER BY f.id ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_feedback_stats():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) c FROM feedback")
    total = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) c FROM feedback WHERE used_in_retrain = 0 AND derived_label IS NOT NULL")
    unused = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) c FROM feedback WHERE used_in_retrain = 1")
    used = cur.fetchone()["c"]
    conn.close()
    return {"total_feedback": total, "unused_labeled_feedback": unused, "used_in_retrain": used}


def mark_feedback_used(row_ids: list):
    if not row_ids:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany("UPDATE feedback SET used_in_retrain = 1 WHERE id = ?", [(i,) for i in row_ids])
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Analytics (live application usage)
# ---------------------------------------------------------------------

def get_analytics():
    conn = get_connection()
    cur = conn.cursor()

    def count(where="", params=()):
        cur.execute(f"SELECT COUNT(*) as c FROM alerts {where}", params)
        return cur.fetchone()["c"]

    total = count()
    fp = count("WHERE ai_recommendation = 'LIKELY FALSE POSITIVE'")
    ti = count("WHERE ai_recommendation = 'LIKELY TRUE INCIDENT'")
    novel = count("WHERE ai_recommendation = 'NOVEL / UNKNOWN — HUMAN INVESTIGATION REQUIRED'")
    overrides = count("WHERE analyst_decision = 'Override'")
    accepted = count("WHERE analyst_decision = 'Accept'")
    rejected = count("WHERE analyst_decision = 'Reject'")
    pending = count("WHERE analyst_decision = 'Pending'")
    requiring_review = count("WHERE ai_recommendation != 'LIKELY FALSE POSITIVE'")

    cur.execute("SELECT alert_source, COUNT(*) c FROM alerts GROUP BY alert_source")
    by_source = {r["alert_source"] or "Unknown": r["c"] for r in cur.fetchall()}

    cur.execute("SELECT severity, COUNT(*) c FROM alerts GROUP BY severity")
    by_severity = {r["severity"] or "Unknown": r["c"] for r in cur.fetchall()}

    cur.execute("SELECT COUNT(*) c FROM change_review WHERE status = 'Pending Review'")
    pending_reviews = cur.fetchone()["c"]

    conn.close()
    return {
        "total": total, "likely_false_positive": fp, "likely_true_incident": ti, "novel": novel,
        "alerts_requiring_review": requiring_review,
        "overrides": overrides, "accepted": accepted, "rejected": rejected, "pending": pending,
        "by_source": by_source, "by_severity": by_severity,
        "pending_change_reviews": pending_reviews,
    }
