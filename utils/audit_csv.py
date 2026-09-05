"""
audit_csv.py
-----------------------------------------
Append-only CSV audit trail (audit_log.csv), kept alongside the SQLite
database as a plain-text, easily-inspectable/exportable record of every
system decision and every analyst action (accept / reject / override).

This satisfies the "CSV file or database" audit-log requirement
independently of the DB: even if the database were lost or unavailable,
audit_log.csv on disk preserves a durable record of what the system
recommended and what the analyst actually did, including override
reasons. Rows are only ever appended -- never edited or deleted.

One row is written:
  - when an alert is first scored (analyst_decision="Pending", override="no")
  - every time an analyst records a decision (Accept / Reject / Override)

Fields (matches the requested schema, plus a small additive set used only
by version_rollback rows -- see append_version_rollback below):
  alert_id, endpoint_risk_score, user_risk_score, previous_incident_count,
  similar_alert_count, severity, alert_type, system_decision, confidence,
  analyst_decision, override, override_reason, timestamp,
  event_type, version_type, target_version, restored_version, reviewer
-----------------------------------------
"""

import os
import csv
from datetime import datetime, timezone

AUDIT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audit_log.csv"
)

FIELDNAMES = [
    "alert_id",
    "endpoint_risk_score",
    "user_risk_score",
    "previous_incident_count",
    "similar_alert_count",
    "severity",
    "alert_type",
    "system_decision",
    "confidence",
    "analyst_decision",
    "override",
    "override_reason",
    "timestamp",
    # --- Additive columns (blank for pre-existing alert/decision rows;
    # csv.DictWriter's default restval='' fills these in automatically for
    # every row written through this module's existing functions below) ---
    "event_type",
    "version_type",
    "target_version",
    "restored_version",
    "reviewer",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ensure_header():
    if not os.path.exists(AUDIT_CSV_PATH) or os.path.getsize(AUDIT_CSV_PATH) == 0:
        with open(AUDIT_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_alert_scored(alert: dict, result: dict) -> None:
    """Log the system's initial recommendation for a newly-scored alert."""
    _ensure_header()
    with open(AUDIT_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({
            "alert_id": alert.get("alert_id", ""),
            "endpoint_risk_score": alert.get("endpoint_risk_score", ""),
            "user_risk_score": alert.get("user_risk_score", ""),
            "previous_incident_count": alert.get("previous_incident_count", ""),
            "similar_alert_count": alert.get("similar_alert_count", ""),
            "severity": alert.get("severity", ""),
            "alert_type": alert.get("alert_type", ""),
            "system_decision": result.get("recommendation", ""),
            "confidence": result.get("confidence", ""),
            "analyst_decision": "Pending",
            "override": "no",
            "override_reason": "",
            "timestamp": _now(),
            "event_type": "alert_scored",
        })


def append_analyst_decision(alert_row: dict, decision: str, override_decision: str = None,
                             override_reason: str = None) -> None:
    """
    Log an analyst action (Accept / Reject / Override) against an
    already-scored alert. `alert_row` is the DB row for the alert (contains
    raw_json / ai_recommendation / confidence). Override reason is always
    logged verbatim -- an override is never recorded silently.
    """
    import json as _json
    _ensure_header()
    raw = {}
    try:
        raw = _json.loads(alert_row.get("raw_json") or "{}")
    except Exception:
        raw = {}

    is_override = decision == "Override"
    analyst_decision = f"Override -> {override_decision}" if is_override else decision

    with open(AUDIT_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({
            "alert_id": alert_row.get("alert_id", ""),
            "endpoint_risk_score": raw.get("endpoint_risk_score", ""),
            "user_risk_score": raw.get("user_risk_score", ""),
            "previous_incident_count": raw.get("previous_incident_count", ""),
            "similar_alert_count": raw.get("similar_alert_count", ""),
            "severity": alert_row.get("severity", ""),
            "alert_type": alert_row.get("alert_type", ""),
            "system_decision": alert_row.get("ai_recommendation", ""),
            "confidence": alert_row.get("confidence", ""),
            "analyst_decision": analyst_decision,
            "override": "yes" if is_override else "no",
            "override_reason": override_reason or "",
            "timestamp": _now(),
            "event_type": "analyst_decision",
        })


def append_version_rollback(version_type: str, target_version: str, restored_version: str,
                             reason: str, reviewer: str) -> None:
    """
    Log a model/rule ROLLBACK event to the same durable, append-only CSV
    audit trail used for every alert score and analyst decision, so
    rollbacks show up in audit_log.csv / /api/audit / /audit without a
    separate file or view. Reason and reviewer are always recorded --
    routes/model_routes.py and routes/change_review.py both make `reason`
    mandatory before a rollback is even performed, so this is never called
    with a blank reason.
    """
    _ensure_header()
    with open(AUDIT_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({
            "alert_id": "",
            "endpoint_risk_score": "",
            "user_risk_score": "",
            "previous_incident_count": "",
            "similar_alert_count": "",
            "severity": "",
            "alert_type": "",
            "system_decision": "",
            "confidence": "",
            "analyst_decision": f"Rollback ({version_type}): -> v{restored_version}",
            "override": "no",
            "override_reason": reason,
            "timestamp": _now(),
            "event_type": "version_rollback",
            "version_type": version_type,
            "target_version": target_version,
            "restored_version": restored_version,
            "reviewer": reviewer,
        })
