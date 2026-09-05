"""
config.py
-----------------------------------------
Holds the current ACTIVE rule-set / operating configuration:

    - novelty_threshold                (novelty safety gate)
    - similar_alert_count_threshold    (insufficient-evidence gate)
    - avg_investigation_minutes        (baseline assumption, configurable)
    - quick_review_minutes             (time to confirm a "likely false
                                         positive" recommendation instead
                                         of a full investigation)
    - missed_incident_target_pct       (safety target, default <= 2%)
    - model_version / rule_version

This file is intentionally a plain JSON file on disk (model/config.json)
rather than hidden inside the database, so it's easy to inspect, and so
the Change Review workflow can snapshot / restore it directly.
Every change goes through routes/change_review.py, which keeps a full
history in the `rule_versions` DB table so nothing is ever silently lost.
-----------------------------------------
"""

import os
import json
import copy

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_CONFIG = {
    "model_version": "1.0",
    "rule_version": "1.0",
    "novelty_threshold": 0.65,
    "similar_alert_count_threshold": 1,
    "avg_investigation_minutes": 5.0,
    "quick_review_minutes": 0.5,
    "missed_incident_target_pct": 2.0,
    # --- High-risk / false-positive safety gates (see model/decision.py) ---
    # An alert meeting ANY of the first three conditions is never allowed to
    # be recommended as a false positive ("Fake"), regardless of what the ML
    # classifier predicts. It must instead be routed to Investigate/Escalate.
    "high_risk_endpoint_threshold": 70,
    "high_risk_user_threshold": 60,
    "high_risk_previous_incident_threshold": 1,
    # An alert may ONLY be recommended as a false positive when it is NOT
    # high-risk AND has more than this many similar prior alerts on record.
    "fake_similar_alert_count_threshold": 5,
    # Below this ML confidence, the UI must show a "Low confidence
    # prediction" warning so the analyst treats the recommendation with
    # appropriate skepticism.
    "low_confidence_threshold": 0.6,
    # --- Strong false-positive rule (see model/decision.py) ---
    # When an alert meets ALL five conditions below, it is FORCED to
    # "Likely False Positive" with high confidence, overriding whatever the
    # ML classifier says. This exists because low-risk, frequently-repeated
    # alerts with a strong track record of being false alarms should not
    # be escalated just because the classifier leans incident.
    "strong_fp_endpoint_risk_max": 30,
    "strong_fp_user_risk_max": 30,
    "strong_fp_previous_incident_max": 0,
    "strong_fp_similar_alert_count_min": 5,
    "strong_fp_historical_fp_rate_min": 0.7,
    "strong_fp_confidence": 0.85,
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    # Backfill any missing keys with defaults (forward-compatible).
    merged = {**DEFAULT_CONFIG, **cfg}
    return merged


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
