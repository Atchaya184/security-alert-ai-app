"""
helpers.py
-----------------------------------------
Shared validation / small utility functions used across routes.
-----------------------------------------
"""

REQUIRED_FIELDS = ["alert_id", "alert_type", "timestamp", "source_ip",
                    "destination_ip", "location", "device", "severity"]

VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}

REQUIRED_CSV_COLUMNS = ["alert_id", "alert_type", "severity"]

RECOMMENDATION_COLORS = {
    "LIKELY FALSE POSITIVE": "gray",
    "LIKELY TRUE INCIDENT": "red",
    "NOVEL / UNKNOWN — HUMAN INVESTIGATION REQUIRED": "purple",
}

VALID_OVERRIDE_DECISIONS = {
    "Likely False Positive",
    "Likely True Incident",
    "Novel / Human Investigation",
}

# Low / Medium / High risk indicator -> badge color (see model/decision.py
# risk_level()).
RISK_COLORS = {
    "Low": "green",
    "Medium": "amber",
    "High": "red",
}


def color_for_risk(level: str) -> str:
    return RISK_COLORS.get(level, "gray")


def validate_alert_input(data: dict):
    """
    Validate a manually-submitted or uploaded alert row.
    Returns (is_valid, error_message_or_None, cleaned_data).
    Missing/invalid fields are filled with safe defaults (rather than
    hard-failing) so the assistant can still produce a recommendation
    for a partial alert -- reduced context instead simply lowers
    confidence and evidence quality, which is reflected transparently.
    """
    if not isinstance(data, dict) or len(data) == 0:
        return False, "No alert data provided.", None

    cleaned = dict(data)

    for field in REQUIRED_FIELDS:
        if field not in cleaned or cleaned.get(field) in (None, "", "nan"):
            cleaned[field] = "Unknown"

    if cleaned.get("severity") not in VALID_SEVERITIES:
        cleaned["severity"] = "Medium"

    return True, None, cleaned


def validate_csv_columns(columns) -> (bool, str):
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in columns]
    if missing:
        return False, f"CSV is missing required column(s): {', '.join(missing)}"
    return True, None


def color_for_recommendation(recommendation: str) -> str:
    return RECOMMENDATION_COLORS.get(recommendation, "gray")


def validate_override(decision: str, override_decision: str, override_reason: str):
    """Enforce: an Override MUST include a decision category and a non-empty reason."""
    if decision != "Override":
        return True, None
    if not override_decision or override_decision not in VALID_OVERRIDE_DECISIONS:
        return False, "Override requires a valid override decision (Likely False Positive / Likely True Incident / Novel / Human Investigation)."
    if not override_reason or not str(override_reason).strip():
        return False, "Override requires a non-empty reason explaining why the analyst disagreed with the AI."
    return True, None


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
