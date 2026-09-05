"""
decision.py
-----------------------------------------
The core recommendation / safety-gate logic, shared by:
  - model/train_model.py    (offline evaluation on the test split)
  - utils/recommend.py      (live inference for the web app)

This keeps evaluation and production behavior identical by construction.

Recommendation categories:
  - "LIKELY FALSE POSITIVE"
  - "LIKELY TRUE INCIDENT"
  - "NOVEL / UNKNOWN — HUMAN INVESTIGATION REQUIRED"

Priority order (see README):
  1. Preserve confirmed/genuine incidents
  2. Preserve novel/unknown threats
  3. Reduce repetitive known false positives

The novelty/insufficient-evidence safety gate is checked BEFORE trusting
the classifier's false-positive call: a highly novel alert, or one with
too little historical evidence, is NEVER recommended as a false positive
suppression, regardless of what the classifier predicts.
-----------------------------------------
"""

from novelty import insufficient_evidence

REC_FALSE_POSITIVE = "LIKELY FALSE POSITIVE"
REC_TRUE_INCIDENT = "LIKELY TRUE INCIDENT"
REC_NOVEL = "NOVEL / UNKNOWN — HUMAN INVESTIGATION REQUIRED"

# Simple, analyst-facing labels for the same three categories (used on the
# dashboard alongside the detailed recommendation string).
SIMPLE_LABELS = {
    REC_FALSE_POSITIVE: "Fake",
    REC_TRUE_INCIDENT: "Escalate",
    REC_NOVEL: "Investigate",
}

RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"

HIGH_IMPACT_ACTIONS = {
    "Quarantine endpoint",
    "Disable user account",
    "Block source IP",
    "Escalate incident",
}


def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def high_risk_reasons(raw_row: dict, config: dict) -> list:
    """
    Returns a list of human-readable reasons this alert is high-risk, or an
    empty list if it isn't. An alert is high-risk if ANY of the following
    hold:
      - endpoint_risk_score > high_risk_endpoint_threshold (default 70)
      - user_risk_score > high_risk_user_threshold (default 60)
      - previous_incident_count > high_risk_previous_incident_threshold (default 1)
    """
    endpoint_risk = _num(raw_row, "endpoint_risk_score")
    user_risk = _num(raw_row, "user_risk_score")
    prev_incidents = _num(raw_row, "previous_incident_count")

    endpoint_thresh = config.get("high_risk_endpoint_threshold", 70)
    user_thresh = config.get("high_risk_user_threshold", 60)
    incident_thresh = config.get("high_risk_previous_incident_threshold", 1)

    reasons = []
    if endpoint_risk > endpoint_thresh:
        reasons.append(f"High endpoint risk ({endpoint_risk:.0f})")
    if user_risk > user_thresh:
        reasons.append(f"High user risk ({user_risk:.0f})")
    if prev_incidents > incident_thresh:
        reasons.append(f"past incidents ({int(prev_incidents)})")
    return reasons


def is_high_risk(raw_row: dict, config: dict) -> bool:
    return len(high_risk_reasons(raw_row, config)) > 0


def risk_level(raw_row: dict, config: dict) -> str:
    """
    Coarse Low / Medium / High risk indicator shown on the dashboard.
    High mirrors the high-risk safety-gate criteria exactly, so the badge
    the analyst sees always matches the gate that drove the recommendation.
    """
    if is_high_risk(raw_row, config):
        return RISK_HIGH

    endpoint_risk = _num(raw_row, "endpoint_risk_score")
    user_risk = _num(raw_row, "user_risk_score")
    prev_incidents = _num(raw_row, "previous_incident_count")

    if endpoint_risk >= 40 or user_risk >= 30 or prev_incidents >= 1:
        return RISK_MEDIUM
    return RISK_LOW


def is_fake_eligible(raw_row: dict, config: dict) -> bool:
    """
    Legacy/soft eligibility check: NOT high-risk AND a high count of
    similar prior alerts on record (default: > 5). Kept as a general
    utility; the live decision path now uses the stricter, explicit
    `strong_false_positive_reasons` rule below instead.
    """
    if is_high_risk(raw_row, config):
        return False
    similar_count = _num(raw_row, "similar_alert_count")
    threshold = config.get("fake_similar_alert_count_threshold", 5)
    return similar_count > threshold


def strong_false_positive_reasons(raw_row: dict, config: dict) -> list:
    """
    Returns a list of human-readable reasons this alert qualifies for the
    STRONG false-positive rule, or an empty list if it doesn't. ALL of the
    following must hold:
      - endpoint_risk_score < strong_fp_endpoint_risk_max (default 30)
      - user_risk_score < strong_fp_user_risk_max (default 30)
      - previous_incident_count == strong_fp_previous_incident_max (default 0)
      - similar_alert_count >= strong_fp_similar_alert_count_min (default 5)
      - historical_false_positive_rate >= strong_fp_historical_fp_rate_min (default 0.7)
    """
    endpoint_risk = _num(raw_row, "endpoint_risk_score")
    user_risk = _num(raw_row, "user_risk_score")
    prev_incidents = _num(raw_row, "previous_incident_count")
    similar_count = _num(raw_row, "similar_alert_count")
    fp_rate = _num(raw_row, "historical_false_positive_rate")

    endpoint_max = config.get("strong_fp_endpoint_risk_max", 30)
    user_max = config.get("strong_fp_user_risk_max", 30)
    incident_max = config.get("strong_fp_previous_incident_max", 0)
    similar_min = config.get("strong_fp_similar_alert_count_min", 5)
    fp_rate_min = config.get("strong_fp_historical_fp_rate_min", 0.7)

    conditions_met = (
        endpoint_risk < endpoint_max
        and user_risk < user_max
        and prev_incidents <= incident_max
        and similar_count >= similar_min
        and fp_rate >= fp_rate_min
    )
    if not conditions_met:
        return []

    return [
        f"Low risk scores (endpoint={endpoint_risk:.0f}, user={user_risk:.0f})",
        f"repeated alerts ({int(similar_count)} occurrences)",
        f"high historical false-positive rate ({fp_rate:.2f})",
    ]


def is_strong_false_positive(raw_row: dict, config: dict) -> bool:
    return len(strong_false_positive_reasons(raw_row, config)) > 0


def classify_from_proba(ml_proba: dict, config: dict) -> str:
    """
    Turn the classifier's P(True Incident) into a binary label using a
    CALIBRATED decision threshold (tuned on the validation split — see
    train_model.py) rather than a naive 0.5 argmax cutoff. Because missing
    a genuine incident is far costlier than over-flagging a false
    positive, the threshold is tuned to reduce missed incidents, i.e. it
    is typically lower than 0.5.
    """
    threshold = config.get("true_incident_threshold", 0.5)
    p_incident = float(ml_proba.get("True Incident", 0.0))
    return "True Incident" if p_incident >= threshold else "False Positive"


def decide(ml_proba: dict, novelty_info: dict, raw_row: dict, config: dict) -> dict:
    """
    Apply the safety gates on top of the raw classifier output, in this
    priority order:

      1. HIGH_RISK_RULE — an alert with high endpoint risk, high user risk,
         or repeat incidents is NEVER recommended as a false positive
         ("Fake"), no matter what the classifier says. It is routed to
         "LIKELY TRUE INCIDENT" (Investigate/Escalate). This always wins,
         even over novelty, because missing a real high-risk incident is
         far costlier than a false alarm.

      2. FALSE_POSITIVE_RULE — the strong false-positive rule. When an
         alert has low endpoint/user risk, zero prior incidents, a high
         count of similar alerts on record, AND a high historical
         false-positive rate, it is FORCED to "Likely False Positive" with
         high confidence (>70%), overriding the ML model's raw prediction.
         This directly targets the over-classification of routine, well
         corroborated noise as "True Incident".

      3. NOVEL_GATE — a highly novel combination of features, or one with
         too little historical evidence, is routed to human investigation
         ("Investigate") rather than trusting either the model or a
         disqualified false-positive suppression.

      4. MODEL_DECISION — fallback to the raw classifier prediction
         (True Incident / False Positive) when none of the rules above
         apply.

    Prints a one-line debug trace naming exactly which rule fired
    ("HIGH_RISK_RULE", "FALSE_POSITIVE_RULE", "NOVEL_GATE", or
    "MODEL_DECISION") so classification behavior is easy to audit.

    Returns a dict: {
        recommendation, confidence, is_novel_gate_triggered,
        insufficient_evidence, ml_prediction, ml_confidence,
        high_risk_gate_triggered, high_risk_reasons, risk_level,
        low_confidence, strong_fp_rule_triggered, strong_fp_reasons,
        rule_triggered
    }
    """
    insuff = insufficient_evidence(raw_row, config.get("similar_alert_count_threshold", 1))
    novelty_score = novelty_info.get("novelty_score", 0.0)
    novelty_gate = novelty_score >= config.get("novelty_threshold", 0.65)

    ml_pred = classify_from_proba(ml_proba, config)
    ml_confidence = float(ml_proba.get(ml_pred, 0.5))

    hr_reasons = high_risk_reasons(raw_row, config)
    high_risk_gate = len(hr_reasons) > 0

    fp_reasons = strong_false_positive_reasons(raw_row, config)
    strong_fp_gate = len(fp_reasons) > 0

    strong_fp_confidence = config.get("strong_fp_confidence", 0.85)

    if high_risk_gate:
        # Rule 1: high-risk alerts must never be classified as Fake,
        # regardless of the classifier's raw output.
        recommendation = REC_TRUE_INCIDENT
        confidence = max(ml_confidence, ml_proba.get("True Incident", ml_confidence))
        rule_triggered = "HIGH_RISK_RULE"
    elif strong_fp_gate:
        # Rule 2: strong, explicit false-positive rule. Overrides the model
        # prediction outright, including novelty/insufficient-evidence.
        recommendation = REC_FALSE_POSITIVE
        confidence = max(strong_fp_confidence, ml_proba.get("False Positive", 0.0))
        rule_triggered = "FALSE_POSITIVE_RULE"
    elif novelty_gate or insuff:
        # Rule 3: too novel or too little evidence to trust either the
        # model's call or a false-positive suppression.
        recommendation = REC_NOVEL
        confidence = novelty_score
        rule_triggered = "NOVEL_GATE"
    else:
        # Rule 4: fall back to the raw classifier prediction.
        recommendation = REC_TRUE_INCIDENT if ml_pred == "True Incident" else REC_FALSE_POSITIVE
        confidence = ml_confidence
        rule_triggered = "MODEL_DECISION"

    confidence = round(float(confidence), 4)
    low_confidence_threshold = config.get("low_confidence_threshold", 0.6)

    print(
        f"[DECISION] alert_id={raw_row.get('alert_id', '?')} rule={rule_triggered} "
        f"recommendation={recommendation} confidence={confidence}"
    )

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "novelty_gate_triggered": bool(novelty_gate),
        "insufficient_evidence": bool(insuff),
        "ml_prediction": ml_pred,
        "ml_confidence": round(ml_confidence, 4),
        "high_risk_gate_triggered": bool(high_risk_gate),
        "high_risk_reasons": hr_reasons,
        "strong_fp_rule_triggered": bool(strong_fp_gate),
        "strong_fp_reasons": fp_reasons,
        "rule_triggered": rule_triggered,
        "risk_level": risk_level(raw_row, config),
        "low_confidence": bool(confidence < low_confidence_threshold),
        "simple_label": SIMPLE_LABELS.get(recommendation, recommendation),
    }


def recommended_action(recommendation: str, alert_type: str, severity: str) -> str:
    """Simulated-only recommended action. Never executed automatically."""
    if recommendation == REC_FALSE_POSITIVE:
        return "No action needed (recommend closing as false positive)"

    if recommendation == REC_NOVEL:
        return "Escalate for human investigation (no automated action)"

    at = (alert_type or "").lower()
    if "ransomware" in at or "malware" in at:
        return "Quarantine endpoint"
    if "credential" in at or "brute force" in at or "unauthorized access" in at:
        return "Disable user account"
    if "exfiltration" in at or "command and control" in at or "c2" in at:
        return "Block source IP"
    if "privilege escalation" in at or "persistence" in at or "phishing" in at:
        return "Escalate incident"
    return "Escalate incident"


def is_high_impact(action: str) -> bool:
    return action in HIGH_IMPACT_ACTIONS
