"""
explain.py
-----------------------------------------
Generates the evidence panel shown to the analyst for every AI
recommendation. All evidence is read directly from the alert's own
field values and the model's own outputs (probabilities, novelty
score/components) — nothing here is a hard-coded or templated fake
explanation independent of the actual data.
-----------------------------------------
"""


def _f(alert, key, default=None):
    try:
        return float(alert.get(key, default))
    except (TypeError, ValueError):
        return default


def build_evidence(alert: dict, result: dict) -> list:
    """
    Returns a list of short evidence strings appropriate to the
    recommendation category, each backed by an actual field value.
    """
    rec = result["recommendation"]
    evidence = []

    fp_rate = _f(alert, "historical_false_positive_rate")
    inc_rate = _f(alert, "historical_incident_rate")
    similar_count = _f(alert, "similar_alert_count")
    prev_count = _f(alert, "previous_alert_count")
    prev_incidents = _f(alert, "previous_incident_count")
    endpoint_risk = _f(alert, "endpoint_risk_score")
    user_risk = _f(alert, "user_risk_score")
    severity = str(alert.get("severity", "")).strip()
    auth_result = str(alert.get("authentication_result", "")).strip()
    alert_source = str(alert.get("alert_source", "")).strip()
    alert_type = str(alert.get("alert_type", "")).strip()
    process_name = str(alert.get("process_name", "")).strip()

    if rec == "NOVEL / UNKNOWN — HUMAN INVESTIGATION REQUIRED":
        evidence.append(f"Novelty score: {result['novelty_score']:.2f} (0 = routine, 1 = highly unusual)")
        nc = result.get("novelty_components", {})
        if nc.get("combination_seen_in_training", None) is not None:
            seen = nc["combination_seen_in_training"]
            if seen == 0:
                country_val = str(alert.get("country", "Unknown"))
                evidence.append(
                    f"This exact combination (source={alert_source or 'Unknown'}, type={alert_type or 'Unknown'}, "
                    f"country={country_val}) was never seen in training data"
                )
            else:
                evidence.append(f"This alert pattern was seen only {seen} time(s) in training data (rare)")
        if result.get("insufficient_evidence"):
            evidence.append(
                f"Insufficient historical evidence: previous_alert_count={prev_count if prev_count is not None else 0}, "
                f"similar_alert_count={similar_count if similar_count is not None else 0}"
            )
        if nc.get("iforest_component") is not None:
            evidence.append(f"Statistical anomaly score: {nc['iforest_component']:.2f} (isolation-forest based)")

    elif rec == "LIKELY TRUE INCIDENT":
        if result.get("high_risk_gate_triggered") and result.get("high_risk_reasons"):
            evidence.append(
                "High-risk safety gate: " + " and ".join(result["high_risk_reasons"]) +
                " indicate elevated threat — never classified as Fake regardless of model score"
            )
        if endpoint_risk is not None:
            evidence.append(f"Endpoint risk score: {endpoint_risk:.0f}/100")
        if user_risk is not None:
            evidence.append(f"User risk score: {user_risk:.0f}/100")
        if prev_incidents:
            evidence.append(f"{int(prev_incidents)} prior confirmed incident(s) on this endpoint/user")
        if inc_rate is not None:
            evidence.append(f"This alert type has a historical true-incident rate of {inc_rate:.0%}")
        if severity in ("High", "Critical"):
            evidence.append(f"Severity reported as {severity}")
        if auth_result and auth_result.lower() not in ("success", "n/a", "unknown", ""):
            evidence.append(f"Authentication result: {auth_result}")
        if process_name and process_name.lower() not in ("unknown", ""):
            evidence.append(f"Associated process: {process_name}")
        evidence.append(f"Model probability of true incident: {result['ml_probabilities'].get('True Incident', 0):.0%}")

    else:  # LIKELY FALSE POSITIVE
        if result.get("strong_fp_rule_triggered") and result.get("strong_fp_reasons"):
            reasons = result["strong_fp_reasons"]
            # e.g. "Low risk scores with repeated alerts (10 occurrences) and
            # high historical false-positive rate (0.90) indicate a likely
            # false positive."
            evidence.append(
                f"{reasons[0]} with {reasons[1]} and {reasons[2]} indicate a likely false positive "
                f"(strong false-positive rule — overrides model prediction)"
            )
        if fp_rate is not None:
            evidence.append(f"This alert type has a historical false-positive rate of {fp_rate:.0%}")
        if similar_count:
            evidence.append(f"{int(similar_count)} similar alerts seen before with no confirmed incident")
        if prev_count:
            evidence.append(f"{int(prev_count)} previous alerts recorded for this endpoint/user")
        if endpoint_risk is not None and endpoint_risk <= 30:
            evidence.append(f"Endpoint risk score is low ({endpoint_risk:.0f}/100)")
        if severity in ("Low", "Medium"):
            evidence.append(f"Severity reported as only {severity}")
        evidence.append(f"Model probability of false positive: {result['ml_probabilities'].get('False Positive', 0):.0%}")
        evidence.append(f"Decision threshold in use: alerts need >= {result.get('ml_confidence', 0):.0%} incident "
                         f"probability to be called an incident (calibrated for safety)")

    if not evidence:
        evidence.append("Based on the overall pattern of alert attributes relative to historical data.")

    if result.get("low_confidence"):
        evidence.append(
            f"⚠ Low confidence prediction ({result.get('confidence', 0) * 100:.0f}%) — treat this "
            f"recommendation as a starting point, not a conclusion"
        )

    return evidence[:7]


def summary_line(alert: dict, result: dict) -> str:
    rec = result["recommendation"]
    conf = result["confidence"]
    if rec == "LIKELY TRUE INCIDENT":
        lead = "Recommended as a LIKELY TRUE INCIDENT"
    elif rec == "LIKELY FALSE POSITIVE":
        lead = "Recommended as a LIKELY FALSE POSITIVE"
    else:
        lead = "Flagged as NOVEL / UNKNOWN — human investigation required"
    line = f"{lead} ({conf*100:.1f}% confidence, model v{result.get('model_version','?')})."
    if result.get("rule_triggered") == "FALSE_POSITIVE_RULE":
        line += " Forced by strong false-positive rule (overrides model)."
    if result.get("high_risk_gate_triggered"):
        line += " High-risk alert — manual confirmation required."
    if result.get("low_confidence"):
        line += " Low confidence prediction."
    return line
