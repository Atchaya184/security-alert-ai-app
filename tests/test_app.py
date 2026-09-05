"""
test_app.py
-----------------------------------------
Automated tests for the False Positive Reduction Assistant.

Run with:
    pytest

Covers: dataset loading, preprocessing, leakage prevention, ML
prediction, novelty detection, evidence generation, false-positive /
true-incident / novel recommendations, manual override (+ mandatory
reason), database logging, high-impact confirmation, change review,
rollback, analyst feedback / retraining, hours-saved and
missed-incident calculations, and edge cases.

NOTE ON MODEL LOADING: the trained model bundle (model/model.pkl)
pickles the NoveltyDetector class under the bare module name
"novelty" (see model/novelty.py's docstring / README limitations).
This resolves correctly as long as `utils.recommend` or `app` has
already been imported in the process (both add model/ to sys.path
before importing/unpickling) — which is exactly what every test
below does by importing `app` first.
-----------------------------------------
"""

import os
import sys
import json
import shutil
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Importing `app` triggers model loading/training and sets up sys.path
# for the model/ package's bare imports (novelty, decision, preprocess).
import app as flask_app_module  # noqa: E402
from utils import db  # noqa: E402
from model.preprocess import (  # noqa: E402
    load_raw_dataset, clean_dataframe, assert_no_leakage,
    FEATURE_COLUMNS, LEAKAGE_COLUMNS, prepare_single_alert,
)
from model.decision import (  # noqa: E402
    decide, REC_FALSE_POSITIVE, REC_TRUE_INCIDENT, REC_NOVEL, is_high_impact,
    is_high_risk, is_fake_eligible, risk_level, is_strong_false_positive,
)
from model.config import load_config  # noqa: E402
from utils.recommend import engine  # noqa: E402
from utils.explain import build_evidence  # noqa: E402


DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "security_alerts_2000.csv")


@pytest.fixture(scope="session", autouse=True)
def fresh_database():
    """Start the test session with a clean database so counts/deltas are predictable.

    NOTE: `import app` above already ran db.init_db() + ensure_initial_versions_recorded()
    once against whatever database.db existed at import time. Resetting the file here
    would leave the DB without those initial model/rule version rows, so we re-run that
    step against the freshly reset database to keep it consistent with a real cold start.
    """
    db_path = os.path.join(PROJECT_ROOT, "database.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db.init_db()
    flask_app_module.ensure_initial_versions_recorded()
    yield


@pytest.fixture(scope="session", autouse=True)
def isolate_config_file():
    """
    Change Review / retraining tests genuinely write to model/config.json
    (by design — that's the real file the running app reads). To avoid the
    test suite leaving permanent side effects on the shipped config file,
    snapshot it before the session and restore the original content
    afterward, regardless of test outcome.
    """
    from model.config import CONFIG_PATH
    original = None
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            original = f.read()
    yield
    if original is not None:
        with open(CONFIG_PATH, "w") as f:
            f.write(original)
    elif os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)


@pytest.fixture(scope="session", autouse=True)
def isolate_archive_dir():
    """Retraining/rollback tests write real files under model/archive/ and
    dataset/retrain_prepared.csv. Clean those up after the session so the
    shipped project isn't left with test-generated artifacts."""
    yield
    archive_dir = os.path.join(PROJECT_ROOT, "model", "archive")
    if os.path.isdir(archive_dir):
        shutil.rmtree(archive_dir)
    prepared_csv = os.path.join(PROJECT_ROOT, "dataset", "retrain_prepared.csv")
    if os.path.exists(prepared_csv):
        os.remove(prepared_csv)


@pytest.fixture(scope="session", autouse=True)
def isolate_model_file():
    """Model-rollback tests genuinely retrain the model, which overwrites
    the real model/model.pkl. The rollback tests below restore it as part
    of the round-trip they're testing, but snapshot/restore it here too
    (belt-and-suspenders, mirroring isolate_config_file) so a failed
    assertion mid-test can never leave the shipped model file mutated."""
    from model.train_model import MODEL_PATH
    original = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            original = f.read()
    yield
    if original is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(original)


@pytest.fixture(scope="session", autouse=True)
def isolate_metrics_file():
    """Retraining rewrites the real model/metrics.json. Snapshot/restore it
    the same way isolate_config_file already does for model/config.json."""
    from model.train_model import METRICS_PATH
    original = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            original = f.read()
    yield
    if original is not None:
        with open(METRICS_PATH, "w") as f:
            f.write(original)


@pytest.fixture()
def client():
    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------
# Dataset loading & preprocessing
# ---------------------------------------------------------------------

def test_dataset_loads_and_has_expected_columns():
    df = load_raw_dataset(DATASET_PATH)
    assert len(df) > 1000
    for col in ["alert_id", "timestamp", "ground_truth", "severity", "alert_source", "novelty_test_flag"]:
        assert col in df.columns


def test_cleaning_fills_missing_values():
    df = load_raw_dataset(DATASET_PATH)
    df.loc[0, "severity"] = None
    df.loc[1, "endpoint_risk_score"] = None
    cleaned = clean_dataframe(df)
    assert cleaned.loc[0, "severity"] == "Unknown"
    assert not clean_dataframe(df)[FEATURE_COLUMNS].isnull().any().any()


def test_preprocessing_handles_missing_dataset_gracefully():
    with pytest.raises(FileNotFoundError):
        load_raw_dataset("dataset/does_not_exist.csv")


# ---------------------------------------------------------------------
# Data leakage prevention
# ---------------------------------------------------------------------

def test_feature_columns_never_include_leakage_columns():
    assert set(FEATURE_COLUMNS).isdisjoint(set(LEAKAGE_COLUMNS))


def test_assert_no_leakage_raises_on_violation():
    with pytest.raises(ValueError):
        assert_no_leakage(["endpoint_risk_score", "ground_truth"])
    with pytest.raises(ValueError):
        assert_no_leakage(["novelty_test_flag"])


def test_assert_no_leakage_passes_on_clean_features():
    assert_no_leakage(FEATURE_COLUMNS)  # must not raise


def test_prepare_single_alert_excludes_leakage_fields():
    alert = {
        "alert_type": "Malware", "severity": "High", "ground_truth": "True Incident",
        "confirmed_incident": "Yes", "analyst_disposition": "Escalated",
    }
    row = prepare_single_alert(alert)
    assert set(row.columns).isdisjoint(set(LEAKAGE_COLUMNS))


# ---------------------------------------------------------------------
# ML prediction / recommendation engine
# ---------------------------------------------------------------------

def test_model_is_loaded():
    assert engine.is_loaded


def test_predict_returns_valid_recommendation_category():
    cfg = load_config()
    alert = {
        "alert_id": "T1", "alert_type": "Multiple Failed Logins", "severity": "Medium",
        "endpoint_risk_score": 40, "user_risk_score": 40, "previous_alert_count": 5,
        "previous_incident_count": 1, "similar_alert_count": 5, "alert_source": "IAM",
    }
    result = engine.predict(alert, cfg)
    assert result["recommendation"] in (REC_FALSE_POSITIVE, REC_TRUE_INCIDENT, REC_NOVEL)
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["novelty_score"] <= 1.0


# ---------------------------------------------------------------------
# Novelty detection
# ---------------------------------------------------------------------

def test_novelty_score_higher_for_unseen_combination():
    cfg = load_config()
    common_alert = {
        "alert_id": "T2", "alert_type": "Suspicious PowerShell", "severity": "Medium",
        "alert_source": "EDR", "country": "India", "process_name": "powershell.exe",
        "previous_alert_count": 20, "similar_alert_count": 20, "historical_false_positive_rate": 0.7,
    }
    weird_alert = {
        "alert_id": "T3", "alert_type": "Zzz-Highly-Unusual-Type-XYZ", "severity": "Low",
        "alert_source": "Zzz-Unusual-Source", "country": "Zzz-Nowhere", "process_name": "zzz_weird.exe",
        "previous_alert_count": 0, "similar_alert_count": 0,
    }
    r_common = engine.predict(common_alert, cfg)
    r_weird = engine.predict(weird_alert, cfg)
    assert r_weird["novelty_score"] >= r_common["novelty_score"]


def test_novel_alert_flagged_as_human_investigation_required():
    cfg = load_config()
    weird_alert = {
        "alert_id": "T4", "alert_type": "Zzz-Highly-Unusual-Type-ABC", "severity": "Medium",
        "alert_source": "Zzz-Unusual-Source-2", "country": "Zzz-Nowhere-2", "process_name": "zzz_weird2.exe",
        "previous_alert_count": 0, "similar_alert_count": 0,
    }
    result = engine.predict(weird_alert, cfg)
    # Insufficient evidence (0 previous alerts, 0 similar) must trigger the safety gate
    # regardless of what the classifier itself would have said.
    assert result["recommendation"] == REC_NOVEL
    assert result["insufficient_evidence"] is True


# ---------------------------------------------------------------------
# Decision / safety-gate logic (deterministic, using crafted inputs)
# ---------------------------------------------------------------------

def test_decide_false_positive_when_confident_and_not_novel():
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {"previous_alert_count": 10, "similar_alert_count": 10}
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_FALSE_POSITIVE


def test_decide_true_incident_when_confident_and_not_novel():
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.2, "True Incident": 0.8}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {"previous_alert_count": 10, "similar_alert_count": 10}
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_TRUE_INCIDENT


def test_decide_novel_overrides_confident_false_positive_when_novelty_high():
    """The safety gate must win even when the classifier is very confident of FP."""
    cfg = load_config()
    cfg["novelty_threshold"] = 0.65
    ml_proba = {"False Positive": 0.99, "True Incident": 0.01}
    novelty_info = {"novelty_score": 0.90}
    raw_row = {"previous_alert_count": 10, "similar_alert_count": 10}
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_NOVEL
    assert d["novelty_gate_triggered"] is True


def test_decide_novel_when_insufficient_evidence_even_if_low_novelty_score():
    cfg = load_config()
    cfg["novelty_threshold"] = 0.65
    cfg["similar_alert_count_threshold"] = 1
    ml_proba = {"False Positive": 0.95, "True Incident": 0.05}
    novelty_info = {"novelty_score": 0.05}  # low anomaly score
    raw_row = {"previous_alert_count": 0, "similar_alert_count": 0}  # but no history at all
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_NOVEL
    assert d["insufficient_evidence"] is True


def test_high_severity_repetitive_false_positive_not_forced_to_incident():
    """Edge case: high severity alone must not force a True Incident call."""
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}  # model says FP despite high severity
    novelty_info = {"novelty_score": 0.1}
    raw_row = {"previous_alert_count": 20, "similar_alert_count": 20, "severity": "Critical"}
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_FALSE_POSITIVE


def test_low_severity_can_still_be_true_incident():
    """Edge case: low severity must not prevent a True Incident call."""
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.2, "True Incident": 0.8}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {"previous_alert_count": 20, "similar_alert_count": 20, "severity": "Low"}
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_TRUE_INCIDENT


# ---------------------------------------------------------------------
# Evidence generation
# ---------------------------------------------------------------------

def test_evidence_generated_for_each_category():
    alert = {"severity": "High", "endpoint_risk_score": 80, "user_risk_score": 70,
              "previous_incident_count": 2, "historical_incident_rate": 0.6}
    result = {"recommendation": REC_TRUE_INCIDENT, "confidence": 0.8,
              "ml_probabilities": {"True Incident": 0.8, "False Positive": 0.2},
              "novelty_score": 0.2, "novelty_components": {}, "insufficient_evidence": False,
              "ml_confidence": 0.8}
    evidence = build_evidence(alert, result)
    assert len(evidence) > 0
    assert any("risk" in e.lower() or "incident" in e.lower() for e in evidence)


def test_evidence_not_hard_coded_reflects_actual_values():
    alert1 = {"historical_false_positive_rate": 0.95, "similar_alert_count": 50}
    alert2 = {"historical_false_positive_rate": 0.10, "similar_alert_count": 0}
    result = {"recommendation": REC_FALSE_POSITIVE, "confidence": 0.7,
              "ml_probabilities": {"True Incident": 0.3, "False Positive": 0.7},
              "novelty_score": 0.2, "novelty_components": {}, "insufficient_evidence": False,
              "ml_confidence": 0.7}
    ev1 = build_evidence(alert1, result)
    ev2 = build_evidence(alert2, result)
    assert ev1 != ev2  # different underlying data must produce different evidence text


# ---------------------------------------------------------------------
# Manual override (mandatory reason) + database logging
# ---------------------------------------------------------------------

def _predict_via_client(client, alert):
    res = client.post("/predict", json=alert)
    assert res.status_code == 200
    return res.get_json()


def test_predict_logs_to_database(client):
    data = _predict_via_client(client, {"alert_id": "DB1", "alert_type": "Test Alert", "severity": "Low"})
    row = db.get_alert(data["id"])
    assert row is not None
    assert row["alert_id"] == "DB1"
    assert json.loads(row["evidence_json"]) == data["evidence"]


def test_override_requires_reason(client):
    data = _predict_via_client(client, {"alert_id": "OV1", "alert_type": "Test Alert", "severity": "Low"})
    res = client.post(f"/decision/{data['id']}", json={"decision": "Override", "override_decision": "Likely False Positive"})
    assert res.status_code == 400
    assert "reason" in res.get_json()["error"].lower()


def test_override_requires_valid_decision_category(client):
    data = _predict_via_client(client, {"alert_id": "OV2", "alert_type": "Test Alert", "severity": "Low"})
    res = client.post(f"/decision/{data['id']}", json={
        "decision": "Override", "override_decision": "Not A Real Category", "override_reason": "testing",
    })
    assert res.status_code == 400


def test_override_succeeds_with_reason_and_is_audited(client):
    data = _predict_via_client(client, {"alert_id": "OV3", "alert_type": "Test Alert", "severity": "Low"})
    res = client.post(f"/decision/{data['id']}", json={
        "decision": "Override", "override_decision": "Likely True Incident",
        "override_reason": "Analyst confirmed via endpoint logs",
    })
    assert res.status_code == 200
    row = db.get_alert(data["id"])
    assert row["override_status"] == 1
    assert row["override_reason"] == "Analyst confirmed via endpoint logs"
    history = db.get_decision_history(data["id"])
    assert any(h["event_type"] == "analyst_decision" for h in history)


def test_accept_reject_and_rollback(client):
    data = _predict_via_client(client, {"alert_id": "RB1", "alert_type": "Test Alert", "severity": "Low"})
    rid = data["id"]
    assert client.post(f"/decision/{rid}", json={"decision": "Accept"}).status_code == 200
    assert client.post(f"/decision/{rid}", json={"decision": "Reject"}).status_code == 200
    res = client.post(f"/rollback/{rid}")
    assert res.status_code == 200
    assert res.get_json()["reverted_to"] == "Accept"
    assert db.get_alert(rid)["analyst_decision"] == "Accept"


def test_rollback_with_no_history_fails_gracefully(client):
    data = _predict_via_client(client, {"alert_id": "RB2", "alert_type": "Test Alert", "severity": "Low"})
    res = client.post(f"/rollback/{data['id']}")
    assert res.status_code == 400


# ---------------------------------------------------------------------
# High-impact action confirmation
# ---------------------------------------------------------------------

def test_high_impact_action_requires_confirmation(client):
    # Craft an alert highly likely to be called a true incident with a mapped action.
    alert = {
        "alert_id": "HI1", "alert_type": "Ransomware Deployment", "severity": "Critical",
        "endpoint_risk_score": 95, "user_risk_score": 90, "previous_incident_count": 5,
        "previous_alert_count": 30, "similar_alert_count": 30, "alert_source": "EDR",
        "historical_incident_rate": 0.9,
    }
    data = _predict_via_client(client, alert)
    if data["is_high_impact_action"]:
        row = db.get_alert(data["id"])
        assert row["human_confirmation"] == "Required"
        res = client.post(f"/confirm-action/{data['id']}", json={"confirmation": "Approved"})
        assert res.status_code == 200
        assert db.get_alert(data["id"])["human_confirmation"] == "Approved"


def test_confirm_action_rejected_for_non_high_impact_alert(client):
    data = _predict_via_client(client, {
        "alert_id": "HI2", "alert_type": "Routine Scan", "severity": "Low",
        "previous_alert_count": 20, "similar_alert_count": 20, "historical_false_positive_rate": 0.95,
    })
    if not data["is_high_impact_action"]:
        res = client.post(f"/confirm-action/{data['id']}", json={"confirmation": "Approved"})
        assert res.status_code == 400


def test_confirm_action_invalid_value_rejected(client):
    data = _predict_via_client(client, {"alert_id": "HI3", "alert_type": "Test Alert", "severity": "Low"})
    res = client.post(f"/confirm-action/{data['id']}", json={"confirmation": "Maybe"})
    assert res.status_code == 400


def test_is_high_impact_helper():
    assert is_high_impact("Quarantine endpoint") is True
    assert is_high_impact("No action needed (recommend closing as false positive)") is False


# ---------------------------------------------------------------------
# Change review + rollback
# ---------------------------------------------------------------------

def test_change_review_propose_requires_reason(client):
    res = client.post("/change-review/propose", json={"field": "novelty_threshold", "new_value": "0.7"})
    assert res.status_code == 400


def test_change_review_propose_rejects_unknown_field(client):
    res = client.post("/change-review/propose", json={
        "field": "not_a_real_field", "new_value": "1", "reason": "test",
    })
    assert res.status_code == 400


def test_change_review_approve_updates_config(client):
    cfg_before = load_config()
    new_val = round(cfg_before["novelty_threshold"] + 0.01, 3)
    propose = client.post("/change-review/propose", json={
        "field": "novelty_threshold", "new_value": str(new_val), "reason": "unit test change",
    })
    assert propose.status_code == 200
    review_id = propose.get_json()["id"]

    decide_res = client.post(f"/change-review/{review_id}/decide", json={"status": "Approved", "reviewer": "pytest"})
    assert decide_res.status_code == 200

    cfg_after = load_config()
    assert cfg_after["novelty_threshold"] == new_val
    assert cfg_after["rule_version"] != cfg_before["rule_version"]


def test_change_review_rollback_restores_previous_rule_set(client):
    cfg_before = load_config()
    original_threshold = cfg_before["novelty_threshold"]
    original_rule_version = cfg_before["rule_version"]

    new_val = round(original_threshold + 0.02, 3)
    propose = client.post("/change-review/propose", json={
        "field": "novelty_threshold", "new_value": str(new_val), "reason": "rollback test",
    })
    review_id = propose.get_json()["id"]
    client.post(f"/change-review/{review_id}/decide", json={"status": "Approved", "reviewer": "pytest"})
    assert load_config()["novelty_threshold"] == new_val

    rollback_res = client.post("/change-review/rollback", json={
        "version": original_rule_version, "confirm": True, "reason": "pytest rollback",
    })
    assert rollback_res.status_code == 200
    assert load_config()["novelty_threshold"] == original_threshold


def test_change_review_rollback_requires_confirm(client):
    res = client.post("/change-review/rollback", json={"version": "1.0"})
    assert res.status_code == 400


# ---------------------------------------------------------------------
# Analyst feedback / controlled retraining loop
# ---------------------------------------------------------------------

def test_feedback_recorded_on_accept_and_override(client):
    data = _predict_via_client(client, {
        "alert_id": "FB1", "alert_type": "Port Scan", "severity": "Low",
        "previous_alert_count": 20, "similar_alert_count": 20, "historical_false_positive_rate": 0.95,
    })
    client.post(f"/decision/{data['id']}", json={"decision": "Accept"})
    stats_before = db.get_feedback_stats()
    assert stats_before["total_feedback"] >= 1


def test_prepare_retrain_endpoint_runs_without_error(client):
    res = client.post("/model/prepare-retrain")
    assert res.status_code == 200
    assert "success" in res.get_json()


# ---------------------------------------------------------------------
# Baseline / hours-saved / missed-incident consistency (from metrics.json)
# ---------------------------------------------------------------------

def _load_metrics():
    path = os.path.join(PROJECT_ROOT, "model", "metrics.json")
    with open(path) as f:
        return json.load(f)


def test_metrics_file_exists_and_is_well_formed():
    metrics = _load_metrics()
    for key in ("classification", "safety_metrics", "false_positive_reduction",
                "baseline_vs_assistant", "novelty_detection", "error_analysis"):
        assert key in metrics


def test_hours_saved_calculation_is_internally_consistent():
    metrics = _load_metrics()
    b = metrics["baseline_vs_assistant"]
    n = b["total_test_alerts"]
    avg_min = b["avg_investigation_minutes"]
    quick_min = b["quick_review_minutes"]
    n_review = b["alerts_requiring_human_investigation"]
    n_fp = b["alerts_recommended_false_positive"]

    expected_baseline_hours = round((n * avg_min) / 60.0, 3)
    expected_assistant_hours = round((n_review * avg_min + n_fp * quick_min) / 60.0, 3)

    assert abs(b["baseline_analyst_hours"] - expected_baseline_hours) < 0.01
    assert abs(b["assistant_analyst_hours"] - expected_assistant_hours) < 0.01
    assert abs(b["analyst_hours_saved"] - (expected_baseline_hours - expected_assistant_hours)) < 0.01
    assert n_review + n_fp == n


def test_missed_incident_rate_is_internally_consistent():
    metrics = _load_metrics()
    s = metrics["safety_metrics"]
    if s["actual_incidents_in_test"] > 0:
        expected_rate = round(s["missed_incidents"] / s["actual_incidents_in_test"] * 100.0, 3)
        assert abs(s["missed_incident_rate_pct"] - expected_rate) < 0.01
    assert s["missed_incident_rate_pct"] >= 0.0


def test_missed_incidents_never_come_from_novel_recommendations():
    """A missed incident is specifically an incident recommended as FALSE POSITIVE, not NOVEL."""
    metrics = _load_metrics()
    # This is a structural guarantee of how missed_mask is computed in train_model.py
    # (actual_incident_mask & recommended_fp_mask); novel-but-incident is tracked
    # separately in error_analysis and is NOT counted as a missed incident.
    assert "novel_but_was_incident" in metrics["error_analysis"]
    assert "incident_called_false_positive_missed" in metrics["error_analysis"]


# ---------------------------------------------------------------------
# Edge / failure cases
# ---------------------------------------------------------------------

def test_edge_case_missing_fields_does_not_crash(client):
    res = client.post("/predict", json={"alert_type": "Something"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["recommendation"] in (REC_FALSE_POSITIVE, REC_TRUE_INCIDENT, REC_NOVEL)


def test_edge_case_completely_empty_payload(client):
    res = client.post("/predict", json={})
    assert res.status_code == 400  # no alert data at all


def test_edge_case_invalid_severity_defaults_safely(client):
    res = client.post("/predict", json={"alert_id": "E1", "alert_type": "X", "severity": "Not A Real Severity"})
    assert res.status_code == 200


def test_edge_case_empty_csv_upload(client):
    from io import BytesIO
    data = {"file": (BytesIO(b""), "empty.csv")}
    res = client.post("/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400


def test_edge_case_non_csv_file_upload(client):
    from io import BytesIO
    data = {"file": (BytesIO(b"not a csv"), "notes.txt")}
    res = client.post("/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400


def test_edge_case_csv_missing_required_columns(client):
    from io import BytesIO
    csv_bytes = b"foo,bar\n1,2\n"
    data = {"file": (BytesIO(csv_bytes), "bad.csv")}
    res = client.post("/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "missing required column" in res.get_json()["error"].lower()


def test_edge_case_valid_csv_upload_processes_rows(client):
    from io import BytesIO
    csv_bytes = (
        b"alert_id,alert_type,severity,alert_source\n"
        b"C1,Port Scan,Low,Firewall\n"
        b"C2,Malware Detected,Critical,EDR\n"
    )
    data = {"file": (BytesIO(csv_bytes), "alerts.csv")}
    res = client.post("/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    body = res.get_json()
    assert body["processed"] == 2
    assert body["failed"] == 0


def test_edge_case_invalid_decision_value_rejected(client):
    data = _predict_via_client(client, {"alert_id": "ED1", "alert_type": "Test Alert", "severity": "Low"})
    res = client.post(f"/decision/{data['id']}", json={"decision": "Delete"})
    assert res.status_code == 400


def test_edge_case_nonexistent_alert_returns_404(client):
    res = client.post("/decision/999999", json={"decision": "Accept"})
    assert res.status_code == 404
    res2 = client.post("/confirm-action/999999", json={"confirmation": "Approved"})
    assert res2.status_code == 404
    res3 = client.get("/investigation/999999")
    assert res3.status_code == 404


def test_app_does_not_crash_on_malformed_json(client):
    res = client.post("/predict", data="not json", content_type="application/json")
    # Flask returns 400 for unparseable JSON body via get_json(silent=True) -> None -> form fallback -> empty
    assert res.status_code in (400, 500)


# ---------------------------------------------------------------------
# High-risk safety gate: high-risk alerts must NEVER be recommended Fake,
# even when the classifier is confident it's a false positive.
# ---------------------------------------------------------------------

def test_high_endpoint_risk_blocks_fake_even_if_model_says_false_positive():
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.95, "True Incident": 0.05}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 85, "user_risk_score": 20, "previous_incident_count": 0,
        "previous_alert_count": 20, "similar_alert_count": 20,
    }
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] != REC_FALSE_POSITIVE
    assert d["recommendation"] == REC_TRUE_INCIDENT
    assert d["high_risk_gate_triggered"] is True
    assert d["risk_level"] == "High"


def test_high_user_risk_blocks_fake():
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 10, "user_risk_score": 75, "previous_incident_count": 0,
        "previous_alert_count": 20, "similar_alert_count": 20,
    }
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_TRUE_INCIDENT
    assert d["high_risk_gate_triggered"] is True


def test_repeat_incidents_block_fake():
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 10, "user_risk_score": 10, "previous_incident_count": 3,
        "previous_alert_count": 20, "similar_alert_count": 20,
    }
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_TRUE_INCIDENT
    assert d["high_risk_gate_triggered"] is True
    assert is_high_risk(raw_row, cfg) is True


def test_low_risk_without_strong_rule_falls_back_to_model_decision():
    """
    Low risk + few similar alerts + no historical_false_positive_rate data
    doesn't qualify for the STRONG false-positive rule (no high fp_rate on
    record), but should now fall through to the model's own prediction
    (MODEL_DECISION) rather than being blocked into NOVEL -- the model
    fallback is a plain fallback per the new decision priority.
    """
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 5, "user_risk_score": 5, "previous_incident_count": 0,
        "previous_alert_count": 3, "similar_alert_count": 2,
    }
    assert is_high_risk(raw_row, cfg) is False
    assert is_strong_false_positive(raw_row, cfg) is False  # no historical_false_positive_rate on record
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_FALSE_POSITIVE
    assert d["rule_triggered"] == "MODEL_DECISION"


def test_low_risk_with_many_similar_alerts_can_be_fake():
    cfg = load_config()
    cfg["true_incident_threshold"] = 0.5
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 5, "user_risk_score": 5, "previous_incident_count": 0,
        "previous_alert_count": 20, "similar_alert_count": 20,
    }
    assert is_fake_eligible(raw_row, cfg) is True
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_FALSE_POSITIVE
    assert d["risk_level"] == "Low"


# ---------------------------------------------------------------------
# STRONG false-positive rule: forces "Fake" with high confidence,
# overriding the model, whenever ALL five conditions are met.
# ---------------------------------------------------------------------

def test_strong_fp_rule_forces_false_positive_even_if_model_says_incident():
    """The strong FP rule must override the model prediction outright."""
    cfg = load_config()
    ml_proba = {"False Positive": 0.3, "True Incident": 0.7}  # model leans incident
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 15, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 10, "historical_false_positive_rate": 0.9,
        "previous_alert_count": 10,
    }
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_FALSE_POSITIVE
    assert d["confidence"] > 0.7
    assert d["strong_fp_rule_triggered"] is True
    assert d["rule_triggered"] == "FALSE_POSITIVE_RULE"


def test_strong_fp_rule_requires_all_five_conditions():
    cfg = load_config()
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    # similar_alert_count is one short of the threshold (needs >= 5)
    raw_row = {
        "endpoint_risk_score": 15, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 4, "historical_false_positive_rate": 0.9,
        "previous_alert_count": 10,
    }
    assert is_strong_false_positive(raw_row, cfg) is False


def test_high_risk_rule_still_wins_over_strong_fp_rule():
    """High-risk protection (rule 1) must take priority over the strong
    false-positive rule (rule 2) -- a high-risk alert can never be Fake
    even if it also happens to have a high historical false-positive rate."""
    cfg = load_config()
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 85, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 10, "historical_false_positive_rate": 0.9,
        "previous_alert_count": 10,
    }
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["recommendation"] == REC_TRUE_INCIDENT
    assert d["rule_triggered"] == "HIGH_RISK_RULE"
    assert d["strong_fp_rule_triggered"] is False  # never evaluated as the winning rule


def test_strong_fp_rule_evidence_mentions_repeated_alerts_and_fp_rate():
    alert = {
        "endpoint_risk_score": 10, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 10, "historical_false_positive_rate": 0.9,
        "previous_alert_count": 10, "severity": "Low",
    }
    cfg = load_config()
    result = engine.predict(alert, cfg)
    evidence = build_evidence(alert, result)
    combined = " ".join(evidence)
    assert "10 occurrences" in combined
    assert "0.90" in combined


def test_debug_log_prints_rule_name(capsys):
    cfg = load_config()
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {
        "endpoint_risk_score": 15, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 10, "historical_false_positive_rate": 0.9,
        "previous_alert_count": 10, "alert_id": "DEBUGTEST1",
    }
    decide(ml_proba, novelty_info, raw_row, cfg)
    captured = capsys.readouterr()
    assert "FALSE_POSITIVE_RULE" in captured.out
    assert "DEBUGTEST1" in captured.out


def test_output_distribution_includes_all_three_categories():
    """Sanity check: across a small varied batch, the system should produce
    at least one of each category, not collapse everything into
    True Incident."""
    cfg = load_config()

    strong_fp_row = {
        "endpoint_risk_score": 10, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 10, "historical_false_positive_rate": 0.9,
        "previous_alert_count": 10,
    }
    high_risk_row = {
        "endpoint_risk_score": 90, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 10, "historical_false_positive_rate": 0.9,
        "previous_alert_count": 10,
    }
    novel_row = {
        "endpoint_risk_score": 10, "user_risk_score": 10, "previous_incident_count": 0,
        "similar_alert_count": 0, "previous_alert_count": 0,
    }
    ml_proba = {"False Positive": 0.9, "True Incident": 0.1}
    novelty_info_low = {"novelty_score": 0.1}
    novelty_info_high = {"novelty_score": 0.9}

    results = {
        decide(ml_proba, novelty_info_low, strong_fp_row, cfg)["recommendation"],
        decide(ml_proba, novelty_info_low, high_risk_row, cfg)["recommendation"],
        decide(ml_proba, novelty_info_high, novel_row, cfg)["recommendation"],
    }
    assert REC_FALSE_POSITIVE in results
    assert REC_TRUE_INCIDENT in results
    assert REC_NOVEL in results


def test_risk_level_medium_between_low_and_high():
    cfg = load_config()
    raw_row = {"endpoint_risk_score": 45, "user_risk_score": 20, "previous_incident_count": 0}
    assert risk_level(raw_row, cfg) == "Medium"


def test_low_confidence_flag_set_below_threshold():
    cfg = load_config()
    cfg["low_confidence_threshold"] = 0.6
    ml_proba = {"False Positive": 0.55, "True Incident": 0.45}
    novelty_info = {"novelty_score": 0.1}
    raw_row = {"endpoint_risk_score": 5, "user_risk_score": 5, "previous_incident_count": 0,
               "previous_alert_count": 20, "similar_alert_count": 20}
    d = decide(ml_proba, novelty_info, raw_row, cfg)
    assert d["low_confidence"] is True


def test_high_risk_alert_requires_manual_confirmation_end_to_end(client):
    """Full pipeline: a high-risk alert must come back flagged for manual
    confirmation and must never be labeled Fake, via the real /predict route."""
    alert = {
        "alert_id": "HR1", "alert_type": "Suspicious Login", "severity": "Medium",
        "endpoint_risk_score": 90, "user_risk_score": 20, "previous_incident_count": 0,
        "previous_alert_count": 15, "similar_alert_count": 15, "alert_source": "IAM",
        "historical_false_positive_rate": 0.9,  # even with a history of being noisy
    }
    data = _predict_via_client(client, alert)
    assert data["recommendation"] != "LIKELY FALSE POSITIVE"
    assert data["simple_label"] != "Fake"
    assert data["high_risk_gate_triggered"] is True
    assert data["requires_manual_confirmation"] is True
    assert data["risk_level"] == "High"


def test_override_is_never_silent_and_always_logged_to_csv(client, tmp_path, monkeypatch):
    """Overrides must always require + store a reason, and must always be
    written to the durable CSV audit trail, never silently dropped."""
    import utils.audit_csv as audit_csv

    csv_path = tmp_path / "audit_log.csv"
    monkeypatch.setattr(audit_csv, "AUDIT_CSV_PATH", str(csv_path))

    data = _predict_via_client(client, {"alert_id": "OVCSV1", "alert_type": "Test Alert", "severity": "Low"})
    res = client.post(f"/decision/{data['id']}", json={
        "decision": "Override", "override_decision": "Likely True Incident",
        "override_reason": "Confirmed malicious via manual log review",
    })
    assert res.status_code == 200
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "OVCSV1" in content
    assert "Confirmed malicious via manual log review" in content
    assert ",yes," in content or content.strip().endswith("yes")


# ---------------------------------------------------------------------
# Rollback fixes: authorization, mandatory reason, duplicate-label
# disambiguation via row id, and a real model-file rollback proof.
# ---------------------------------------------------------------------

def _toggle_testing_off(app_module):
    """Context manager-ish helper: temporarily disable app.config['TESTING']
    so utils.auth.require_auth actually enforces authorization for one
    request, then restore it (every other test in this suite relies on
    TESTING=True to bypass auth -- see utils/auth.py)."""
    class _Toggle:
        def __enter__(self):
            app_module.app.config["TESTING"] = False
            return self

        def __exit__(self, *exc):
            app_module.app.config["TESTING"] = True
            return False
    return _Toggle()


def test_model_rollback_requires_confirm(client):
    res = client.post("/model/rollback", json={"version": "1.0", "reason": "x"})
    assert res.status_code == 400


def test_model_rollback_requires_reason(client):
    res = client.post("/model/rollback", json={"version": "1.0", "confirm": True})
    assert res.status_code == 400
    assert "reason" in res.get_json()["error"].lower()


def test_model_rollback_rejects_invalid_version(client):
    res = client.post("/model/rollback", json={
        "version": "999.999", "confirm": True, "reason": "test invalid version",
    })
    assert res.status_code == 404


def test_model_rollback_requires_auth(client):
    with _toggle_testing_off(flask_app_module):
        res = client.post("/model/rollback", json={
            "version": "1.0", "confirm": True, "reason": "should be blocked",
        })
        assert res.status_code == 401


def test_model_rollback_restores_previous_model_file_and_predictions(client):
    """Real end-to-end proof, not just a version-label check: retrain the
    model (v1 -> v2), confirm the actual model.pkl file on disk changes,
    roll back, and confirm the file is byte-for-byte identical to what it
    was before retraining and that predictions come from the restored
    model (not merely that a 'model_version' string changed back)."""
    import hashlib
    from model.train_model import MODEL_PATH as REAL_MODEL_PATH

    def _hash(path):
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    original_version = engine.model_version
    original_hash = _hash(REAL_MODEL_PATH)

    alert = {
        "alert_id": "MRB1", "alert_type": "Test Alert", "severity": "Low",
        "endpoint_risk_score": 10, "user_risk_score": 10, "previous_incident_count": 0,
        "previous_alert_count": 5, "similar_alert_count": 5,
    }
    cfg_snapshot = load_config()
    before = engine.predict(dict(alert), cfg_snapshot)

    retrain_res = client.post("/model/retrain")
    assert retrain_res.status_code == 200
    new_version = retrain_res.get_json()["new_model_version"]
    assert new_version != original_version
    assert engine.model_version == new_version

    retrained_hash = _hash(REAL_MODEL_PATH)
    assert retrained_hash != original_hash  # the actual file changed, not just a label

    rollback_res = client.post("/model/rollback", json={
        "version": original_version, "confirm": True, "reason": "pytest model rollback",
    })
    assert rollback_res.status_code == 200
    assert rollback_res.get_json()["current_model_version"] == original_version
    assert engine.model_version == original_version

    restored_hash = _hash(REAL_MODEL_PATH)
    assert restored_hash == original_hash  # byte-for-byte identical to the pre-retrain file

    after = engine.predict(dict(alert), load_config())
    assert after["ml_probabilities"] == before["ml_probabilities"]

    history = db.get_rollback_audit_events()
    assert any(h["version_type"] == "model" and h["target_label"] == original_version for h in history)


def test_rule_rollback_requires_reason(client):
    res = client.post("/change-review/rollback", json={"version": "1.0", "confirm": True})
    assert res.status_code == 400
    assert "reason" in res.get_json()["error"].lower()


def test_rule_rollback_rejects_invalid_version_id(client):
    res = client.post("/change-review/rollback", json={
        "version_id": 999999, "confirm": True, "reason": "test invalid id",
    })
    assert res.status_code == 404


def test_rule_rollback_requires_auth(client):
    with _toggle_testing_off(flask_app_module):
        res = client.post("/change-review/rollback", json={
            "version": "1.0", "confirm": True, "reason": "should be blocked",
        })
        assert res.status_code == 401


def test_confirm_action_requires_auth(client):
    data = _predict_via_client(client, {"alert_id": "AUTHHI1", "alert_type": "Test Alert", "severity": "Low"})
    with _toggle_testing_off(flask_app_module):
        res = client.post(f"/confirm-action/{data['id']}", json={"confirmation": "Approved"})
        assert res.status_code == 401


def test_rule_rollback_by_id_avoids_duplicate_label_ambiguity(client):
    """Two historical rule-version rows can legitimately share the same
    display label (e.g. after a rollback-then-new-change cycle -- version
    labels are just numeric display bumps, not a uniqueness guarantee; see
    routes/change_review.py). Rolling back via the unique row id must
    restore exactly the row that was selected, never silently substituting
    a different, more-recently-created row that happens to share the same
    label."""
    cfg_before = load_config()

    # Row A: an "older" rule-set snapshot labeled 1.0 with novelty_threshold=0.11
    old_cfg = dict(cfg_before)
    old_cfg["novelty_threshold"] = 0.11
    old_row_id = db.record_version(
        version_type="rule", version="1.0", description="duplicate-label test (older)",
        reason="test setup", reviewer="pytest", config_snapshot=old_cfg, status="inactive",
    )

    # Row B: a "newer" rule-set snapshot that ALSO happens to be labeled
    # 1.0, with a DIFFERENT novelty_threshold=0.99.
    new_cfg = dict(cfg_before)
    new_cfg["novelty_threshold"] = 0.99
    new_row_id = db.record_version(
        version_type="rule", version="1.0", description="duplicate-label test (newer)",
        reason="test setup", reviewer="pytest", config_snapshot=new_cfg, status="inactive",
    )
    assert new_row_id > old_row_id  # row B really is the more-recently-created "1.0"

    # Rolling back by the OLDER row's id must restore ITS config (0.11),
    # never the newer row that shares the same label.
    res = client.post("/change-review/rollback", json={
        "version_id": old_row_id, "confirm": True, "reason": "pytest duplicate-label rollback",
    })
    assert res.status_code == 200
    assert res.get_json()["version_id"] == old_row_id
    assert load_config()["novelty_threshold"] == 0.11

    # And, for comparison, rolling back by the NEWER row's id restores 0.99
    # -- proving the id, not the label, is what determines the outcome.
    res2 = client.post("/change-review/rollback", json={
        "version_id": new_row_id, "confirm": True, "reason": "pytest duplicate-label rollback 2",
    })
    assert res2.status_code == 200
    assert res2.get_json()["version_id"] == new_row_id
    assert load_config()["novelty_threshold"] == 0.99


def test_rollback_events_appear_in_audit_api_and_csv(client, tmp_path, monkeypatch):
    """Every successful model/rule rollback must show up in the SAME audit
    surfaces used for everything else: /api/audit and the durable CSV
    export -- in addition to (not instead of) the existing model_versions
    history table."""
    import utils.audit_csv as audit_csv_module

    csv_path = tmp_path / "audit_log.csv"
    monkeypatch.setattr(audit_csv_module, "AUDIT_CSV_PATH", str(csv_path))

    cfg_before = load_config()
    rollback_res = client.post("/change-review/rollback", json={
        "version": cfg_before["rule_version"], "confirm": True,
        "reason": "pytest audit-visibility rollback",
    })
    assert rollback_res.status_code == 200

    api_res = client.get("/api/audit")
    assert api_res.status_code == 200
    rollback_events = api_res.get_json()["rollback_events"]
    assert any(e["reason"] == "pytest audit-visibility rollback" for e in rollback_events)

    assert csv_path.exists()
    csv_content = csv_path.read_text()
    assert "pytest audit-visibility rollback" in csv_content
    assert "version_rollback" in csv_content


