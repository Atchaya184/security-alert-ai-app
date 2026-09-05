"""
train_model.py
-----------------------------------------
Trains the binary classifier (False Positive / True Incident), fits the
novelty detector, applies the safety-gate recommendation logic to a
held-out CHRONOLOGICAL test split, and computes every metric genuinely
from the data (no fabricated numbers):

  - Classification report + confusion matrix (test split)
  - Novelty-detection recovery of the held-out `novelty_test_flag` cases
  - False-positive reduction
  - Missed-incident count & rate (safety metric)
  - Baseline vs. assistant analyst-hours comparison
  - Error analysis with sample alert IDs per error category

Run directly:
    python model/train_model.py
-----------------------------------------
"""

import os
import sys
import json
import copy
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_and_clean, chronological_split, get_features_and_target,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET_CLASSES,
)
from novelty import NoveltyDetector
from decision import decide, classify_from_proba, recommended_action, REC_FALSE_POSITIVE, REC_TRUE_INCIDENT, REC_NOVEL
from config import load_config, DEFAULT_CONFIG

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _proba_dict(proba_row, classes):
    return {cls: float(p) for cls, p in zip(classes, proba_row)}


def train_and_save(dataset_path="dataset/security_alerts_2000.csv",
                    extra_df: pd.DataFrame = None,
                    config: dict = None,
                    model_version: str = None,
                    trigger: str = "initial_training"):
    """
    Full pipeline: load -> chronological split -> train classifier ->
    fit novelty detector -> evaluate on test split -> compute all
    measurable metrics -> save model bundle + metrics.json.

    `extra_df` (optional) lets the analyst-feedback retraining path
    append accumulated feedback rows to the original dataset before
    re-splitting and re-training (see routes/model_routes.py).
    """
    cfg = config or load_config()

    df = load_and_clean(dataset_path)
    df["_source"] = "original_dataset"

    # Chronological split happens on the ORIGINAL dataset only. Accumulated
    # analyst feedback is confirmed historical ground truth by definition
    # (an analyst has already reviewed it), so it is appended directly to
    # the TRAINING split rather than re-inserted into the timeline and
    # potentially landing in the "future" validation/test slices, which
    # would leak analyst review effort into the evaluation of alerts that
    # are supposed to simulate not-yet-reviewed future alerts.
    train_df, val_df, test_df = chronological_split(df)

    if extra_df is not None and len(extra_df) > 0:
        extra_df = extra_df.copy()
        extra_df["_source"] = "analyst_feedback"
        train_df = pd.concat([train_df, extra_df], ignore_index=True, sort=False)

    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(val_df)
    X_test, y_test = get_features_and_target(test_df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    classes = list(pipeline.classes_)

    # ---- Novelty detector, fit ONLY on training-split feature space ----
    preproc = pipeline.named_steps["preprocessor"]
    X_train_transformed = preproc.transform(X_train)
    train_raw_rows = train_df[FEATURE_COLUMNS].to_dict("records")

    novelty_detector = NoveltyDetector(contamination=0.08)
    novelty_detector.fit(X_train_transformed, train_raw_rows)

    # ---- Calibrate the True-Incident decision threshold on VALIDATION ----
    # (never on test). Missing a genuine incident is far costlier than an
    # extra false-positive review, so we search for the threshold that
    # keeps the validation missed-incident rate at/near the configured
    # target while preserving as much false-positive reduction as
    # possible. This is a real, data-driven calibration step, not a
    # hard-coded constant.
    X_val_transformed = preproc.transform(X_val)
    val_raw_rows = val_df[FEATURE_COLUMNS].to_dict("records")
    val_proba = pipeline.predict_proba(X_val)
    val_novelty_infos = novelty_detector.score_batch(X_val_transformed, val_raw_rows)
    val_ground_truth = val_df["ground_truth"].values
    missed_target_pct = float(cfg.get("missed_incident_target_pct", 2.0))

    candidate_thresholds = [round(t, 2) for t in np.arange(0.02, 0.55, 0.02)]
    threshold_search = []

    for thr in candidate_thresholds:
        trial_cfg = {**cfg, "true_incident_threshold": thr}
        recs = [
            decide(_proba_dict(val_proba[i], classes), val_novelty_infos[i], val_raw_rows[i], trial_cfg)
            for i in range(len(val_df))
        ]
        rec_labels = np.array([d["recommendation"] for d in recs])
        actual_incident = val_ground_truth == "True Incident"
        actual_fp = val_ground_truth == "False Positive"
        n_incidents = int(actual_incident.sum())
        missed = int((actual_incident & (rec_labels == REC_FALSE_POSITIVE)).sum())
        missed_rate = (missed / n_incidents) if n_incidents > 0 else 0.0
        fp_flagged = int((actual_fp & (rec_labels == REC_FALSE_POSITIVE)).sum())
        n_fp = int(actual_fp.sum())
        fp_rate = (fp_flagged / n_fp) if n_fp > 0 else 0.0

        threshold_search.append({
            "threshold": thr,
            "val_missed_incident_rate_pct": round(missed_rate * 100, 3),
            "val_false_positive_reduction_pct": round(fp_rate * 100, 3),
        })

    # Selection rule (decided purely from validation results, never test):
    #   - Among thresholds whose VALIDATION missed-incident rate meets the
    #     safety target, pick the one with the highest false-positive
    #     reduction (best analyst-time benefit that still meets the target).
    #   - If none meet the target, fall back to the threshold with the
    #     lowest validation missed-incident rate (safest available option),
    #     and this is reported honestly as "target not met" rather than
    #     forced to look successful.
    meeting_target = [c for c in threshold_search if c["val_missed_incident_rate_pct"] <= missed_target_pct]
    if meeting_target:
        best = max(meeting_target, key=lambda c: c["val_false_positive_reduction_pct"])
    else:
        best = min(threshold_search, key=lambda c: c["val_missed_incident_rate_pct"])
    best_threshold = best["threshold"]

    cfg = {**cfg, "true_incident_threshold": best_threshold}

    # ---- Evaluate raw classifier on the TEST split (using calibrated threshold) ----
    y_test_proba = pipeline.predict_proba(X_test)
    y_test_pred = np.array([classify_from_proba(_proba_dict(y_test_proba[i], classes), cfg)
                             for i in range(len(y_test))])

    # Threshold-INDEPENDENT discrimination quality (how well the model
    # ranks incidents above false positives, regardless of where the
    # decision threshold is set). Reported separately from the calibrated
    # operating-point accuracy below, since those two numbers answer
    # different questions and shouldn't be conflated.
    true_incident_idx = classes.index("True Incident")
    roc_auc = roc_auc_score((y_test == "True Incident").astype(int), y_test_proba[:, true_incident_idx])

    # Reference report at the naive 0.5 cutoff, for comparison against the
    # safety-calibrated operating point used operationally (see below).
    y_test_pred_uncalibrated = pipeline.predict(X_test)
    uncalibrated_accuracy = accuracy_score(y_test, y_test_pred_uncalibrated)
    uncalibrated_cm = confusion_matrix(y_test, y_test_pred_uncalibrated, labels=TARGET_CLASSES)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, labels=TARGET_CLASSES, zero_division=0
    )
    cm = confusion_matrix(y_test, y_test_pred, labels=TARGET_CLASSES)
    report_dict = classification_report(
        y_test, y_test_pred, labels=TARGET_CLASSES, output_dict=True, zero_division=0
    )

    # ---- Novelty scores + full safety-gated recommendation on TEST ----
    X_test_transformed = preproc.transform(X_test)
    test_raw_rows = test_df[FEATURE_COLUMNS].to_dict("records")
    novelty_infos = novelty_detector.score_batch(X_test_transformed, test_raw_rows)

    decisions = []
    for i in range(len(test_df)):
        ml_proba = _proba_dict(y_test_proba[i], classes)
        d = decide(ml_proba, novelty_infos[i], test_raw_rows[i], cfg)
        decisions.append(d)

    test_df = test_df.reset_index(drop=True)
    test_df["_ml_pred"] = y_test_pred
    test_df["_recommendation"] = [d["recommendation"] for d in decisions]
    test_df["_confidence"] = [d["confidence"] for d in decisions]
    test_df["_novelty_score"] = [ni["novelty_score"] for ni in novelty_infos]
    test_df["_novelty_gate"] = [d["novelty_gate_triggered"] for d in decisions]
    test_df["_insufficient_evidence"] = [d["insufficient_evidence"] for d in decisions]

    n_test = len(test_df)
    ground_truth = test_df["ground_truth"].values

    # ---- False positive reduction ----
    actual_fp_mask = ground_truth == "False Positive"
    n_actual_fp = int(actual_fp_mask.sum())
    recommended_fp_mask = (test_df["_recommendation"] == REC_FALSE_POSITIVE).values
    n_recommended_fp = int(recommended_fp_mask.sum())
    # Of the ACTUAL false positives, how many did the assistant correctly
    # flag for suppression (i.e. no longer need full manual investigation)?
    fp_correctly_flagged = int((recommended_fp_mask & actual_fp_mask).sum())
    false_positive_reduction_rate = (
        fp_correctly_flagged / n_actual_fp if n_actual_fp > 0 else 0.0
    )

    # ---- Missed incidents (critical safety metric) ----
    actual_incident_mask = ground_truth == "True Incident"
    n_actual_incidents = int(actual_incident_mask.sum())
    missed_mask = actual_incident_mask & recommended_fp_mask
    n_missed = int(missed_mask.sum())
    missed_incident_rate = (n_missed / n_actual_incidents if n_actual_incidents > 0 else 0.0)
    missed_alert_ids = test_df.loc[missed_mask, "alert_id"].astype(str).head(10).tolist()

    # ---- Novelty detection evaluation vs. held-out novelty_test_flag ----
    # novelty_test_flag is used ONLY here, for evaluation, never as a feature.
    novel_flag_test = (test_df["novelty_test_flag"] == "Yes").values
    novel_predicted = (test_df["_recommendation"] == REC_NOVEL).values
    n_flagged_novel_in_test = int(novel_flag_test.sum())
    novelty_recall = (
        float((novel_flag_test & novel_predicted).sum()) / n_flagged_novel_in_test
        if n_flagged_novel_in_test > 0 else None
    )
    n_predicted_novel = int(novel_predicted.sum())
    novelty_precision = (
        float((novel_flag_test & novel_predicted).sum()) / n_predicted_novel
        if n_predicted_novel > 0 else None
    )

    # ---- Baseline vs. Assistant: analyst hours ----
    avg_min = float(cfg.get("avg_investigation_minutes", 5.0))
    quick_min = float(cfg.get("quick_review_minutes", 0.5))

    baseline_hours = (n_test * avg_min) / 60.0

    n_requires_review = int((test_df["_recommendation"] != REC_FALSE_POSITIVE).sum())
    n_quick_review = int((test_df["_recommendation"] == REC_FALSE_POSITIVE).sum())
    assistant_hours = (n_requires_review * avg_min + n_quick_review * quick_min) / 60.0

    hours_saved = baseline_hours - assistant_hours
    pct_saved = (hours_saved / baseline_hours * 100.0) if baseline_hours > 0 else 0.0

    missed_target_pct = float(cfg.get("missed_incident_target_pct", 2.0))
    target_achieved = (missed_incident_rate * 100.0) <= missed_target_pct

    # ---- Error analysis ----
    fp_as_incident_mask = actual_fp_mask & (test_df["_recommendation"] == REC_TRUE_INCIDENT)
    incident_as_fp_mask = missed_mask  # same definition as missed incidents
    novel_but_was_fp = novel_predicted & actual_fp_mask
    novel_but_was_incident = novel_predicted & actual_incident_mask
    low_confidence_mask = test_df["_confidence"] < 0.6

    def _sample_ids(mask, n=5):
        return test_df.loc[mask, "alert_id"].astype(str).head(n).tolist()

    error_analysis = {
        "false_positive_called_incident": {
            "count": int(fp_as_incident_mask.sum()),
            "pct_of_test": round(100 * int(fp_as_incident_mask.sum()) / n_test, 2) if n_test else 0,
            "sample_alert_ids": _sample_ids(fp_as_incident_mask),
            "explanation": "Actual false positives that the classifier scored as a true incident "
                           "(analyst time is wasted, but no incident is missed).",
        },
        "incident_called_false_positive_missed": {
            "count": int(incident_as_fp_mask.sum()),
            "pct_of_test": round(100 * int(incident_as_fp_mask.sum()) / n_test, 2) if n_test else 0,
            "sample_alert_ids": missed_alert_ids,
            "explanation": "MISSED INCIDENTS: confirmed true incidents recommended as a false "
                           "positive suppression. This is the critical safety metric to minimize.",
        },
        "novel_but_was_false_positive": {
            "count": int(novel_but_was_fp.sum()),
            "pct_of_test": round(100 * int(novel_but_was_fp.sum()) / n_test, 2) if n_test else 0,
            "sample_alert_ids": _sample_ids(novel_but_was_fp),
            "explanation": "Flagged NOVEL/UNKNOWN by the safety gate but turned out to be a false "
                           "positive. Costs analyst time but is a safe, intentional trade-off.",
        },
        "novel_but_was_incident": {
            "count": int(novel_but_was_incident.sum()),
            "pct_of_test": round(100 * int(novel_but_was_incident.sum()) / n_test, 2) if n_test else 0,
            "sample_alert_ids": _sample_ids(novel_but_was_incident),
            "explanation": "Flagged NOVEL/UNKNOWN and turned out to be a genuine incident — exactly "
                           "the case the novelty safety gate exists to protect against.",
        },
        "low_confidence_predictions": {
            "count": int(low_confidence_mask.sum()),
            "pct_of_test": round(100 * int(low_confidence_mask.sum()) / n_test, 2) if n_test else 0,
            "sample_alert_ids": _sample_ids(low_confidence_mask),
            "explanation": "Recommendations made with confidence below 0.60 — worth extra analyst "
                           "attention even if the recommendation label happens to be correct.",
        },
    }

    trained_at = datetime.now(timezone.utc).isoformat()
    resolved_model_version = model_version or cfg.get("model_version", "1.0")

    metrics = {
        "trained_at": trained_at,
        "trigger": trigger,
        "model_version": resolved_model_version,
        "rule_version": cfg.get("rule_version", "1.0"),
        "dataset_path": dataset_path,
        "threshold_calibration": {
            "note": "True-Incident decision threshold calibrated on the VALIDATION split "
                    "(never on test) to minimize missed incidents relative to the safety target.",
            "candidates": threshold_search,
            "selected_threshold": best_threshold,
        },
        "row_counts": {
            "total": int(len(train_df) + len(val_df) + len(test_df)),
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
            "analyst_feedback_rows_included": int((train_df["_source"] == "analyst_feedback").sum()),
        },
        "classification": {
            "classes": TARGET_CLASSES,
            "roc_auc_true_incident": round(float(roc_auc), 4),
            "roc_auc_note": "Threshold-independent ranking quality (0.5 = random, 1.0 = perfect). "
                            "Not affected by the safety-threshold calibration below.",
            "uncalibrated_0.5_threshold": {
                "accuracy": round(float(uncalibrated_accuracy), 4),
                "confusion_matrix": uncalibrated_cm.tolist(),
                "note": "Reference performance at a naive 0.5 probability cutoff, shown for "
                        "comparison against the safety-calibrated operating point actually used.",
            },
            "calibrated_operating_point": {
                "true_incident_threshold": best_threshold,
                "accuracy": round(float(accuracy), 4),
                "precision": {c: round(float(p), 4) for c, p in zip(TARGET_CLASSES, precision)},
                "recall": {c: round(float(r), 4) for c, r in zip(TARGET_CLASSES, recall)},
                "f1_score": {c: round(float(f), 4) for c, f in zip(TARGET_CLASSES, f1)},
                "support": {c: int(s) for c, s in zip(TARGET_CLASSES, support)},
                "confusion_matrix": cm.tolist(),
                "confusion_matrix_labels": TARGET_CLASSES,
                "full_report": report_dict,
            },
        },
        "novelty_detection": {
            "n_test_rows": n_test,
            "n_flagged_as_novelty_test_case": n_flagged_novel_in_test,
            "n_predicted_novel": n_predicted_novel,
            "novelty_recall_on_flagged_cases": round(novelty_recall, 4) if novelty_recall is not None else None,
            "novelty_precision_on_predicted_novel": round(novelty_precision, 4) if novelty_precision is not None else None,
            "novelty_threshold_used": cfg.get("novelty_threshold"),
            "note": "novelty_test_flag was used ONLY for this evaluation, never as a model input feature.",
        },
        "safety_metrics": {
            "actual_incidents_in_test": n_actual_incidents,
            "missed_incidents": n_missed,
            "missed_incident_rate_pct": round(missed_incident_rate * 100.0, 3),
            "missed_incident_target_pct": missed_target_pct,
            "target_achieved": bool(target_achieved),
            "missed_incident_sample_alert_ids": missed_alert_ids,
        },
        "false_positive_reduction": {
            "actual_false_positives_in_test": n_actual_fp,
            "correctly_flagged_false_positives": fp_correctly_flagged,
            "false_positive_reduction_rate_pct": round(false_positive_reduction_rate * 100.0, 3),
        },
        "baseline_vs_assistant": {
            "avg_investigation_minutes": avg_min,
            "quick_review_minutes": quick_min,
            "total_test_alerts": n_test,
            "alerts_requiring_human_investigation": n_requires_review,
            "alerts_recommended_false_positive": n_quick_review,
            "baseline_analyst_hours": round(baseline_hours, 3),
            "assistant_analyst_hours": round(assistant_hours, 3),
            "analyst_hours_saved": round(hours_saved, 3),
            "pct_hours_saved": round(pct_saved, 3),
        },
        "error_analysis": error_analysis,
    }

    joblib.dump({
        "pipeline": pipeline,
        "novelty_detector": novelty_detector,
        "model_version": resolved_model_version,
        "trained_at": trained_at,
        "feature_columns": FEATURE_COLUMNS,
        "classes": classes,
    }, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Persist the calibrated threshold as part of the active rule config so
    # live inference (utils/recommend.py) uses the same decision rule that
    # was just evaluated above.
    from config import save_config
    persisted_cfg = {**cfg, "model_version": resolved_model_version}
    save_config(persisted_cfg)

    print(f"Model v{resolved_model_version} trained and saved to {MODEL_PATH}")
    print(f"ROC-AUC (True Incident): {roc_auc:.4f} | Calibrated threshold: {best_threshold} | "
          f"Calibrated test accuracy: {accuracy:.4f}")
    print(f"Missed-incident rate: {missed_incident_rate*100:.2f}% (target <= {missed_target_pct}%, "
          f"achieved={target_achieved}) | FP reduction: {false_positive_reduction_rate*100:.1f}% | "
          f"Hours saved: {hours_saved:.2f}h ({pct_saved:.1f}%)")

    return pipeline, novelty_detector, metrics


if __name__ == "__main__":
    project_root = os.path.dirname(MODEL_DIR)
    dataset_path = os.path.join(project_root, "dataset", "security_alerts_2000.csv")
    if not os.path.exists(dataset_path):
        dataset_path = "dataset/security_alerts_2000.csv"
    train_and_save(dataset_path)
