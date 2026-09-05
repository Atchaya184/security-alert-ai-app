"""
model_routes.py
-----------------------------------------
Model / Feedback page: shows the current model version + metrics,
and implements the CONTROLLED analyst-feedback learning loop:

    [Prepare Retraining Dataset]  ->  [Retrain Model]

Retraining never happens automatically on every click; an analyst must
explicitly prepare the dataset and then explicitly trigger retraining.
A new model version is recorded (with metrics + timestamp) and a
rollback to the previous model version is always available.
-----------------------------------------
"""

import os
import json
import shutil
import pandas as pd
from flask import Blueprint, request, jsonify, render_template

from utils import db
from utils import audit_csv
from utils.auth import require_auth, current_analyst_name
from model.config import load_config
from model.preprocess import FEATURE_COLUMNS, clean_dataframe

model_bp = Blueprint("model_bp", __name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "security_alerts_2000.csv")
PREPARED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "retrain_prepared.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")


def _next_model_version(current_version: str) -> str:
    """Simple, intuitive numeric bump (1.0 -> 1.1 -> 1.2 -> ...)."""
    try:
        major, minor = current_version.split(".")
        return f"{major}.{int(minor) + 1}"
    except Exception:
        return current_version + ".1"


def _archive_filename(version_label: str) -> str:
    """
    Archived model files are named with BOTH the version label and a
    timestamp. This guarantees the archive is collision-proof (never
    silently overwrites a different model file) even if a version label
    is reused after a rollback followed by a new training run — which
    can legitimately happen since labels are just numeric display bumps,
    not a uniqueness guarantee.
    """
    ts = datetime_now_compact()
    return f"model_v{version_label}_{ts}.pkl"


def datetime_now_compact():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _find_archive_for_version(version_label: str):
    """Return the path to the MOST RECENTLY archived file matching this version
    label, or None. Using mtime keeps behavior intuitive ('roll back to the
    latest thing that was called v1.1') without ever deleting older entries."""
    if not os.path.isdir(ARCHIVE_DIR):
        return None
    candidates = [
        os.path.join(ARCHIVE_DIR, f) for f in os.listdir(ARCHIVE_DIR)
        if f == f"model_v{version_label}.pkl" or f.startswith(f"model_v{version_label}_")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def register_model_routes(app, engine, get_config):

    @app.route("/model")
    def model_page():
        cfg = get_config()
        metrics = _load_metrics()
        feedback_stats = db.get_feedback_stats()
        model_versions = db.get_versions("model")
        return render_template("model.html", active="model", config=cfg, metrics=metrics,
                                feedback_stats=feedback_stats, model_versions=model_versions)

    @app.route("/api/model/status")
    def model_status():
        cfg = get_config()
        metrics = _load_metrics()
        feedback_stats = db.get_feedback_stats()
        model_versions = db.get_versions("model")
        prepared_exists = os.path.exists(PREPARED_PATH)
        prepared_count = 0
        if prepared_exists:
            try:
                prepared_count = len(pd.read_csv(PREPARED_PATH))
            except Exception:
                prepared_count = 0
        return jsonify({
            "current_model_version": engine.model_version,
            "config": cfg,
            "metrics_summary": {
                "trained_at": metrics.get("trained_at") if metrics else None,
                "accuracy": metrics.get("classification", {}).get("calibrated_operating_point", {}).get("accuracy") if metrics else None,
                "missed_incident_rate_pct": metrics.get("safety_metrics", {}).get("missed_incident_rate_pct") if metrics else None,
                "hours_saved": metrics.get("baseline_vs_assistant", {}).get("analyst_hours_saved") if metrics else None,
            } if metrics else None,
            "feedback_stats": feedback_stats,
            "model_versions": model_versions,
            "retrain_dataset_prepared": prepared_exists,
            "prepared_row_count": prepared_count,
        })

    @app.route("/model/prepare-retrain", methods=["POST"])
    def prepare_retrain():
        feedback_rows = db.get_unused_feedback()
        if not feedback_rows:
            return jsonify({"success": True, "prepared_rows": 0,
                             "message": "No new labeled analyst feedback available to prepare."})

        records = []
        used_ids = []
        for fb in feedback_rows:
            try:
                raw = json.loads(fb["raw_json"])
            except Exception:
                continue
            row = {col: raw.get(col, "Unknown") for col in FEATURE_COLUMNS}
            row["alert_id"] = fb.get("source_alert_id", f"FEEDBACK-{fb['id']}")
            row["ground_truth"] = fb["derived_label"]
            row["timestamp"] = fb["created_at"]
            row["novelty_test_flag"] = "No"
            records.append(row)
            used_ids.append(fb["id"])

        if not records:
            return jsonify({"success": True, "prepared_rows": 0,
                             "message": "No feedback rows had a usable derived label."})

        prepared_df = pd.DataFrame(records)
        prepared_df.to_csv(PREPARED_PATH, index=False)
        db.mark_feedback_used(used_ids)

        return jsonify({"success": True, "prepared_rows": len(records), "path": "dataset/retrain_prepared.csv"})

    @app.route("/model/retrain", methods=["POST"])
    def retrain():
        from model.train_model import train_and_save  # local import: avoids circular import at app startup

        cfg = get_config()
        extra_df = None
        n_feedback = 0
        if os.path.exists(PREPARED_PATH):
            try:
                extra_df = pd.read_csv(PREPARED_PATH)
                extra_df = clean_dataframe(extra_df)
                n_feedback = len(extra_df)
            except Exception:
                extra_df = None

        new_version = _next_model_version(cfg.get("model_version", "1.0"))

        # Archive the CURRENTLY ACTIVE model file under a timestamped name
        # (collision-proof regardless of version-label reuse) so rollback
        # can restore it later. Labeled using the ACTUALLY LOADED bundle's
        # own internal version (ground truth), not the config file, since
        # the two could in principle drift.
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        current_model_path = os.path.join(MODEL_DIR, "model.pkl")
        active_label = engine.model_version if engine.is_loaded else cfg.get("model_version", "1.0")
        if os.path.exists(current_model_path):
            archive_path = os.path.join(ARCHIVE_DIR, _archive_filename(active_label))
            shutil.copy(current_model_path, archive_path)

        try:
            _, _, metrics = train_and_save(
                dataset_path=DATASET_PATH,
                extra_df=extra_df,
                config=cfg,
                model_version=new_version,
                trigger="analyst_feedback_retrain",
            )
        except Exception as e:
            return jsonify({"error": f"Retraining failed: {str(e)}"}), 500

        engine.load()  # hot-reload the newly trained bundle into the running app

        db.record_version(
            version_type="model", version=new_version,
            description=f"Retrained with {n_feedback} accumulated analyst feedback row(s)",
            reason="Controlled retraining triggered from Model/Feedback page",
            reviewer="Analyst", config_snapshot=cfg, metrics_snapshot=metrics, status="active",
        )

        return jsonify({
            "success": True,
            "new_model_version": new_version,
            "feedback_rows_used": n_feedback,
            "metrics_summary": {
                "accuracy": metrics["classification"]["calibrated_operating_point"]["accuracy"],
                "missed_incident_rate_pct": metrics["safety_metrics"]["missed_incident_rate_pct"],
                "hours_saved": metrics["baseline_vs_assistant"]["analyst_hours_saved"],
                "false_positive_reduction_pct": metrics["false_positive_reduction"]["false_positive_reduction_rate_pct"],
            },
        })

    @app.route("/model/rollback", methods=["POST"])
    @require_auth
    def rollback_model():
        data = request.get_json(silent=True) or request.form.to_dict()
        target_version = data.get("version")
        confirm = data.get("confirm")
        reason = (data.get("reason") or "").strip()

        if not confirm:
            return jsonify({"error": "Rollback requires confirm=true."}), 400
        if not reason:
            return jsonify({"error": "A rollback reason is required."}), 400
        if not target_version:
            return jsonify({"error": "Target version is required."}), 400

        archive_path = _find_archive_for_version(target_version)
        if not archive_path:
            return jsonify({"error": f"No archived model file found for version {target_version}."}), 404

        current_model_path = os.path.join(MODEL_DIR, "model.pkl")
        cfg = get_config()

        # Preserve the model we're rolling back FROM, in case of a future
        # re-rollback. Timestamped filename guarantees this never
        # overwrites an existing archive entry.
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        active_label = engine.model_version if engine.is_loaded else cfg.get("model_version", "unknown")
        from_archive_path = os.path.join(ARCHIVE_DIR, _archive_filename(active_label))
        shutil.copy(current_model_path, from_archive_path)
        shutil.copy(archive_path, current_model_path)

        engine.load()

        # Keep config.json's model_version in sync with the actually-active
        # model file, so the Change Review / Model page and every new
        # recommendation's recorded model_version reflect reality.
        from model.config import save_config
        cfg["model_version"] = engine.model_version
        save_config(cfg)

        current_active = db.get_active_version("model")
        if current_active:
            db.mark_version_rolled_back(current_active["id"])

        reviewer = current_analyst_name()
        restored_label = f"{target_version}-restored"
        new_row_id = db.record_version(
            version_type="model", version=restored_label,
            description=f"Rolled back to model v{target_version}",
            reason=reason,
            reviewer=reviewer, config_snapshot=cfg, status="active",
        )

        # Surface this rollback in the SAME audit trail analysts already use
        # for every other event (in addition to, not instead of, the
        # model_versions history table above). Model rollback matches
        # archived files by label + timestamp rather than a DB row id (see
        # _find_archive_for_version), so source_version_row_id is left
        # unset here -- only rule rollback has an unambiguous source row.
        db.record_rollback_audit(
            version_type="model", target_label=target_version, restored_version=restored_label,
            reason=reason, reviewer=reviewer, resulting_version_row_id=new_row_id,
        )
        try:
            audit_csv.append_version_rollback(
                version_type="model", target_version=target_version,
                restored_version=restored_label, reason=reason, reviewer=reviewer,
            )
        except Exception:
            pass  # never let audit-log I/O errors break a rollback that already succeeded

        return jsonify({"success": True, "restored_version": target_version,
                         "current_model_version": engine.model_version})

    return model_bp
