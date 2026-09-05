"""
audit_routes.py
-----------------------------------------
Audit trail page + the Analytics dashboard, which combines:
  - Live application usage stats (from the SQLite DB)
  - The offline, chronologically-evaluated experiment results computed
    once at training time and stored in model/metrics.json (baseline vs.
    assistant, hours saved, missed-incident rate, error analysis).

The two are kept clearly separate in the API response so the UI never
conflates "what happened in this demo session" with "what was measured
on the held-out chronological test split."
-----------------------------------------
"""

import json
import os
from flask import Blueprint, jsonify, render_template, send_file

from utils import db
from utils.audit_csv import AUDIT_CSV_PATH

audit_bp = Blueprint("audit_bp", __name__)

METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "metrics.json"
)


def _load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def register_audit_routes(app, engine, get_config):

    @app.route("/audit")
    def audit_page():
        return render_template("audit.html", active="audit")

    @app.route("/api/audit")
    def audit_data():
        alerts = db.get_all_alerts()
        for a in alerts:
            try:
                a["evidence"] = json.loads(a.get("evidence_json") or "[]")
            except Exception:
                a["evidence"] = []
        # Additive: model/rule rollback events, in the SAME audit response
        # analysts already read (in addition to, not instead of, the
        # existing model_versions history shown on /model and
        # /change-review).
        rollback_events = db.get_rollback_audit_events()
        return jsonify({"alerts": alerts, "rollback_events": rollback_events})

    @app.route("/api/audit/<int:row_id>/history")
    def audit_history(row_id):
        history = db.get_decision_history(row_id)
        return jsonify({"history": history})

    @app.route("/audit/export")
    def audit_export():
        """
        Download the durable, append-only, plain-text audit trail
        (audit_log.csv): every system recommendation plus every analyst
        action (accept/reject/override, with reason), never edited or
        deleted after the fact.
        """
        if not os.path.exists(AUDIT_CSV_PATH):
            return jsonify({"error": "No audit log entries yet."}), 404
        return send_file(AUDIT_CSV_PATH, mimetype="text/csv",
                          as_attachment=True, download_name="audit_log.csv")

    @app.route("/analytics")
    def analytics_page():
        return render_template("analytics.html", active="analytics")

    @app.route("/api/analytics")
    def analytics_data():
        live = db.get_analytics()
        metrics = _load_metrics()

        experiment = None
        error_analysis = None
        if metrics:
            experiment = {
                "trained_at": metrics.get("trained_at"),
                "model_version": metrics.get("model_version"),
                "row_counts": metrics.get("row_counts"),
                "classification": metrics.get("classification"),
                "novelty_detection": metrics.get("novelty_detection"),
                "safety_metrics": metrics.get("safety_metrics"),
                "false_positive_reduction": metrics.get("false_positive_reduction"),
                "baseline_vs_assistant": metrics.get("baseline_vs_assistant"),
            }
            error_analysis = metrics.get("error_analysis")

        return jsonify({
            "live": live,
            "experiment": experiment,
            "error_analysis": error_analysis,
            "config": get_config(),
        })

    return audit_bp
