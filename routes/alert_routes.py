"""
alert_routes.py
-----------------------------------------
Dashboard, manual alert prediction, analyst decision (with mandatory
override reason), high-impact action confirmation, and the
Investigation detail page.
-----------------------------------------
"""

import json
from flask import Blueprint, request, jsonify, render_template, abort

from utils.helpers import validate_alert_input, color_for_recommendation, validate_override, color_for_risk
from utils.explain import build_evidence, summary_line
from utils import db
from utils.auth import require_auth

alert_bp = Blueprint("alert_bp", __name__)


def register_alert_routes(app, engine, get_config):

    @app.route("/")
    def index():
        return render_template("index.html", active="home")

    @app.route("/predict", methods=["POST"])
    def predict():
        if not engine.is_loaded:
            return jsonify({"error": "Model is not loaded. Train the model first (python model/train_model.py)."}), 500

        data = request.get_json(silent=True) or request.form.to_dict()
        is_valid, error, cleaned = validate_alert_input(data)
        if not is_valid:
            return jsonify({"error": error}), 400

        cfg = get_config()

        try:
            result = engine.predict(cleaned, cfg)
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        evidence = build_evidence(cleaned, result)
        explanation = summary_line(cleaned, result)
        color = color_for_recommendation(result["recommendation"])

        try:
            row_id = db.insert_alert(cleaned, result, evidence, engine.model_version, cfg.get("rule_version"))
        except Exception as e:
            return jsonify({"error": f"Failed to save alert: {str(e)}"}), 500

        return jsonify({
            "id": row_id,
            "recommendation": result["recommendation"],
            "simple_label": result["simple_label"],
            "confidence": round(result["confidence"] * 100, 2),
            "explanation": explanation,
            "evidence": evidence,
            "color": color,
            "novelty_score": result["novelty_score"],
            "recommended_action": result["recommended_action"],
            "is_high_impact_action": result["is_high_impact_action"],
            "requires_manual_confirmation": result["requires_manual_confirmation"],
            "high_risk_gate_triggered": result["high_risk_gate_triggered"],
            "high_risk_reasons": result["high_risk_reasons"],
            "strong_fp_rule_triggered": result["strong_fp_rule_triggered"],
            "strong_fp_reasons": result["strong_fp_reasons"],
            "rule_triggered": result["rule_triggered"],
            "risk_level": result["risk_level"],
            "risk_color": color_for_risk(result["risk_level"]),
            "low_confidence": result["low_confidence"],
            "ml_probabilities": {k: round(v * 100, 2) for k, v in result["ml_probabilities"].items()},
            "model_version": result["model_version"],
            "rule_version": cfg.get("rule_version"),
        })

    @app.route("/investigation/<int:row_id>")
    def investigation_page(row_id):
        alert = db.get_alert(row_id)
        if alert is None:
            abort(404)
        alert["evidence"] = json.loads(alert.get("evidence_json") or "[]")
        alert["raw"] = json.loads(alert.get("raw_json") or "{}")
        history = db.get_decision_history(row_id)
        high_impact = db.get_high_impact_action(row_id)
        return render_template("investigation.html", active="investigation",
                                alert=alert, history=history, high_impact=high_impact)

    @app.route("/decision/<int:row_id>", methods=["POST"])
    def decision(row_id):
        data = request.get_json(silent=True) or request.form.to_dict()
        decision_value = data.get("decision")
        override_decision = data.get("override_decision")
        override_reason = data.get("override_reason")

        if decision_value not in ("Accept", "Reject", "Override"):
            return jsonify({"error": "Invalid decision. Must be Accept, Reject, or Override."}), 400

        ok, err = validate_override(decision_value, override_decision, override_reason)
        if not ok:
            return jsonify({"error": err}), 400

        alert = db.get_alert(row_id)
        if alert is None:
            return jsonify({"error": "Alert not found."}), 404

        db.update_decision(row_id, decision_value, override_decision, override_reason)
        return jsonify({"success": True, "id": row_id, "decision": decision_value,
                         "override_decision": override_decision})

    @app.route("/rollback/<int:row_id>", methods=["POST"])
    def rollback(row_id):
        alert = db.get_alert(row_id)
        if alert is None:
            return jsonify({"error": "Alert not found."}), 404

        previous = db.rollback_decision(row_id)
        if previous is None:
            return jsonify({"error": "No previous decision to roll back to."}), 400

        return jsonify({"success": True, "id": row_id, "reverted_to": previous})

    @app.route("/confirm-action/<int:row_id>", methods=["POST"])
    @require_auth
    def confirm_action(row_id):
        data = request.get_json(silent=True) or request.form.to_dict()
        confirmation = data.get("confirmation")
        if confirmation not in ("Approved", "Rejected"):
            return jsonify({"error": "confirmation must be 'Approved' or 'Rejected'."}), 400

        alert = db.get_alert(row_id)
        if alert is None:
            return jsonify({"error": "Alert not found."}), 404
        if not alert.get("is_high_impact_action"):
            return jsonify({"error": "This alert has no high-impact action requiring confirmation."}), 400

        db.confirm_high_impact_action(row_id, confirmation)
        return jsonify({"success": True, "id": row_id, "confirmation": confirmation})

    return alert_bp
