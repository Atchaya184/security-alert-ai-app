"""
upload_routes.py
-----------------------------------------
Bulk CSV alert upload and classification, representing alerts coming
from multiple disconnected security tools (via `alert_source`),
normalized into one unified workflow.
-----------------------------------------
"""

import io
import pandas as pd
from flask import Blueprint, request, jsonify, render_template

from utils.helpers import validate_alert_input, color_for_recommendation, validate_csv_columns
from utils.explain import build_evidence, summary_line
from utils import db

upload_bp = Blueprint("upload_bp", __name__)


def register_upload_routes(app, engine, get_config):

    @app.route("/upload", methods=["GET"])
    def upload_page():
        return render_template("upload.html", active="upload")

    @app.route("/upload", methods=["POST"])
    def upload_csv():
        if not engine.is_loaded:
            return jsonify({"error": "Model is not loaded. Train the model first."}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only .csv files are supported."}), 400

        try:
            content = file.read()
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty."}), 400

        col_ok, col_err = validate_csv_columns(df.columns)
        if not col_ok:
            return jsonify({"error": col_err}), 400

        cfg = get_config()
        results = []
        errors = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            is_valid, error, cleaned = validate_alert_input(row_dict)
            if not is_valid:
                errors.append({"row": int(idx), "error": error})
                continue

            try:
                result = engine.predict(cleaned, cfg)
                evidence = build_evidence(cleaned, result)
                row_id = db.insert_alert(cleaned, result, evidence, engine.model_version, cfg.get("rule_version"))

                results.append({
                    "id": row_id,
                    "alert_id": cleaned.get("alert_id", "Unknown"),
                    "recommendation": result["recommendation"],
                    "confidence": round(result["confidence"] * 100, 2),
                    "explanation": summary_line(cleaned, result),
                    "color": color_for_recommendation(result["recommendation"]),
                    "recommended_action": result["recommended_action"],
                    "is_high_impact_action": result["is_high_impact_action"],
                })
            except Exception as e:
                errors.append({"row": int(idx), "error": str(e)})

        return jsonify({
            "success": True,
            "processed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        })

    return upload_bp
