"""
change_review.py
-----------------------------------------
Change Review workflow for proposed rule-set changes (thresholds,
baseline assumptions, safety targets). Nothing takes effect until
explicitly approved here. Every approved change bumps the rule
version and is recorded so it can later be rolled back.
-----------------------------------------
"""

import json
from flask import Blueprint, request, jsonify, render_template

from utils import db
from utils import audit_csv
from utils.auth import require_auth, current_analyst_name
from model.config import load_config, save_config

change_bp = Blueprint("change_bp", __name__)

EDITABLE_FIELDS = {
    "novelty_threshold": float,
    "similar_alert_count_threshold": int,
    "avg_investigation_minutes": float,
    "quick_review_minutes": float,
    "missed_incident_target_pct": float,
}


def _next_rule_version(current_version: str) -> str:
    """Simple, intuitive numeric bump (1.0 -> 1.1 -> 1.2 -> ...). Safe to reuse a
    label after a rollback-then-new-change (each DB row keeps its own id and full
    config snapshot regardless of label, so no history is lost even if two rows
    happen to share a display label)."""
    try:
        major, minor = current_version.split(".")
        return f"{major}.{int(minor) + 1}"
    except Exception:
        return current_version + ".1"


def register_change_review_routes(app, engine, get_config):

    @app.route("/change-review")
    def change_review_page():
        reviews = db.get_change_reviews()
        for r in reviews:
            try:
                r["change_payload"] = json.loads(r.get("change_payload") or "{}")
            except Exception:
                r["change_payload"] = {}
        cfg = get_config()
        rule_versions = db.get_versions("rule")
        return render_template("change_review.html", active="change_review",
                                reviews=reviews, config=cfg, rule_versions=rule_versions,
                                editable_fields=list(EDITABLE_FIELDS.keys()))

    @app.route("/change-review/propose", methods=["POST"])
    def propose():
        data = request.get_json(silent=True) or request.form.to_dict()
        field = data.get("field")
        new_value_raw = data.get("new_value")
        reason = data.get("reason", "").strip()

        if field not in EDITABLE_FIELDS:
            return jsonify({"error": f"Unknown/unsupported field: {field}"}), 400
        if not reason:
            return jsonify({"error": "A reason is required for the proposed change."}), 400

        try:
            new_value = EDITABLE_FIELDS[field](new_value_raw)
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid value for {field}: {new_value_raw}"}), 400

        cfg = get_config()
        old_value = cfg.get(field)
        current_version = cfg.get("rule_version", "1.0")
        proposed_version = _next_rule_version(current_version)

        proposed_change = f"{field}: {old_value} -> {new_value}"
        row_id = db.propose_change(
            proposed_change=proposed_change,
            reason=reason,
            current_version=current_version,
            proposed_version=proposed_version,
            change_payload={"field": field, "old_value": old_value, "new_value": new_value},
        )
        return jsonify({"success": True, "id": row_id, "proposed_change": proposed_change,
                         "proposed_version": proposed_version})

    @app.route("/change-review/<int:row_id>/decide", methods=["POST"])
    def decide_review(row_id):
        data = request.get_json(silent=True) or request.form.to_dict()
        status = data.get("status")
        reviewer = data.get("reviewer", "Analyst").strip() or "Analyst"

        if status not in ("Approved", "Rejected"):
            return jsonify({"error": "status must be 'Approved' or 'Rejected'."}), 400

        review = db.get_change_review(row_id)
        if review is None:
            return jsonify({"error": "Change review not found."}), 404
        if review["status"] != "Pending Review":
            return jsonify({"error": f"This change was already {review['status']}."}), 400

        db.decide_change_review(row_id, status, reviewer)

        if status == "Approved":
            payload = json.loads(review["change_payload"])
            cfg = get_config()

            # Snapshot the OLD config before mutating it, so rollback can restore it.
            db.record_version(
                version_type="rule", version=review["current_version"],
                description=f"Rule set before applying: {review['proposed_change']}",
                reason="Pre-change snapshot", reviewer=reviewer,
                config_snapshot=cfg, status="inactive",
            )

            cfg[payload["field"]] = payload["new_value"]
            cfg["rule_version"] = review["proposed_version"]
            save_config(cfg)

            db.record_version(
                version_type="rule", version=review["proposed_version"],
                description=review["proposed_change"], reason=review["reason"],
                reviewer=reviewer, config_snapshot=cfg, status="active",
            )

        return jsonify({"success": True, "id": row_id, "status": status})

    @app.route("/change-review/rollback", methods=["POST"])
    @require_auth
    def rollback_rule_version():
        data = request.get_json(silent=True) or request.form.to_dict()
        target_version_id = data.get("version_id")
        target_version_label = data.get("version")
        confirm = data.get("confirm")
        reason = (data.get("reason") or "").strip()

        if not confirm:
            return jsonify({"error": "Rollback requires confirm=true."}), 400
        if not reason:
            return jsonify({"error": "A rollback reason is required."}), 400
        if not target_version_id and not target_version_label:
            return jsonify({"error": "Target version is required."}), 400

        target = None
        if target_version_id is not None and target_version_id != "":
            # Preferred path: look up the EXACT historical row by its unique
            # database id. This is the only way to disambiguate two rows
            # that happen to share the same display label (version labels
            # are just numeric display bumps, not a uniqueness guarantee --
            # see _next_rule_version above) and guarantees rollback always
            # restores the row the analyst actually selected in the UI.
            try:
                row_id_int = int(target_version_id)
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid version_id: {target_version_id}"}), 400
            target = db.get_version_by_id(row_id_int)
            if target is None or target["version_type"] != "rule":
                return jsonify({"error": f"Rule version_id {target_version_id} not found in history."}), 404
        else:
            # Backward-compatible fallback for older clients that only send
            # a display label rather than a row id. If more than one row
            # shares this label, this matches the most-recently-created one
            # (see db.get_versions ordering) -- callers that need to
            # disambiguate duplicate labels precisely must pass version_id
            # instead. The shipped UI always sends version_id now.
            versions = db.get_versions("rule")
            target = next((v for v in versions if v["version"] == target_version_label), None)
            if target is None:
                return jsonify({"error": f"Rule version {target_version_label} not found in history."}), 404

        restored_cfg = json.loads(target["config_snapshot"])
        current_active = db.get_active_version("rule")

        # Rule-set rollback must not clobber the model_version field, since
        # model versioning and rule versioning are independent tracks (a
        # model retrain may have happened after this rule snapshot was
        # taken). Only the rule-related fields are restored.
        live_cfg = get_config()
        restored_cfg["model_version"] = live_cfg.get("model_version")
        save_config(restored_cfg)

        if current_active:
            db.mark_version_rolled_back(current_active["id"])

        reviewer = current_analyst_name()
        target_label = target["version"]
        restored_label = f"{target_label}-restored"

        new_row_id = db.record_version(
            version_type="rule", version=restored_label,
            description=f"Rolled back to rule set v{target_label}", reason=reason,
            reviewer=reviewer, config_snapshot=restored_cfg, status="active",
        )

        # Surface this rollback in the SAME audit trail analysts already use
        # for every other event (in addition to, not instead of, the
        # model_versions history table above).
        db.record_rollback_audit(
            version_type="rule", target_label=target_label, restored_version=restored_label,
            reason=reason, reviewer=reviewer,
            source_version_row_id=target["id"], resulting_version_row_id=new_row_id,
        )
        try:
            audit_csv.append_version_rollback(
                version_type="rule", target_version=target_label,
                restored_version=restored_label, reason=reason, reviewer=reviewer,
            )
        except Exception:
            pass  # never let audit-log I/O errors break a rollback that already succeeded

        return jsonify({"success": True, "restored_version": target_label, "version_id": target["id"]})

    return change_bp
