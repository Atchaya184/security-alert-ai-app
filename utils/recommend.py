"""
recommend.py
-----------------------------------------
Live inference for the web app: loads the trained model bundle
(classifier pipeline + novelty detector) and applies the exact same
decide() safety-gate logic used during offline evaluation, so that
what gets reported in metrics.json matches what the app actually does.
-----------------------------------------
"""

import os
import sys
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
sys.path.append(MODEL_DIR)

from preprocess import prepare_single_alert, assert_no_leakage  # noqa: E402
from decision import decide, recommended_action, is_high_impact  # noqa: E402

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


class RecommendationEngine:
    def __init__(self):
        self.bundle = None
        self.pipeline = None
        self.novelty_detector = None
        self.model_version = None

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}. Run model/train_model.py first.")
        self.bundle = joblib.load(MODEL_PATH)
        self.pipeline = self.bundle["pipeline"]
        self.novelty_detector = self.bundle["novelty_detector"]
        self.model_version = self.bundle.get("model_version", "unknown")
        return self

    @property
    def is_loaded(self):
        return self.pipeline is not None

    def predict(self, alert: dict, config: dict) -> dict:
        """
        Run the full pipeline for one alert dict:
          clean -> features -> ML probability -> novelty score ->
          safety-gated recommendation -> simulated recommended action.

        Returns a dict ready to store in the DB / return via the API.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        X = prepare_single_alert(alert)
        assert_no_leakage(X)

        proba_row = self.pipeline.predict_proba(X)[0]
        classes = list(self.pipeline.classes_)
        ml_proba = {cls: float(p) for cls, p in zip(classes, proba_row)}

        preproc = self.pipeline.named_steps["preprocessor"]
        x_transformed = preproc.transform(X)[0]
        novelty_info = self.novelty_detector.score_one(x_transformed, alert)

        decision = decide(ml_proba, novelty_info, alert, config)

        action = recommended_action(
            decision["recommendation"],
            alert.get("alert_type", ""),
            alert.get("severity", ""),
        )

        is_high_impact_action = is_high_impact(action)
        # An alert requires manual confirmation (no quick Accept) whenever
        # the recommended action is high-impact OR the high-risk safety
        # gate fired — even if the resulting action itself looks routine.
        requires_manual_confirmation = bool(
            is_high_impact_action or decision["high_risk_gate_triggered"]
        )

        return {
            "recommendation": decision["recommendation"],
            "simple_label": decision["simple_label"],
            "confidence": decision["confidence"],
            "ml_prediction": decision["ml_prediction"],
            "ml_confidence": decision["ml_confidence"],
            "ml_probabilities": {k: round(v, 4) for k, v in ml_proba.items()},
            "novelty_score": novelty_info["novelty_score"],
            "novelty_components": {
                "iforest_component": novelty_info["iforest_component"],
                "rarity_component": novelty_info["rarity_component"],
                "combination_seen_in_training": novelty_info["combination_seen_in_training"],
            },
            "novelty_gate_triggered": decision["novelty_gate_triggered"],
            "insufficient_evidence": decision["insufficient_evidence"],
            "high_risk_gate_triggered": decision["high_risk_gate_triggered"],
            "high_risk_reasons": decision["high_risk_reasons"],
            "strong_fp_rule_triggered": decision["strong_fp_rule_triggered"],
            "strong_fp_reasons": decision["strong_fp_reasons"],
            "rule_triggered": decision["rule_triggered"],
            "risk_level": decision["risk_level"],
            "low_confidence": decision["low_confidence"],
            "recommended_action": action,
            "is_high_impact_action": is_high_impact_action,
            "requires_manual_confirmation": requires_manual_confirmation,
            "model_version": self.model_version,
        }


# Module-level singleton used by the Flask app.
engine = RecommendationEngine()
