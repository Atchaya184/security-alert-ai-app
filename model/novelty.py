"""
novelty.py
-----------------------------------------
Novelty / anomaly detection for the "NOVEL / UNKNOWN - HUMAN
INVESTIGATION REQUIRED" safety layer.

Combines two independent signals into a single novelty_score in [0, 1]:

1. Isolation Forest anomaly score over the same preprocessed feature
   space used by the classifier (numeric + one-hot categorical).
   Captures alerts that are statistically unusual overall.

2. Combination rarity: how often the (alert_source, alert_type,
   country, process_name) combination was seen in the training data.
   Captures alerts whose *specific pattern* is new even if individual
   fields look ordinary.

`novelty_test_flag` from the dataset is NEVER used as a model input.
It is only used later (in train_model.py) to evaluate how well this
detector recovers alerts that were deliberately held out as "novel"
test cases.
-----------------------------------------
"""

import numpy as np
from collections import Counter
from sklearn.ensemble import IsolationForest

from preprocess import rarity_key


class NoveltyDetector:
    def __init__(self, contamination=0.08, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.iforest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self.rarity_counts = Counter()
        self.total_train_rows = 0
        self._score_min = None
        self._score_max = None

    def fit(self, X_transformed: np.ndarray, raw_rows: list):
        """
        X_transformed: dense numeric array (output of the classifier's
                       fitted preprocessing ColumnTransformer) for the
                       TRAINING split only.
        raw_rows: list of dicts (original row values) aligned with
                  X_transformed, used to build the rarity table.
        """
        self.iforest.fit(X_transformed)

        raw_scores = -self.iforest.score_samples(X_transformed)  # higher = more anomalous
        self._score_min = float(np.min(raw_scores))
        self._score_max = float(np.max(raw_scores))

        self.rarity_counts = Counter(rarity_key(r) for r in raw_rows)
        self.total_train_rows = len(raw_rows)
        return self

    def _normalize_iforest_score(self, raw_score: float) -> float:
        if self._score_max is None or self._score_max == self._score_min:
            return 0.5
        val = (raw_score - self._score_min) / (self._score_max - self._score_min)
        return float(np.clip(val, 0.0, 1.0))

    def score_one(self, x_transformed_row: np.ndarray, raw_row: dict) -> dict:
        """
        Returns a dict with the combined novelty_score plus the two
        underlying components, for transparency in the evidence panel.
        """
        raw_iforest = float(-self.iforest.score_samples(x_transformed_row.reshape(1, -1))[0])
        iforest_component = self._normalize_iforest_score(raw_iforest)

        key = rarity_key(raw_row)
        count = self.rarity_counts.get(key, 0)
        # Rarity component: 1.0 if never seen in training, decaying toward 0
        # as the combination becomes common. +1 smoothing avoids div-by-zero.
        rarity_component = 1.0 / (1.0 + count)

        novelty_score = float(0.6 * iforest_component + 0.4 * rarity_component)
        novelty_score = float(np.clip(novelty_score, 0.0, 1.0))

        return {
            "novelty_score": round(novelty_score, 4),
            "iforest_component": round(iforest_component, 4),
            "rarity_component": round(rarity_component, 4),
            "combination_seen_in_training": count,
        }

    def score_batch(self, X_transformed: np.ndarray, raw_rows: list) -> list:
        return [self.score_one(X_transformed[i], raw_rows[i]) for i in range(len(raw_rows))]


def insufficient_evidence(raw_row: dict, similar_alert_count_threshold: int = 1) -> bool:
    """
    True when there simply isn't enough historical evidence about this
    endpoint/alert pattern to trust a false-positive suppression
    recommendation, regardless of what the classifier says.
    """
    try:
        previous_alert_count = float(raw_row.get("previous_alert_count", 0))
    except (TypeError, ValueError):
        previous_alert_count = 0
    try:
        similar_alert_count = float(raw_row.get("similar_alert_count", 0))
    except (TypeError, ValueError):
        similar_alert_count = 0

    return previous_alert_count == 0 and similar_alert_count <= similar_alert_count_threshold
