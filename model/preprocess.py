"""
preprocess.py
-----------------------------------------
Loading, cleaning, feature definition, and leakage-safe preparation
of the security alerts dataset.

IMPORTANT - DATA LEAKAGE PREVENTION
------------------------------------
The following columns exist in the dataset but must NEVER be used as
INPUT FEATURES for predicting an unseen/new alert, because they only
become known *after* an analyst has investigated the alert (i.e. they
are outcome / label information, not evidence available at alert time):

    - analyst_disposition
    - confirmed_incident
    - ground_truth            (this IS the training target, not a feature)
    - investigation_reason
    - incident_type
    - novelty_test_flag       (reserved ONLY for evaluating novelty detection)

`FEATURE_COLUMNS` below intentionally excludes all of these. The
`assert_no_leakage()` helper is called before every model fit/predict
to make this a hard guarantee rather than a convention that can silently
be violated by a future edit.
-----------------------------------------
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Columns that must never be used as model input features.
# ---------------------------------------------------------------------
LEAKAGE_COLUMNS = [
    "analyst_disposition",
    "confirmed_incident",
    "ground_truth",
    "investigation_reason",
    "incident_type",
    "novelty_test_flag",
]

# Features available at alert-creation time (before any human investigation).
NUMERIC_FEATURES = [
    "endpoint_risk_score",
    "user_risk_score",
    "previous_alert_count",
    "previous_incident_count",
    "similar_alert_count",
    "historical_false_positive_rate",
    "historical_incident_rate",
]

CATEGORICAL_FEATURES = [
    "alert_source",
    "alert_type",
    "severity",
    "endpoint_type",
    "operating_system",
    "user_role",
    "department",
    "country",
    "authentication_result",
    "process_name",
    "parent_process",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "ground_truth"  # binary: "False Positive" / "True Incident"
TARGET_CLASSES = ["False Positive", "True Incident"]

# Fields used only to build a rarity/novelty signature (NOT fed to the
# classifier as one-hot categorical features directly, since the
# combination as a whole is what matters for novelty rather than each
# field independently).
#
# NOTE: process_name is intentionally EXCLUDED from the rarity key even
# though it IS a classifier feature. process_name is missing/"Unknown"
# for many realistic alerts (e.g. Firewall, IAM alerts don't carry a
# process), and "Unknown" itself is rare in the training data (~1.5% of
# rows) purely because most training rows happen to have a real process
# name. Including it here would make any alert lacking optional process
# context look artificially "novel" regardless of how routine the rest
# of the alert is -- a spurious signal, not a genuine one.
RARITY_KEY_FIELDS = ["alert_source", "alert_type", "country"]


def assert_no_leakage(df_or_columns) -> None:
    """
    Hard guard: raises ValueError if any leakage column is present among
    the given feature columns. Call this immediately before fit/predict.
    """
    cols = list(df_or_columns.columns) if hasattr(df_or_columns, "columns") else list(df_or_columns)
    leaked = [c for c in cols if c in LEAKAGE_COLUMNS]
    if leaked:
        raise ValueError(
            f"Data leakage guard triggered: {leaked} must never be used as "
            f"model input features. This indicates a bug in feature preparation."
        )


def load_raw_dataset(path="dataset/security_alerts_2000.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values without touching ground-truth/leakage columns."""
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).replace("", "Unknown").replace("nan", "Unknown")
        else:
            df[col] = "Unknown"

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            fill_val = median_val if not np.isnan(median_val) else 0
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = 0.0

    if "timestamp" in df.columns:
        df["_parsed_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["_parsed_timestamp"] = pd.NaT

    return df


def load_and_clean(path="dataset/security_alerts_2000.csv") -> pd.DataFrame:
    return clean_dataframe(load_raw_dataset(path))


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """
    Sort by timestamp and split into train / validation / test sets that
    simulate: past alerts (train) -> near-future alerts (validation) ->
    future alerts (test). Returns (train_df, val_df, test_df).
    Rows with unparseable timestamps are pushed to the end (treated as
    "most recent / unknown time") rather than dropped.
    """
    df = df.copy()
    df["_parsed_timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["_sort_key"] = df["_parsed_timestamp"].fillna(pd.Timestamp.max)
    df = df.sort_values("_sort_key").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def get_features_and_target(df: pd.DataFrame):
    """
    Return (X, y) using ONLY leakage-safe feature columns and the binary
    ground_truth target. Guarded by assert_no_leakage.
    """
    X = df[FEATURE_COLUMNS].copy()
    assert_no_leakage(X)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset missing target column '{TARGET_COLUMN}'")
    y = df[TARGET_COLUMN].astype(str)
    return X, y


def prepare_single_alert(alert: dict) -> pd.DataFrame:
    """
    Convert a single alert dict (manual form / CSV row / API payload) into
    a one-row DataFrame matching FEATURE_COLUMNS. Missing fields default
    safely rather than raising.
    """
    row = {}
    for col in CATEGORICAL_FEATURES:
        val = alert.get(col, "Unknown")
        if val is None or str(val).strip() in ("", "nan"):
            val = "Unknown"
        row[col] = str(val)

    for col in NUMERIC_FEATURES:
        val = alert.get(col, 0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        row[col] = val

    df = pd.DataFrame([row])
    assert_no_leakage(df)
    return df


def rarity_key(row: dict) -> tuple:
    """Signature used for combination-rarity scoring (novelty detection)."""
    return tuple(str(row.get(f, "Unknown")) for f in RARITY_KEY_FIELDS)
