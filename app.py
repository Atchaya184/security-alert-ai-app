"""
app.py
-----------------------------------------
Main entry point for the False Positive Reduction Assistant.

Run with:
    python app.py

On first run, if model/model.pkl does not exist, the app automatically
trains it from dataset/security_alerts_2000.csv (chronological
evaluation, novelty detection, safety-gated recommendations — see
model/train_model.py). The database (database.db) is auto-created;
no external DB server is required.
-----------------------------------------
"""

import os
import sys
from flask import Flask

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

from utils import db
from utils.recommend import engine
from model.config import load_config
from routes.alert_routes import register_alert_routes
from routes.upload_routes import register_upload_routes
from routes.audit_routes import register_audit_routes
from routes.change_review import register_change_review_routes
from routes.model_routes import register_model_routes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "security_alerts_2000.csv")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def get_config():
    """Always re-read config.json so Change Review updates apply immediately."""
    return load_config()


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            engine.load()
            print(f"Loaded existing model v{engine.model_version} from {MODEL_PATH}")
            return
        except Exception as e:
            print(f"Failed to load existing model ({e}). Retraining...")

    if os.path.exists(DATASET_PATH):
        from model.train_model import train_and_save
        print("No valid model found. Training a new model from the dataset...")
        train_and_save(DATASET_PATH)
        engine.load()
    else:
        print(f"WARNING: Dataset not found at {DATASET_PATH}. "
              f"Predictions will be unavailable until a model is trained.")


def ensure_initial_versions_recorded():
    """Record the initial model/rule versions in the DB if not already present,
    so the Model and Change Review pages have a starting point to show/rollback to."""
    cfg = get_config()
    if db.get_active_version("model") is None and engine.is_loaded:
        db.record_version(
            version_type="model", version=engine.model_version,
            description="Initial trained model", reason="Initial training",
            reviewer="System", config_snapshot=cfg, status="active",
        )
    if db.get_active_version("rule") is None:
        db.record_version(
            version_type="rule", version=cfg.get("rule_version", "1.0"),
            description="Initial rule set", reason="Initial configuration",
            reviewer="System", config_snapshot=cfg, status="active",
        )


# Initialize database (auto-creates database.db; no external DB server needed)
db.init_db()

# Load / train model at startup
load_or_train_model()

ensure_initial_versions_recorded()

# Register all route blueprints (functions attach routes directly to `app`)
register_alert_routes(app, engine, get_config)
register_upload_routes(app, engine, get_config)
register_audit_routes(app, engine, get_config)
register_change_review_routes(app, engine, get_config)
register_model_routes(app, engine, get_config)


@app.errorhandler(404)
def not_found(e):
    return {"error": "Not found"}, 404


@app.errorhandler(500)
def server_error(e):
    return {"error": "Internal server error"}, 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
