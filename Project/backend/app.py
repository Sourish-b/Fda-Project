import os

import joblib
import pandas as pd
from flask import Flask
from flask_cors import CORS


cluster_df = None
profiles_df = None
seasonal_df = None
rf_model = None


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _warn_missing(file_path):
    print(f"Warning: missing file: {file_path}")


def _safe_read_csv(file_path):
    if not os.path.exists(file_path):
        _warn_missing(file_path)
        return None
    return pd.read_csv(file_path)


def _safe_load_model(file_path):
    if not os.path.exists(file_path):
        _warn_missing(file_path)
        return None
    return joblib.load(file_path)


def _load_startup_data():
    global cluster_df, profiles_df, seasonal_df, rf_model

    root = _project_root()
    cluster_path = os.path.join(root, "model", "saved", "cluster_labels.csv")
    profiles_path = os.path.join(root, "model", "saved", "state_profiles.csv")
    seasonal_path = os.path.join(root, "data", "india_renewable.csv")
    model_path = os.path.join(root, "model", "saved", "rf_model.pkl")

    cluster_df = _safe_read_csv(cluster_path)
    profiles_df = _safe_read_csv(profiles_path)
    seasonal_df = _safe_read_csv(seasonal_path)
    rf_model = _safe_load_model(model_path)

    if all(x is not None for x in [cluster_df, profiles_df, seasonal_df, rf_model]):
        print("All data loaded successfully.")


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config["cluster_df"] = cluster_df
    app.config["profiles_df"] = profiles_df
    app.config["seasonal_df"] = seasonal_df
    app.config["rf_model"] = rf_model

    try:
        from routes import api_bp

        app.register_blueprint(api_bp, url_prefix="/api")
    except Exception as exc:
        print(f"Warning: could not register backend/routes.py blueprint: {exc}")

    return app


_load_startup_data()
app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
