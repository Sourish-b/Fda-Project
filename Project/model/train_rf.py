import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from model.train_test.split import create_splits


if os.path.exists("model/saved/rf_model.pkl"):
    print("Model already trained. Delete model/saved/rf_model.pkl to retrain.")
    sys.exit(0)


def train_and_save(df):
    """Train Random Forest model, evaluate, and persist artifacts."""
    X_train, X_test, y_train, y_test = create_splits(df)

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "rf_model.pkl")
    joblib.dump(model, model_path)

    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind="bar", color="#2a9d8f")
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()

    fi_path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Model saved. Use predict.py for predictions — do not retrain.")
    return model
