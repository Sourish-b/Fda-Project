import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _paths():
    train_test_dir = os.path.dirname(__file__)
    model_dir = os.path.abspath(os.path.join(train_test_dir, ".."))
    saved_dir = os.path.join(model_dir, "saved")
    results_dir = os.path.join(train_test_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    return {
        "model_path": os.path.join(saved_dir, "rf_model.pkl"),
        "x_test_path": os.path.join(results_dir, "X_test.csv"),
        "y_test_path": os.path.join(results_dir, "y_test.csv"),
        "report_path": os.path.join(results_dir, "evaluation_report.txt"),
        "actual_pred_plot_path": os.path.join(results_dir, "actual_vs_predicted.png"),
        "residuals_plot_path": os.path.join(results_dir, "residuals_plot.png"),
    }


def _load_model_and_test_data():
    p = _paths()

    if not os.path.exists(p["model_path"]):
        raise FileNotFoundError(f"Model file not found: {p['model_path']}")
    if not os.path.exists(p["x_test_path"]):
        raise FileNotFoundError(f"X_test file not found: {p['x_test_path']}")
    if not os.path.exists(p["y_test_path"]):
        raise FileNotFoundError(f"y_test file not found: {p['y_test_path']}")

    model = joblib.load(p["model_path"])
    X_test = pd.read_csv(p["x_test_path"])
    y_test_df = pd.read_csv(p["y_test_path"])

    if y_test_df.shape[1] == 0:
        raise ValueError("y_test.csv has no columns.")

    y_test = pd.to_numeric(y_test_df.iloc[:, 0], errors="coerce")
    valid_mask = y_test.notna()
    X_test = X_test.loc[valid_mask].reset_index(drop=True)
    y_test = y_test.loc[valid_mask].reset_index(drop=True)

    return model, X_test, y_test


def _get_predictions():
    model, X_test, y_test = _load_model_and_test_data()
    y_pred = model.predict(X_test)
    return y_test.to_numpy(), np.asarray(y_pred)


def evaluate_model():
    """Evaluate saved model on test split and write metrics report."""
    p = _paths()
    y_test, y_pred = _get_predictions()

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    non_zero_mask = y_test != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100)
    else:
        mape = float("nan")

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")

    report_lines = [
        "Evaluation Report",
        "=================",
        f"RMSE: {rmse:.6f}",
        f"MAE: {mae:.6f}",
        f"R2: {r2:.6f}",
        f"MAPE: {mape:.6f}",
    ]

    with open(p["report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


def plot_actual_vs_predicted():
    """Plot actual vs predicted values with reference diagonal and R2 annotation."""
    p = _paths()
    y_test, y_pred = _get_predictions()

    r2 = float(r2_score(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.7)

    diag_min = min(float(np.min(y_test)), float(np.min(y_pred)))
    diag_max = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([diag_min, diag_max], [diag_min, diag_max], "r--", linewidth=1.5)

    plt.text(0.05, 0.95, f"R2 = {r2:.4f}", transform=plt.gca().transAxes, va="top", ha="left")
    plt.xlabel("Actual (y_test)")
    plt.ylabel("Predicted (y_pred)")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(p["actual_pred_plot_path"], dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals():
    """Plot residual diagnostics with residual scatter and histogram."""
    p = _paths()
    y_test, y_pred = _get_predictions()

    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, color="steelblue", alpha=0.7)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted (y_pred)")
    axes[0].set_ylabel("Residuals (y_test - y_pred)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=20, color="slategray", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(p["residuals_plot_path"], dpi=300, bbox_inches="tight")
    plt.close()
