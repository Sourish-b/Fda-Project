import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate


def _results_dir():
    return os.path.join(os.path.dirname(__file__), "results")


def _load_train_data():
    results_dir = _results_dir()
    x_train_path = os.path.join(results_dir, "X_train.csv")
    y_train_path = os.path.join(results_dir, "y_train.csv")

    if not os.path.exists(x_train_path):
        raise FileNotFoundError(f"X_train.csv not found at: {x_train_path}")
    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"y_train.csv not found at: {y_train_path}")

    X_train = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)

    if y_train_df.shape[1] == 0:
        raise ValueError("y_train.csv has no columns.")

    y_train = pd.to_numeric(y_train_df.iloc[:, 0], errors="coerce")
    valid_mask = y_train.notna()

    X_train = X_train.loc[valid_mask].reset_index(drop=True)
    y_train = y_train.loc[valid_mask].reset_index(drop=True)

    return X_train, y_train


def run_cross_validation():
    """Run 5-fold cross-validation with a fresh RandomForest model and save report."""
    X_train, y_train = _load_train_data()

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        return_train_score=False,
    )

    r2_scores = cv_results["test_r2"]
    rmse_scores = -cv_results["test_neg_root_mean_squared_error"]
    mae_scores = -cv_results["test_neg_mean_absolute_error"]

    lines = []
    header = f"{'Fold':<8}{'R2':>12}{'RMSE':>12}{'MAE':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for i in range(5):
        lines.append(f"{f'Fold {i + 1}':<8}{r2_scores[i]:>12.4f}{rmse_scores[i]:>12.4f}{mae_scores[i]:>12.4f}")

    lines.append("-" * len(header))
    lines.append(
        f"{'Mean':<8}{np.mean(r2_scores):>12.4f}{np.mean(rmse_scores):>12.4f}{np.mean(mae_scores):>12.4f}"
    )
    lines.append(
        f"{'Std':<8}{np.std(r2_scores):>12.4f}{np.std(rmse_scores):>12.4f}{np.std(mae_scores):>12.4f}"
    )

    report_text = "\n".join(lines)
    print(report_text)

    results_dir = _results_dir()
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "cv_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    return cv_results


def plot_cv_scores(cv_results):
    """Plot fold-wise CV scores for R2, RMSE, and MAE and save chart."""
    r2_scores = np.asarray(cv_results["test_r2"])
    rmse_scores = -np.asarray(cv_results["test_neg_root_mean_squared_error"])
    mae_scores = -np.asarray(cv_results["test_neg_mean_absolute_error"])

    folds = np.arange(1, 6)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(folds, r2_scores, color="teal")
    axes[0].axhline(np.mean(r2_scores), color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"R2\nmean={np.mean(r2_scores):.4f}, std={np.std(r2_scores):.4f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Score")

    axes[1].bar(folds, rmse_scores, color="coral")
    axes[1].axhline(np.mean(rmse_scores), color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"RMSE\nmean={np.mean(rmse_scores):.4f}, std={np.std(rmse_scores):.4f}")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Score")

    axes[2].bar(folds, mae_scores, color="#FFBF00")
    axes[2].axhline(np.mean(mae_scores), color="black", linestyle="--", linewidth=1)
    axes[2].set_title(f"MAE\nmean={np.mean(mae_scores):.4f}, std={np.std(mae_scores):.4f}")
    axes[2].set_xlabel("Fold")
    axes[2].set_ylabel("Score")

    fig.suptitle("5-Fold Cross Validation Scores")
    plt.tight_layout()

    results_dir = _results_dir()
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "cv_scores.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
