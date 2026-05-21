from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics_scratch import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix_binary,
    f1_score,
    precision_score,
    recall_score,
    true_negative_rate,
)
from svm_scratch import LinearSVMScratch


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed_data"
FIGURE_DIR = BASE_DIR / "figures"


def load_processed_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"]).to_numpy(dtype=float)
    y = df["Churn"].to_numpy(dtype=int)
    return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    cm = confusion_matrix_binary(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tnr": true_negative_rate(y_true, y_pred),
        "tp": cm["tp"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
    }


def run_single_experiment(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float,
    lambda_param: float,
    n_epochs: int,
    batch_size: int,
) -> tuple[dict[str, float | int | str], list[float]]:
    model = LinearSVMScratch(
        learning_rate=learning_rate,
        lambda_param=lambda_param,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=42,
    )

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start

    pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - pred_start

    metrics = evaluate_predictions(y_test, y_pred)
    row: dict[str, float | int | str] = {
        "experiment_name": name,
        "learning_rate": learning_rate,
        "lambda_param": lambda_param,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "final_loss": model.loss_history[-1],
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
    }
    row.update(metrics)
    return row, model.loss_history


def save_loss_curve(loss_history: list[float], filename: str, title: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="#E45756", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Hinge Loss + L2 Regularization")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_path = PROCESSED_DIR / "train_processed.csv"
    test_path = PROCESSED_DIR / "test_processed.csv"
    oversampled_path = PROCESSED_DIR / "train_processed_oversampled.csv"

    if not train_path.exists() or not test_path.exists() or not oversampled_path.exists():
        raise FileNotFoundError(
            "Processed data files are missing. Please run preprocess_telco.py first."
        )

    X_train, y_train = load_processed_csv(train_path)
    X_test, y_test = load_processed_csv(test_path)
    X_train_over, y_train_over = load_processed_csv(oversampled_path)

    experiments = [
        {
            "name": "baseline_original_train",
            "X_train": X_train,
            "y_train": y_train,
            "learning_rate": 0.001,
            "lambda_param": 0.01,
            "n_epochs": 300,
            "batch_size": 128,
        },
        {
            "name": "oversampled_train",
            "X_train": X_train_over,
            "y_train": y_train_over,
            "learning_rate": 0.001,
            "lambda_param": 0.01,
            "n_epochs": 300,
            "batch_size": 128,
        },
        {
            "name": "oversampled_stronger_reg",
            "X_train": X_train_over,
            "y_train": y_train_over,
            "learning_rate": 0.001,
            "lambda_param": 0.05,
            "n_epochs": 350,
            "batch_size": 128,
        },
    ]

    rows: list[dict[str, float | int | str]] = []
    best_row: dict[str, float | int | str] | None = None
    best_loss_history: list[float] = []

    for exp in experiments:
        row, loss_history = run_single_experiment(
            name=exp["name"],
            X_train=exp["X_train"],
            y_train=exp["y_train"],
            X_test=X_test,
            y_test=y_test,
            learning_rate=exp["learning_rate"],
            lambda_param=exp["lambda_param"],
            n_epochs=exp["n_epochs"],
            batch_size=exp["batch_size"],
        )
        rows.append(row)
        if best_row is None or (row["f1"], row["balanced_accuracy"]) > (
            best_row["f1"],
            best_row["balanced_accuracy"],
        ):
            best_row = row
            best_loss_history = loss_history

    assert best_row is not None

    save_csv(rows, OUTPUT_DIR / "svm_experiments.csv")
    save_csv([best_row], OUTPUT_DIR / "svm_metrics.csv")
    save_loss_curve(
        best_loss_history,
        filename="svm_loss_curve.png",
        title=f"SVM Loss Curve ({best_row['experiment_name']})",
    )

    print("Best SVM experiment:")
    for key, value in best_row.items():
        print(f"{key}: {value}")
    print(f"Saved experiment table to: {OUTPUT_DIR / 'svm_experiments.csv'}")
    print(f"Saved best metrics to: {OUTPUT_DIR / 'svm_metrics.csv'}")
    print(f"Saved loss curve to: {FIGURE_DIR / 'svm_loss_curve.png'}")


if __name__ == "__main__":
    main()
