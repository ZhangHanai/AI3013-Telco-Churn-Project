from pathlib import Path
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cross_validation import cross_validate_knn
from src.knn_scratch import KNearestNeighbors
from src.metrics import print_classification_report
from src.preprocessing import (
    find_dataset_path,
    load_preprocessed_split,
)


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]


def save_line_plot(cv_summary, metric, output_path):
    sorted_df = cv_summary.sort_values("k")
    plt.figure(figsize=(7, 4.5))
    plt.plot(sorted_df["k"], sorted_df[metric], marker="o")
    plt.xlabel("Number of neighbors (k)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"KNN Cross-Validation {metric.replace('_', ' ').title()}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix_plot(report, output_path):
    matrix = np.array([[report["TN"], report["FP"]], [report["FN"], report["TP"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.xticks([0, 1], ["Predicted No", "Predicted Yes"])
    plt.yticks([0, 1], ["Actual No", "Actual Yes"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run KNN from scratch on Telco churn data.")
    parser.add_argument("--fast", action="store_true", help="Run a quick test with fewer k values and fewer CV folds.")
    parser.add_argument("--weighted", action="store_true", help="Use distance-weighted voting.")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = PROJECT_ROOT / "outputs" / "knn"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    dataset_path = find_dataset_path(PROJECT_ROOT)
    X_train, X_test, y_train, y_test, metadata = load_preprocessed_split(dataset_path, test_size=0.2, random_state=42)
    feature_names = metadata["feature_names"]
    train_indices = metadata["train_indices"]
    test_indices = metadata["test_indices"]

    if args.fast:
        k_values = [3, 5, 11]
        n_splits = 3
    else:
        k_values = [1, 3, 5, 7, 9, 11]
        n_splits = 5

    print("Running KNN cross-validation...")
    cv_summary, cv_detailed = cross_validate_knn(
        X_train, y_train, k_values=k_values, feature_names=feature_names, n_splits=n_splits, random_state=42, weighted=args.weighted
    )

    cv_summary.to_csv(results_dir / "knn_cv_summary.csv", index=False)
    cv_detailed.to_csv(results_dir / "knn_cv_detailed.csv", index=False)

    best_row = cv_summary.iloc[0]
    best_k = int(best_row["k"])

    print("\nKNN cross-validation summary:")
    print(cv_summary.round(4).to_string(index=False))
    print(f"\nBest k selected by mean F1-score: {best_k}")

    X_train_scaled, X_test_scaled = X_train, X_test
    model = KNearestNeighbors(k=best_k, weighted=args.weighted)

    train_start = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    train_time_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    prediction_time_seconds = time.perf_counter() - predict_start

    test_report = print_classification_report(y_test, y_pred, model_name=f"KNN from scratch (k={best_k})")

    memory_usage_mb = (X_train_scaled.nbytes + y_train.nbytes) / (1024 ** 2)

    test_metrics = {
        "model": f"KNN from scratch (k={best_k})",
        "best_k": best_k,
        "accuracy": test_report["accuracy"],
        "precision": test_report["precision"],
        "recall": test_report["recall"],
        "f1": test_report["f1"],
        "TP": test_report["TP"],
        "TN": test_report["TN"],
        "FP": test_report["FP"],
        "FN": test_report["FN"],
        "training_time_seconds": train_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
        "estimated_memory_usage_mb": memory_usage_mb,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features_after_encoding": X_train_scaled.shape[1],
    }

    pd.DataFrame([test_metrics]).to_csv(results_dir / "knn_test_metrics.csv", index=False)

    predictions_df = pd.DataFrame({
        "test_index": test_indices,
        "true_churn": y_test,
        "predicted_churn": y_pred,
        "positive_probability": y_proba,
    })
    predictions_df.to_csv(results_dir / "knn_test_predictions.csv", index=False)

    save_line_plot(cv_summary, "mean_f1", figures_dir / "knn_cv_mean_f1_by_k.png")
    save_line_plot(cv_summary, "mean_accuracy", figures_dir / "knn_cv_mean_accuracy_by_k.png")
    save_confusion_matrix_plot(test_report, figures_dir / "knn_confusion_matrix.png")

    comparison_template = pd.DataFrame([
        {"model": "Logistic Regression from scratch", "accuracy": "", "precision": "", "recall": "", "f1": "", "training_time_seconds": "", "prediction_time_seconds": "", "estimated_memory_usage_mb": "", "owner": "3号"},
        {"model": f"KNN from scratch (k={best_k})", "accuracy": test_report["accuracy"], "precision": test_report["precision"], "recall": test_report["recall"], "f1": test_report["f1"], "training_time_seconds": train_time_seconds, "prediction_time_seconds": prediction_time_seconds, "estimated_memory_usage_mb": memory_usage_mb, "owner": "4号"},
        {"model": "Linear SVM from scratch", "accuracy": "", "precision": "", "recall": "", "f1": "", "training_time_seconds": "", "prediction_time_seconds": "", "estimated_memory_usage_mb": "", "owner": "1号+2号"},
    ])
    comparison_template.to_csv(results_dir / "model_comparison_template.csv", index=False)

    summary_text = f"""KNN experiment summary

Dataset path:
{dataset_path}

Encoded feature shape:
X_train = {X_train.shape}, X_test = {X_test.shape}

Train-test split:
Training samples = {len(y_train)}
Test samples = {len(y_test)}
Features after encoding = {X_train_scaled.shape[1]}

Cross-validation:
k values tested = {k_values}
Number of folds = {n_splits}
Best k by mean F1-score = {best_k}
Best CV mean F1-score = {best_row['mean_f1']:.4f}

Final test result:
Accuracy = {test_report['accuracy']:.4f}
Precision = {test_report['precision']:.4f}
Recall = {test_report['recall']:.4f}
F1-score = {test_report['f1']:.4f}
TP = {test_report['TP']}
TN = {test_report['TN']}
FP = {test_report['FP']}
FN = {test_report['FN']}

Runtime:
Training time = {train_time_seconds:.6f} seconds
Prediction time = {prediction_time_seconds:.6f} seconds
Estimated memory usage = {memory_usage_mb:.4f} MB
"""
    (results_dir / "knn_experiment_summary.txt").write_text(summary_text, encoding="utf-8")

    print("\nSaved files under outputs/knn and figures/.")


if __name__ == "__main__":
    main()
