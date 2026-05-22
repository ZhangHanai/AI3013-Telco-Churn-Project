from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[run_all] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_one_row(csv_path: Path, selector: str | None = None) -> pd.Series:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if selector:
        selected = df[df["model"].astype(str).str.contains(selector, regex=False)]
        if not selected.empty:
            return selected.iloc[0]
    return df.iloc[0]


def build_results_summary(project_root: Path) -> Path:
    outputs = project_root / "outputs"

    lr_row = load_one_row(outputs / "logistic_regression" / "lr_metrics.csv", selector="Novel weighted LR")
    knn_row = load_one_row(outputs / "knn" / "knn_test_metrics.csv")
    svm_row = load_one_row(outputs / "svm" / "svm_metrics.csv")

    rows = [
        {
            "model": "Logistic Regression",
            "accuracy": lr_row.get("accuracy"),
            "precision": lr_row.get("precision"),
            "recall": lr_row.get("recall"),
            "f1": lr_row.get("f1"),
            "training_time_sec": lr_row.get("training_time_sec"),
            "prediction_time_sec": lr_row.get("prediction_time_sec"),
            "memory_usage_mb": lr_row.get("peak_memory_mb"),
            "TP": lr_row.get("tp"),
            "TN": lr_row.get("tn"),
            "FP": lr_row.get("fp"),
            "FN": lr_row.get("fn"),
            "notes": f"threshold={lr_row.get('threshold'):.2f}",
        },
        {
            "model": "KNN",
            "accuracy": knn_row.get("accuracy"),
            "precision": knn_row.get("precision"),
            "recall": knn_row.get("recall"),
            "f1": knn_row.get("f1"),
            "training_time_sec": knn_row.get("training_time_seconds"),
            "prediction_time_sec": knn_row.get("prediction_time_seconds"),
            "memory_usage_mb": knn_row.get("estimated_memory_usage_mb"),
            "TP": knn_row.get("TP"),
            "TN": knn_row.get("TN"),
            "FP": knn_row.get("FP"),
            "FN": knn_row.get("FN"),
            "notes": f"best_k={int(knn_row.get('best_k'))}",
        },
        {
            "model": "Linear SVM",
            "accuracy": svm_row.get("accuracy"),
            "precision": svm_row.get("precision"),
            "recall": svm_row.get("recall"),
            "f1": svm_row.get("f1"),
            "training_time_sec": svm_row.get("train_time_sec"),
            "prediction_time_sec": svm_row.get("predict_time_sec"),
            "memory_usage_mb": svm_row.get("estimated_memory_usage_mb"),
            "TP": svm_row.get("TP"),
            "TN": svm_row.get("TN"),
            "FP": svm_row.get("FP"),
            "FN": svm_row.get("FN"),
            "notes": str(svm_row.get("experiment_name")),
        },
    ]

    summary_df = pd.DataFrame(rows)
    output_path = outputs / "results_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all churn models and build unified summary.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Fast demo mode.")
    mode.add_argument("--full", action="store_true", help="Full experiment mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    if args.demo:
        print("[run_all] Mode: demo")
        run([sys.executable, "-m", "src.run_logistic_regression", "--mode", "demo"], project_root)
        run([sys.executable, "-m", "src.run_knn_experiment", "--fast"], project_root)
        run([sys.executable, "-m", "src.run_svm_experiments", "--mode", "demo"], project_root)
    else:
        print("[run_all] Mode: full")
        run([sys.executable, "-m", "src.run_logistic_regression", "--mode", "full"], project_root)
        run([sys.executable, "-m", "src.run_knn_experiment"], project_root)
        run([sys.executable, "-m", "src.run_svm_experiments", "--mode", "full"], project_root)

    summary_path = build_results_summary(project_root)
    print(f"[run_all] Unified summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
