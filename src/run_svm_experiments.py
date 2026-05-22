from __future__ import annotations

import csv
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import classification_report_binary
from src.preprocessing import load_preprocessed_split
from src.svm_scratch import LinearSVMScratch

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "svm"
FIGURE_DIR = PROJECT_ROOT / "figures"


def oversample_training_only(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_train, y_train

    if len(pos_idx) < len(neg_idx):
        extra = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
        all_idx = np.concatenate([neg_idx, pos_idx, extra])
    else:
        all_idx = np.arange(len(y_train))
    rng.shuffle(all_idx)
    return X_train[all_idx], y_train[all_idx]


def run_experiment(name, X_train, y_train, X_test, y_test, learning_rate, lambda_param, n_epochs, batch_size):
    model = LinearSVMScratch(learning_rate=learning_rate, lambda_param=lambda_param, n_epochs=n_epochs, batch_size=batch_size, random_state=42)
    t0 = time.perf_counter(); model.fit(X_train, y_train); train_t = time.perf_counter() - t0
    t1 = time.perf_counter(); y_pred = model.predict(X_test); pred_t = time.perf_counter() - t1
    report = classification_report_binary(y_test, y_pred)
    row = {
        "experiment_name": name, "learning_rate": learning_rate, "lambda_param": lambda_param,
        "n_epochs": n_epochs, "batch_size": batch_size, "final_loss": model.loss_history[-1],
        "train_time_sec": train_t, "predict_time_sec": pred_t, **report
    }
    return row, model.loss_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="full")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, metadata = load_preprocessed_split(test_size=0.2, random_state=42)
    X_train_over, y_train_over = oversample_training_only(X_train, y_train, random_state=42)

    epochs = 120 if args.mode == "demo" else 300
    experiments = [
        ("baseline_original_train", X_train, y_train, 0.001, 0.01, epochs, 128),
        ("oversampled_train", X_train_over, y_train_over, 0.001, 0.01, epochs, 128),
    ]
    rows=[]; best=None; best_loss=None
    for e in experiments:
        row, loss = run_experiment(e[0], e[1], e[2], X_test, y_test, e[3], e[4], e[5], e[6])
        row["n_train"]=len(e[2]); row["n_test"]=len(y_test); row["n_features_after_encoding"]=X_train.shape[1]
        row["estimated_memory_usage_mb"] = (e[1].nbytes + e[2].nbytes) / (1024 ** 2)
        rows.append(row)
        if best is None or row["f1"] > best["f1"]:
            best=row; best_loss=loss

    with open(OUTPUT_DIR / "svm_experiments.csv", "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(OUTPUT_DIR / "svm_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(best.keys())); w.writeheader(); w.writerow(best)

    # lightweight CV summary (from scratch fold split)
    from src.cross_validation import make_stratified_k_folds
    n_splits = 3 if args.mode == "demo" else 5
    folds = make_stratified_k_folds(y_train, n_splits=n_splits, random_state=42)
    all_idx = np.arange(len(y_train))
    cv_rows = []
    for i, val_idx in enumerate(folds, start=1):
        tr_idx = np.setdiff1d(all_idx, val_idx)
        model = LinearSVMScratch(learning_rate=0.001, lambda_param=0.01, n_epochs=epochs, batch_size=128, random_state=42)
        model.fit(X_train[tr_idx], y_train[tr_idx])
        rep = classification_report_binary(y_train[val_idx], model.predict(X_train[val_idx]))
        cv_rows.append({"fold": i, "accuracy": rep["accuracy"], "precision": rep["precision"], "recall": rep["recall"], "f1": rep["f1"]})
    cv_df = np.array([[r["accuracy"], r["precision"], r["recall"], r["f1"]] for r in cv_rows], dtype=float)
    with open(OUTPUT_DIR / "svm_cv_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n_splits", "mean_accuracy", "mean_precision", "mean_recall", "mean_f1"])
        w.writeheader()
        w.writerow({"n_splits": n_splits, "mean_accuracy": cv_df[:, 0].mean(), "mean_precision": cv_df[:, 1].mean(), "mean_recall": cv_df[:, 2].mean(), "mean_f1": cv_df[:, 3].mean()})

    plt.figure(figsize=(7,4.5)); plt.plot(best_loss); plt.title(f"SVM Loss Curve ({best['experiment_name']})")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.tight_layout(); plt.savefig(FIGURE_DIR / "svm_loss_curve.png", dpi=300); plt.close()

    src_rank = PROJECT_ROOT / "svm_package_for_comparison(1)" / "outputs" / "information_gain_ranking.csv"
    if src_rank.exists():
        (OUTPUT_DIR / "information_gain_ranking.csv").write_bytes(src_rank.read_bytes())

    print(f"SVM done. Train/Test: {X_train.shape} / {X_test.shape}")
    print(f"Oversampling applies only to training set: {len(y_train_over)} train rows, {len(y_test)} test rows")


if __name__ == "__main__":
    main()
