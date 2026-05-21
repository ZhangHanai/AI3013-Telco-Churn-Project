import numpy as np
import pandas as pd

from src.knn_scratch import KNearestNeighbors
from src.metrics import classification_report_binary
from src.preprocessing import standardize_selected_features


def make_stratified_k_folds(y, n_splits=5, random_state=42):
    y = np.asarray(y).astype(int)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        for fold_id, index in enumerate(label_indices):
            folds[fold_id % n_splits].append(index)

    fold_arrays = []
    for fold in folds:
        fold_array = np.asarray(fold, dtype=int)
        rng.shuffle(fold_array)
        fold_arrays.append(fold_array)
    return fold_arrays


def cross_validate_knn(X, y, k_values, feature_names, n_splits=5, random_state=42, weighted=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    folds = make_stratified_k_folds(y, n_splits=n_splits, random_state=random_state)

    detailed_rows = []
    all_indices = np.arange(len(y))

    for k in k_values:
        for fold_id, val_indices in enumerate(folds, start=1):
            train_indices = np.setdiff1d(all_indices, val_indices)
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]

            X_train_scaled, X_val_scaled, _, _, _ = standardize_selected_features(
                X_train_fold, X_val_fold, feature_names
            )

            model = KNearestNeighbors(k=k, weighted=weighted)
            model.fit(X_train_scaled, y_train_fold)
            y_pred = model.predict(X_val_scaled)
            report = classification_report_binary(y_val_fold, y_pred)

            detailed_rows.append({
                "k": k,
                "fold": fold_id,
                "accuracy": report["accuracy"],
                "precision": report["precision"],
                "recall": report["recall"],
                "f1": report["f1"],
                "TP": report["TP"],
                "TN": report["TN"],
                "FP": report["FP"],
                "FN": report["FN"],
            })

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = (
        detailed_df
        .groupby("k", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
        )
        .sort_values(["mean_f1", "mean_accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    return summary_df, detailed_df
