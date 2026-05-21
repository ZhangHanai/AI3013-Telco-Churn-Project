from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "Churn"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC_SCALE_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def find_dataset_path(project_root=None):
    project_root = Path.cwd() if project_root is None else Path(project_root)
    candidates = [
        project_root / "data" / DATASET_FILENAME,
        project_root / DATASET_FILENAME,
        project_root.parent / "data" / DATASET_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Dataset not found under data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_telco_dataframe(csv_path=None):
    if csv_path is None:
        csv_path = find_dataset_path()
    return pd.read_csv(csv_path)


def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(random_state)

    train_indices, test_indices = [], []
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        n_test = int(round(len(label_indices) * test_size))
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices], train_indices, test_indices


def fit_standardizer(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    return (X - mean) / std


def standardize_selected_features(X_train, X_test, feature_names, columns_to_scale=None):
    columns_to_scale = NUMERIC_SCALE_COLUMNS if columns_to_scale is None else columns_to_scale
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    scale_indices = [name_to_idx[name] for name in columns_to_scale if name in name_to_idx]

    X_train_scaled = np.asarray(X_train, dtype=float).copy()
    X_test_scaled = np.asarray(X_test, dtype=float).copy()

    if scale_indices:
        mean, std = fit_standardizer(X_train_scaled[:, scale_indices])
        X_train_scaled[:, scale_indices] = apply_standardizer(X_train_scaled[:, scale_indices], mean, std)
        X_test_scaled[:, scale_indices] = apply_standardizer(X_test_scaled[:, scale_indices], mean, std)
    else:
        mean = np.array([])
        std = np.array([])

    return X_train_scaled, X_test_scaled, np.asarray(scale_indices, dtype=int), mean, std


def load_preprocessed_split(csv_path=None, test_size=0.2, random_state=42):
    df = load_telco_dataframe(csv_path).copy()
    raw_rows = len(df)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int).to_numpy()
    X_df = df.drop(columns=[TARGET_COLUMN])

    categorical_columns = X_df.select_dtypes(include=["object"]).columns.tolist()
    X_df = pd.get_dummies(X_df, columns=categorical_columns, drop_first=False)
    X_df = X_df.astype(float)

    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, train_idx, test_idx = stratified_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_scaled, X_test_scaled, scale_indices, mean, std = standardize_selected_features(
        X_train, X_test, feature_names
    )

    metadata = {
        "raw_rows": raw_rows,
        "cleaned_rows": len(df),
        "dropped_totalcharges_rows": raw_rows - len(df),
        "feature_names": feature_names,
        "scaled_feature_indices": scale_indices,
        "scaled_feature_names": [feature_names[i] for i in scale_indices],
        "train_indices": train_idx,
        "test_indices": test_idx,
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, metadata
