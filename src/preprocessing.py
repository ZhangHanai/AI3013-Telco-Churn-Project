from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "Churn"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def find_dataset_path(project_root=None):
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    candidates = [
        project_root / "data" / DATASET_FILENAME,
        project_root / DATASET_FILENAME,
        project_root.parent / "data" / DATASET_FILENAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset not found. Please place WA_Fn-UseC_-Telco-Customer-Churn.csv under the data/ folder."
    )


def load_telco_dataframe(csv_path=None):
    if csv_path is None:
        csv_path = find_dataset_path()
    return pd.read_csv(csv_path)


def encode_telco_features(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    yes_no_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1}).astype(int)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("The dataset must contain the target column: Churn.")

    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int).to_numpy()
    X_df = df.drop(columns=[TARGET_COLUMN])

    categorical_columns = X_df.select_dtypes(include=["object"]).columns.tolist()
    X_df = pd.get_dummies(X_df, columns=categorical_columns, drop_first=False)
    X_df = X_df.astype(float)

    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)
    return X, y, feature_names, X_df


def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        n_test = int(round(len(label_indices) * test_size))
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    train_indices = np.asarray(train_indices)
    test_indices = np.asarray(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices], train_indices, test_indices


def fit_standardizer(X_train):
    X_train = np.asarray(X_train, dtype=float)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    X = np.asarray(X, dtype=float)
    return (X - mean) / std


def standardize_train_test(X_train, X_test):
    mean, std = fit_standardizer(X_train)
    X_train_scaled = apply_standardizer(X_train, mean, std)
    X_test_scaled = apply_standardizer(X_test, mean, std)
    return X_train_scaled, X_test_scaled, mean, std


def load_encoded_telco_data(csv_path=None):
    df = load_telco_dataframe(csv_path)
    return encode_telco_features(df)
