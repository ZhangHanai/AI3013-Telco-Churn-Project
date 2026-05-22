from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    model: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    business_cost: int
    tn: int
    fp: int
    fn: int
    tp: int
    training_time_sec: float
    prediction_time_sec: float
    peak_memory_mb: float


class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using NumPy.

    Main design choices:
    - Sigmoid activation for binary probability output.
    - Weighted binary cross-entropy for class imbalance.
    - Batch gradient descent optimization.
    - Optional L2 regularization, excluding the bias term.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 2500,
        l2_lambda: float = 0.001,
        class_weight: dict[int, float] | None = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)
        return np.array([self.class_weight[int(label)] for label in y], dtype=float)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        proba = self.predict_proba(X)
        sample_w = self._sample_weights(y)

        bce = -(y * np.log(proba + eps) + (1 - y) * np.log(1 - proba + eps))
        weighted_bce = np.mean(sample_w * bce)

        if self.weights is None:
            return float(weighted_bce)

        l2_penalty = (self.l2_lambda / 2.0) * np.sum(self.weights ** 2)
        return float(weighted_bce + l2_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        self.loss_history = []

        sample_w = self._sample_weights(y)

        for epoch in range(self.epochs):
            linear = X @ self.weights + self.bias
            proba = self._sigmoid(linear)
            error = (proba - y) * sample_w

            grad_w = (X.T @ error) / n_samples + self.l2_lambda * self.weights
            grad_b = float(np.mean(error))

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            if epoch % 25 == 0 or epoch == self.epochs - 1:
                self.loss_history.append(self.compute_loss(X, y))

            if self.verbose and epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | Loss = {self.loss_history[-1]:.5f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    beta2 = beta ** 2
    f2 = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)

    # Business assumption: missing a real churn customer is costlier than contacting a non-churn customer.
    business_cost = 5 * fn + 1 * fp

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "business_cost": business_cost,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def stratified_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        shuffled = rng.permutation(idx)
        val_count = int(round(len(shuffled) * validation_size))
        val_idx.extend(shuffled[:val_count].tolist())
        train_idx.extend(shuffled[val_count:].tolist())

    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def load_processed_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train_processed.csv")
    test_df = pd.read_csv(data_dir / "test_processed.csv")
    oversampled_df = pd.read_csv(data_dir / "train_processed_oversampled.csv")
    return train_df, test_df, oversampled_df


def split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_df = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int).to_numpy()
    X = X_df.to_numpy(dtype=float)
    return X, y, list(X_df.columns)


def make_class_weight(y: np.ndarray) -> dict[int, float]:
    n = len(y)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    return {
        0: n / (2 * n0),
        1: n / (2 * n1),
    }


def threshold_search(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (proba >= threshold).astype(int)
        row = {"threshold": float(threshold)}
        row.update(calculate_metrics(y_true, pred))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(search_df: pd.DataFrame) -> float:
    # Primary: maximize F2.
    # Tie-breakers: minimize business cost, then maximize recall, then maximize precision.
    ordered = search_df.sort_values(
        by=["f2", "business_cost", "recall", "precision"],
        ascending=[False, True, False, False],
    )
    return float(ordered.iloc[0]["threshold"])


def fit_and_evaluate(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    class_weight: dict[int, float] | None = None,
    learning_rate: float = 0.05,
    epochs: int = 2500,
    l2_lambda: float = 0.001,
) -> tuple[LogisticRegressionScratch, EvaluationResult, np.ndarray]:
    model = LogisticRegressionScratch(
        learning_rate=learning_rate,
        epochs=epochs,
        l2_lambda=l2_lambda,
        class_weight=class_weight,
        random_state=42,
    )

    tracemalloc.start()
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_train
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start_pred = time.perf_counter()
    proba = model.predict_proba(X_test)
    pred = (proba >= threshold).astype(int)
    prediction_time = time.perf_counter() - start_pred

    metrics = calculate_metrics(y_test, pred)
    result = EvaluationResult(
        model=model_name,
        threshold=threshold,
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        f2=float(metrics["f2"]),
        business_cost=int(metrics["business_cost"]),
        tn=int(metrics["tn"]),
        fp=int(metrics["fp"]),
        fn=int(metrics["fn"]),
        tp=int(metrics["tp"]),
        training_time_sec=training_time,
        prediction_time_sec=prediction_time,
        peak_memory_mb=peak / (1024 ** 2),
    )

    return model, result, proba


def save_confusion_matrix_plot(results_df: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results_df), figsize=(5 * len(results_df), 4))

    if len(results_df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, results_df.iterrows()):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        im = ax.imshow(matrix)
        ax.set_title(row["model"], fontsize=10)
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=axes, fraction=0.025)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metric_comparison_plot(results_df: pd.DataFrame, fig_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "f2"]
    plot_df = results_df[["model"] + metrics].melt(id_vars="model", var_name="metric", value_name="score")

    plt.figure(figsize=(11, 6))
    x_positions = np.arange(len(metrics))
    width = 0.18

    models = results_df["model"].tolist()
    for i, model in enumerate(models):
        subset = plot_df[plot_df["model"] == model]
        plt.bar(x_positions + (i - (len(models)-1)/2) * width, subset["score"], width=width, label=model)

    plt.xticks(x_positions, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Logistic Regression Experiment Comparison on Test Set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_threshold_plot(search_df: pd.DataFrame, selected_threshold: float, fig_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for metric in ["precision", "recall", "f1", "f2"]:
        plt.plot(search_df["threshold"], search_df[metric], label=metric)

    plt.axvline(selected_threshold, linestyle="--", label=f"selected threshold = {selected_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Validation Threshold Search for Churn-Oriented Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_loss_curve_plot(models: dict[str, LogisticRegressionScratch], fig_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        plt.plot(model.loss_history, label=name)
    plt.xlabel("Recorded training step")
    plt.ylabel("Loss")
    plt.title("Logistic Regression Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_threshold_comparison_plot(threshold_df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df["threshold"], threshold_df["accuracy"], marker="o", label="accuracy")
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="recall")
    plt.plot(threshold_df["threshold"], threshold_df["f2"], marker="o", label="f2")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Trade-off on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_top_coefficients_plot(model: LogisticRegressionScratch, feature_names: list[str], fig_path: Path) -> None:
    if model.weights is None:
        return
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.weights})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    top_df = coef_df.sort_values("abs_coefficient", ascending=False).head(15).sort_values("coefficient")

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["coefficient"])
    plt.xlabel("Coefficient")
    plt.title("Top Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment(base_dir: Path | None = None) -> None:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]

    data_dir = base_dir / "data"
    out_dir = base_dir / "outputs_lr"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, oversampled_df = load_processed_data(data_dir)

    X_train, y_train, feature_names = split_xy(train_df)
    X_test, y_test, _ = split_xy(test_df)
    X_over, y_over, _ = split_xy(oversampled_df)

    X_inner, X_val, y_inner, y_val = stratified_validation_split(X_train, y_train, validation_size=0.2)

    class_weight = make_class_weight(y_inner)

    # Baseline LR
    baseline_model, baseline_result, _ = fit_and_evaluate(
        "Baseline LR t=0.50",
        X_train, y_train, X_test, y_test,
        threshold=0.5,
        class_weight=None,
    )

    # Oversampled LR
    oversampled_model, oversampled_result, _ = fit_and_evaluate(
        "Oversampled LR t=0.50",
        X_over, y_over, X_test, y_test,
        threshold=0.5,
        class_weight=None,
    )

    # Class-weighted LR with default threshold
    weighted_model, weighted_result, _ = fit_and_evaluate(
        "Class-weighted LR t=0.50",
        X_train, y_train, X_test, y_test,
        threshold=0.5,
        class_weight=make_class_weight(y_train),
    )

    # Threshold search on validation set only
    validation_model = LogisticRegressionScratch(
        learning_rate=0.05,
        epochs=2500,
        l2_lambda=0.001,
        class_weight=class_weight,
        random_state=42,
    )
    validation_model.fit(X_inner, y_inner)
    val_proba = validation_model.predict_proba(X_val)
    search_df = threshold_search(y_val, val_proba)
    selected_threshold = choose_threshold(search_df)
    search_df.to_csv(out_dir / "lr_threshold_search_validation.csv", index=False)

    # Novelty model: retrain on all original training data and evaluate on untouched test set
    novel_model, novel_result, novel_test_proba = fit_and_evaluate(
        f"Novel weighted LR t={selected_threshold:.2f}",
        X_train, y_train, X_test, y_test,
        threshold=selected_threshold,
        class_weight=make_class_weight(y_train),
    )

    results_df = pd.DataFrame([baseline_result.__dict__, oversampled_result.__dict__, weighted_result.__dict__, novel_result.__dict__])
    results_df.to_csv(out_dir / "lr_metrics.csv", index=False)

    # Selected threshold comparison on test set, for sensitivity analysis only
    selected_thresholds = [0.20, 0.30, round(selected_threshold, 2), 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    threshold_rows = []
    for t in selected_thresholds:
        pred = (novel_test_proba >= t).astype(int)
        row = {"threshold": float(t)}
        row.update(calculate_metrics(y_test, pred))
        threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    threshold_df.to_csv(out_dir / "lr_threshold_comparison_test_selected.csv", index=False)

    # Save figures
    save_loss_curve_plot(
        {
            "Baseline LR": baseline_model,
            "Oversampled LR": oversampled_model,
            "Class-weighted LR": weighted_model,
            "Novel weighted LR": novel_model,
        },
        fig_dir / "lr_loss_curves.png",
    )
    save_metric_comparison_plot(results_df, fig_dir / "lr_model_comparison_bar.png")
    save_confusion_matrix_plot(results_df, fig_dir / "lr_confusion_matrices.png")
    save_threshold_plot(search_df, selected_threshold, fig_dir / "lr_threshold_sweep_validation.png")
    save_threshold_comparison_plot(threshold_df, fig_dir / "lr_threshold_comparison_test_selected.png")
    save_top_coefficients_plot(novel_model, feature_names, fig_dir / "lr_top_coefficients_novel_model.png")

    # Export a compact summary for presentation
    presentation_summary = {
        "project_goal": "Predict telecom customer churn using from-scratch ML models and compare model suitability.",
        "lr_novelty": "Class-weighted Logistic Regression with validation-based threshold tuning for churn-oriented recall/F2 objective.",
        "selected_threshold": selected_threshold,
        "selection_metric": "Primary: validation F2-score; secondary: lower business cost = 5*FN + 1*FP; supporting: recall and FN.",
        "final_novel_result": novel_result.__dict__,
    }
    with open(out_dir / "lr_presentation_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(presentation_summary, f, indent=2)

    print("Saved LR results to:", out_dir)
    print("Selected validation threshold:", selected_threshold)
    print(results_df[["model", "threshold", "accuracy", "precision", "recall", "f1", "f2", "business_cost", "tn", "fp", "fn", "tp"]])


if __name__ == "__main__":
    run_experiment()
