from __future__ import annotations

import numpy as np


class LinearSVMScratch:
    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        n_epochs: int = 300,
        batch_size: int = 128,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _to_margin_labels(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).astype(int)
        return np.where(y == 1, 1.0, -1.0)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.weights is None:
            raise ValueError("Model has not been fitted.")
        y_margin = self._to_margin_labels(y)
        margins = y_margin * (X @ self.weights + self.bias)
        hinge_loss = np.maximum(0.0, 1.0 - margins).mean()
        reg_loss = 0.5 * self.lambda_param * np.dot(self.weights, self.weights)
        return float(reg_loss + hinge_loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        X = np.asarray(X, dtype=float)
        y_margin = self._to_margin_labels(np.asarray(y))

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []
        rng = np.random.default_rng(self.random_state)

        batch_size = min(self.batch_size, n_samples)
        for _ in range(self.n_epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_margin[indices]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                margins = y_batch * (X_batch @ self.weights + self.bias)
                active = margins < 1.0

                grad_w = self.lambda_param * self.weights
                if np.any(active):
                    grad_w -= np.mean((y_batch[active, None] * X_batch[active]), axis=0)
                    grad_b = -float(np.mean(y_batch[active]))
                else:
                    grad_b = 0.0

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            epoch_loss = self.compute_loss(X, np.where(y_margin == 1, 1, 0))
            self.loss_history.append(epoch_loss)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been fitted.")
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)

