import numpy as np


class KNearestNeighbors:
    """K-Nearest Neighbors classifier implemented from scratch.

    This implementation only uses NumPy. It supports binary classification
    labels encoded as 0 and 1, which matches the Telco churn setting:
    Churn = Yes -> 1, Churn = No -> 0.
    """

    def __init__(self, k=5, weighted=False, batch_size=512):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        self.k = k
        self.weighted = weighted
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data.

        KNN is a lazy learner: training only stores X and y. The main
        computation happens during prediction, when distances are calculated.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if self.k > X.shape[0]:
            raise ValueError("k cannot be larger than the number of training samples.")

        unique_labels = set(np.unique(y).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError("This binary KNN expects labels encoded as 0 and 1.")

        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distances_squared(self, X_batch):
        """Compute squared Euclidean distances from test rows to training rows."""
        test_sq = np.sum(X_batch ** 2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train ** 2, axis=1)
        cross_term = X_batch @ self.X_train.T
        distances_sq = test_sq + train_sq - 2 * cross_term
        return np.maximum(distances_sq, 0.0)

    def _vote_unweighted(self, neighbor_labels):
        positive_votes = np.sum(neighbor_labels == 1)
        negative_votes = np.sum(neighbor_labels == 0)

        if positive_votes > negative_votes:
            return 1
        if negative_votes > positive_votes:
            return 0
        # Tie-breaking rule: predict churn class 1 (Churn = Yes).
        return 1

    def _vote_weighted(self, neighbor_labels, neighbor_distances_sq):
        eps = 1e-12
        weights = 1.0 / (np.sqrt(neighbor_distances_sq) + eps)
        positive_weight = np.sum(weights[neighbor_labels == 1])
        negative_weight = np.sum(weights[neighbor_labels == 0])

        if positive_weight > negative_weight:
            return 1
        if negative_weight > positive_weight:
            return 0
        # Tie-breaking rule: predict churn class 1 (Churn = Yes).
        return 1

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X must have the same number of features as the training data.")

        predictions = []
        for start in range(0, X.shape[0], self.batch_size):
            end = start + self.batch_size
            X_batch = X[start:end]
            distances_sq = self._euclidean_distances_squared(X_batch)

            for row_distances_sq in distances_sq:
                neighbor_indices = np.argpartition(row_distances_sq, self.k - 1)[:self.k]
                neighbor_labels = self.y_train[neighbor_indices]
                neighbor_distances_sq = row_distances_sq[neighbor_indices]

                if self.weighted:
                    prediction = self._vote_weighted(neighbor_labels, neighbor_distances_sq)
                else:
                    prediction = self._vote_unweighted(neighbor_labels)
                predictions.append(prediction)

        return np.asarray(predictions, dtype=int)

    def predict_proba(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X must have the same number of features as the training data.")

        probabilities = []
        for start in range(0, X.shape[0], self.batch_size):
            end = start + self.batch_size
            X_batch = X[start:end]
            distances_sq = self._euclidean_distances_squared(X_batch)

            for row_distances_sq in distances_sq:
                neighbor_indices = np.argpartition(row_distances_sq, self.k - 1)[:self.k]
                neighbor_labels = self.y_train[neighbor_indices]
                neighbor_distances_sq = row_distances_sq[neighbor_indices]

                if self.weighted:
                    eps = 1e-12
                    weights = 1.0 / (np.sqrt(neighbor_distances_sq) + eps)
                    probability = np.sum(weights[neighbor_labels == 1]) / np.sum(weights)
                else:
                    probability = np.mean(neighbor_labels == 1)
                probabilities.append(probability)

        return np.asarray(probabilities, dtype=float)
