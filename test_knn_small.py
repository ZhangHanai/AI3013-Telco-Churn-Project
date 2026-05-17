import numpy as np

from src.knn_scratch import KNearestNeighbors
from src.metrics import print_classification_report


X_train = np.array([[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [3.0, 3.0], [3.2, 3.1], [2.9, 3.3]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[0.1, 0.2], [3.1, 3.2], [2.8, 3.0], [0.3, 0.2]])
y_test = np.array([0, 1, 1, 0])

model = KNearestNeighbors(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions:", predictions.tolist())
print("Positive-class probabilities:", probabilities.round(4).tolist())
print_classification_report(y_test, predictions, model_name="KNN small test")
