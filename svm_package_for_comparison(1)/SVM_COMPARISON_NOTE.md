# SVM Comparison Note

This note is prepared for the group member who will write the comparison among Logistic Regression, KNN, and SVM.

## Model type

- Model: `Linear SVM from scratch`
- Implementation file: `svm_scratch.py`
- Training script: `run_svm_experiments.py`
- Metrics file: `outputs/svm_metrics.csv`
- Full experiment table: `outputs/svm_experiments.csv`

## Preprocessing setting used by SVM

- Same shared preprocessing pipeline as the other models
- `customerID` removed
- `TotalCharges` converted to numeric
- 11 rows with missing `TotalCharges` removed
- One-hot encoding applied
- Numerical features standardized:
  - `tenure`
  - `MonthlyCharges`
  - `TotalCharges`
- Stratified 80/20 train-test split
- Additional oversampled training set prepared

## Why SVM is included

- SVM is a margin-based linear classifier
- It provides a useful contrast to:
  - Logistic Regression: probabilistic linear classifier
  - KNN: distance-based non-parametric classifier

## Best current SVM result

The current best experiment is `oversampled_train`.

- Accuracy: `0.7427`
- Precision: `0.5099`
- Recall: `0.8262`
- F1-score: `0.6306`
- Balanced Accuracy: `0.7693`
- TNR: `0.7125`

Confusion matrix values:

- TP = `309`
- TN = `736`
- FP = `297`
- FN = `65`

## Key comparison observations

- Compared with the baseline original-train SVM, the oversampled setting improves:
  - Recall
  - F1-score
  - Balanced Accuracy
- But it reduces:
  - Accuracy
  - Precision
  - TNR

This means the oversampled SVM is more aggressive in identifying churn customers. It catches more positive cases, but it also produces more false positives.

## Suggested wording for the comparison section

You can describe the SVM result like this:

`The Linear SVM achieved its best result when trained on the oversampled training set. This setting improved recall and F1-score, which is useful for churn detection because missing churn customers is costly. However, the gain in recall came with lower precision and lower true negative rate, indicating that the model became more aggressive in predicting churn. Therefore, SVM can be interpreted as a margin-based classifier that offers stronger positive-class detection than the more conservative baseline, but it does not necessarily dominate the other models on all evaluation metrics.`

## Files to cite or inspect

- `svm_scratch.py`
- `run_svm_experiments.py`
- `metrics_scratch.py`
- `outputs/svm_metrics.csv`
- `outputs/svm_experiments.csv`
- `figures/svm_loss_curve.png`
