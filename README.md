# AI3013 Telco Customer Churn Project - Member 4 KNN Package

This folder contains the integrated code package for Member 4's KNN part of the AI3013 Machine Learning course project.

## Task

Binary classification on the Telco Customer Churn dataset.

- Positive class: `Churn = Yes` -> `1`
- Negative class: `Churn = No` -> `0`
- Main model: K-Nearest Neighbors from scratch
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix, and cross-validation

## Folder structure

```text
AI3013_Telco_Churn_4_KNN_Integrated/
├── data/
├── src/
├── figures/
├── results/
├── report_notes/
├── run_all.py
├── run_fast_test.py
├── requirements.txt
└── README.md
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick check:

```bash
python test_knn_small.py
```

Fast experiment:

```bash
python run_fast_test.py
```

Full experiment for the final report:

```bash
python run_all.py
```

## Expected outputs

The scripts save outputs to `results/` and `figures/`, including:

```text
results/knn_cv_summary.csv
results/knn_cv_detailed.csv
results/knn_test_metrics.csv
results/knn_test_predictions.csv
results/model_comparison_template.csv
figures/knn_cv_mean_f1_by_k.png
figures/knn_cv_mean_accuracy_by_k.png
figures/knn_confusion_matrix.png
```

## Notes

This package is for Member 4's part. The final group package should later merge this folder with:

- Member 3: Logistic Regression from scratch
- Members 1 and 2: Linear SVM from scratch
- Final report
- Presentation slides
