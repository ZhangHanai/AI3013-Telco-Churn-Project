# AI3013 Telco Customer Churn Project

This project contains from-scratch implementations for Telco churn prediction experiments (KNN and Linear SVM), using one **shared preprocessing pipeline**.

## Shared preprocessing (used by KNN/SVM and reusable for Logistic Regression)
Pipeline in `src/preprocessing.py`:
- read `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- drop `customerID`
- convert `TotalCharges` to numeric
- drop missing `TotalCharges` rows (11 rows)
- encode `Churn` as Yes=1, No=0
- keep `SeniorCitizen` numeric
- one-hot encode categorical features
- stratified split (`test_size=0.2`, `random_state=42`)
- standardize only `tenure`, `MonthlyCharges`, `TotalCharges` using **training-only statistics**

Expected split shape with the official raw dataset:
- Train: `5625 x 45`
- Test: `1407 x 45`

## Setup
```bash
pip install -r requirements.txt
```

## Run KNN
```bash
python -m src.run_knn_experiment
```
Fast mode:
```bash
python -m src.run_knn_experiment --fast
```

## Run SVM
```bash
python -m src.run_svm_experiments
```
Notes:
- Oversampling is applied **only** to SVM training experiments.
- Test set is never oversampled.

## Run all experiments
```bash
python run_all.py
```

## Checks
```bash
python -m py_compile src/*.py
python -m src.run_knn_experiment --fast
python -m src.run_svm_experiments
```

## Main outputs
- `results/knn_cv_summary.csv`
- `results/knn_test_metrics.csv`
- `outputs/svm_experiments.csv`
- `outputs/svm_metrics.csv`
- `outputs/information_gain_ranking.csv`
- `figures/knn_confusion_matrix.png`
- `figures/svm_loss_curve.png`
