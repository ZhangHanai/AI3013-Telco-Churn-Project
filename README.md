# AI3013 Telco Customer Churn Project

This project implements **Logistic Regression, KNN, and Linear SVM from scratch** using NumPy/Pandas/Matplotlib only (no sklearn/TensorFlow/Keras/PyTorch model implementations).

## Preprocessing (shared by all three models)
`src/preprocessing.py` is the canonical pipeline used by LR/KNN/SVM:
- load `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- drop `customerID`
- convert `TotalCharges` to numeric and **drop rows with missing `TotalCharges`**
- encode target: `Churn` Yes=1, No=0
- one-hot encode categorical features
- stratified split (`test_size=0.2`, `random_state=42`)
- standardize only `tenure`, `MonthlyCharges`, `TotalCharges` using train-only stats

## How to Run
```bash
python run_all.py --demo
python run_all.py --full
python -m src.run_logistic_regression
python -m src.run_knn_experiment --fast
python -m src.run_svm_experiments
```

### What each command produces
- `python run_all.py --demo`: runs LR (demo CV), KNN fast mode, SVM demo CV, then writes unified summary.
- `python run_all.py --full`: runs full LR/KNN/SVM experiments with full CV settings, then writes unified summary. In full mode, KNN CV tests k = 1, 3, 5, 7, 9, 11 with 5 folds.
- `python -m src.run_logistic_regression`: LR outputs in `outputs/logistic_regression/` including `lr_metrics.csv` and `lr_cv_summary.csv`.
- `python -m src.run_knn_experiment --fast`: KNN fast outputs in `outputs/knn/`.
- `python -m src.run_svm_experiments`: SVM outputs in `outputs/svm/` including `svm_metrics.csv` and `svm_cv_summary.csv`.

## Canonical final outputs
- Unified model comparison: `outputs/results_summary.csv`
- LR: `outputs/logistic_regression/`
- KNN: `outputs/knn/`
- SVM: `outputs/svm/`
- SVM final selected model may be `oversampled_train`, which means **only the training split is oversampled**; the held-out test set remains unchanged for evaluation.

## Demo guide
1. Run `python run_all.py --demo`.
2. Show `outputs/results_summary.csv` for 3-model comparison.
3. Show per-model files: `outputs/logistic_regression/lr_metrics.csv`, `outputs/knn/knn_test_metrics.csv`, `outputs/svm/svm_metrics.csv`.
