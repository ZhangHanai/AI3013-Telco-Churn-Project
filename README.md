# AI3013 Telco Customer Churn Project

This project contains from-scratch implementations for Telco churn prediction experiments (Logistic Regression, KNN, and Linear SVM), using one **shared preprocessing pipeline**.

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


## Repository structure policy
- This repository is maintained as **one unified project** (no nested standalone model packages).
- All source code lives in `src/`.
- All SVM CSV outputs are saved in `outputs/`.
- Any model-specific notes should be consolidated into this README or `docs/`.

## Setup
```bash
pip install -r requirements.txt
```


## Run Logistic Regression (from scratch)
```bash
python -m src.run_logistic_regression
```
Notes:
- Uses only NumPy/Pandas/Matplotlib (no sklearn/TensorFlow/Keras model training).
- Uses project-root relative paths (`Path.cwd()`).
- Reads processed CSVs from `data/` if present, otherwise falls back to `data/processed_reference/`.
- Saves outputs to `outputs/logistic_regression/` and figures to `figures/logistic_regression/`.
- This LR handoff is recall-oriented / churn-sensitive: threshold is selected on validation by F2 and business-cost rule (`5*FN + 1*FP`).

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
(Current `run_all.py` executes KNN + SVM. Run LR separately using the command above.)


## Checks
```bash
python -m py_compile src/*.py
python -m src.run_knn_experiment --fast
python -m src.run_svm_experiments
```

## Main outputs
- `results/model_comparison_rows/lr_row_for_results_summary.csv` (recommended LR row for final 3-model table)
- `results/logistic_regression/lr_metrics.csv` (complete LR experiment comparison file)
- `results/knn_cv_summary.csv`
- `results/knn_test_metrics.csv`
- `outputs/svm_experiments.csv`
- `outputs/svm_metrics.csv`
- `outputs/information_gain_ranking.csv`
- `figures/knn_confusion_matrix.png`
- `figures/svm_loss_curve.png`

## Logistic Regression handoff integration map
- LR code: `src/logistic_regression_scratch.py`, `src/run_logistic_regression.py`.
- LR comparison row source: `results/model_comparison_rows/lr_row_for_results_summary.csv`.
- LR report figures: `figures/logistic_regression/`.
- LR handoff notes (EN/CN/checklist): `docs/logistic_regression/`.
- This handoff only provides LR outputs. KNN + SVM rows still need to be merged with the LR row into one final summary table containing: accuracy, precision, recall, F1, training_time_sec, prediction_time_sec, peak_memory_mb, TN, FP, FN, TP, threshold.
