# AI3013 Telco Customer Churn Prediction (Final Submission)

## 1. Project Overview
- This project predicts whether a telecom customer will churn.
- It is a **binary classification** task.
- Positive class definition: **Churn = Yes (1)**.
- The project compares three **from-scratch** machine learning models under one shared preprocessing pipeline.

## 2. Repository Structure
- `data/`  
  Raw dataset and processed reference files.
- `src/`  
  Source code for preprocessing, model implementations, metrics, and experiment runners.
- `outputs/`  
  **Canonical final result folder** for model metrics, CV summaries, and final comparison outputs.
- `figures/`  
  Generated plots for analysis/demo.
- `run_all.py`  
  Top-level script that runs all three models and rebuilds `outputs/results_summary.csv`.
- `requirements.txt`  
  Python dependencies.

## 3. Dataset
- Dataset: IBM Telco Customer Churn.
- File used: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- Shared preprocessing includes:
  - remove `customerID`
  - convert `TotalCharges` to numeric
  - drop rows with invalid/missing `TotalCharges`
  - encode `Churn` as `No = 0`, `Yes = 1`
  - encode categorical features
  - scale numerical features using training-set statistics only

## 4. Environment Setup
```bash
python -m pip install -r requirements.txt
```

## 5. How to Run
```bash
python run_all.py --demo
python run_all.py --full
```

## 6. Demo Mode
- `python run_all.py --demo` is the faster mode for class/project demo.
- It still runs:
  - Logistic Regression (from scratch)
  - K-Nearest Neighbors (from scratch)
  - Linear SVM (from scratch)
- It generates per-model outputs and rebuilds the unified summary file.

## 7. Full Experiment Mode
- `python run_all.py --full` regenerates the full final experimental outputs.
- This mode may take longer, especially for KNN, because prediction is distance-based.

## 8. Output Files (Final)
- `outputs/results_summary.csv`
- `outputs/logistic_regression/lr_metrics.csv`
- `outputs/logistic_regression/lr_cv_summary.csv`
- `outputs/knn/knn_cv_summary.csv`
- `outputs/knn/knn_test_metrics.csv`
- `outputs/svm/svm_experiments.csv`
- `outputs/svm/svm_metrics.csv`
- `outputs/svm/svm_cv_summary.csv`

## 9. Model Summary
- **Logistic Regression (from scratch)**
  - sigmoid probability output
  - weighted binary cross-entropy
  - gradient descent optimization
  - final threshold fixed at **0.36** to match the validated LR handoff experiment and final report
  - final report metrics: Accuracy = 0.673774, Precision = 0.443258, Recall = 0.887701, F1 = 0.591273, F2 = 0.739421, TN = 616, FP = 417, FN = 42, TP = 332
- **KNN (from scratch)**
  - Euclidean distance
  - k-nearest neighbors majority voting
  - `k` selected by cross-validation
- **Linear SVM (from scratch)**
  - linear decision boundary
  - hinge loss with L2 regularization
  - oversampled training setting for churn-oriented detection (training only)

## 10. Notes for Project Code Demo
Recommended demo flow:
1. Show shared preprocessing in `src/preprocessing.py`.
2. Show from-scratch model files in `src/`.
3. Show orchestration in `run_all.py`.
4. Show final comparison in `outputs/results_summary.csv`; the Logistic Regression row should show `threshold=0.36`, Recall ≈ 0.8877, F1 ≈ 0.5913, FN = 42, TP = 332.
5. Explain conclusion:
   - Logistic Regression: churn-oriented threshold-tuned model aligned with the final report
   - KNN: highest accuracy and precision
   - Linear SVM: highest F1-score
   - No single model dominates all metrics; model choice depends on business objective.

---
This repository is prepared as a reproducible, from-scratch, leakage-safe, business-oriented model comparison for final submission.
