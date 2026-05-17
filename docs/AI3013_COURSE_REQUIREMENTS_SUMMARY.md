# AI3013 Course Requirements Summary

This file summarizes the course constraints for Codex and teammates.

## Project

Course: AI3013 Machine Learning Course Project  
Topic: Telco Customer Churn Prediction  
Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv  
Task type: Binary classification  
Target: Churn  
Positive class: Churn = Yes -> 1  
Negative class: Churn = No -> 0  

## Required models

The final report compares:

1. Logistic Regression from scratch
2. K-Nearest Neighbors from scratch
3. Linear Support Vector Machine from scratch

## Important course constraints

- The implementation should be from scratch.
- Do not use existing machine learning model libraries for the main models.
- Do not use scikit-learn, TensorFlow, or Keras model classes for Logistic Regression, KNN, or SVM.
- NumPy, Pandas, and Matplotlib are allowed.
- The project should include preprocessing, model theory, implementation, evaluation, analysis, and comparison.
- Use suitable metrics for binary classification:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Use cross-validation to check robustness.
- Compare models using:
  - Performance metrics
  - Training time
  - Prediction time
  - Memory usage
  - Suitability for the dataset
  - Bias-variance behavior
- The final report should be 9-12 pages.
- The final ZIP submission should include source code, presentation PPT, and project report.
- Code should run correctly from a clean folder.
