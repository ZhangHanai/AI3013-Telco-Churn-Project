# LR Handoff Notes for Model Analysis Teammate

## Purpose

This package provides the Logistic Regression outputs needed for Model Performance Comparison, Computational Complexity Comparison, Bias-Variance discussion, and presentation.

## Key file for final comparison

Use:

`02_results_for_model_comparison/lr_row_for_results_summary.csv`

This file contains the recommended LR row for the final three-model comparison table.

## Recommended LR model

Recommended row: `Novel weighted LR t=0.36`

This is the final Logistic Regression variant because it represents the novelty of this part:
- Logistic Regression from scratch
- class-weighted binary cross-entropy
- validation-based threshold tuning
- churn-oriented F2-score / business cost objective

## Why threshold = 0.36?

The threshold was selected on the validation set, not on the test set.
The selection criterion was:
1. Maximize F2-score
2. Minimize business cost = 5 × FN + 1 × FP
3. Maintain high recall and reduce false negatives

## Suggested interpretation

The threshold-tuned Logistic Regression sacrifices some accuracy and precision, but it improves recall and reduces false negatives. This makes it more suitable for telecom churn prediction, because missing an actual churn customer can be more costly than contacting a non-churn customer.

## Missing items for the full group comparison

This package only contains Logistic Regression. The final comparison still needs:
- KNN metrics and CV results
- SVM metrics and parameter results
- unified `results_summary.csv`
- unified `cv_results.csv`