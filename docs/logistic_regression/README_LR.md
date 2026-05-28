# Logistic Regression Code Pack - Telco Customer Churn

This folder is the clean submission version for the Logistic Regression part.

## Project purpose

The project predicts telecom customer churn using from-scratch machine learning models.  
My part focuses on Logistic Regression and explains whether a churn-oriented decision threshold is more suitable than the default threshold of 0.50.

## Novelty in the Logistic Regression part

The novelty is not inventing a new algorithm. Logistic Regression is a common method.  
The novelty is a churn-oriented adaptation:

1. Logistic Regression implemented from scratch using NumPy.
2. Class-weighted binary cross-entropy to address class imbalance.
3. Validation-based threshold analysis.
4. Final report threshold fixed at **0.36** to match the validated LR handoff experiment and final report.
5. F2-score and business cost used to explain why a lower churn-oriented threshold reduces missed churn customers.

The final threshold is 0.36. The test set is used only for final evaluation.

## Required data files

The following processed files are already included in the `data/` folder:

- `train_processed.csv`
- `test_processed.csv`
- `train_processed_oversampled.csv`
- `information_gain_ranking.csv`
- `preprocessing_summary.json`

The target column is `Churn`.

## How to run

From this folder:

```bash
python run_logistic_regression.py
```

Or run the notebook:

```text
logistic_regression_churn_from_scratch_novelty_v2.ipynb
```

## Outputs

After running through this repository, results are saved to `outputs/logistic_regression/`.

Main CSV files:

- `lr_metrics.csv`
- `lr_threshold_search_validation.csv`
- `lr_threshold_comparison_test_selected.csv`
- `lr_presentation_summary.json`

Main figures:

- `figures/lr_loss_curves.png`
- `figures/lr_model_comparison_bar.png`
- `figures/lr_confusion_matrices.png`
- `figures/lr_threshold_sweep_validation.png`
- `figures/lr_threshold_comparison_test_selected.png`
- `figures/lr_top_coefficients_novel_model.png`

## Important explanation for report and presentation

The final threshold used for the report is **0.36**. In this repository it is fixed in code so rerunning `python run_all.py --full` remains consistent with the validated LR handoff experiment and final report.  
F2-score is discussed because it gives more weight to recall, which is important in churn prediction because missing a real churn customer is costly.

Final LR report metrics: Accuracy = 0.673774, Precision = 0.443258, Recall = 0.887701, F1 = 0.591273, F2 = 0.739421, TN = 616, FP = 417, FN = 42, TP = 332.

Business cost is defined as:

```text
Business cost = 5 * FN + 1 * FP
```

This reflects the assumption that missing a real churn customer is more costly than contacting a non-churn customer.
