SVM package for model comparison

Start here:
1. Read `SVM_COMPARISON_NOTE.md`
2. Check `outputs/svm_metrics.csv` for the best current SVM result
3. Check `outputs/svm_experiments.csv` for all tested SVM settings
4. Use `figures/svm_loss_curve.png` if you want to mention training behavior

Main current conclusion:
- Best setting: `oversampled_train`
- Best F1: `0.6306`
- Recall is high: `0.8262`
- TNR is `0.7125`

Useful interpretation:
- SVM on the oversampled training set is more aggressive in detecting churn customers
- It improves recall and F1 compared with the baseline original-train SVM
- But it sacrifices some precision, accuracy, and true negative rate

Code files included:
- `svm_scratch.py`
- `run_svm_experiments.py`
- `metrics_scratch.py`
