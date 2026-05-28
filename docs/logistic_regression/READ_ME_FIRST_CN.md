# 给模型分析同学的 LR Handoff Notes

## 1. 这个包的作用

这是 Logistic Regression 部分给 Model Performance Comparison / Bias-Variance / Suitability Analysis 同学使用的材料包。
它包含 LR from scratch 代码、最终指标、threshold 对比表、图，以及可以直接合并进 results_summary.csv 的 LR 行。

## 2. 你需要优先看的文件

1. `outputs/results_summary.csv`
   - 这是三模型总对比表；Logistic Regression 行必须是 `threshold=0.36`。
   - 已包含 accuracy, precision, recall, F1, training time, prediction time, memory, confusion matrix。

2. `outputs/logistic_regression/lr_metrics.csv`
   - 这是 LR 所有实验版本：
     - Baseline LR t=0.50
     - Oversampled LR t=0.50
     - Class-weighted LR t=0.50
     - Novel weighted LR t=0.36

3. `figures/logistic_regression/lr_threshold_sweep_validation.png`
   - 展示 validation threshold analysis，并标出最终报告采用的 threshold=0.36。

4. `figures/logistic_regression/lr_confusion_matrices.png`
   - 展示不同 LR 版本的 confusion matrix。

5. `figures/logistic_regression/lr_model_comparison_bar.png`
   - 展示不同 LR 版本的 accuracy, precision, recall, F1, F2 对比。

## 3. 推荐用于三模型总对比的 LR 结果

推荐使用：`Novel weighted LR t=0.36`

原因：
- 这是 Logistic Regression 部分的 novelty version。
- 它不是 accuracy 最高，但更符合 churn prediction 的业务目标。
- 它减少 false negatives，提升 recall，更适合找出潜在流失客户。

最终推荐行：
- Accuracy: 0.673773987206823
- Precision: 0.4432576769025367
- Recall: 0.8877005347593583
- F1: 0.591273374888691
- F2: 0.7394209354120267
- Confusion Matrix: TN=616, FP=417, FN=42, TP=332

## 4. 为什么 LR 用 0.36 threshold

默认 Logistic Regression 用 threshold=0.50。
但是 churn prediction 的业务目标不是只追求 accuracy，而是尽量不要漏掉真正会流失的客户。

所以 LR 部分做了：
- class-weighted binary cross-entropy
- validation-based threshold tuning
- 用 F2-score 作为主要选择指标
- 用 business cost = 5 × FN + 1 × FP 作为辅助指标

0.36 是三号 LR handoff experiment 和最终报告确认的 threshold。为了保证 `python run_all.py --full` 重新运行后不会回到后续 cleanup 的 0.31 版本，本仓库代码中将最终 LR threshold 固定为 0.36。test set 只用于最终评估，避免 data leakage。

## 5. 给模型分析同学写 comparison 时可以用的表述

英文：
For Logistic Regression, the final selected variant was the class-weighted and threshold-tuned model. Although it did not achieve the highest accuracy, it produced a much higher recall and fewer false negatives, making it more suitable for churn prediction where missing an actual churn customer is more costly than contacting a non-churn customer.

中文：
Logistic Regression 最终选择的是 class-weighted + threshold-tuned 版本。它不是 accuracy 最高的版本，但 recall 更高、FN 更少，所以更适合客户流失预测这种更重视提前识别风险客户的场景。

## 6. 还缺什么

这个包只包含 Logistic Regression 部分。
最终三模型总对比还需要：
- KNN 的最终 metrics / k-fold CV results / K 值对比
- SVM 的最终 metrics / SVM 参数实验表
- 最终统一的 `results_summary.csv`
- 最终统一的 `cv_results.csv`

如果组里要保持最公平比较，可以让 KNN 和 SVM 也用同一套 test set，并提供同样列名：
model, accuracy, precision, recall, f1, training_time_sec, prediction_time_sec, peak_memory_mb, tn, fp, fn, tp