# 4号报告部分写作底稿

## 你负责的正式报告标题

### 5.2 K-Nearest Neighbors from Scratch
#### 5.2.1 Distance Calculation
#### 5.2.2 Majority Voting
#### 5.2.3 Choice of k and Scaling Requirement

### 6.2 KNN Results and Cross-Validation

### 6.4 Model Performance Comparison

### 6.6 Bias-Variance and Suitability Analysis

### 7. Future Work and Conclusion

## 你的代码输出对应关系

- `results/knn_cv_summary.csv`: 写 6.2 的核心表格
- `results/knn_test_metrics.csv`: 写 KNN test result
- `figures/knn_cv_mean_f1_by_k.png`: 展示 k 对 F1 的影响
- `figures/knn_confusion_matrix.png`: 展示 KNN 最终分类错误类型
- `results/model_comparison_template.csv`: 等 3号 LR 和 1/2号 SVM 结果出来后，补齐三模型总对比表

## 解释口径

KNN is a lazy learning method. It does not learn explicit parameters during training. Instead, it stores the training data and classifies a new sample based on the labels of the nearest training samples. Therefore, KNN has very low training cost but relatively high prediction cost.

Because KNN depends directly on distance, feature scaling is necessary. Without scaling, features with larger numerical ranges, such as TotalCharges and MonthlyCharges, would dominate the Euclidean distance.

The value of k controls the bias-variance trade-off. A small k may fit local noise and cause overfitting, while a large k may smooth the decision boundary too much and cause underfitting. Therefore, this project uses cross-validation to compare different k values.
