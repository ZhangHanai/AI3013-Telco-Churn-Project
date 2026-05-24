# FINAL_CODE_CHECKLIST

## 1) Run Commands
```bash
python -m pip install -r requirements.txt
python run_all.py --demo
python run_all.py --full
```

## 2) Expected Output Files
- `outputs/results_summary.csv`
- `outputs/logistic_regression/lr_metrics.csv`
- `outputs/logistic_regression/lr_cv_summary.csv`
- `outputs/knn/knn_cv_summary.csv`
- `outputs/knn/knn_test_metrics.csv`
- `outputs/svm/svm_experiments.csv`
- `outputs/svm/svm_metrics.csv`
- `outputs/svm/svm_cv_summary.csv`

## 3) What to Show During Code Demo
- `README.md`
- `run_all.py`
- `src/preprocessing.py`
- `src/logistic_regression_scratch.py`
- `src/knn_scratch.py`
- `src/svm_scratch.py`
- `outputs/results_summary.csv`

## 4) Common Q&A Notes
- **Why not only accuracy?**  
  Churn data is class-imbalanced, so accuracy alone can hide poor minority-class detection.
- **Why use recall/F1 for churn?**  
  Missing true churners is costly; recall and F1 better reflect churn-capture performance.
- **Why KNN needs scaling?**  
  KNN uses distance directly; unscaled features can dominate Euclidean distance.
- **Why SVM uses oversampling only on training data?**  
  To improve minority learning while keeping evaluation on untouched test distribution.
- **Why are these models from scratch?**  
  Core training/inference logic is implemented manually with NumPy (no sklearn/TensorFlow/Keras/PyTorch model classes).
- **How is data leakage avoided?**  
  Train/test split is done first; scaling parameters are fitted on training set only and then applied to test set.
