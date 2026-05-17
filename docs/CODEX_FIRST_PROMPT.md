# Suggested Codex Prompt

I am working on an AI3013 Machine Learning course project about Telco Customer Churn Prediction.

Please read these context files first:

- `docs/AI3013_COURSE_REQUIREMENTS_SUMMARY.md`
- `docs/PROJECT_CONTEXT_FOR_CODEX.md`

Please audit this repository only. Do not rewrite the whole project yet.

Course constraints:
1. The final models must be implemented from scratch.
2. Do not use scikit-learn models for KNN, Logistic Regression, or Linear SVM.
3. Basic libraries such as NumPy, Pandas, and Matplotlib are allowed.
4. The final code should be reproducible from a clean clone.
5. The final report will compare Logistic Regression, KNN, and Linear SVM using accuracy, precision, recall, F1-score, confusion matrix, cross-validation, training time, prediction time, and memory usage.

Current repository status:
- This repo currently contains Member 4's KNN part.
- The KNN implementation is in `src/knn_scratch.py`.
- Metrics are in `src/metrics.py`.
- Cross-validation is in `src/cross_validation.py`.
- Preprocessing is in `src/preprocessing.py`.
- The main runnable scripts are `run_fast_test.py` and `run_all.py`.

Please check the following:
1. Whether the folder structure is reasonable for a course project.
2. Whether all file paths are portable and do not depend on my local computer.
3. Whether `README.md` is clear enough for a teacher or teammate to run the code.
4. Whether `requirements.txt` is complete.
5. Whether `run_fast_test.py` and `run_all.py` can run from the repository root.
6. Whether the KNN code is truly from scratch and does not violate the course rule.
7. Whether the generated outputs in `results/` and `figures/` are suitable for the final report.
8. Whether there are any obvious bugs, redundant files, or unclear naming problems.

Please give me an audit report first. If changes are needed, list them clearly and explain why. Do not modify files until I approve.
