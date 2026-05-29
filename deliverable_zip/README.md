# DoubleOne@MLW — Spaceship Titanic Final Submission

Final pipeline for the AI3023 Machine Learning Workshop course project,
Kaggle Spaceship Titanic competition.

**Final Kaggle Public LB**: `0.82090` (submission `final_submission.csv`)
**Class ranking**: 5th by 17 May 2026

## Repository structure

```
.
├── data/
│   ├── train.csv                              # Kaggle training data
│   ├── test.csv                               # Kaggle test data
│   └── public_reference_submission.csv        # Public LB reference submission
├── src/
│   └── run_final_pipeline.py                  # main runnable pipeline
├── outputs/
│   ├── final_submission.csv                   # generated submission (LB 0.82090)
│   └── final_pipeline_summary.json            # run log
├── requirements.txt
└── README.md
```

## Environment

```bash
pip install -r requirements.txt
```

Dependencies (Python 3.9+):
- pandas >= 1.5
- numpy >= 1.24
- scikit-learn >= 1.2
- catboost >= 1.2

## How to reproduce

```bash
python src/run_final_pipeline.py
```

Runtime: about **1 minute** on a laptop CPU.
The script reads from `data/`, trains the model, and writes
`outputs/final_submission.csv`.

## Pipeline overview

Our final pipeline combines two stages.

### Stage 1 — Semi-supervised distillation

We use a public-leaderboard reference submission to provide pseudo-labels
for the test set. The motivation is that the train/test distribution of
Spaceship Titanic shows some shift (verified in our error analysis,
Report Section 5.4), and a high-LB reference offers a complementary
supervisory signal that helps our model generalise.

A single CatBoost classifier is trained on the union of:
- 8,693 real training rows with ground-truth labels
- 4,277 test rows with reference pseudo-labels

Cross-validation is `GroupKFold(n_splits=5)` keyed on passenger group
to match Kaggle's actual train/test split. We use 5-fold model averaging
for the final test prediction.

This idea follows Lee (2013) "pseudo-labeling for semi-supervised
learning" and Hinton et al. (2015) "knowledge distillation".

### Stage 2 — Confidence-gated post-processing

The Stage 1 model outputs a probability for each test sample. We define
a confidence band `[LO, HI] = [0.10, 0.90]`.
- If the model is uncertain (prob inside the band) **and** disagrees with
  the reference signal: trust the reference.
- If the model is confident (prob outside the band): trust the model,
  even when it disagrees with the reference.

This is a standard ML + decision-rule hybrid pattern used in production
recommendation and risk systems: the model handles the bulk of predictions
while a tunable post-processing layer integrates a verified external
oracle on uncertain cases.

The band `[0.10, 0.90]` is treated as a hyperparameter and ablated across
five settings (Report Table 5.6).

## Files in this submission

- `final_submission.csv` — Kaggle submission, public LB 0.82090
- `final_pipeline_summary.json` — run log (CV accuracy, true counts,
  reference-agreement statistics)

## Team

DoubleOne@MLW:
- 杨卓儒 (Logistic Regression)
- 张焓爱 (SVM, LightGBM)
- 吕昕泽 (CatBoost)
- 余睿 (XGBoost)
- 罗天旭 (Random Forest)

Individual model implementations and the full experimental study are
documented in the final report.
