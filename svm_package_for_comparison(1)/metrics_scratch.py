from __future__ import annotations

import numpy as np


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    denom = cm["tp"] + cm["fp"]
    return float(cm["tp"] / denom) if denom else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    denom = cm["tp"] + cm["fn"]
    return float(cm["tp"] / denom) if denom else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom else 0.0


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    tpr_denom = cm["tp"] + cm["fn"]
    tnr_denom = cm["tn"] + cm["fp"]
    tpr = cm["tp"] / tpr_denom if tpr_denom else 0.0
    tnr = cm["tn"] / tnr_denom if tnr_denom else 0.0
    return float((tpr + tnr) / 2)


def true_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    denom = cm["tn"] + cm["fp"]
    return float(cm["tn"] / denom) if denom else 0.0

