import numpy as np


def confusion_matrix_binary(y_true, y_pred):
    """Return TP, TN, FP, FN for binary classification.

    Positive class: 1
    Negative class: 0
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        raise ValueError("y_true is empty.")

    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)
    tp = cm["TP"]
    fp = cm["FP"]

    if tp + fp == 0:
        return 0.0

    return float(tp / (tp + fp))


def recall_score(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)
    tp = cm["TP"]
    fn = cm["FN"]

    if tp + fn == 0:
        return 0.0

    return float(tp / (tp + fn))


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def classification_report_binary(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "TP": cm["TP"],
        "TN": cm["TN"],
        "FP": cm["FP"],
        "FN": cm["FN"],
    }


def print_classification_report(y_true, y_pred, model_name="Model"):
    report = classification_report_binary(y_true, y_pred)

    print(f"{model_name} evaluation results")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall   : {report['recall']:.4f}")
    print(f"F1-score : {report['f1']:.4f}")
    print("Confusion Matrix:")
    print(f"TP={report['TP']}, FP={report['FP']}")
    print(f"FN={report['FN']}, TN={report['TN']}")

    return report
