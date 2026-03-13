"""
Evaluation functions for regression and classification models.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate regression R² score.
    """
    return float(r2_score(y_true, y_pred))


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Returns
    -------
    dict
        accuracy, precision, recall, f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }


def calculate_auroc_score(y_true, y_scores):
    """
    Calculate AUROC.
    """
    return float(roc_auc_score(y_true, y_scores))


def calculate_auprc_score(y_true, y_scores):
    """
    Calculate average precision score.
    """
    return float(average_precision_score(y_true, y_scores))


def generate_auroc_curve(y_true, y_scores, title="AUROC Curve"):
    """
    Generate ROC curve figure.

    Returns
    -------
    matplotlib.figure.Figure
        ROC curve figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    return fig


def generate_auprc_curve(y_true, y_scores, title="AUPRC Curve"):
    """
    Generate precision-recall curve figure.

    Returns
    -------
    matplotlib.figure.Figure
        PR curve figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_value = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap_value:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    plt.tight_layout()

    return fig
