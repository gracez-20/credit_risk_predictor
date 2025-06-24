from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import numpy as np

def evaluate_model_performance(y_true, y_pred, y_prob=None):
    positive_label = 1  
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cost_matrix = np.array([[0, 1],  # Actual Good → Pred Good, Pred Bad
                            [5, 0]]) # Actual Bad  → Pred Good, Pred Bad
    total_cost = np.sum(cm * cost_matrix)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label),
        "recall": recall_score(y_true, y_pred, pos_label=positive_label),
        "f1": f1_score(y_true, y_pred, pos_label=positive_label),
        "confusion_matrix": cm,
        "total_cost": total_cost,
    }

    if y_prob is not None:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    return metrics