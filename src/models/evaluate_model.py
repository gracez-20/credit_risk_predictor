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



from sklearn.model_selection import StratifiedKFold

def evaluate_model_performance_cv(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for train_index, val_index in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)

        try:
            y_prob = model.predict_proba(X_val_fold)[:, 1]
        except AttributeError:
            y_prob = None

        metrics = evaluate_model_performance(y_val_fold, y_pred, y_prob)
        results.append(metrics)

    # 对所有折的结果取平均和标准差，并格式化输出为 "mean ± std"
    avg_metrics = {}
    for key in results[0].keys():
        if key == "confusion_matrix":
            avg_metrics[key] = np.mean([m[key] for m in results], axis=0).astype(int)
        else:
            values = [m[key] for m in results]
            mean = float(np.mean(values))
            std = float(np.std(values))
            avg_metrics[key] = f"{mean:.4f} ± {std:.4f}"

    return avg_metrics