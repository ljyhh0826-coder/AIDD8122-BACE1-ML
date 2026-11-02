"""Evaluation metrics for the classification (active/inactive) and regression (pIC50) tasks."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

PRIMARY = {"cls": "ROC_AUC", "reg": "RMSE"}
HIGHER_IS_BETTER = {"ROC_AUC": True, "PR_AUC": True, "MCC": True, "F1": True, "BACC": True,
                    "RMSE": False, "MAE": False, "R2": True, "Pearson": True, "Spearman": True}


def classification_metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y, prob),
        "PR_AUC": average_precision_score(y, prob),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred),
        "BACC": balanced_accuracy_score(y, pred),
    }


def regression_metrics(y, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": mean_absolute_error(y, pred),
        "R2": r2_score(y, pred),
        "Pearson": pearsonr(y, pred)[0],
        "Spearman": spearmanr(y, pred)[0],
    }


def evaluate(task, y, pred):
    return classification_metrics(y, pred) if task == "cls" else regression_metrics(y, pred)


def enrichment_factor(y, score, fraction=0.05):
    """EF at the top `fraction` of the ranked list (virtual-screening view of a classifier)."""
    n_top = max(1, int(round(fraction * len(y))))
    top = np.argsort(-score)[:n_top]
    return y[top].mean() / y.mean()
