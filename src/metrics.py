"""Binary classification metrics + run summarisation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out = {"n": int(len(y_true)), "pos_rate": float(y_true.mean()) if len(y_true) else float("nan")}
    if len(y_true) and y_true.min() != y_true.max():
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0
    return out


def summarize(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if k not in ("n", "pos_rate")]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))
    return out
