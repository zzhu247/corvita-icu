"""Metrics, threshold choice, confidence intervals, calibration and missing-vital checks.

No function here fits anything. They only score predictions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)


def _safe_div(a, b):
    return round(float(a / b), 4) if b else None


def choose_threshold(y, p, min_sensitivity: float) -> float:
    """Highest cutoff whose sensitivity on (y, p) is still >= min_sensitivity.

    Higher cutoff = fewer alerts, so among all cutoffs that catch enough deaths,
    take the one that raises the fewest false alarms.
    """
    # drop_intermediate=False keeps every cutoff; the default can skip the exact crossing point.
    _, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    ok = (tpr >= min_sensitivity) & np.isfinite(thr)
    if not ok.any():
        raise ValueError("no threshold reaches the target sensitivity")
    return float(thr[ok].max())


def threshold_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y)
    pred = np.asarray(p) >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "alert_rate": _safe_div(tp + fp, len(y)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def all_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y)
    both = 0 < y.sum() < len(y)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if both else None,
        "auroc": round(float(roc_auc_score(y, p)), 4) if both else None,
        "brier": round(float(brier_score_loss(y, p)), 4),
        "mean_predicted": round(float(np.mean(p)), 4),
        "observed_rate": round(float(y.mean()), 4),
    }
    out.update(threshold_metrics(y, p, thr))
    return out


def bootstrap_ci(y, p, thr: float, reps: int, seed: int) -> dict:
    """95% percentile intervals, resampling patients with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    keys = ["pr_auc", "auroc", "brier", "sensitivity", "specificity"]
    draws = {k: [] for k in keys}
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if not 0 < yb.sum() < len(yb):
            continue
        m = all_metrics(yb, pb, thr)
        for k in keys:
            if m[k] is not None:
                draws[k].append(m[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in draws.items() if v}


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    """Patients grouped by predicted risk (equal-size bins): predicted vs observed death rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    return [{"n": int(len(grp)), "mean_predicted": round(float(grp["p"].mean()), 4),
             "observed_rate": round(float(grp["y"].mean()), 4)} for _, grp in g]


def mask_vitals(X: pd.DataFrame, vitals: list[str]) -> pd.DataFrame:
    """Make the listed vitals look 'never measured', exactly as features.py encodes it."""
    X = X.copy()
    for v in vitals:
        X[f"{v}_count"] = 0
        X[f"{v}_mean"] = np.nan
        X[f"{v}_last"] = np.nan
        X[f"{v}_missing"] = 1
    return X


def subgroup_metrics(X: pd.DataFrame, y, p, thr: float, groups: dict) -> dict:
    """groups: name -> boolean mask over rows of X."""
    y, p = np.asarray(y), np.asarray(p)
    out = {}
    for name, mask in groups.items():
        mask = np.asarray(mask)
        if mask.sum() == 0:
            continue
        out[name] = all_metrics(y[mask], p[mask], thr)
    return out
