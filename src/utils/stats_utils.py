"""Survey-weighted statistics, bootstrap CI, DeLong AUROC test, NRI/IDI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Survey-weighted summary statistics
# ---------------------------------------------------------------------------

def weighted_proportion(
    mask: pd.Series,
    weights: pd.Series,
) -> tuple[float, int]:
    """Return (weighted proportion, unweighted count) for boolean mask."""
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0
    prop = (mask * weights).sum() / weights.sum()
    return float(prop), n


def weighted_mean_std(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[float, float]:
    valid = values.notna()
    v = values[valid]
    w = weights[valid]
    mean = float((v * w).sum() / w.sum())
    var = float(((v - mean) ** 2 * w).sum() / w.sum())
    return mean, float(np.sqrt(var))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: pd.DataFrame,
    stat_fn,
    n_iter: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap CI for any statistic computed from a DataFrame.
    stat_fn(df) -> float
    """
    rng = np.random.default_rng(random_state)
    stats_boot = []
    n = len(data)
    for _ in range(n_iter):
        sample = data.iloc[rng.integers(0, n, size=n)]
        try:
            stats_boot.append(stat_fn(sample))
        except Exception:
            continue
    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(stats_boot, alpha * 100))
    hi = float(np.nanpercentile(stats_boot, (1 - alpha) * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# ROC / AUROC utilities
# ---------------------------------------------------------------------------

def compute_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iter: int = 1000,
    random_state: int = 42,
) -> dict:
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    rng = np.random.default_rng(random_state)
    boot_aurocs = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
    ci_lo = float(np.nanpercentile(boot_aurocs, 2.5))
    ci_hi = float(np.nanpercentile(boot_aurocs, 97.5))
    return {
        "auroc": float(auroc),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> tuple[float, float]:
    """
    DeLong's test for comparing two AUROCs on the same dataset.
    Returns (z_statistic, p_value).
    Simplified implementation via bootstrap variance.
    """
    auc_a = roc_auc_score(y_true, y_score_a)
    auc_b = roc_auc_score(y_true, y_score_b)
    rng = np.random.default_rng(42)
    n = len(y_true)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        da = roc_auc_score(yt, y_score_a[idx]) - roc_auc_score(yt, y_score_b[idx])
        diffs.append(da)
    se = float(np.std(diffs))
    z = (auc_a - auc_b) / (se + 1e-10)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return z, p


# ---------------------------------------------------------------------------
# Youden's index optimal cutoff
# ---------------------------------------------------------------------------

def youden_cutoff(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ---------------------------------------------------------------------------
# Net Reclassification Improvement (NRI) & IDI
# ---------------------------------------------------------------------------

def compute_nri_idi(
    y_true: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    cutoff: float = 0.5,
) -> dict:
    """Categorical NRI and IDI between old and new probability scores."""
    events = y_true == 1
    non_events = y_true == 0

    # Categorical NRI
    up_events = ((p_new > cutoff) & (p_old <= cutoff) & events).sum()
    down_events = ((p_new <= cutoff) & (p_old > cutoff) & events).sum()
    up_non = ((p_new > cutoff) & (p_old <= cutoff) & non_events).sum()
    down_non = ((p_new <= cutoff) & (p_old > cutoff) & non_events).sum()

    nri_events = (up_events - down_events) / events.sum() if events.sum() > 0 else 0.0
    nri_non = (down_non - up_non) / non_events.sum() if non_events.sum() > 0 else 0.0
    nri = float(nri_events + nri_non)

    # IDI
    idi = float(
        (p_new[events].mean() - p_old[events].mean())
        - (p_new[non_events].mean() - p_old[non_events].mean())
    )

    return {
        "nri": nri,
        "nri_events": float(nri_events),
        "nri_non_events": float(nri_non),
        "idi": idi,
    }


# ---------------------------------------------------------------------------
# Sensitivity / Specificity table
# ---------------------------------------------------------------------------

def sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }
