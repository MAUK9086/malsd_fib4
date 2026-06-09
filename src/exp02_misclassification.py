"""EXP-02: FIB-4 misclassification rate, survey-weighted, with subgroup stratification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.utils.stats_utils import (
    bootstrap_ci,
    compute_auroc_ci,
    sensitivity_specificity,
    weighted_proportion,
)
from src.utils.plot_utils import plot_roc_curves
from src.utils.nhanes_codebook import RACE_LABELS


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def misclassification_matrix(df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Weighted 2×3 matrix: FIB4_CAT (rows) × LSM_CAT (cols)."""
    rows = []
    for fib4_cat in [0, 1, 2]:
        row = {"FIB4_CAT": fib4_cat}
        for lsm_cat in [0, 1, 2]:
            mask = (df["FIB4_CAT"] == fib4_cat) & (df["LSM_CAT"] == lsm_cat)
            w_sum = weights[mask].sum()
            row[f"LSM_CAT_{lsm_cat}"] = float(w_sum)
        rows.append(row)
    mat = pd.DataFrame(rows).set_index("FIB4_CAT")
    # Convert to proportions of row totals
    mat_pct = mat.div(mat.sum(axis=1), axis=0) * 100
    return mat_pct


def subgroup_false_negative_rate(
    df: pd.DataFrame,
    weights: pd.Series,
    stratify_col: str,
) -> pd.DataFrame:
    """Weighted FN rate within FIB-4 low-risk stratum, by subgroup."""
    low_risk = df[df["FIB4_CAT"] == 0].copy()
    low_w = weights[low_risk.index]

    rows = []
    for val, grp in low_risk.groupby(stratify_col, dropna=True):
        grp_w = low_w[grp.index]
        fn_rate, fn_n = weighted_proportion(grp["FIB4_FALSE_NEGATIVE"] == 1, grp_w)
        total_n = len(grp)
        rows.append({
            stratify_col: val,
            "n_total": total_n,
            "n_false_negative": fn_n,
            "weighted_fn_rate_pct": fn_rate * 100,
        })
    return pd.DataFrame(rows)


def run_exp02(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir = Path(config["paths"]["results"]) / "exp02"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    weights = df["WTMECPRP"].fillna(df["WTMECPRP"].median())
    lsm_thresh = config["thresholds"]["lsm_significant_fibrosis"]
    n_boot = config["bootstrap"]["n_iterations"]

    print(f"EXP-02: N={len(df):,}, LSM threshold={lsm_thresh} kPa")

    # --- Weighted misclassification matrix ---
    mat = misclassification_matrix(df, weights)
    mat.to_csv(results_dir / "misclassification_matrix_weighted.csv")
    print("Misclassification matrix (weighted %):")
    print(mat.round(1))

    # --- Core FN rate in low-risk stratum ---
    low_risk = df[df["FIB4_CAT"] == 0]
    low_w = weights[low_risk.index]
    fn_rate, fn_n = weighted_proportion(low_risk["FIB4_FALSE_NEGATIVE"] == 1, low_w)
    fn_ci = bootstrap_ci(
        low_risk.assign(_w=low_w),
        lambda d: weighted_proportion(d["FIB4_FALSE_NEGATIVE"] == 1, d["_w"])[0],
        n_iter=n_boot,
    )
    print(f"\nFIB-4 False Negative Rate (low-risk stratum): {fn_rate*100:.1f}% "
          f"(95% CI: {fn_ci[0]*100:.1f}–{fn_ci[1]*100:.1f}%)")
    print(f"Unweighted FN count: {fn_n:,} / {len(low_risk):,}")

    # US population extrapolation (NHANES MEC weight represents US adults)
    total_us_weight = weights.sum()
    low_risk_us_est = low_w.sum()
    fn_us_est = (low_risk["FIB4_FALSE_NEGATIVE"] == 1).multiply(low_w).sum()
    pop_text = (
        f"Population Extrapolation\n"
        f"========================\n"
        f"Survey-weighted MASLD cohort represents: {low_risk_us_est/1e6:.1f}M US adults (FIB-4 low-risk)\n"
        f"Estimated false negatives: {fn_us_est/1e6:.1f}M US adults\n"
        f"False negative rate: {fn_rate*100:.1f}% (95% CI: {fn_ci[0]*100:.1f}–{fn_ci[1]*100:.1f}%)\n"
    )
    (results_dir / "population_extrapolation.txt").write_text(pop_text)
    print(pop_text)

    # --- ROC: FIB-4 vs LSM ≥ 8 kPa ---
    # Use only patients where FIB-4 is meaningful (not indeterminate zone)
    valid = df.dropna(subset=["FIB4", "LUXSMED"])
    y_true = (valid["LUXSMED"] >= lsm_thresh).astype(int).values
    y_fib4 = valid["FIB4"].values

    roc_fib4 = compute_auroc_ci(y_true, y_fib4, n_iter=n_boot)
    roc_nfs = compute_auroc_ci(y_true, valid["NFS"].fillna(valid["NFS"].median()).values, n_iter=n_boot)
    roc_apri = compute_auroc_ci(y_true, valid["APRI"].fillna(valid["APRI"].median()).values, n_iter=n_boot)

    plot_roc_curves(
        [
            {"label": f"FIB-4", "auroc": roc_fib4["auroc"], "fpr": roc_fib4["fpr"], "tpr": roc_fib4["tpr"]},
            {"label": f"NFS", "auroc": roc_nfs["auroc"], "fpr": roc_nfs["fpr"], "tpr": roc_nfs["tpr"]},
            {"label": f"APRI", "auroc": roc_apri["auroc"], "fpr": roc_apri["fpr"], "tpr": roc_apri["tpr"]},
        ],
        output_path=results_dir / "roc_curve_fib4.png",
        title="FIB-4 vs LSM ≥ 8 kPa — ROC Curves",
    )

    # --- Sensitivity/specificity ---
    fib4_binary = (valid["FIB4_CAT"] == 2).astype(int).values  # high-risk as positive
    ss = sensitivity_specificity(y_true, fib4_binary)
    print(f"\nFIB-4 (high-risk cutoff >2.67) vs LSM≥8 kPa:")
    print(f"  Sensitivity: {ss['sensitivity']:.3f}, Specificity: {ss['specificity']:.3f}")
    print(f"  PPV: {ss['ppv']:.3f}, NPV: {ss['npv']:.3f}")

    # --- Subgroup stratifications ---
    subgroup_results = []

    # Age groups
    df["AGE_GROUP"] = pd.cut(df["RIDAGEYR"], bins=[0, 45, 60, 120], labels=["18-45", "46-60", "61+"])
    for col, label in [
        ("RIAGENDR", "Sex"),
        ("AGE_GROUP", "Age Group"),
        ("RIDRETH3", "Race/Ethnicity"),
    ]:
        sg = subgroup_false_negative_rate(df, weights, col)
        sg["subgroup_variable"] = label
        subgroup_results.append(sg.rename(columns={col: "subgroup_value"}))

    pd.concat(subgroup_results, ignore_index=True).to_csv(
        results_dir / "sensitivity_specificity_by_subgroup.csv", index=False
    )

    print(f"\nEXP-02 results saved to {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp02(cfg)
