"""EXP-09: FIB-4 Plus — augmented triage rule using top SHAP features (H4).

Baselines:
  - NFS within low-risk stratum (clinical next-step recommendation)
  - BMI only (simple anthropometric proxy)
  - FIB-4 on FULL MASLD cohort (reference — shows overall performance separately)
NOTE: FIB-4 within the low-risk stratum (FIB-4 < 1.30) has AUROC ≈ 0.50 by
construction and is NOT a valid comparison. The NFS is the correct comparison.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.stats_utils import (
    compute_auroc_ci,
    delong_test,
    youden_cutoff,
    compute_nri_idi,
    sensitivity_specificity,
)
from src.utils.plot_utils import plot_roc_curves, plot_decision_curve


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Mapping from common SHAP feature names to human readable labels
FEATURE_LABELS = {
    "LBXGH": "HbA1c",
    "LBXSGTSI": "GGT",
    "BMXWAIST": "Waist circumference",
    "BMXBMI": "BMI",
    "LBXTR": "Triglycerides",
    "LBDHDD": "HDL-C",
    "LBXSGL": "Glucose",
    "TYG": "TyG index",
    "DE_RITIS": "AST/ALT ratio",
    "LBXSATSI": "ALT",
    "LBXSASSI": "AST",
    "LBXPLTSI": "Platelets",
    "FIB4": "FIB-4",
    "RIDAGEYR": "Age",
}


def run_exp09(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir_04 = Path(config["paths"]["results"]) / "exp04"
    results_dir = Path(config["paths"]["results"]) / "exp09"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_parquet(data_dir / "features_fib4_low_risk.parquet")
    cohort = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    low_risk_cohort = cohort[cohort["FIB4_CAT"] == 0]

    # Load SHAP ranking for top features
    shap_path = results_dir_04 / "shap_feature_ranking.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        top_features = shap_df["feature"].head(3).tolist()
    else:
        print("SHAP ranking not found, using default expected top features")
        top_features = ["LBXGH", "LBXSGTSI", "BMXWAIST"]  # Expected from SRS

    print(f"EXP-09: FIB-4 Plus features: {top_features}")
    top_labels = [FEATURE_LABELS.get(f, f) for f in top_features]

    # Build combined feature set: FIB-4 + top SHAP features
    all_features = ["FIB4"] + top_features
    available = [f for f in all_features if f in feature_df.columns]
    missing = [f for f in all_features if f not in feature_df.columns]
    if missing:
        print(f"Missing features: {missing} — using available subset")

    X_all = feature_df[available].copy()
    y_all = feature_df["FIB4_FALSE_NEGATIVE"].astype(int)

    # Drop rows with any missing values in these features
    valid_mask = X_all.notna().all(axis=1)
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]

    print(f"N for FIB-4 Plus: {len(X_all):,}, positive rate: {y_all.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # Standardise for logistic regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # FIB-4 Plus model (logistic regression)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    p_plus = lr.predict_proba(X_test_s)[:, 1]

    n_boot = config["bootstrap"]["n_iterations"]
    lsm_thresh = config["thresholds"]["lsm_significant_fibrosis"]

    # --- Correct baselines ---
    # Baseline 1: NFS within the same low-risk test stratum
    nfs_test = feature_df.loc[X_test.index, "NFS"].fillna(
        feature_df["NFS"].median()
    ).values if "NFS" in feature_df.columns else np.full(len(X_test), 0.0)

    # Baseline 2: BMI only
    bmi_test = feature_df.loc[X_test.index, "BMXBMI"].fillna(
        feature_df["BMXBMI"].median()
    ).values if "BMXBMI" in feature_df.columns else np.full(len(X_test), 0.0)

    # Baseline 3: FIB-4 on FULL MASLD cohort (not stratum — for reference context only)
    cohort_full = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    y_full = (cohort_full["LUXSMED"] >= lsm_thresh).astype(int)
    fib4_full_valid = cohort_full.dropna(subset=["FIB4", "LUXSMED"])
    roc_fib4_full = compute_auroc_ci(
        (fib4_full_valid["LUXSMED"] >= lsm_thresh).astype(int).values,
        fib4_full_valid["FIB4"].values,
        n_iter=200,
    )

    # Spearman r of FIB-4 vs LSM within the low-risk stratum (informational)
    if "FIB4" in feature_df.columns:
        fib4_lr = feature_df.loc[X_all.index, "FIB4"].fillna(0)
        lsm_lr = cohort_full.set_index("SEQN").reindex(
            feature_df.loc[X_all.index, "SEQN"] if "SEQN" in feature_df.columns else pd.Index([])
        )["LUXSMED"] if "SEQN" in feature_df.columns else pd.Series(dtype=float)
        if len(lsm_lr) == len(fib4_lr):
            rho, p_rho = spearmanr(fib4_lr.values, lsm_lr.fillna(0).values)
            print(f"FIB-4 vs LSM Spearman within low-risk stratum: r={rho:.3f}, p={p_rho:.4f}")

    roc_plus = compute_auroc_ci(y_test.values, p_plus, n_iter=n_boot)
    roc_nfs = compute_auroc_ci(y_test.values, nfs_test, n_iter=n_boot)
    roc_bmi = compute_auroc_ci(y_test.values, bmi_test, n_iter=n_boot)

    print(f"NFS AUROC (baseline): {roc_nfs['auroc']:.4f} (95% CI: {roc_nfs['ci_lo']:.3f}–{roc_nfs['ci_hi']:.3f})")
    print(f"BMI-only AUROC:       {roc_bmi['auroc']:.4f} (95% CI: {roc_bmi['ci_lo']:.3f}–{roc_bmi['ci_hi']:.3f})")
    print(f"FIB-4 Plus AUROC:     {roc_plus['auroc']:.4f} (95% CI: {roc_plus['ci_lo']:.3f}–{roc_plus['ci_hi']:.3f})")
    print(f"FIB-4 (full cohort):  {roc_fib4_full['auroc']:.4f} [reference only, different denominator]")

    # DeLong's test: FIB-4 Plus vs NFS (the correct primary comparison)
    z, p_val = delong_test(y_test.values, p_plus, nfs_test)
    print(f"DeLong's test (FIB-4 Plus vs NFS): z={z:.3f}, p={p_val:.4f}")

    auroc_df = pd.DataFrame([
        {"model": "NFS (low-risk stratum)",
         "auroc": roc_nfs["auroc"], "ci_lo": roc_nfs["ci_lo"], "ci_hi": roc_nfs["ci_hi"],
         "note": "primary comparison — clinical next-step"},
        {"model": "BMI only (low-risk stratum)",
         "auroc": roc_bmi["auroc"], "ci_lo": roc_bmi["ci_lo"], "ci_hi": roc_bmi["ci_hi"],
         "note": "simple anthropometric baseline"},
        {"model": f"FIB-4 Plus ({'+'.join(top_labels)})",
         "auroc": roc_plus["auroc"], "ci_lo": roc_plus["ci_lo"], "ci_hi": roc_plus["ci_hi"],
         "delong_z_vs_nfs": z, "delong_p_vs_nfs": p_val,
         "note": "primary model"},
        {"model": "FIB-4 (full MASLD cohort, reference only)",
         "auroc": roc_fib4_full["auroc"],
         "ci_lo": roc_fib4_full["ci_lo"], "ci_hi": roc_fib4_full["ci_hi"],
         "note": "different denominator — all FIB-4 ranges, not comparable directly"},
    ])
    auroc_df.to_csv(results_dir / "fib4plus_vs_fib4_auroc.csv", index=False)

    # Optimal cutoff for FIB-4 Plus using Youden's index
    cutoff = youden_cutoff(y_test.values, p_plus)
    (results_dir / "fib4plus_cutoffs.txt").write_text(
        f"FIB-4 Plus optimal cutoff (Youden's index): {cutoff:.4f}\n"
        f"Features: FIB-4 + {', '.join(top_features)}\n"
        f"Logistic regression, standardised inputs\n"
        f"Primary baseline comparison: NFS (DeLong z={z:.3f}, p={p_val:.4f})\n"
    )
    print(f"FIB-4 Plus optimal cutoff: {cutoff:.4f}")

    # Reclassification table (NRI/IDI) vs NFS baseline
    nfs_norm = (nfs_test - nfs_test.min()) / (nfs_test.max() - nfs_test.min() + 1e-10)
    nri_idi = compute_nri_idi(y_test.values, nfs_norm, p_plus, cutoff=cutoff)
    pd.DataFrame([nri_idi]).to_csv(results_dir / "reclassification_table.csv", index=False)
    print(f"NRI vs NFS: {nri_idi['nri']:.3f}, IDI: {nri_idi['idi']:.3f}")

    # ROC comparison plot (3 curves)
    plot_roc_curves(
        [
            {"label": "NFS (baseline)",
             "auroc": roc_nfs["auroc"], "fpr": roc_nfs["fpr"], "tpr": roc_nfs["tpr"]},
            {"label": "BMI only",
             "auroc": roc_bmi["auroc"], "fpr": roc_bmi["fpr"], "tpr": roc_bmi["tpr"]},
            {"label": "FIB-4 Plus",
             "auroc": roc_plus["auroc"], "fpr": roc_plus["fpr"], "tpr": roc_plus["tpr"]},
        ],
        output_path=results_dir / "fib4plus_roc.png",
        title="FIB-4 Plus vs Baselines — ROC Comparison (within FIB-4 low-risk stratum)",
    )

    # Decision curve analysis
    plot_decision_curve(
        y_test.values,
        {"NFS": nfs_norm, "FIB-4 Plus": p_plus},
        output_path=results_dir / "decision_curve_analysis.png",
        title="Decision Curve Analysis — FIB-4 Plus vs NFS",
    )

    # Save model
    with open(results_dir / "fib4plus_model.pkl", "wb") as f:
        pickle.dump({"model": lr, "scaler": scaler, "features": available}, f)

    print(f"\nEXP-09 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp09(cfg)
