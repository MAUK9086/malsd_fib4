"""EXP-09: FIB-4 Plus — augmented triage rule using top SHAP features (H4)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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

    # FIB-4 alone on same test set
    fib4_idx = available.index("FIB4") if "FIB4" in available else None
    if fib4_idx is not None:
        p_fib4 = X_test["FIB4"].values
    else:
        p_fib4 = np.full(len(X_test), 0.5)

    n_boot = config["bootstrap"]["n_iterations"]
    roc_plus = compute_auroc_ci(y_test.values, p_plus, n_iter=n_boot)
    roc_fib4 = compute_auroc_ci(y_test.values, p_fib4, n_iter=n_boot)

    print(f"FIB-4 AUROC: {roc_fib4['auroc']:.4f} (95% CI: {roc_fib4['ci_lo']:.3f}–{roc_fib4['ci_hi']:.3f})")
    print(f"FIB-4 Plus AUROC: {roc_plus['auroc']:.4f} (95% CI: {roc_plus['ci_lo']:.3f}–{roc_plus['ci_hi']:.3f})")

    # DeLong's test
    z, p_val = delong_test(y_test.values, p_plus, p_fib4)
    print(f"DeLong's test: z={z:.3f}, p={p_val:.4f}")

    auroc_df = pd.DataFrame([
        {"model": "FIB-4", "auroc": roc_fib4["auroc"], "ci_lo": roc_fib4["ci_lo"], "ci_hi": roc_fib4["ci_hi"]},
        {"model": f"FIB-4 Plus ({'+'.join(top_labels)})",
         "auroc": roc_plus["auroc"], "ci_lo": roc_plus["ci_lo"], "ci_hi": roc_plus["ci_hi"],
         "delong_z": z, "delong_p": p_val},
    ])
    auroc_df.to_csv(results_dir / "fib4plus_vs_fib4_auroc.csv", index=False)

    # Optimal cutoff for FIB-4 Plus using Youden's index
    cutoff = youden_cutoff(y_test.values, p_plus)
    (results_dir / "fib4plus_cutoffs.txt").write_text(
        f"FIB-4 Plus optimal cutoff (Youden's index): {cutoff:.4f}\n"
        f"Features: FIB-4 + {', '.join(top_features)}\n"
        f"Logistic regression, standardised inputs\n"
    )
    print(f"FIB-4 Plus optimal cutoff: {cutoff:.4f}")

    # Reclassification table (NRI/IDI)
    p_fib4_norm = (p_fib4 - p_fib4.min()) / (p_fib4.max() - p_fib4.min() + 1e-10)
    nri_idi = compute_nri_idi(y_test.values, p_fib4_norm, p_plus, cutoff=cutoff)
    pd.DataFrame([nri_idi]).to_csv(results_dir / "reclassification_table.csv", index=False)
    print(f"NRI: {nri_idi['nri']:.3f}, IDI: {nri_idi['idi']:.3f}")

    # ROC comparison plot
    plot_roc_curves(
        [
            {"label": "FIB-4", "auroc": roc_fib4["auroc"],
             "fpr": roc_fib4["fpr"], "tpr": roc_fib4["tpr"]},
            {"label": f"FIB-4 Plus", "auroc": roc_plus["auroc"],
             "fpr": roc_plus["fpr"], "tpr": roc_plus["tpr"]},
        ],
        output_path=results_dir / "fib4plus_roc.png",
        title="FIB-4 Plus vs FIB-4 — ROC Comparison",
    )

    # Decision curve analysis
    plot_decision_curve(
        y_test.values,
        {"FIB-4": p_fib4_norm, "FIB-4 Plus": p_plus},
        output_path=results_dir / "decision_curve_analysis.png",
        title="Decision Curve Analysis — FIB-4 Plus vs FIB-4",
    )

    # Save model
    with open(results_dir / "fib4plus_model.pkl", "wb") as f:
        pickle.dump({"model": lr, "scaler": scaler, "features": available}, f)

    print(f"\nEXP-09 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp09(cfg)
