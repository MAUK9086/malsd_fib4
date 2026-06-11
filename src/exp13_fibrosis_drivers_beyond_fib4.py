"""EXP-13: FIB-4 Blindspot Analysis — what FIB-4 cannot see.

Produces:
- Univariate AUROC for all non-FIB-4 variables
- FIB-4 Blindspot Score (0–4): BMI, HbA1c, waist, GGT
- Forest-plot-style figure
- Comparison: Blindspot Score vs FIB-4 Plus vs XGBoost
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.utils.stats_utils import compute_auroc_ci


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


FIB4_FORMULA_VARS = {"RIDAGEYR", "LBXSASSI", "LBXSATSI", "LBXPLTSI"}

FEATURE_LABELS = {
    "LBXGH": "HbA1c", "LBXSGTSI": "GGT", "BMXWAIST": "Waist circ.",
    "BMXBMI": "BMI", "LBXTR": "Triglycerides", "LBDHDD": "HDL-C",
    "LBXSGL": "Glucose", "DE_RITIS": "AST/ALT ratio", "LBXSATSI": "ALT",
    "LBXSASSI": "AST", "LBXPLTSI": "Platelets", "FIB4": "FIB-4",
    "RIDAGEYR": "Age", "LBXSAL": "Albumin", "LBXHGB": "Haemoglobin",
    "LBXWBCSI": "WBC", "SII": "SII", "LBXSCR": "Creatinine",
    "LBXTC": "Total cholesterol", "LBXMCVSI": "MCV", "LBXRDW": "RDW",
    "WHTR": "WHtR", "TYG": "TyG index", "NFS": "NFS", "APRI": "APRI",
}


def compute_univariate_auroc(X: pd.DataFrame, y: pd.Series, feature: str, n_boot: int = 200) -> dict:
    col = X[feature].dropna()
    common = col.index.intersection(y.index)
    if len(common) < 30:
        return {"feature": feature, "auroc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    result = compute_auroc_ci(y.loc[common].values, col.loc[common].values, n_iter=n_boot)
    return {
        "feature": feature,
        "label": FEATURE_LABELS.get(feature, feature),
        "auroc": result["auroc"],
        "ci_lo": result["ci_lo"],
        "ci_hi": result["ci_hi"],
        "in_fib4_formula": feature in FIB4_FORMULA_VARS,
    }


def compute_blindspot_score(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    FIB-4 Blindspot Score (0–4):
      +1 if BMI >= 30
      +1 if HbA1c >= 5.7 (prediabetes)
      +1 if waist >= 102 cm (men) / 88 cm (women)
      +1 if GGT > 36 (men) / > 25 (women)
    """
    t = config["thresholds"]
    score = pd.Series(0, index=df.index, dtype=int)

    if "BMXBMI" in df.columns:
        score += (df["BMXBMI"] >= 30).astype(int)

    if "LBXGH" in df.columns:
        score += (df["LBXGH"] >= 5.7).astype(int)

    if "BMXWAIST" in df.columns and "RIAGENDR" in df.columns:
        male = df["RIAGENDR"] == 1
        waist_thresh = np.where(male, t.get("waist_western_men", 102), t.get("waist_western_women", 88))
        score += (df["BMXWAIST"] >= waist_thresh).astype(int)

    if "LBXSGTSI" in df.columns and "RIAGENDR" in df.columns:
        male = df["RIAGENDR"] == 1
        ggt_thresh = np.where(male, t.get("ggt_elevated_men", 36), t.get("ggt_elevated_women", 25))
        score += (df["LBXSGTSI"] > ggt_thresh).astype(int)

    return score


def run_exp13(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir_09 = Path(config["paths"]["results"]) / "exp09"
    results_dir_11 = Path(config["paths"]["results"]) / "exp11"
    results_dir = Path(config["paths"]["results"]) / "exp13"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_parquet(data_dir / "features_fib4_low_risk.parquet")
    cohort = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")

    # Merge sex info from cohort (needed for waist / GGT thresholds)
    if "RIAGENDR" not in feature_df.columns and "SEQN" in feature_df.columns:
        sex_col = cohort[["SEQN", "RIAGENDR"]].drop_duplicates()
        feature_df = feature_df.merge(sex_col, on="SEQN", how="left")

    y = feature_df["FIB4_FALSE_NEGATIVE"].astype(int)
    n_boot = min(config["bootstrap"]["n_iterations"], 500)

    # --- Determine NON_FIB4_VARS from EXP-11 SHAP if available ---
    shap_path = results_dir_11 / "shap_raw_feature_ranking.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        top10 = shap_df["feature"].head(10).tolist()
        non_fib4_vars = [f for f in top10 if f not in FIB4_FORMULA_VARS][:10]
    else:
        # Sensible defaults (expected EXP-11 output based on SRS)
        non_fib4_vars = ["LBXGH", "LBXSGTSI", "BMXWAIST", "BMXBMI",
                         "LBXTR", "LBDHDD", "LBXSGL", "DE_RITIS", "SII", "LBXHGB"]

    all_vars = list(FIB4_FORMULA_VARS) + [v for v in non_fib4_vars if v not in FIB4_FORMULA_VARS]
    all_vars = [v for v in all_vars if v in feature_df.columns]

    # --- Univariate AUROCs ---
    print("Computing univariate AUROCs...")
    uni_records = []
    for feat in all_vars:
        rec = compute_univariate_auroc(feature_df, y, feat, n_boot=n_boot)
        uni_records.append(rec)

    uni_df = pd.DataFrame(uni_records).dropna(subset=["auroc"]).sort_values("auroc", ascending=False)
    uni_df.to_csv(results_dir / "univariate_auroc_table.csv", index=False)
    print(f"Top-5 univariate predictors:\n{uni_df.head(5)[['label','auroc','in_fib4_formula']].to_string(index=False)}")

    # Non-FIB-4 formula vars ranked
    non_fib4_ranked = uni_df[~uni_df["in_fib4_formula"]].reset_index(drop=True)
    non_fib4_ranked.to_csv(results_dir / "non_fib4_predictors_ranked.csv", index=False)

    # --- Spearman r vs LSM in full cohort ---
    lsm_thresh = config["thresholds"]["lsm_significant_fibrosis"]
    full_valid = cohort.dropna(subset=["LUXSMED"])
    y_full = (full_valid["LUXSMED"] >= lsm_thresh).astype(int)
    spearman_rows = []
    for feat in all_vars:
        if feat in full_valid.columns:
            col = full_valid[feat].dropna()
            common = col.index.intersection(y_full.index)
            if len(common) > 30:
                rho, p = spearmanr(full_valid.loc[common, feat].values, y_full.loc[common].values)
                spearman_rows.append({"feature": feat, "label": FEATURE_LABELS.get(feat, feat),
                                      "spearman_r": round(rho, 3), "p_value": round(p, 4)})
    if spearman_rows:
        pd.DataFrame(spearman_rows).to_csv(results_dir / "spearman_vs_lsm.csv", index=False)

    # --- FIB-4 Blindspot Score ---
    feature_df_with_sex = feature_df.copy()
    bs_score = compute_blindspot_score(feature_df_with_sex, config)
    feature_df_with_sex["BLINDSPOT_SCORE"] = bs_score

    valid_bs = feature_df_with_sex["BLINDSPOT_SCORE"].notna()
    roc_bs = compute_auroc_ci(y[valid_bs].values, bs_score[valid_bs].values, n_iter=n_boot)
    print(f"Blindspot Score AUROC: {roc_bs['auroc']:.4f} "
          f"(95% CI: {roc_bs['ci_lo']:.3f}–{roc_bs['ci_hi']:.3f})")

    # Compare against FIB-4 Plus and XGBoost
    auroc_rows = [
        {"model": "FIB-4 Blindspot Score (0–4)",
         "auroc": roc_bs["auroc"], "ci_lo": roc_bs["ci_lo"], "ci_hi": roc_bs["ci_hi"],
         "note": "BMI≥30 + HbA1c≥5.7 + waist≥threshold + GGT>threshold"},
    ]

    fib4plus_path = results_dir_09 / "fib4plus_vs_fib4_auroc.csv"
    if fib4plus_path.exists():
        try:
            f9 = pd.read_csv(fib4plus_path)
            plus_row = f9[f9["model"].str.contains("Plus")]
            if len(plus_row) > 0:
                auroc_rows.append({
                    "model": "FIB-4 Plus (logistic regression)",
                    "auroc": plus_row["auroc"].iloc[0],
                    "ci_lo": plus_row["ci_lo"].iloc[0],
                    "ci_hi": plus_row["ci_hi"].iloc[0],
                    "note": "EXP-09",
                })
        except Exception:
            pass

    pd.DataFrame(auroc_rows).to_csv(results_dir / "blindspot_score_auroc.csv", index=False)

    # --- Forest-plot figure ---
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        plot_df = uni_df.dropna(subset=["ci_lo", "ci_hi"]).head(20).copy()
        plot_df = plot_df.sort_values("auroc")

        fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.4)))
        colors = ["#d62728" if r else "#1f77b4" for r in plot_df["in_fib4_formula"]]
        y_pos = np.arange(len(plot_df))

        ax.barh(y_pos, plot_df["auroc"] - 0.5,
                xerr=[plot_df["auroc"] - plot_df["ci_lo"],
                      plot_df["ci_hi"] - plot_df["auroc"]],
                left=0.5, color=colors, alpha=0.7, capsize=3, height=0.6)
        ax.axvline(0.5, color="grey", linewidth=1, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["label"].tolist(), fontsize=9)
        ax.set_xlabel("AUROC for FIB-4 False Negative")
        ax.set_title("Univariate AUROC — FIB-4 Formula Vars vs Others")
        in_fib4_patch = mpatches.Patch(color="#d62728", alpha=0.7, label="In FIB-4 formula")
        not_fib4_patch = mpatches.Patch(color="#1f77b4", alpha=0.7, label="NOT in FIB-4 formula")
        ax.legend(handles=[in_fib4_patch, not_fib4_patch], loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(results_dir / "fib4_blindspot_figure.png", dpi=150)
        plt.close(fig)
        print("Saved fib4_blindspot_figure.png")
    except Exception as e:
        print(f"Forest plot failed: {e}")

    print(f"\nEXP-13 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp13(cfg)
