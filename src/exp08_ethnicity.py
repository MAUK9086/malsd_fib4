"""EXP-08: Ethnicity-stratified subgroup analysis — Asian American focus (H3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.utils.stats_utils import weighted_proportion, bootstrap_ci
from src.utils.plot_utils import plot_shap_summary
from src.utils.nhanes_codebook import RACE_LABELS, MODEL_FEATURES


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fisher_or(
    n_fn_group: int, n_total_group: int,
    n_fn_other: int, n_total_other: int,
) -> dict:
    """Fisher's exact test and odds ratio."""
    table = [
        [n_fn_group, n_total_group - n_fn_group],
        [n_fn_other, n_total_other - n_fn_other],
    ]
    oddsratio, pvalue = stats.fisher_exact(table)
    # Confidence interval via log-odds method
    log_or = np.log(oddsratio + 1e-10)
    n11, n12, n21, n22 = table[0][0], table[0][1], table[1][0], table[1][1]
    se = np.sqrt(1/(n11+1e-5) + 1/(n12+1e-5) + 1/(n21+1e-5) + 1/(n22+1e-5))
    ci_lo = np.exp(log_or - 1.96 * se)
    ci_hi = np.exp(log_or + 1.96 * se)
    return {
        "odds_ratio": float(oddsratio),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(pvalue),
    }


def run_exp08(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir = Path(config["paths"]["results"]) / "exp08"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    weights = df["WTMECPRP"].fillna(df["WTMECPRP"].median())
    lsm_thresh = config["thresholds"]["lsm_significant_fibrosis"]

    low_risk = df[df["FIB4_CAT"] == 0].copy()
    low_w = weights[low_risk.index]

    print(f"EXP-08: FIB-4 low-risk stratum N={len(low_risk):,}")

    # --- FN rates by race/ethnicity ---
    eth_records = []
    for code, label in RACE_LABELS.items():
        grp = low_risk[low_risk["RIDRETH3"] == code]
        if len(grp) < 10:
            continue
        grp_w = low_w[grp.index]
        fn_rate, fn_n = weighted_proportion(grp["FIB4_FALSE_NEGATIVE"] == 1, grp_w)
        eth_records.append({
            "race_code": code,
            "race_label": label,
            "n_low_risk": len(grp),
            "n_false_negative": fn_n,
            "weighted_fn_rate_pct": fn_rate * 100,
        })

    eth_df = pd.DataFrame(eth_records).sort_values("weighted_fn_rate_pct", ascending=False)
    eth_df.to_csv(results_dir / "false_negative_rates_by_ethnicity.csv", index=False)
    print("\nFalse negative rates by race/ethnicity:")
    print(eth_df.to_string(index=False))

    # --- Asian vs all others: Fisher's exact + OR ---
    asian_mask = low_risk["RIDRETH3"] == 6
    other_mask = low_risk["RIDRETH3"] != 6

    n_asian_fn = int((low_risk.loc[asian_mask, "FIB4_FALSE_NEGATIVE"] == 1).sum())
    n_asian_total = int(asian_mask.sum())
    n_other_fn = int((low_risk.loc[other_mask, "FIB4_FALSE_NEGATIVE"] == 1).sum())
    n_other_total = int(other_mask.sum())

    if n_asian_total > 0 and n_other_total > 0:
        or_result = fisher_or(n_asian_fn, n_asian_total, n_other_fn, n_other_total)
        or_row = pd.DataFrame([{
            "comparison": "Asian vs Non-Asian",
            "n_asian": n_asian_total,
            "n_asian_fn": n_asian_fn,
            "asian_fn_rate_pct": n_asian_fn / n_asian_total * 100,
            "n_other": n_other_total,
            "n_other_fn": n_other_fn,
            "other_fn_rate_pct": n_other_fn / n_other_total * 100,
            **or_result,
        }])
        or_row.to_csv(results_dir / "asian_vs_others_odds_ratio.csv", index=False)
        print(f"\nAsian vs Others:")
        print(f"  OR={or_result['odds_ratio']:.2f} "
              f"(95% CI: {or_result['ci_lo']:.2f}–{or_result['ci_hi']:.2f}), "
              f"p={or_result['p_value']:.4f}")

    # --- Asian BMI ≥ 23 re-analysis ---
    # Compare MASLD prevalence / FN rates with standard vs Asian BMI cutoff
    asian_cohort = df[df["RIDRETH3"] == 6].copy()
    asian_w = weights[asian_cohort.index]

    # Standard MASLD (BMI ≥ 25)
    asian_std_masld = asian_cohort[(asian_cohort["BMXBMI"] >= 25) & (asian_cohort["LUXCAPM"] >= config["thresholds"]["cap_steatosis"])]
    # Asian-specific MASLD (BMI ≥ 23)
    asian_specific_masld = asian_cohort[(asian_cohort["BMXBMI"] >= 23) & (asian_cohort["LUXCAPM"] >= config["thresholds"]["cap_steatosis"])]

    extra_n = len(asian_specific_masld) - len(asian_std_masld)
    comparison_df = pd.DataFrame([{
        "definition": "Standard (BMI≥25)",
        "n_masld_eligible": len(asian_std_masld),
    }, {
        "definition": "Asian-specific (BMI≥23)",
        "n_masld_eligible": len(asian_specific_masld),
        "extra_patients_identified": extra_n,
    }])
    comparison_df.to_csv(results_dir / "asian_masld_vs_standard_masld_comparison.csv", index=False)
    print(f"\nAsian BMI re-analysis: {extra_n} additional patients with BMI 23–25 reclassified as MASLD-eligible")

    # --- Subgroup SHAP for Asian Americans ---
    try:
        import pickle
        import shap
        model_path = Path(config["paths"]["results"]) / "exp04" / "xgboost_model.pkl"
        feature_path = data_dir / "features_fib4_low_risk.parquet"

        if model_path.exists() and feature_path.exists():
            with open(model_path, "rb") as f:
                xgb_model = pickle.load(f)
            feature_df = pd.read_parquet(feature_path)
            feat_cols = [c for c in MODEL_FEATURES if c in feature_df.columns]

            asian_fn = feature_df[
                (feature_df["FIB4_FALSE_NEGATIVE"] == 1) &
                (feature_df.get("RIDRETH3", pd.Series(0, index=feature_df.index)) == 6)
            ][feat_cols].fillna(feature_df[feat_cols].median())

            min_size = config["thresholds"]["min_false_negatives_for_subgroup_clustering"]
            if len(asian_fn) >= min_size:
                explainer = shap.TreeExplainer(xgb_model)
                shap_vals = explainer.shap_values(asian_fn)
                plot_shap_summary(
                    shap_vals, asian_fn,
                    output_path=results_dir / "asian_specific_shap.png",
                    title=f"SHAP — Asian-American False Negatives (N={len(asian_fn)})",
                )
                print(f"Asian-specific SHAP computed (N={len(asian_fn)})")
            else:
                print(f"Asian subgroup too small for SHAP (N={len(asian_fn)} < {min_size})")
        else:
            print("XGBoost model not found — skipping Asian SHAP (run exp04 first)")
    except Exception as exc:
        print(f"SHAP subgroup analysis failed: {exc}")

    print(f"\nEXP-08 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp08(cfg)
