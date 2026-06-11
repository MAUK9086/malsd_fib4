"""EXP-12: BMI-stratified false-negative profiles, forced k=3 clustering,
and lean MASLD hypothesis analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.plot_utils import plot_bar_chart
from src.utils.nhanes_codebook import MODEL_FEATURES
from src.exp05_clustering import CLUSTERING_FEATURES, cluster_profile_stats


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


BMI_BINS = [0, 25, 30, 40, float("inf")]
BMI_LABELS = ["<25", "25–30", "30–40", "≥40"]


def bmi_stratified_analysis(fn_cohort: pd.DataFrame) -> pd.DataFrame:
    """Median biomarker profiles by BMI group among false negatives."""
    fn = fn_cohort.copy()
    fn["bmi_group"] = pd.cut(fn["BMXBMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False)

    summary_rows = []
    feat_cols = [c for c in MODEL_FEATURES if c in fn.columns]
    for group, grp in fn.groupby("bmi_group", observed=True):
        row = {"bmi_group": str(group), "n": len(grp)}
        for f in feat_cols:
            if f in grp.columns:
                row[f"{f}_median"] = grp[f].median()
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def three_cluster_analysis(
    feature_df: pd.DataFrame,
    fn_cohort: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Force k=3 K-Means on false-negative feature space."""
    fn_idx = feature_df.index[feature_df["FIB4_FALSE_NEGATIVE"] == 1]
    fn_features = feature_df.loc[fn_idx]

    feat_cols = [c for c in CLUSTERING_FEATURES if c in fn_features.columns]
    X = fn_features[feat_cols].fillna(fn_features[feat_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km3 = KMeans(n_clusters=3, random_state=random_state, n_init=20)
    labels = km3.fit_predict(X_scaled)

    fn_with_k3 = fn_cohort.loc[fn_idx].copy()
    fn_with_k3["cluster_k3"] = labels

    feat_all = [c for c in MODEL_FEATURES if c in fn_with_k3.columns]
    profiles = cluster_profile_stats(fn_with_k3, feat_all, cluster_col="cluster_k3")
    profiles = profiles.rename(columns={"cluster": "cluster_k3"})
    return profiles


def lean_masld_analysis(fn_cohort: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Identify 'lean MASLD' subset: BMI<25 + central adiposity."""
    t = config["thresholds"]
    male = fn_cohort["RIAGENDR"] == 1 if "RIAGENDR" in fn_cohort.columns else pd.Series(True, index=fn_cohort.index)
    female = ~male

    waist_asian_men = t.get("waist_asian_men", 90)
    waist_asian_women = t.get("waist_asian_women", 80)

    central_adiposity = (
        (male & (fn_cohort["BMXWAIST"] >= waist_asian_men)) |
        (female & (fn_cohort["BMXWAIST"] >= waist_asian_women))
    ) if "BMXWAIST" in fn_cohort.columns else pd.Series(False, index=fn_cohort.index)

    lean_mask = (fn_cohort["BMXBMI"] < 25) & central_adiposity
    lean_fn = fn_cohort[lean_mask].copy()

    if len(lean_fn) == 0:
        return pd.DataFrame([{"n_lean_fn": 0, "note": "No lean MASLD FN found"}])

    # Percentage meeting Asian BMI ≥23
    asian_bmi_cutoff = t.get("asian_bmi_cutoff", 23.0)
    pct_asian_bmi = (lean_fn["BMXBMI"] >= asian_bmi_cutoff).mean() * 100

    feat_cols = [c for c in MODEL_FEATURES if c in lean_fn.columns]
    profile = {"n_lean_fn": len(lean_fn), "pct_asian_bmi_ge23": round(pct_asian_bmi, 1)}
    for f in feat_cols:
        if f in lean_fn.columns:
            profile[f"{f}_median"] = lean_fn[f].median()

    return pd.DataFrame([profile])


def run_exp12(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir_05 = Path(config["paths"]["results"]) / "exp05"
    results_dir = Path(config["paths"]["results"]) / "exp12"
    results_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    feature_df = pd.read_parquet(data_dir / "features_fib4_low_risk.parquet")

    fn_cohort = cohort[(cohort["FIB4_CAT"] == 0) & (cohort["FIB4_FALSE_NEGATIVE"] == 1)].copy()
    print(f"EXP-12: FIB-4 false negatives N={len(fn_cohort):,}")

    # --- Part A: BMI-stratified profiles ---
    bmi_profiles = bmi_stratified_analysis(fn_cohort)
    bmi_profiles.to_csv(results_dir / "bmi_stratified_fn_profiles.csv", index=False)
    print(f"BMI groups:\n{bmi_profiles[['bmi_group', 'n']].to_string(index=False)}")

    # FN rate bar chart by BMI group
    low_risk = cohort[cohort["FIB4_CAT"] == 0].copy()
    low_risk["bmi_group"] = pd.cut(low_risk["BMXBMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False)
    fn_rates = (
        low_risk.groupby("bmi_group", observed=True)["FIB4_FALSE_NEGATIVE"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"FIB4_FALSE_NEGATIVE": "fn_rate_pct"})
    )
    fn_rates["bmi_group"] = fn_rates["bmi_group"].astype(str)
    fn_rates.to_csv(results_dir / "bmi_group_fn_rates.csv", index=False)

    try:
        plot_bar_chart(
            fn_rates,
            x_col="bmi_group",
            y_col="fn_rate_pct",
            output_path=results_dir / "bmi_group_fn_rates.png",
            title="FIB-4 False Negative Rate by BMI Group",
            xlabel="BMI group (kg/m²)",
            ylabel="FN rate (%)",
        )
    except Exception as e:
        print(f"Bar chart failed: {e}")

    # --- Part B: Forced k=3 clustering ---
    try:
        k3_profiles = three_cluster_analysis(feature_df, fn_cohort)
        k3_profiles.to_csv(results_dir / "three_cluster_profiles.csv", index=False)
        print(f"Forced k=3 cluster sizes: {k3_profiles['n'].tolist()}")
    except Exception as e:
        print(f"k=3 clustering failed: {e}")

    # --- Part C: Lean MASLD analysis ---
    lean_profile = lean_masld_analysis(fn_cohort, config)
    lean_profile.to_csv(results_dir / "lean_masld_fn_profile.csv", index=False)
    if "n_lean_fn" in lean_profile.columns:
        n_lean = lean_profile["n_lean_fn"].iloc[0]
        pct_asian = lean_profile.get("pct_asian_bmi_ge23", pd.Series([float("nan")])).iloc[0]
        print(f"Lean MASLD FN: N={n_lean}, {pct_asian:.1f}% meet Asian BMI ≥23 threshold")

    # --- Part D: Publication-ready summary table ---
    # Combine BMI stratification + key biomarker medians
    key_cols = ["bmi_group", "n"]
    key_features = ["LBXGH_median", "LBXSGTSI_median", "BMXWAIST_median",
                    "FIB4_median", "LUXSMED_median", "BMXBMI_median"]
    avail_key = [c for c in key_features if c in bmi_profiles.columns]
    summary_cols = key_cols + avail_key
    summary = bmi_profiles[[c for c in summary_cols if c in bmi_profiles.columns]]
    summary.to_csv(results_dir / "phenotype_clinical_summary_table.csv", index=False)

    print(f"\nEXP-12 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp12(cfg)
