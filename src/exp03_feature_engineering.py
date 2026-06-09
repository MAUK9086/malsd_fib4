"""EXP-03: Feature engineering — imputation, Winsorising, log-transforms, binary target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.nhanes_codebook import MODEL_FEATURES, LOG_TRANSFORM_VARS
from src.utils.plot_utils import plot_correlation_heatmap, plot_feature_distributions


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def winsorise(series: pd.Series, lo_pct: float = 1.0, hi_pct: float = 99.0) -> pd.Series:
    lo = series.quantile(lo_pct / 100)
    hi = series.quantile(hi_pct / 100)
    return series.clip(lower=lo, upper=hi)


def run_exp03(config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir = Path(config["paths"]["results"]) / "exp03"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")

    # Work within the FIB-4 low-risk stratum only
    low_risk = df[df["FIB4_CAT"] == 0].copy()
    print(f"EXP-03: FIB-4 low-risk stratum: N={len(low_risk):,}")

    # Select features that exist in the data
    available_features = [f for f in MODEL_FEATURES if f in low_risk.columns]
    missing_features = [f for f in MODEL_FEATURES if f not in low_risk.columns]
    if missing_features:
        print(f"Warning: features not in data: {missing_features}")

    X = low_risk[available_features].copy()
    y = low_risk["FIB4_FALSE_NEGATIVE"].astype(int)

    # Report high-missingness features (> 20%)
    miss_pct = X.isnull().mean() * 100
    high_miss = miss_pct[miss_pct > 20]
    if len(high_miss) > 0:
        print(f"High missingness (>20%) features:\n{high_miss.round(1)}")
        (results_dir / "high_missingness_features.txt").write_text(high_miss.to_string())

    # Step 1: Median imputation
    medians = X.median()
    X = X.fillna(medians)

    # Step 2: Winsorise at 1st / 99th percentile
    for col in X.columns:
        X[col] = winsorise(X[col])

    # Step 3: Log-transform right-skewed variables (only those present)
    log_cols = [c for c in LOG_TRANSFORM_VARS if c in X.columns]
    for col in log_cols:
        X[f"LOG_{col}"] = np.log1p(X[col].clip(lower=0))
    print(f"Log-transformed: {log_cols}")

    # Step 4: Assemble final feature matrix with target
    feature_df = X.copy()
    feature_df["FIB4_FALSE_NEGATIVE"] = y.values
    feature_df["SEQN"] = low_risk["SEQN"].values

    out_path = data_dir / "features_fib4_low_risk.parquet"
    feature_df.to_parquet(out_path, index=False)
    print(f"Saved feature matrix: {out_path}  ({len(feature_df):,} rows × {len(feature_df.columns)} cols)")
    print(f"Class balance: {y.mean()*100:.1f}% false negatives")

    # Plots
    plot_feature_distributions(
        X, available_features[:20], hue_col=None,
        output_path=results_dir / "feature_distributions.png",
    )
    plot_correlation_heatmap(
        X[available_features],
        output_path=results_dir / "correlation_heatmap.png",
    )

    print(f"EXP-03 complete → {results_dir}/")
    return feature_df


if __name__ == "__main__":
    cfg = load_config()
    run_exp03(cfg)
