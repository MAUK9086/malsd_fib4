"""EXP-17: Sensitivity analyses — tests core findings across 4 parameter variations.

Pre-empts reviewer questions by showing robustness of FN rate and AUROC to
threshold choices and cohort restrictions.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils.stats_utils import compute_auroc_ci
from src.utils.nhanes_codebook import MODEL_FEATURES
from src.compute_scores import add_all_scores


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_xgb_model_and_features(results_dir_04: Path, results_dir_11: Path):
    """Load saved XGBoost model from EXP-04 or EXP-11."""
    for pkl_path in [results_dir_04 / "xgboost_model.pkl", results_dir_11 / "xgboost_raw_model.pkl"]:
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    return data.get("model"), data.get("features", [])
                return data, []
            except Exception as e:
                print(f"Could not load {pkl_path}: {e}")
    return None, []


def build_variant_cohort(
    raw_cohort: pd.DataFrame,
    config: dict,
    lsm_thresh: float = 8.0,
    fib4_low: float = 1.30,
    cap_thresh: float = 274.0,
    age_min: int = 18,
    age_max: int = 120,
) -> pd.DataFrame:
    """Rebuild a cohort variant with modified thresholds (no re-downloading)."""
    df = raw_cohort.copy()

    # Age restriction
    df = df[(df["RIDAGEYR"] >= age_min) & (df["RIDAGEYR"] <= age_max)]

    # CAP threshold
    df = df[df["LUXCAPM"] >= cap_thresh]

    # FIB-4 re-categorise
    df["FIB4_CAT_VAR"] = pd.cut(
        df["FIB4"],
        bins=[-np.inf, fib4_low, np.inf],
        labels=[0, 1],
    ).astype(float).astype("Int64")

    # FN at new LSM threshold
    df["LSM_ELEVATED_VAR"] = (df["LUXSMED"] >= lsm_thresh).astype(int)
    df["FIB4_FN_VAR"] = ((df["FIB4_CAT_VAR"] == 0) & (df["LSM_ELEVATED_VAR"] == 1)).astype(int)

    return df


def compute_variant_metrics(
    variant_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    n_boot: int = 200,
) -> dict:
    """Compute FN rate and AUROC for a cohort variant."""
    low_risk = variant_df[variant_df["FIB4_CAT_VAR"] == 0].copy()
    n_lr = len(low_risk)
    fn_rate = low_risk["FIB4_FN_VAR"].mean() * 100 if n_lr > 0 else float("nan")

    auroc = float("nan")
    top3_shap = []
    if model is not None and n_lr > 30:
        avail = [f for f in feature_cols if f in low_risk.columns]
        X = low_risk[avail].fillna(low_risk[avail].median())
        y = low_risk["FIB4_FN_VAR"]
        valid = y.notna() & X.notna().all(axis=1)
        if valid.sum() > 30:
            try:
                roc = compute_auroc_ci(y[valid].values, model.predict_proba(X[valid])[:, 1], n_iter=n_boot)
                auroc = roc["auroc"]
            except Exception:
                pass

    return {
        "n_low_risk": n_lr,
        "fn_rate_pct": round(fn_rate, 1),
        "auroc": round(auroc, 4) if not np.isnan(auroc) else "N/A",
    }


def run_exp17(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir_04 = Path(config["paths"]["results"]) / "exp04"
    results_dir_11 = Path(config["paths"]["results"]) / "exp11"
    results_dir = Path(config["paths"]["results"]) / "exp17"
    results_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    model, feature_cols = load_xgb_model_and_features(results_dir_04, results_dir_11)

    if model is None:
        print("WARNING: No XGBoost model found — AUROC columns will be N/A")

    n_boot = min(config["bootstrap"]["n_iterations"], 200)

    # Primary analysis values
    primary = {
        "lsm_thresh": config["thresholds"]["lsm_significant_fibrosis"],
        "fib4_low": config["thresholds"]["fib4_low"],
        "cap_thresh": config["thresholds"]["cap_steatosis"],
        "age_min": 18,
        "age_max": 120,
    }
    primary_variant = build_variant_cohort(cohort, config, **primary)
    primary_metrics = compute_variant_metrics(primary_variant, model, feature_cols, n_boot)
    primary_metrics.update({"sensitivity": "Primary", **primary})

    all_rows = [primary_metrics]

    # --- Sensitivity 1: LSM threshold ---
    print("Sensitivity 1: LSM threshold...")
    lsm_results = []
    for lsm_t in [7.9, 8.0, 12.0]:
        if lsm_t == primary["lsm_thresh"]:
            continue
        v = build_variant_cohort(cohort, config, lsm_thresh=lsm_t,
                                  fib4_low=primary["fib4_low"],
                                  cap_thresh=primary["cap_thresh"])
        m = compute_variant_metrics(v, model, feature_cols, n_boot)
        m.update({"sensitivity": f"LSM≥{lsm_t}", "lsm_thresh": lsm_t,
                  "fib4_low": primary["fib4_low"], "cap_thresh": primary["cap_thresh"],
                  "age_min": 18, "age_max": 120})
        lsm_results.append(m)
        all_rows.append(m)
    pd.DataFrame(lsm_results + [primary_metrics]).to_csv(
        results_dir / "sensitivity_lsm_threshold.csv", index=False
    )

    # --- Sensitivity 2: FIB-4 cutoff ---
    print("Sensitivity 2: FIB-4 cutoff...")
    fib4_results = []
    for fib4_c in [1.30, 1.45]:
        if fib4_c == primary["fib4_low"]:
            continue
        v = build_variant_cohort(cohort, config, fib4_low=fib4_c,
                                  lsm_thresh=primary["lsm_thresh"],
                                  cap_thresh=primary["cap_thresh"])
        m = compute_variant_metrics(v, model, feature_cols, n_boot)
        m.update({"sensitivity": f"FIB-4<{fib4_c}", "fib4_low": fib4_c,
                  "lsm_thresh": primary["lsm_thresh"], "cap_thresh": primary["cap_thresh"],
                  "age_min": 18, "age_max": 120})
        fib4_results.append(m)
        all_rows.append(m)
    pd.DataFrame(fib4_results + [primary_metrics]).to_csv(
        results_dir / "sensitivity_fib4_cutoff.csv", index=False
    )

    # --- Sensitivity 3: CAP threshold ---
    print("Sensitivity 3: CAP threshold...")
    cap_results = []
    for cap_t in [248, 274]:
        if cap_t == primary["cap_thresh"]:
            continue
        v = build_variant_cohort(cohort, config, cap_thresh=cap_t,
                                  lsm_thresh=primary["lsm_thresh"],
                                  fib4_low=primary["fib4_low"])
        m = compute_variant_metrics(v, model, feature_cols, n_boot)
        m.update({"sensitivity": f"CAP≥{cap_t}", "cap_thresh": cap_t,
                  "lsm_thresh": primary["lsm_thresh"], "fib4_low": primary["fib4_low"],
                  "age_min": 18, "age_max": 120})
        cap_results.append(m)
        all_rows.append(m)
    pd.DataFrame(cap_results + [primary_metrics]).to_csv(
        results_dir / "sensitivity_masld_definition.csv", index=False
    )

    # --- Sensitivity 4: Age restriction ---
    print("Sensitivity 4: Age restriction...")
    age_results = []
    for (amin, amax, label) in [(18, 120, "18+"), (40, 75, "40–75")]:
        if amin == 18 and amax == 120:
            continue
        v = build_variant_cohort(cohort, config,
                                  lsm_thresh=primary["lsm_thresh"],
                                  fib4_low=primary["fib4_low"],
                                  cap_thresh=primary["cap_thresh"],
                                  age_min=amin, age_max=amax)
        m = compute_variant_metrics(v, model, feature_cols, n_boot)
        m.update({"sensitivity": f"Age {label}", "age_min": amin, "age_max": amax,
                  "lsm_thresh": primary["lsm_thresh"], "fib4_low": primary["fib4_low"],
                  "cap_thresh": primary["cap_thresh"]})
        age_results.append(m)
        all_rows.append(m)
    pd.DataFrame(age_results + [primary_metrics]).to_csv(
        results_dir / "sensitivity_age_restriction.csv", index=False
    )

    # --- Summary table with % change from primary ---
    primary_fn = primary_metrics["fn_rate_pct"]
    summary_df = pd.DataFrame(all_rows)
    summary_df["delta_fn_rate_pp"] = (summary_df["fn_rate_pct"] - primary_fn).round(1)
    summary_df.to_csv(results_dir / "sensitivity_summary.csv", index=False)

    print(f"\nSensitivity summary:\n{summary_df[['sensitivity', 'n_low_risk', 'fn_rate_pct', 'delta_fn_rate_pp', 'auroc']].to_string(index=False)}")
    print(f"\nEXP-17 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp17(cfg)
