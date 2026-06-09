"""EXP-01: Apply MASLD eligibility criteria, quality filters, and exclusions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.compute_scores import add_all_scores


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_lux_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only complete FibroScan exams (LUAXSTAT == 1)."""
    return df[df["LUAXSTAT"] == 1].copy()


def apply_age_filter(df: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    return df[df["RIDAGEYR"] >= min_age].copy()


def apply_masld_eligibility(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    MASLD 2023 criteria:
    1. CAP ≥ 274 dB/m
    2. ≥ 1 cardiometabolic criterion
    3. No exclusion criteria
    """
    t = config["thresholds"]

    # --- Criterion 1: Hepatic steatosis (CAP) ---
    cap_ok = df["LUXCAPM"] >= t["cap_steatosis"]

    # --- Criterion 2: Cardiometabolic (≥ 1 of 5) ---
    # BMI: standard ≥ 25, Asian ≥ 23 (RIDRETH3 == 6)
    is_asian = df["RIDRETH3"] == 6
    bmi_cutoff = np.where(is_asian, t["asian_bmi_cutoff"], t["standard_bmi_cutoff"])
    cm_bmi = df["BMXBMI"] >= bmi_cutoff

    # Fasting glucose ≥ 100 OR diagnosed diabetes
    cm_glucose = (df["LBXSGL"] >= t["fasting_glucose_cutoff"]) | (df["DIQ010"] == 1)

    # BP ≥ 130/85
    cm_bp = (df["BPXOSY1"] >= t["systolic_bp_cutoff"]) | (df["BPXODI1"] >= t["diastolic_bp_cutoff"])

    # Triglycerides ≥ 150
    cm_trig = df["LBXTR"] >= t["triglycerides_cutoff"]

    # HDL-C: < 40 men, < 50 women
    female = df["RIAGENDR"] == 2
    hdl_cutoff = np.where(female, t["hdl_women_cutoff"], t["hdl_men_cutoff"])
    cm_hdl = df["LBDHDD"] < hdl_cutoff

    cardiometabolic_ok = (cm_bmi | cm_glucose | cm_bp | cm_trig | cm_hdl)

    # --- Exclusion criteria ---
    # Hepatitis B or C
    no_hep = ~((df.get("HEQ010", pd.Series(np.nan, index=df.index)) == 1) |
               (df.get("HEQ030", pd.Series(np.nan, index=df.index)) == 1))

    # Heavy alcohol: > 2.14 drinks/day men, > 1.43 drinks/day women
    alq = df.get("ALQ130", pd.Series(np.nan, index=df.index)).fillna(0)
    alc_male_ok = ~((df["RIAGENDR"] == 1) & (alq > t["alcohol_men_cutoff"]))
    alc_female_ok = ~((df["RIAGENDR"] == 2) & (alq > t["alcohol_women_cutoff"]))
    no_heavy_alcohol = alc_male_ok & alc_female_ok

    mask = cap_ok & cardiometabolic_ok & no_hep & no_heavy_alcohol
    return df[mask].copy()


def define_cohort(config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir = Path(config["paths"]["results"]) / "exp01"
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "nhanes_merged_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run merge_nhanes.py first: {raw_path}")

    df = pd.read_parquet(raw_path)
    n0 = len(df)
    print(f"Starting N: {n0:,}")

    flowchart: list[str] = [f"Step 0 — Starting cohort (all with FibroScan): N={n0:,}"]

    # Step 1: Age ≥ 18
    df = apply_age_filter(df)
    n1 = len(df)
    flowchart.append(f"Step 1 — Age ≥ 18: N={n1:,} (excluded {n0-n1:,})")

    # Step 2: Complete FibroScan (LUAXSTAT == 1)
    df = apply_lux_quality_filter(df)
    n2 = len(df)
    flowchart.append(f"Step 2 — Complete FibroScan (LUAXSTAT=1): N={n2:,} (excluded {n1-n2:,})")

    # Step 3: Compute derived scores (needed for MASLD criteria)
    df = add_all_scores(df, config)
    flowchart.append(f"Step 3 — Derived scores computed")

    # Step 4: MASLD eligibility
    df = apply_masld_eligibility(df, config)
    n4 = len(df)
    flowchart.append(f"Step 4 — MASLD-eligible (CAP≥274 + cardiometabolic − exclusions): N={n4:,} (excluded {n2-n4:,})")

    # Step 5: Complete data for FIB-4 computation
    fib4_vars = ["RIDAGEYR", "LBXSASSI", "LBXSATSI", "LBXPLTSI"]
    df = df.dropna(subset=fib4_vars)
    n5 = len(df)
    flowchart.append(f"Step 5 — Complete FIB-4 variables: N={n5:,} (excluded {n4-n5:,})")

    # Step 6: Valid LSM
    df = df.dropna(subset=["LUXSMED"])
    n6 = len(df)
    flowchart.append(f"Step 6 — Valid LSM (LUXSMED present): N={n6:,} (excluded {n5-n6:,})")

    flowchart.append(f"\nFinal analysis cohort: N={n6:,}")

    # Save flowchart
    flowchart_text = "\n".join(flowchart)
    print("\n" + flowchart_text)
    (results_dir / "cohort_flowchart.txt").write_text(flowchart_text)

    # Missing data audit
    missing = (df.isnull().sum() / len(df) * 100).round(2)
    missing.name = "missing_pct"
    missing.to_csv(results_dir / "missing_data_audit.csv", header=True)
    print(f"\nMissing data audit saved to {results_dir}/missing_data_audit.csv")

    # Save final cohort
    out_path = data_dir / "nhanes_masld_cohort.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved MASLD cohort: {out_path}  ({len(df):,} rows × {len(df.columns)} cols)")

    return df


if __name__ == "__main__":
    cfg = load_config()
    df = define_cohort(cfg)
    print(f"\nFIB-4 false negatives: {df['FIB4_FALSE_NEGATIVE'].sum():,}")
    print(f"FIB-4 false positives: {df['FIB4_FALSE_POSITIVE'].sum():,}")
    print(df["MISCLASS_LABEL"].value_counts())
