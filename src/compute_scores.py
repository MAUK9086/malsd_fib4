"""EXP-01/03: Compute derived clinical scores — FIB-4, NFS, APRI, de Ritis, etc."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fib4(df: pd.DataFrame) -> pd.Series:
    """FIB-4 = (Age × AST) / (Platelets × √ALT)"""
    age = df["RIDAGEYR"]
    ast = df["LBXSASSI"]
    alt = df["LBXSATSI"].clip(lower=0.01)  # avoid sqrt(0)
    plt = df["LBXPLTSI"].clip(lower=0.01)  # avoid div-by-zero
    return (age * ast) / (plt * np.sqrt(alt))


def compute_fib4_category(fib4: pd.Series, fib4_low: float = 1.30, fib4_high: float = 2.67) -> pd.Series:
    """0=Low, 1=Indeterminate, 2=High"""
    cat = pd.cut(
        fib4,
        bins=[-np.inf, fib4_low, fib4_high, np.inf],
        labels=[0, 1, 2],
    ).astype("Int64")
    return cat


def compute_nfs(df: pd.DataFrame) -> pd.Series:
    """NAFLD Fibrosis Score"""
    age = df["RIDAGEYR"]
    bmi = df["BMXBMI"]
    dm = (df["DIQ010"] == 1).astype(float)
    ast_alt = df["LBXSASSI"] / df["LBXSATSI"].clip(lower=0.01)
    plt = df["LBXPLTSI"]
    alb = df["LBXSAL"]
    return -1.675 + 0.037 * age + 0.094 * bmi + 1.13 * dm + 0.99 * ast_alt - 0.013 * plt - 0.66 * alb


def compute_apri(df: pd.DataFrame, ast_uln: float = 40.0) -> pd.Series:
    """APRI = (AST / AST_ULN) / Platelets × 100"""
    return (df["LBXSASSI"] / ast_uln) / df["LBXPLTSI"].clip(lower=0.01) * 100


def compute_de_ritis(df: pd.DataFrame) -> pd.Series:
    """AST/ALT ratio"""
    return df["LBXSASSI"] / df["LBXSATSI"].clip(lower=0.01)


def compute_hsi(df: pd.DataFrame) -> pd.Series:
    """Hepatic Steatosis Index"""
    alt_ast = df["LBXSATSI"] / df["LBXSASSI"].clip(lower=0.01)
    bmi = df["BMXBMI"]
    female = (df["RIAGENDR"] == 2).astype(float)
    dm = (df["DIQ010"] == 1).astype(float)
    return 8 * alt_ast + bmi + 2 * female + 2 * dm


def compute_sii(df: pd.DataFrame) -> pd.Series:
    """Systemic Immune-Inflammation Index = Platelets × Neutrophils / Lymphocytes"""
    plt = df["LBXPLTSI"]
    wbc = df["LBXWBCSI"].clip(lower=0.01)
    # Approximate neutrophil count from WBC × neutrophil%
    # If neutrophil % not available, use WBC as proxy (conservative)
    if "LBXNEPCT" in df.columns and "LBXLYPCT" in df.columns:
        neutrophils = wbc * df["LBXNEPCT"] / 100
        lymphocytes = (wbc * df["LBXLYPCT"] / 100).clip(lower=0.01)
        return plt * neutrophils / lymphocytes
    else:
        # Rough proxy when differential not available
        return plt * wbc


def compute_tyg(df: pd.DataFrame) -> pd.Series:
    """Triglyceride-Glucose Index = ln(TG × glucose / 2)"""
    tg = df["LBXTR"].clip(lower=0.01)
    glu = df["LBXSGL"].clip(lower=0.01)
    return np.log(tg * glu / 2)


def compute_whtr(df: pd.DataFrame) -> pd.Series:
    """Waist-to-Height Ratio"""
    return df["BMXWAIST"] / df["BMXHT"].clip(lower=0.01)


def compute_egfr_ckd_epi(df: pd.DataFrame) -> pd.Series:
    """CKD-EPI 2021 (race-free) eGFR estimation."""
    cr = df["LBXSCR"].clip(lower=0.01)
    age = df["RIDAGEYR"]
    female = (df["RIAGENDR"] == 2).astype(float)

    # CKD-EPI 2021 formula (race-free)
    kappa = np.where(female == 1, 0.7, 0.9)
    alpha = np.where(female == 1, -0.241, -0.302)
    cr_k = cr / kappa

    egfr = (
        142
        * np.minimum(cr_k, 1) ** alpha
        * np.maximum(cr_k, 1) ** (-1.200)
        * 0.9938 ** age
        * np.where(female == 1, 1.012, 1.0)
    )
    return pd.Series(egfr, index=df.index)


def compute_lsm_category(df: pd.DataFrame, lsm_sig: float = 8.0, lsm_adv: float = 12.0) -> pd.Series:
    """0=No/Mild, 1=Significant, 2=Advanced"""
    lsm = df["LUXSMED"]
    cat = pd.Series(np.nan, index=df.index)
    cat[lsm < lsm_sig] = 0
    cat[(lsm >= lsm_sig) & (lsm < lsm_adv)] = 1
    cat[lsm >= lsm_adv] = 2
    return cat.astype("Int64")


def add_all_scores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute and add all derived scores to dataframe."""
    t = config["thresholds"]
    ast_uln = config.get("ast_uln", 40.0)

    df = df.copy()
    df["FIB4"] = compute_fib4(df)
    df["FIB4_CAT"] = compute_fib4_category(df["FIB4"], t["fib4_low"], t["fib4_high"])
    df["NFS"] = compute_nfs(df)
    df["APRI"] = compute_apri(df, ast_uln)
    df["DE_RITIS"] = compute_de_ritis(df)
    df["HSI"] = compute_hsi(df)
    df["SII"] = compute_sii(df)
    df["TYG"] = compute_tyg(df)
    df["WHTR"] = compute_whtr(df)
    df["EGFR"] = compute_egfr_ckd_epi(df)
    df["LSM_CAT"] = compute_lsm_category(df, t["lsm_significant_fibrosis"], t["lsm_advanced_fibrosis"])

    # Misclassification labels (requires LSM and FIB-4)
    lsm = df["LUXSMED"]
    fib4_cat = df["FIB4_CAT"]
    lsm_thresh = t["lsm_significant_fibrosis"]

    df["FIB4_FALSE_NEGATIVE"] = ((fib4_cat == 0) & (lsm >= lsm_thresh)).astype("Int64")
    df["FIB4_FALSE_POSITIVE"] = ((fib4_cat == 2) & (lsm < lsm_thresh)).astype("Int64")
    df["FIB4_TRUE_NEGATIVE"] = ((fib4_cat == 0) & (lsm < lsm_thresh)).astype("Int64")
    df["FIB4_TRUE_POSITIVE"] = ((fib4_cat == 2) & (lsm >= lsm_thresh)).astype("Int64")

    # Human-readable misclass label
    conditions = [
        df["FIB4_FALSE_NEGATIVE"] == 1,
        df["FIB4_FALSE_POSITIVE"] == 1,
        df["FIB4_TRUE_NEGATIVE"] == 1,
        df["FIB4_TRUE_POSITIVE"] == 1,
        (fib4_cat == 1) & (lsm < lsm_thresh),
        (fib4_cat == 1) & (lsm >= lsm_thresh),
    ]
    labels = [
        "false_negative", "false_positive", "true_negative", "true_positive",
        "indeterminate_low", "indeterminate_high",
    ]
    df["MISCLASS_LABEL"] = np.select([pd.Series(c).fillna(False).to_numpy(dtype=bool) for c in conditions], labels, default="unknown")
    return df
