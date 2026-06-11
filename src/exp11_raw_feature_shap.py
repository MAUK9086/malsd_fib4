"""EXP-11: XGBoost + SHAP using RAW features only (no composite scores).

Produces the interpretable "which raw biomarkers predict FIB-4 failure" story
for the abstract. Compares AUROC against the full-feature EXP-04 model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils.stats_utils import compute_auroc_ci
from src.utils.plot_utils import plot_shap_beeswarm, plot_shap_importance_bar


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Raw features only — no composite scores (HSI, NFS, APRI, TYG are excluded).
# These are the 27 individually measured biomarkers.
RAW_FEATURES_ONLY = [
    # FIB-4 formula components (raw)
    "RIDAGEYR", "LBXSASSI", "LBXSATSI", "LBXPLTSI",
    # Hepatic enzymes (raw)
    "LBXSGTSI",
    # Protein / renal / metabolic (raw)
    "LBXSAL", "LBXSAPSI", "LBXSTB", "LBXSCR", "LBXSBU", "LBXSGL", "LBXSUA",
    # Metabolic (raw — excludes TYG which depends on LBXTR + LBXSGL)
    "LBXGH", "LBXTR", "LBDHDD", "LBXTC",
    # Anthropometric (raw — excludes WHTR which is BMXWAIST/height)
    "BMXBMI", "BMXWAIST",
    # Haematological (raw)
    "LBXHGB", "LBXWBCSI", "LBXMCVSI", "LBXRDW", "LBXMPSI",
    # Inflammatory ratios (not simple functions of the above individually)
    "SII", "DE_RITIS",
    # FIB-4 score itself (kept so we can show it is NOT the top predictor here)
    "FIB4",
]

FEATURE_LABELS = {
    "LBXGH": "HbA1c", "LBXSGTSI": "GGT", "BMXWAIST": "Waist",
    "BMXBMI": "BMI", "LBXTR": "Triglycerides", "LBDHDD": "HDL-C",
    "LBXSGL": "Glucose", "DE_RITIS": "AST/ALT ratio", "LBXSATSI": "ALT",
    "LBXSASSI": "AST", "LBXPLTSI": "Platelets", "FIB4": "FIB-4",
    "RIDAGEYR": "Age", "LBXSAL": "Albumin", "LBXHGB": "Haemoglobin",
    "LBXWBCSI": "WBC", "SII": "SII", "LBXSCR": "Creatinine",
    "LBXSBU": "BUN", "LBXSUA": "Uric acid", "LBXTC": "Total cholesterol",
    "LBXMCVSI": "MCV", "LBXRDW": "RDW", "LBXMPSI": "MPV",
    "LBXSAPSI": "ALP", "LBXSTB": "Total bilirubin",
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5,
    "eval_metric": "logloss",
    "random_state": 42,
    "use_label_encoder": False,
}


def run_exp11(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir_04 = Path(config["paths"]["results"]) / "exp04"
    results_dir = Path(config["paths"]["results"]) / "exp11"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_parquet(data_dir / "features_fib4_low_risk.parquet")

    available = [f for f in RAW_FEATURES_ONLY if f in feature_df.columns]
    missing = [f for f in RAW_FEATURES_ONLY if f not in feature_df.columns]
    if missing:
        print(f"EXP-11: features not in data: {missing}")

    X = feature_df[available].copy()
    y = feature_df["FIB4_FALSE_NEGATIVE"].astype(int)

    valid = X.notna().all(axis=1)
    X, y = X[valid], y[valid]
    print(f"EXP-11: N={len(X):,}, positive rate={y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load EXP-04 best params if available
    params = DEFAULT_XGB_PARAMS.copy()
    exp04_pkl = results_dir_04 / "xgboost_model.pkl"
    if exp04_pkl.exists():
        try:
            with open(exp04_pkl, "rb") as f:
                exp04_data = pickle.load(f)
            if isinstance(exp04_data, dict) and "best_params" in exp04_data:
                saved = exp04_data["best_params"]
                for k in ["n_estimators", "max_depth", "learning_rate",
                          "subsample", "colsample_bytree"]:
                    if k in saved:
                        params[k] = saved[k]
                print(f"Loaded EXP-04 params: {params}")
        except Exception as e:
            print(f"Could not load EXP-04 params ({e}), using defaults")
    else:
        print("EXP-04 model not found — using default params")

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    p_raw = model.predict_proba(X_test)[:, 1]

    n_boot = config["bootstrap"]["n_iterations"]
    roc_raw = compute_auroc_ci(y_test.values, p_raw, n_iter=n_boot)
    print(f"Raw-only model AUROC: {roc_raw['auroc']:.4f} "
          f"(95% CI: {roc_raw['ci_lo']:.3f}–{roc_raw['ci_hi']:.3f})")

    # Compare against EXP-04 AUROC if available
    auroc_composite = None
    exp04_auroc_path = results_dir_04 / "auroc_comparison.csv"
    if exp04_auroc_path.exists():
        try:
            comp_df = pd.read_csv(exp04_auroc_path)
            xgb_row = comp_df[comp_df["model"].str.contains("XGBoost", case=False)]
            if len(xgb_row) > 0:
                auroc_composite = xgb_row["auroc"].iloc[0]
        except Exception:
            pass

    auroc_rows = [
        {"model": "XGBoost (raw features only)", "auroc": roc_raw["auroc"],
         "ci_lo": roc_raw["ci_lo"], "ci_hi": roc_raw["ci_hi"],
         "note": "EXP-11 — no composite scores"},
    ]
    if auroc_composite is not None:
        auroc_rows.append({
            "model": "XGBoost (composite + raw, EXP-04)",
            "auroc": auroc_composite,
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "note": "EXP-04 reference",
        })
    pd.DataFrame(auroc_rows).to_csv(results_dir / "auroc_raw_vs_composite.csv", index=False)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_ranking = pd.DataFrame({
        "feature": available,
        "label": [FEATURE_LABELS.get(f, f) for f in available],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_ranking.to_csv(results_dir / "shap_raw_feature_ranking.csv", index=False)
    print(f"Top-5 raw SHAP features:\n{shap_ranking.head(5)[['feature', 'label', 'mean_abs_shap']].to_string(index=False)}")

    # Plots
    plot_shap_beeswarm(
        shap_values, X_test.values, available,
        output_path=results_dir / "shap_beeswarm_raw.png",
        title="SHAP Values — Raw Features Only (EXP-11)",
    )
    plot_shap_importance_bar(
        shap_ranking,
        output_path=results_dir / "shap_bar_raw.png",
        title="Mean |SHAP| — Raw Features (EXP-11)",
        top_n=20,
    )

    # Dependence plots for top-5
    dep_dir = results_dir / "shap_dependence_top5_raw"
    dep_dir.mkdir(exist_ok=True)
    top5 = shap_ranking["feature"].head(5).tolist()
    for feat in top5:
        if feat in available:
            idx = available.index(feat)
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(X_test[feat].values, shap_values[:, idx], alpha=0.3, s=10)
                ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
                ax.set_ylabel(f"SHAP value for {FEATURE_LABELS.get(feat, feat)}")
                ax.axhline(0, color="grey", linewidth=0.5)
                fig.tight_layout()
                fig.savefig(dep_dir / f"shap_dep_{feat}.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"Dependence plot failed for {feat}: {e}")

    # Save model
    with open(results_dir / "xgboost_raw_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": available, "params": params}, f)

    print(f"\nEXP-11 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp11(cfg)
