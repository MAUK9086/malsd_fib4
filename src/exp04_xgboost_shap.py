"""EXP-04: XGBoost misclassification predictor + SHAP explainability."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
import optuna

from src.utils.stats_utils import compute_auroc_ci, sensitivity_specificity
from src.utils.plot_utils import (
    plot_roc_curves,
    plot_calibration_curve,
    plot_shap_summary,
    plot_shap_bar,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"FIB4_FALSE_NEGATIVE", "SEQN", "FIB4_FALSE_POSITIVE",
               "FIB4_TRUE_NEGATIVE", "FIB4_TRUE_POSITIVE", "MISCLASS_LABEL"}
    return [c for c in df.columns if c not in exclude]


def objective(trial: optuna.Trial, X_tr, y_tr, scale_pos_weight: float, cv: int = 5) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 2.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "eval_metric": "auc",
        "use_label_encoder": False,
    }
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        model = xgb.XGBClassifier(**params, verbosity=0)
        model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])], verbose=False)
        from sklearn.metrics import roc_auc_score
        preds = model.predict_proba(X_tr.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_tr.iloc[val_idx], preds))
    return float(np.mean(aucs))


def run_shap_subgroup(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    mask: pd.Series,
    output_path: Path,
    title: str,
) -> np.ndarray | None:
    X_sub = X[mask]
    if len(X_sub) < 10:
        print(f"  Subgroup too small for SHAP ({len(X_sub)}), skipping: {title}")
        return None
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sub)
    plot_shap_summary(sv, X_sub, output_path=output_path, title=title)
    return sv


def run_exp04(config: dict | None = None) -> dict:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir = Path(config["paths"]["results"]) / "exp04"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_parquet(data_dir / "features_fib4_low_risk.parquet")
    feat_cols = get_feature_cols(feature_df)
    X = feature_df[feat_cols]
    y = feature_df["FIB4_FALSE_NEGATIVE"].astype(int)

    print(f"EXP-04: N={len(X):,}, features={len(feat_cols)}, "
          f"positive rate={y.mean()*100:.1f}%")

    cfg_xgb = config["xgboost"]
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg_xgb["test_size"],
        stratify=y,
        random_state=cfg_xgb["random_state"],
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # --- Optuna HPO ---
    print(f"Running Optuna ({cfg_xgb['n_trials']} trials)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, scale_pos_weight, cfg_xgb["cv_folds"]),
        n_trials=cfg_xgb["n_trials"],
        show_progress_bar=True,
    )
    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = 42
    print(f"Best CV AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # --- Final model training ---
    model = xgb.XGBClassifier(**best_params, verbosity=0, use_label_encoder=False)
    model.fit(X_train, y_train)

    # Save model
    with open(results_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # --- Test set evaluation ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_result = compute_auroc_ci(y_test.values, y_pred_proba, n_iter=config["bootstrap"]["n_iterations"])
    print(f"Test AUROC: {roc_result['auroc']:.4f} "
          f"(95% CI: {roc_result['ci_lo']:.3f}–{roc_result['ci_hi']:.3f})")

    # Benchmark: NFS, APRI scores from original cohort
    cohort = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    low_risk_cohort = cohort[cohort["FIB4_CAT"] == 0]
    test_seqn = feature_df.iloc[y_test.index]["SEQN"].values
    test_cohort = low_risk_cohort[low_risk_cohort["SEQN"].isin(test_seqn)]

    benchmarks = []
    for score_col in ["FIB4", "NFS", "APRI"]:
        if score_col in test_cohort.columns:
            valid = test_cohort.dropna(subset=[score_col, "FIB4_FALSE_NEGATIVE"])
            yt = valid["FIB4_FALSE_NEGATIVE"].astype(int).values
            ys = valid[score_col].values
            if len(np.unique(yt)) > 1:
                r = compute_auroc_ci(yt, ys, n_iter=200)
                benchmarks.append({"model": score_col, "auroc": r["auroc"],
                                   "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})

    benchmarks.append({
        "model": "XGBoost",
        "auroc": roc_result["auroc"],
        "ci_lo": roc_result["ci_lo"],
        "ci_hi": roc_result["ci_hi"],
    })
    perf_df = pd.DataFrame(benchmarks)
    perf_df.to_csv(results_dir / "model_performance_table.csv", index=False)
    print(perf_df.to_string(index=False))

    # ROC plot
    plot_roc_curves(
        [{"label": r["model"], "auroc": r["auroc"],
          "fpr": roc_result["fpr"] if r["model"] == "XGBoost" else [],
          "tpr": roc_result["tpr"] if r["model"] == "XGBoost" else []}
         for r in benchmarks],
        output_path=results_dir / "roc_curve_xgboost.png",
        title="XGBoost vs Benchmarks — ROC",
    )

    # Calibration
    plot_calibration_curve(
        y_test.values, y_pred_proba,
        output_path=results_dir / "calibration_curve.png",
    )
    brier = brier_score_loss(y_test.values, y_pred_proba)
    print(f"Brier score: {brier:.4f}")

    # --- SHAP global ---
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plot_shap_summary(
        shap_values, X_test,
        output_path=results_dir / "shap_beeswarm.png",
        title="SHAP Beeswarm — FIB-4 False Negative Predictor",
    )
    plot_shap_bar(
        shap_values, feat_cols,
        output_path=results_dir / "shap_global_summary.png",
        title="SHAP Feature Importance (Global)",
    )

    # Top-5 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
    top5_features = [feat_cols[i] for i in top5_idx]
    print(f"Top-5 SHAP features: {top5_features}")

    # Dependence plots for top-5
    for feat in top5_features:
        if feat in X_test.columns:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            feat_idx = feat_cols.index(feat)
            shap.dependence_plot(feat_idx, shap_values, X_test, ax=ax, show=False)
            ax.set_title(f"SHAP Dependence: {feat}")
            fig.savefig(results_dir / f"shap_dependence_{feat}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    # --- Subgroup SHAP ---
    # Need original cohort info for subgroup masks
    test_idx_in_feature_df = y_test.index
    asian_mask_test = pd.Series(False, index=X_test.index)
    dm_mask_test = pd.Series(False, index=X_test.index)
    normal_bmi_mask_test = pd.Series(False, index=X_test.index)

    if "RIDRETH3" in feature_df.columns:
        asian_mask_test = feature_df.loc[X_test.index, "RIDRETH3"] == 6
    if "DIQ010" in feature_df.columns:
        dm_mask_test = feature_df.loc[X_test.index, "DIQ010"] == 1
    if "BMXBMI" in feature_df.columns:
        normal_bmi_mask_test = feature_df.loc[X_test.index, "BMXBMI"] < 25

    run_shap_subgroup(model, X_test, asian_mask_test, results_dir / "shap_asian_subgroup.png", "SHAP — Asian Americans")
    run_shap_subgroup(model, X_test, dm_mask_test, results_dir / "shap_diabetic_subgroup.png", "SHAP — Diabetics")
    run_shap_subgroup(model, X_test, normal_bmi_mask_test, results_dir / "shap_normal_bmi_subgroup.png", "SHAP — Normal BMI")

    # Save top SHAP features for downstream use
    shap_df = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(results_dir / "shap_feature_ranking.csv", index=False)

    print(f"\nEXP-04 complete → {results_dir}/")
    return {"model": model, "shap_values": shap_values, "top5_features": top5_features}


if __name__ == "__main__":
    cfg = load_config()
    run_exp04(cfg)
