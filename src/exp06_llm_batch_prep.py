"""EXP-06 Part 1: Prepare all LLM prompts as a single JSON batch file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prompts.prompt_templates import PROMPT_REGISTRY
from src.utils.nhanes_codebook import RACE_LABELS


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_cluster_context(
    profile_row: pd.Series,
    fn_full: pd.DataFrame,
    cluster_id: int,
) -> dict:
    """
    Build the template variables dict for a single cluster.

    profile_row: row from cluster_profiles.csv (lab medians from feature_df)
    fn_full: full cohort false-negative rows with cluster assignments,
             includes FIB4, LUXSMED, RIAGENDR, DIQ010, RIDRETH3
    """
    grp = fn_full[fn_full["cluster"] == cluster_id]
    n = len(grp)

    # Demographics — derived directly from full cohort group
    pct_female = (grp["RIAGENDR"] == 2).mean() * 100 if "RIAGENDR" in grp.columns else float("nan")
    pct_dm = (grp["DIQ010"] == 1).mean() * 100 if "DIQ010" in grp.columns else float("nan")

    # FIB-4 and LSM from full cohort (not from profile_row which may lack these)
    fib4_median = grp["FIB4"].median() if "FIB4" in grp.columns else float("nan")
    lsm_median = grp["LUXSMED"].median() if "LUXSMED" in grp.columns else float("nan")

    # Race/ethnicity distribution
    if "RIDRETH3" in grp.columns:
        race_counts = grp["RIDRETH3"].value_counts()
        race_dist_parts = []
        for code, count in race_counts.items():
            label = RACE_LABELS.get(int(code), f"Code {code}")
            race_dist_parts.append(f"{label}: {count/n*100:.0f}%")
        race_dist = ", ".join(race_dist_parts[:4])
    else:
        race_dist = "Not available"

    # Lab values from profile_row (cluster_profiles.csv medians from feature_df)
    def safe_get(col: str, default: str = "N/A") -> str:
        # Try profile_row first, then compute from grp
        val = profile_row.get(f"{col}_median", np.nan)
        if pd.isna(val) and col in grp.columns:
            val = grp[col].median()
        if pd.isna(val):
            return default
        return f"{val:.1f}"

    def fmt(val: float, default: str = "N/A") -> str:
        return f"{val:.1f}" if not (val is None or np.isnan(val)) else default

    return {
        "n_patients": n,
        "age": safe_get("RIDAGEYR"),
        "pct_female": fmt(pct_female),
        "bmi": safe_get("BMXBMI"),
        "waist": safe_get("BMXWAIST"),
        "alt": safe_get("LBXSATSI"),
        "ast": safe_get("LBXSASSI"),
        "ggt": safe_get("LBXSGTSI"),
        "plt": safe_get("LBXPLTSI"),
        "hba1c": safe_get("LBXGH"),
        "trig": safe_get("LBXTR"),
        "hdl": safe_get("LBDHDD"),
        "alb": safe_get("LBXSAL"),
        "hgb": safe_get("LBXHGB"),
        "fib4": fmt(fib4_median),
        "lsm": fmt(lsm_median),
        "pct_dm": fmt(pct_dm),
        "race_dist": race_dist,
    }


def prepare_batch(
    config: dict | None = None,
    output_dir: str | None = None,
    experiment_label: str = "EXP-06",
) -> str:
    """
    Prepare LLM batch JSON. Returns path to batch file.

    output_dir: override results/exp06 with a different directory (e.g. results/exp15)
    """
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir_05 = Path(config["paths"]["results"]) / "exp05"
    results_dir_out = Path(output_dir) if output_dir else Path(config["paths"]["results"]) / "exp06"
    results_dir_out.mkdir(parents=True, exist_ok=True)

    # Load cluster profiles (from EXP-05 — always read from exp05)
    profiles_path = results_dir_05 / "cluster_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Run exp05_clustering.py first: {profiles_path}")

    profiles = pd.read_csv(profiles_path)
    cluster_ids = sorted(profiles["cluster"].tolist())
    print(f"Found {len(cluster_ids)} clusters: {cluster_ids}")

    # Load FULL cohort (has FIB4, LUXSMED, RIAGENDR, DIQ010, RIDRETH3)
    cohort = pd.read_parquet(data_dir / "nhanes_masld_cohort.parquet")
    cohort_fn = cohort[(cohort["FIB4_CAT"] == 0) & (cohort["FIB4_FALSE_NEGATIVE"] == 1)].copy()

    # Merge cluster assignments onto full cohort false negatives
    assign_df = pd.read_parquet(results_dir_05 / "cluster_assignments.parquet")
    if "SEQN" in cohort_fn.columns and "SEQN" in assign_df.columns:
        fn_full = cohort_fn.merge(assign_df[["SEQN", "cluster"]], on="SEQN", how="left")
    else:
        fn_full = cohort_fn.copy()
        fn_full["cluster"] = assign_df["cluster"].values

    models_config = [
        {"name": config["models"]["primary_name"], "path": config["models"]["primary_path"]},
    ]
    if config["models"].get("use_secondary_model", True):
        models_config.append(
            {"name": config["models"]["secondary_name"], "path": config["models"]["secondary_path"]}
        )

    batch = []
    # DIFFERENTIATION is a cross-cluster prompt — skip it in the per-cluster loop
    per_cluster_types = [pt for pt in PROMPT_REGISTRY.keys() if pt != "DIFFERENTIATION"]

    for cluster_id in cluster_ids:
        profile_row = profiles[profiles["cluster"] == cluster_id].iloc[0]
        ctx = build_cluster_context(profile_row, fn_full, cluster_id)

        # Verify FIB-4 and LSM are populated
        if ctx["fib4"] == "N/A" or ctx["lsm"] == "N/A":
            print(f"  WARNING: cluster {cluster_id} missing FIB-4 ({ctx['fib4']}) or LSM ({ctx['lsm']})")

        for model_info in models_config:
            for prompt_type in per_cluster_types:
                fill_fn = PROMPT_REGISTRY[prompt_type]
                system_prompt, user_prompt = fill_fn(ctx)

                batch.append({
                    "experiment": experiment_label,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "metadata": {
                        "cluster_id": cluster_id,
                        "model_name": model_info["name"],
                        "model_path": model_info["path"],
                        "prompt_type": prompt_type,
                        "n_patients": ctx["n_patients"],
                        "fib4_check": ctx["fib4"],
                        "lsm_check": ctx["lsm"],
                    },
                })

    # DIFFERENTIATION: one prompt per model covering all clusters simultaneously
    if "DIFFERENTIATION" in PROMPT_REGISTRY and len(cluster_ids) > 1:
        # Build a simple comparison table from profiles
        key_cols = ["fib4", "lsm", "bmi", "hba1c", "ggt", "waist"]
        table_lines = ["Cluster | N | " + " | ".join(k.upper() for k in key_cols)]
        table_lines.append("-" * 60)
        for cluster_id in cluster_ids:
            profile_row = profiles[profiles["cluster"] == cluster_id].iloc[0]
            ctx = build_cluster_context(profile_row, fn_full, cluster_id)
            row = f"  {cluster_id}   | {ctx['n_patients']} | " + " | ".join(ctx.get(k, "N/A") for k in key_cols)
            table_lines.append(row)

        diff_ctx = {
            "n_clusters": len(cluster_ids),
            "cluster_table": "\n".join(table_lines),
        }
        fill_fn = PROMPT_REGISTRY["DIFFERENTIATION"]
        for model_info in models_config:
            system_prompt, user_prompt = fill_fn(diff_ctx)
            batch.append({
                "experiment": experiment_label,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": {
                    "cluster_id": -1,  # -1 = all clusters
                    "model_name": model_info["name"],
                    "model_path": model_info["path"],
                    "prompt_type": "DIFFERENTIATION",
                    "n_patients": sum(
                        build_cluster_context(
                            profiles[profiles["cluster"] == c].iloc[0], fn_full, c
                        )["n_patients"] for c in cluster_ids
                    ),
                    "fib4_check": "all",
                    "lsm_check": "all",
                },
            })

    batch_path = results_dir_out / "llm_batch.json"
    with open(batch_path, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"\nBatch prepared: {len(batch)} prompts → {batch_path}")
    print(f"  Clusters: {len(cluster_ids)}, Models: {len(models_config)}, Prompt types: {len(PROMPT_REGISTRY)}")
    return str(batch_path)


if __name__ == "__main__":
    cfg = load_config()
    prepare_batch(cfg)
