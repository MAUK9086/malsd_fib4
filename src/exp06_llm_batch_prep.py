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
    cohort_subset: pd.DataFrame,
    cluster_id: int,
) -> dict:
    """Build the template variables dict for a single cluster."""
    grp = cohort_subset[cohort_subset["cluster"] == cluster_id]
    n = len(grp)

    # Female percentage
    pct_female = (grp.get("RIAGENDR", pd.Series([])) == 2).mean() * 100 if "RIAGENDR" in grp.columns else float("nan")

    # Diabetes percentage
    pct_dm = (grp.get("DIQ010", pd.Series([])) == 1).mean() * 100 if "DIQ010" in grp.columns else float("nan")

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

    def safe_get(col: str, default: str = "N/A") -> str:
        val = profile_row.get(f"{col}_median", np.nan)
        if pd.isna(val):
            return default
        return f"{val:.1f}"

    return {
        "n_patients": n,
        "age": safe_get("RIDAGEYR"),
        "pct_female": f"{pct_female:.0f}" if not np.isnan(pct_female) else "N/A",
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
        "fib4": safe_get("FIB4"),
        "lsm": safe_get("LUXSMED"),
        "pct_dm": f"{pct_dm:.0f}" if not np.isnan(pct_dm) else "N/A",
        "race_dist": race_dist,
    }


def prepare_batch(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    data_dir = Path(config["paths"]["data_processed"])
    results_dir_05 = Path(config["paths"]["results"]) / "exp05"
    results_dir_06 = Path(config["paths"]["results"]) / "exp06"
    results_dir_06.mkdir(parents=True, exist_ok=True)

    # Load cluster profiles
    profiles_path = results_dir_05 / "cluster_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Run exp05_clustering.py first: {profiles_path}")

    profiles = pd.read_csv(profiles_path)
    cluster_ids = sorted(profiles["cluster"].tolist())
    print(f"Found {len(cluster_ids)} clusters: {cluster_ids}")

    # Load full cohort to compute per-cluster demographics
    feature_df = pd.read_parquet(data_dir / "features_fib4_low_risk.parquet")
    assign_df = pd.read_parquet(results_dir_05 / "cluster_assignments.parquet")

    # Merge cluster assignments back to feature data
    if "SEQN" in feature_df.columns and "SEQN" in assign_df.columns:
        fn_df = feature_df[feature_df["FIB4_FALSE_NEGATIVE"] == 1].merge(
            assign_df[["SEQN", "cluster"]], on="SEQN", how="left"
        )
    else:
        fn_df = feature_df[feature_df["FIB4_FALSE_NEGATIVE"] == 1].copy()
        fn_df["cluster"] = assign_df["cluster"].values

    models_config = [
        {"name": config["models"]["primary_name"], "path": config["models"]["primary_path"]},
        {"name": config["models"]["secondary_name"], "path": config["models"]["secondary_path"]},
    ]

    batch = []
    prompt_types = list(PROMPT_REGISTRY.keys())

    for cluster_id in cluster_ids:
        profile_row = profiles[profiles["cluster"] == cluster_id].iloc[0]
        ctx = build_cluster_context(profile_row, fn_df, cluster_id)

        for model_info in models_config:
            for prompt_type in prompt_types:
                fill_fn = PROMPT_REGISTRY[prompt_type]
                system_prompt, user_prompt = fill_fn(ctx)

                batch.append({
                    "experiment": "EXP-06",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "metadata": {
                        "cluster_id": cluster_id,
                        "model_name": model_info["name"],
                        "model_path": model_info["path"],
                        "prompt_type": prompt_type,
                        "n_patients": ctx["n_patients"],
                    },
                })

    batch_path = results_dir_06 / "llm_batch.json"
    with open(batch_path, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"\nBatch prepared: {len(batch)} prompts → {batch_path}")
    print(f"  Clusters: {len(cluster_ids)}")
    print(f"  Models: {len(models_config)}")
    print(f"  Prompt types: {len(prompt_types)}")


if __name__ == "__main__":
    cfg = load_config()
    prepare_batch(cfg)
