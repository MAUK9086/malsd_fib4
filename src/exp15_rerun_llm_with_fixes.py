"""EXP-15: Re-run LLM phenotyping after BUG FIX 1 + 2 (new clusters, real FIB-4/LSM values).

Produces v2 responses for EXP-07 concordance validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.exp06_llm_batch_prep import prepare_batch
from src.exp07_concordance import (
    compute_concordance,
    load_shap_top5_per_cluster,
)
from src.utils.llm_utils import run_llm_batch, validate_llm_response


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_exp15(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    results_dir_05 = Path(config["paths"]["results"]) / "exp05"
    results_dir_04 = Path(config["paths"]["results"]) / "exp04"
    results_dir = Path(config["paths"]["results"]) / "exp15"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight checks
    profiles_path = results_dir_05 / "cluster_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"cluster_profiles.csv not found — run exp05_clustering.py first: {profiles_path}"
        )

    profiles = pd.read_csv(profiles_path)
    if "FIB4_median" not in profiles.columns or "LUXSMED_median" not in profiles.columns:
        raise ValueError(
            "cluster_profiles.csv missing FIB4_median or LUXSMED_median — "
            "re-run exp05_clustering.py with BUG FIX 1 applied."
        )

    # Check that FIB-4 values are not all NaN (BUG FIX 2 validation)
    fib4_ok = profiles["FIB4_median"].notna().any()
    if not fib4_ok:
        raise ValueError("FIB4_median is all NaN — BUG FIX 2 not applied correctly.")

    print(f"EXP-15: {len(profiles)} clusters, FIB4 medians: {profiles['FIB4_median'].tolist()}")

    # Prepare batch using exp06 infrastructure with exp15 output directory
    print("Preparing LLM batch (v2)...")
    batch_path = prepare_batch(config, output_dir=str(results_dir), experiment_label="EXP-15")

    # Run LLM inference
    print("Running LLM batch (v2)...")
    model_names = [config["models"]["primary_name"]]
    if config["models"].get("use_secondary_model", True):
        model_names.append(config["models"]["secondary_name"])

    for model_name in model_names:
        out_path = results_dir / f"llm_responses_{model_name}_v2.json"
        print(f"Running {model_name}...")
        try:
            responses = run_llm_batch(
                batch_path=batch_path,
                model_name=model_name,
                model_path=config["models"]["primary_path"] if model_name == config["models"]["primary_name"]
                           else config["models"]["secondary_path"],
                config=config,
                filter_model_name=model_name,
            )
            with open(out_path, "w") as f:
                json.dump(responses, f, indent=2)
            print(f"  Saved {len(responses)} responses → {out_path}")
        except Exception as e:
            print(f"  LLM inference failed for {model_name}: {e}")
            continue

    # Re-run concordance computation on v2 responses
    print("\nComputing concordance (v2)...")
    shap_top5 = load_shap_top5_per_cluster(results_dir_04, results_dir_05)

    concordance_records = []
    phenotype_names_v2 = []
    concordance_threshold = config["thresholds"]["llm_concordance_threshold"]

    for model_name in model_names:
        resp_path = results_dir / f"llm_responses_{model_name}_v2.json"
        if not resp_path.exists():
            continue
        with open(resp_path) as f:
            responses = json.load(f)

        for resp in responses:
            if resp.get("status") != "success" or resp.get("parsed") is None:
                continue
            cluster_id = resp.get("cluster_id")
            prompt_type = resp.get("prompt_type")
            parsed = resp["parsed"]
            all_text = " ".join(str(v) for v in parsed.values() if isinstance(v, str))

            top5 = shap_top5.get(cluster_id, [])
            concordance = compute_concordance(all_text, top5) if top5 else float("nan")
            validation = validate_llm_response(resp)

            concordance_records.append({
                "model": model_name,
                "cluster_id": cluster_id,
                "prompt_type": prompt_type,
                "concordance": concordance,
                "shap_top5": str(top5),
            })

            if prompt_type == "PHENOTYPE":
                name = parsed.get("phenotype_name", "")
                if name:
                    phenotype_names_v2.append({
                        "cluster_id": cluster_id,
                        "model": model_name,
                        "phenotype_name": name,
                        "concordance": concordance,
                    })

    conc_df = pd.DataFrame(concordance_records)
    conc_df.to_csv(results_dir / "llm_shap_concordance_v2.csv", index=False)

    if not conc_df.empty:
        mean_conc = conc_df["concordance"].mean()
        print(f"Mean concordance (v2): {mean_conc*100:.1f}%")
        above_threshold = (conc_df["concordance"] >= concordance_threshold).mean() * 100
        print(f"Above threshold ({concordance_threshold}): {above_threshold:.1f}%")

    # Save phenotype names
    if phenotype_names_v2:
        names_df = pd.DataFrame(phenotype_names_v2)
        names_df.to_csv(results_dir / "phenotype_names_v2.csv", index=False)
        # Plain text version for exp10
        unique_names = names_df.drop_duplicates(subset=["cluster_id"]).sort_values("cluster_id")
        names_text = "\n".join(
            f"Cluster {row.cluster_id}: {row.phenotype_name}" for row in unique_names.itertuples()
        )
        (results_dir / "phenotype_names_v2.txt").write_text(names_text)
        print(f"\nPhenotype names (v2):\n{names_text}")

    print(f"\nEXP-15 complete → {results_dir}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp15(cfg)
