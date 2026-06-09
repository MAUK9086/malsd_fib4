"""EXP-06 Part 2: Execute LLM batch — run overnight on Blackwell GPU."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.utils.llm_utils import load_llm, run_llm_batch


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_exp06(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    results_dir_06 = Path(config["paths"]["results"]) / "exp06"
    batch_path = results_dir_06 / "llm_batch.json"

    if not batch_path.exists():
        raise FileNotFoundError(f"Run exp06_llm_batch_prep.py first: {batch_path}")

    with open(batch_path) as f:
        batch = json.load(f)

    # Group by model path so we load each model once
    model_groups: dict[str, list] = {}
    for item in batch:
        mp = item["metadata"]["model_path"]
        model_groups.setdefault(mp, []).append(item)

    cfg_m = config["models"]

    for model_path, items in model_groups.items():
        model_name = items[0]["metadata"]["model_name"]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({len(items)} prompts)")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        # Determine GPU layers based on model size
        is_large = "72b" in model_path.lower() or "70b" in model_path.lower()
        n_gpu_layers = cfg_m["n_gpu_layers_70b"] if is_large else cfg_m["n_gpu_layers_32b"]

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"WARNING: Model file not found: {model_path}")
            print("Skipping this model. Download with huggingface-cli (see SRS §10.4).")
            continue

        llm = load_llm(
            model_path=str(model_path_obj),
            n_gpu_layers=n_gpu_layers,
            n_ctx=cfg_m["n_ctx"],
            seed=cfg_m["seed"],
            n_threads=8,
        )

        # Write model-specific batch subset
        subset_path = results_dir_06 / f"batch_{model_name}.json"
        with open(subset_path, "w") as f:
            json.dump(items, f, indent=2)

        output_path = results_dir_06 / f"llm_responses_{model_name}.json"
        run_llm_batch(
            batch_file=subset_path,
            output_file=output_path,
            model=llm,
            temperature=cfg_m["temperature"],
            max_tokens=cfg_m["max_tokens"],
            seed=cfg_m["seed"],
        )

        # Free model from memory between runs
        del llm
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print(f"\nEXP-06 complete. Results in {results_dir_06}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp06(cfg)
