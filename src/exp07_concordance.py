"""EXP-07: LLM–SHAP concordance analysis (hallucination detection)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.llm_utils import extract_biomarkers_from_text, validate_llm_response
from src.utils.plot_utils import plot_concordance_heatmap

# Maps composite score feature names to their component biomarker canonical names.
# A composite feature is counted as "mentioned" if ANY of its components appear
# in the LLM response text — the LLM correctly names components, not the score.
COMPOSITE_SCORE_COMPONENTS: dict[str, set[str]] = {
    "HSI":      {"alt", "ast", "bmi", "glucose", "haemoglobin"},
    "NFS":      {"age", "bmi", "glucose", "ast", "alt", "platelets", "albumin"},
    "APRI":     {"ast", "platelets"},
    "DE_RITIS": {"ast", "alt"},
    "TYG":      {"triglycerides", "glucose"},
    "SII":      {"platelets", "wbc"},
    "WHTR":     {"waist"},
}


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_shap_top5_per_cluster(results_dir_04: Path, results_dir_05: Path) -> dict[int, list[str]]:
    """
    Load per-cluster top-5 SHAP features.
    Falls back to global SHAP ranking if cluster-specific not available.
    """
    global_shap_path = results_dir_04 / "shap_feature_ranking.csv"
    if not global_shap_path.exists():
        raise FileNotFoundError(f"Run exp04_xgboost_shap.py first: {global_shap_path}")

    global_shap = pd.read_csv(global_shap_path)
    global_top5 = global_shap["feature"].head(5).tolist()

    cluster_path = results_dir_05 / "cluster_assignments.parquet"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Run exp05_clustering.py first: {cluster_path}")

    assign_df = pd.read_parquet(cluster_path)
    cluster_ids = sorted(assign_df["cluster"].dropna().unique())

    # For now, use global top-5 for all clusters
    # In a full run, per-cluster SHAP from exp04 subgroup analysis would be used
    return {int(c): global_top5 for c in cluster_ids}


def compute_concordance(llm_text: str, shap_top5: list[str]) -> float:
    """
    Fraction of SHAP top-5 features 'covered' by the LLM response.

    For composite scores (HSI, NFS, APRI, etc.), the feature is counted as
    covered if ANY of its component biomarkers appear in the LLM text —
    the LLM correctly names components rather than the composite formula name.
    """
    mentioned = extract_biomarkers_from_text(llm_text)

    feature_to_canonical = {
        "LBXSATSI": "alt", "LBXSASSI": "ast", "LBXSGTSI": "ggt",
        "LBXPLTSI": "platelets", "LBXSAL": "albumin", "LBXGH": "hba1c",
        "BMXBMI": "bmi", "BMXWAIST": "waist", "LBXTR": "triglycerides",
        "LBDHDD": "hdl", "LBXSGL": "glucose", "LBXHGB": "haemoglobin",
        "LBXWBCSI": "wbc", "LBXSCR": "creatinine", "RIDAGEYR": "age",
        "LBXMCVSI": "mcv", "LBXRDW": "rdw",
        "NFS": "nfs", "APRI": "apri", "DE_RITIS": "ast",
    }

    if not shap_top5:
        return 0.0

    matched = 0
    for feat in shap_top5:
        if feat in COMPOSITE_SCORE_COMPONENTS:
            # Composite: count as match if ANY component is mentioned
            if mentioned & COMPOSITE_SCORE_COMPONENTS[feat]:
                matched += 1
        else:
            canonical = feature_to_canonical.get(feat, feat.lower())
            if canonical in mentioned:
                matched += 1

    return matched / len(shap_top5)


def compute_semantic_similarity(text_a: str, text_b: str, model) -> float:
    """Cosine similarity between two texts using sentence-transformers."""
    emb_a = model.encode([text_a], normalize_embeddings=True)
    emb_b = model.encode([text_b], normalize_embeddings=True)
    return float(np.dot(emb_a[0], emb_b[0]))


def run_exp07(config: dict | None = None) -> None:
    if config is None:
        config = load_config()

    results_dir_04 = Path(config["paths"]["results"]) / "exp04"
    results_dir_05 = Path(config["paths"]["results"]) / "exp05"
    results_dir_06 = Path(config["paths"]["results"]) / "exp06"
    results_dir_07 = Path(config["paths"]["results"]) / "exp07"
    results_dir_07.mkdir(parents=True, exist_ok=True)

    concordance_threshold = config["thresholds"]["llm_concordance_threshold"]
    similarity_threshold = config["thresholds"]["inter_model_similarity_threshold"]

    # Load per-cluster SHAP top-5
    shap_top5 = load_shap_top5_per_cluster(results_dir_04, results_dir_05)
    print(f"Loaded SHAP top-5 for {len(shap_top5)} clusters")

    # Load LLM responses (both models if available)
    model_names = [config["models"]["primary_name"], config["models"]["secondary_name"]]
    responses_by_model: dict[str, list] = {}

    for model_name in model_names:
        response_path = results_dir_06 / f"llm_responses_{model_name}.json"
        if response_path.exists():
            with open(response_path) as f:
                responses_by_model[model_name] = json.load(f)
            print(f"Loaded {len(responses_by_model[model_name])} responses from {model_name}")
        else:
            print(f"WARNING: No responses found for {model_name} at {response_path}")

    if not responses_by_model:
        print("No LLM responses found. Run exp06_llm_run.py first.")
        return

    # --- Concordance analysis ---
    concordance_records = []
    hallucination_flags = []

    for model_name, responses in responses_by_model.items():
        for resp in responses:
            cluster_id = resp.get("cluster_id")
            prompt_type = resp.get("prompt_type")

            if resp.get("status") != "success" or resp.get("parsed") is None:
                continue

            # Extract all text from parsed response
            parsed = resp["parsed"]
            all_text = " ".join(str(v) for v in parsed.values() if isinstance(v, str))

            # Concordance
            top5 = shap_top5.get(cluster_id, [])
            concordance = compute_concordance(all_text, top5) if top5 else float("nan")

            # Hallucination validation
            validation = validate_llm_response(resp)

            concordance_records.append({
                "model": model_name,
                "cluster_id": cluster_id,
                "prompt_type": prompt_type,
                "concordance": concordance,
                "shap_top5": str(top5),
                "mentioned_vars": str(validation["mentioned_variables"]),
            })

            if concordance < concordance_threshold:
                hallucination_flags.append({
                    "model": model_name,
                    "cluster_id": cluster_id,
                    "prompt_type": prompt_type,
                    "concordance": concordance,
                    "hallucinated_vars": str(validation["hallucinated_variables"]),
                    "flag": "high_hallucination_risk",
                })

    conc_df = pd.DataFrame(concordance_records)
    conc_df.to_csv(results_dir_07 / "llm_shap_concordance.csv", index=False)

    if hallucination_flags:
        pd.DataFrame(hallucination_flags).to_csv(results_dir_07 / "hallucination_flags.csv", index=False)
        print(f"Hallucination flags: {len(hallucination_flags)} responses flagged")
    else:
        print("No high-hallucination-risk responses flagged.")

    if not conc_df.empty:
        mean_conc = conc_df.groupby(["cluster_id", "model"])["concordance"].mean()
        print(f"\nMean concordance by cluster/model:\n{mean_conc.round(2)}")

        # Concordance heatmap
        conc_pivot = (
            conc_df[conc_df["prompt_type"] == "PHENOTYPE"]
            .groupby(["cluster_id", "model"])["concordance"]
            .mean()
            .unstack("model")
        )
        if len(conc_pivot) > 0:
            plot_concordance_heatmap(conc_pivot, results_dir_07 / "concordance_heatmap.png")

    # --- Inter-model agreement (semantic similarity) ---
    if len(responses_by_model) >= 2:
        print("\nComputing inter-model semantic similarity...")
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(config["sentence_transformer_model"])
        except Exception as exc:
            print(f"sentence-transformers not available: {exc}")
            st_model = None

        if st_model:
            model_names_list = list(responses_by_model.keys())
            m1, m2 = model_names_list[0], model_names_list[1]

            def responses_to_dict(responses: list) -> dict:
                d = {}
                for r in responses:
                    if r.get("status") == "success" and r.get("parsed"):
                        key = (r.get("cluster_id"), r.get("prompt_type"))
                        text = " ".join(str(v) for v in r["parsed"].values() if isinstance(v, str))
                        d[key] = text
                return d

            d1 = responses_to_dict(responses_by_model[m1])
            d2 = responses_to_dict(responses_by_model[m2])

            agreement_records = []
            for key in set(d1.keys()) & set(d2.keys()):
                sim = compute_semantic_similarity(d1[key], d2[key], st_model)
                agreement_records.append({
                    "cluster_id": key[0],
                    "prompt_type": key[1],
                    f"similarity_{m1}_vs_{m2}": sim,
                    "agreement": sim >= similarity_threshold,
                })

            if agreement_records:
                agr_df = pd.DataFrame(agreement_records)
                agr_df.to_csv(results_dir_07 / "inter_model_agreement.csv", index=False)
                print(f"Agreement rate (≥{similarity_threshold}): "
                      f"{agr_df['agreement'].mean()*100:.0f}%")

    print(f"\nEXP-07 complete → {results_dir_07}/")


if __name__ == "__main__":
    cfg = load_config()
    run_exp07(cfg)
