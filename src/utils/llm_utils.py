"""LLM model factory, batch runner, JSON validation, biomarker extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Biomarker keyword dictionary (SRS §8.2)
# ---------------------------------------------------------------------------

BIOMARKER_KEYWORDS: dict[str, list[str]] = {
    "alt": ["alt", "alanine", "aminotransferase"],
    "ast": ["ast", "aspartate"],
    "ggt": ["ggt", "gamma", "glutamyl", "transpeptidase"],
    "platelets": ["platelet", "thrombocyte", "thrombocytopenia"],
    "albumin": ["albumin", "hypoalbuminaemia", "hypoalbuminemia"],
    "hba1c": ["hba1c", "glycated haemoglobin", "glycated hemoglobin", "glycosylated", "a1c"],
    "bmi": ["bmi", "body mass index", "obese", "obesity", "overweight"],
    "waist": ["waist", "abdominal", "central adiposity", "truncal"],
    "triglycerides": ["triglyceride", "tg", "hypertriglyceridaemia", "hypertriglyceridemia"],
    "hdl": ["hdl", "high density"],
    "glucose": ["glucose", "glycaemia", "glycemia", "hyperglycaemia", "hyperglycemia", "fasting glucose"],
    "haemoglobin": ["haemoglobin", "hemoglobin", "anaemia", "anemia"],
    "wbc": ["white blood cell", "wbc", "leukocyte"],
    "creatinine": ["creatinine", "renal", "kidney"],
    "fibrosis": ["fibrosis", "cirrhosis", "steatohepatitis"],
    "ldl": ["ldl", "low density"],
    "cholesterol": ["cholesterol", "dyslipidaemia", "dyslipidemia"],
    "apri": ["apri"],
    "nfs": ["nafld fibrosis score", "nfs"],
    "age": ["age", "older", "younger", "elderly"],
}

ALLOWED_VARIABLES = set(BIOMARKER_KEYWORDS.keys())


def extract_biomarkers_from_text(text: str) -> set[str]:
    """Return set of canonical biomarker names mentioned in text."""
    text_lower = text.lower()
    found = set()
    for canonical, keywords in BIOMARKER_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.add(canonical)
                break
    return found


# ---------------------------------------------------------------------------
# LLM response validation
# ---------------------------------------------------------------------------

def validate_llm_response(
    response: dict,
    cluster_profile: dict | None = None,
) -> dict:
    """
    Checks LLM response for hallucinated biomarkers.
    Returns validation metadata dict.
    """
    text_to_check = ""
    parsed = response.get("parsed") or {}
    for key in ["description", "formula_explanation", "missing_variable_explanation",
                "asian_relevance_statement", "phenotype_name"]:
        if key in parsed:
            text_to_check += " " + str(parsed[key])

    mentioned = extract_biomarkers_from_text(text_to_check)
    hallucinated = mentioned - ALLOWED_VARIABLES

    return {
        "mentioned_variables": sorted(mentioned),
        "hallucinated_variables": sorted(hallucinated),
        "hallucination_detected": len(hallucinated) > 0,
    }


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_llm_json(raw_text: str) -> dict | None:
    """Strip markdown fences and parse JSON from LLM output."""
    clean = raw_text.strip()
    # Remove ```json ... ``` or ``` ... ```
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try extracting first {...} block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def load_llm(model_path: str, n_gpu_layers: int, n_ctx: int, seed: int, n_threads: int = 8):
    """Load a GGUF model with llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python not installed. Run setup.sh first."
        ) from e

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Download with: huggingface-cli download ... (see SRS §10.4)"
        )

    print(f"Loading model: {model_path}")
    print(f"  n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=True,   # verbose=False crashes on Windows (stdout suppression null-deref)
        seed=seed,
    )
    print("Model loaded.")
    return llm


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_llm_batch(
    batch_file: str | Path,
    output_file: str | Path,
    model,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = 42,
) -> None:
    """
    Read all prompts from batch_file, run each, write results to output_file.
    Never stops on individual failure — records error and continues.
    """
    with open(batch_file) as f:
        batch = json.load(f)

    results: list[dict] = []
    n_total = len(batch)

    for i, item in enumerate(batch):
        print(
            f"[{i+1}/{n_total}] "
            f"{item.get('experiment', '?')} | "
            f"cluster={item.get('metadata', {}).get('cluster_id', '?')} | "
            f"prompt={item.get('metadata', {}).get('prompt_type', '?')}"
        )
        try:
            response = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_prompt"]},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            raw_text: str = response["choices"][0]["message"]["content"]
            parsed = parse_llm_json(raw_text)
            results.append(
                {
                    **item.get("metadata", {}),
                    "raw_response": raw_text,
                    "parsed": parsed,
                    "status": "success" if parsed is not None else "parse_error",
                }
            )
        except Exception as exc:
            results.append(
                {
                    **item.get("metadata", {}),
                    "raw_response": None,
                    "parsed": None,
                    "status": "error",
                    "error": str(exc),
                }
            )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(r["status"] == "success" for r in results)
    print(f"\nDone. {n_ok}/{n_total} successful → {output_file}")
