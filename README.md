# MASLD FIB-4 Failure Mode Phenotyping Study

**Decoding Who FIB-4 Fails: An LLM-Augmented Clinical Phenotyping Study Using NHANES 2017–2020**

Target venue: **APASL STC 2026 Kumamoto** — September 18–19, 2026  
Abstract deadline: **June 19, 2026**

---

## Overview

FIB-4 misclassifies ~10% of MASLD patients with significant fibrosis as "low risk". This study characterises the distinct clinical phenotypes of these false negatives using XGBoost + SHAP + unsupervised clustering on NHANES 2017–2020 data, then generates human-readable phenotype descriptions using locally-deployed LLMs (Qwen2.5-32B-Instruct), validated against SHAP via a novel concordance scoring methodology.

## Hardware Requirements

- NVIDIA RTX Pro Blackwell 4000 (24 GB VRAM)
- CUDA 12.8, Ubuntu 22.04+, Python 3.12
- 32 GB RAM, 50 GB disk space

## Quickstart

```bash
git clone https://github.com/MAUK9086/malsd_fib4.git
cd malsd_fib4
git checkout v1

# 1. Set up environment (builds llama-cpp-python for CUDA 12.8)
chmod +x setup.sh
./setup.sh
source .venv/bin/activate

# 2. Download NHANES 2017-2020 data (~200 MB, ~30 min)
python src/download_nhanes.py

# 3. Build analysis cohort
python src/merge_nhanes.py
python src/define_cohort.py

# 4. Run experiments (non-LLM, ~2-3 hours)
python src/exp02_misclassification.py
python src/exp03_feature_engineering.py
python src/exp04_xgboost_shap.py      # ~30-60 min (Optuna HPO)
python src/exp05_clustering.py
python src/exp08_ethnicity.py
python src/exp09_fib4plus.py

# 5. Download LLMs (see setup notes below)
# Then prepare and run LLM batch overnight:
python src/exp06_llm_batch_prep.py
nohup python src/exp06_llm_run.py > logs/llm_run.log 2>&1 &

# 6. Morning: validation + abstract
python src/exp07_concordance.py
python src/exp10_abstract_numbers.py
```

## LLM Model Downloads

```bash
source .venv/bin/activate

# Recommended primary (fits fully in 24 GB VRAM at Q4_K_M ~19 GB)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF \
    qwen2.5-32b-instruct-q4_k_m.gguf --local-dir models/

# Secondary for cross-model validation
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
    Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf --local-dir models/
```

## Output Structure

```
results/
  exp01/  cohort_flowchart.txt, missing_data_audit.csv
  exp02/  misclassification_matrix_weighted.csv, roc_curve_fib4.png
  exp03/  feature_distributions.png, correlation_heatmap.png
  exp04/  xgboost_model.pkl, shap_beeswarm.png, shap_global_summary.png
  exp05/  umap_clusters.png, cluster_profiles.csv, cluster_radar_charts.png
  exp06/  llm_responses_qwen2.5-32b.json, llm_responses_llama-3.1-70b.json
  exp07/  llm_shap_concordance.csv, concordance_heatmap.png
  exp08/  false_negative_rates_by_ethnicity.csv, asian_specific_shap.png
  exp09/  fib4plus_vs_fib4_auroc.csv, decision_curve_analysis.png
  exp10/  abstract_key_numbers.json, abstract_draft.txt
```

## Project Structure

```
src/
  download_nhanes.py      # EXP-01: Download NHANES XPT files
  merge_nhanes.py         # EXP-01: Merge by SEQN
  define_cohort.py        # EXP-01: MASLD eligibility + quality filters
  compute_scores.py       # FIB-4, NFS, APRI, de Ritis, TyG, SII, ...
  exp02_misclassification.py
  exp03_feature_engineering.py
  exp04_xgboost_shap.py
  exp05_clustering.py
  exp06_llm_batch_prep.py
  exp06_llm_run.py
  exp07_concordance.py
  exp08_ethnicity.py
  exp09_fib4plus.py
  exp10_abstract_numbers.py
  utils/
    llm_utils.py          # LLM factory, batch runner, hallucination detector
    stats_utils.py        # Survey weights, bootstrap CI, DeLong test, NRI/IDI
    plot_utils.py         # All visualisation helpers
    nhanes_codebook.py    # NHANES variable to human label mapping
prompts/
  prompt_templates.py     # LLM prompt templates (PHENOTYPE, MECHANISM, APASL)
config/
  config.yaml             # All paths, thresholds, model settings
```

## Key Findings (populated after analysis run)

- FIB-4 false negative rate: **[TBD]%** of low-risk patients
- Number of distinct phenotype clusters: **[TBD]**
- LLM-SHAP concordance: **[TBD]%**
- FIB-4 Plus AUROC improvement: **[TBD]**
- Asian American false negative rate: **[TBD]x** overall rate
