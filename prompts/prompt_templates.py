"""LLM prompt templates for EXP-06 (versioned). See SRS §7.2."""

# ---------------------------------------------------------------------------
# PROMPT_PHENOTYPE — Clinical Phenotype Naming
# ---------------------------------------------------------------------------

SYSTEM_PHENOTYPE = """\
You are a clinical hepatologist assistant performing structured data analysis.
Your task is STRICTLY LIMITED to interpreting the numerical data provided.
You MUST NOT reference any patient history, clinical notes, or information
not explicitly provided in this prompt.
You MUST NOT invent symptoms, diagnoses, medications, or clinical context.
If a value is not provided, you MUST NOT assume or infer it.\
"""

USER_PHENOTYPE = """\
You are analysing a cluster of patients where FIB-4 incorrectly classified
them as "low risk" for liver fibrosis, but their liver stiffness measurement
(LSM) was elevated (≥ 8 kPa), indicating significant fibrosis.
Below are the MEDIAN laboratory and clinical values for this patient cluster.
Only these values exist. Do not assume anything else about these patients.

CLUSTER PROFILE:
- N patients in cluster: {n_patients}
- Median Age: {age} years
- Sex: {pct_female}% female
- Median BMI: {bmi} kg/m²
- Median Waist circumference: {waist} cm
- Median ALT: {alt} U/L  [Reference: Male 7–56 U/L, Female 7–45 U/L]
- Median AST: {ast} U/L  [Reference: 10–40 U/L]
- Median GGT: {ggt} U/L  [Reference: Male 8–61 U/L, Female 5–36 U/L]
- Median Platelet count: {plt} × 10³/μL  [Reference: 150–400]
- Median HbA1c: {hba1c}%  [Reference: <5.7% normal, 5.7–6.4% prediabetes, ≥6.5% diabetes]
- Median Triglycerides: {trig} mg/dL  [Reference: <150 normal]
- Median HDL cholesterol: {hdl} mg/dL  [Reference: Male >40, Female >50]
- Median Albumin: {alb} g/dL  [Reference: 3.5–5.0]
- Median Haemoglobin: {hgb} g/dL  [Reference: Male 13.5–17.5, Female 12.0–15.5]
- Computed FIB-4 score: {fib4} [Classified as: LOW RISK < 1.30]
- Median Liver Stiffness (LSM): {lsm} kPa [ELEVATED ≥ 8.0 kPa = significant fibrosis]
- % with Diabetes diagnosis: {pct_dm}%
- Race/ethnicity distribution: {race_dist}

TASK:
Based ONLY on the values above:
1. Propose a SHORT clinical phenotype name (3–5 words, e.g., "Metabolic Silent Fibroser")
2. Write EXACTLY 3 sentences explaining which specific values deviate from normal,
   and why these deviations might cause FIB-4 to UNDERESTIMATE fibrosis risk.
   Reference ONLY values that are actually abnormal in the profile above.
3. State which ONE variable in this profile most likely confounds FIB-4.

RESPONSE FORMAT (strict JSON, no other text):
{{
  "phenotype_name": "...",
  "description": "sentence1. sentence2. sentence3.",
  "primary_confounder": "variable_name",
  "confidence": "high|medium|low"
}}\
"""

# ---------------------------------------------------------------------------
# PROMPT_MECHANISM — FIB-4 Formula Failure Explanation
# ---------------------------------------------------------------------------

SYSTEM_MECHANISM = """\
You are a clinical hepatology expert. You are explaining why a mathematical
formula failed. Answer only with information derivable from the data provided.
Do not use generic statements. Every claim must reference a specific number
from the cluster profile below.\
"""

USER_MECHANISM = """\
FIB-4 formula: FIB-4 = (Age × AST) / (Platelet count × √ALT)

For FIB-4 to classify a patient as LOW RISK (< 1.30), the denominator
(Platelet count × √ALT) must be relatively large compared to the numerator.

Cluster profile:
- Computed FIB-4: {fib4} (LOW RISK threshold: < 1.30) ← patient was INCORRECTLY classified
- Actual LSM: {lsm} kPa (≥ 8.0 = SIGNIFICANT FIBROSIS) ← fibrosis IS present
- Age: {age} years
- AST: {ast} U/L
- ALT: {alt} U/L
- Platelet count: {plt} × 10³/μL
- GGT: {ggt} U/L (NOT in FIB-4 formula, but reference: Male 8–61, Female 5–36 U/L)
- HbA1c: {hba1c}% (NOT in FIB-4 formula, reference: <5.7% normal)
- BMI: {bmi} kg/m²

TASK:
In EXACTLY 2 sentences, explain mathematically why this cluster's specific
values caused FIB-4 to fall below 1.30 despite actual fibrosis being present.
Name the specific variable or ratio that is most responsible.
Then in 1 sentence, state what variable (not in FIB-4) would have helped detect
the fibrosis in this patient, and why, using only values from the profile.

RESPONSE FORMAT (strict JSON):
{{
  "formula_explanation": "sentence1. sentence2.",
  "missing_variable_explanation": "sentence3.",
  "key_misleading_variable": "variable_name"
}}\
"""

# ---------------------------------------------------------------------------
# PROMPT_APASL_RELEVANCE — Asia-Pacific Clinical Relevance
# ---------------------------------------------------------------------------

SYSTEM_APASL = """\
You are reviewing a liver disease patient cluster for its relevance to
Asia-Pacific populations. Use ONLY the data provided. Do not reference
specific country names unless the data supports it.\
"""

USER_APASL = """\
Cluster demographic and metabolic profile:
- Race: {race_dist}
- BMI: {bmi} kg/m² (Asian obesity threshold: ≥23 kg/m² vs ≥25 kg/m² standard)
- HbA1c: {hba1c}%
- Waist circumference: {waist} cm (Asian-specific abdominal obesity: ≥90 cm men, ≥80 cm women)
- Triglycerides: {trig} mg/dL
- LSM: {lsm} kPa (significant fibrosis ≥ 8.0)
- FIB-4: {fib4} (incorrectly classified as low risk)

TASK:
In EXACTLY 2 sentences, describe whether the anthropometric values in this
cluster are consistent with phenotypes commonly described in Asian populations
with MASLD (lower BMI, normal-weight metabolic syndrome, central adiposity).
Reference ONLY the numerical values provided.

RESPONSE FORMAT (strict JSON):
{{
  "asian_relevance_statement": "sentence1. sentence2.",
  "asian_bmi_concern": true,
  "central_adiposity_concern": true
}}\
"""


def fill_phenotype_prompt(cluster: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for PHENOTYPE task."""
    return SYSTEM_PHENOTYPE, USER_PHENOTYPE.format(**cluster)


def fill_mechanism_prompt(cluster: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for MECHANISM task."""
    return SYSTEM_MECHANISM, USER_MECHANISM.format(**cluster)


def fill_apasl_prompt(cluster: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for APASL_RELEVANCE task."""
    return SYSTEM_APASL, USER_APASL.format(**cluster)


# ---------------------------------------------------------------------------
# PROMPT_DIFFERENTIATION — Force unique, differentiated cluster names
# ---------------------------------------------------------------------------

SYSTEM_DIFFERENTIATION = """\
You are naming distinct patient clusters. Each cluster name MUST be UNIQUE
and capture what makes THAT cluster different from the others in the table.
Use only the data provided. Do not invent clinical details.\
"""

USER_DIFFERENTIATION = """\
You are naming {n_clusters} patient clusters of FIB-4 false negatives (patients
FIB-4 classified as low risk but who have elevated liver stiffness ≥ 8 kPa).
Each cluster represents a distinct metabolic phenotype.

CLUSTER COMPARISON TABLE (median values):
{cluster_table}

TASK:
For each cluster, provide a unique clinical name (3–5 words) that captures
the KEY distinguishing feature of THAT cluster vs the others.
Names must differ — do not use the same primary descriptor for two clusters.

RESPONSE FORMAT (strict JSON, one entry per cluster):
{{
  "cluster_names": [
    {{"cluster_id": 0, "name": "...", "key_difference": "1 sentence"}},
    {{"cluster_id": 1, "name": "...", "key_difference": "1 sentence"}}
  ]
}}\
"""


def fill_differentiation_prompt(ctx: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for DIFFERENTIATION task."""
    return SYSTEM_DIFFERENTIATION, USER_DIFFERENTIATION.format(**ctx)


PROMPT_REGISTRY = {
    "PHENOTYPE": fill_phenotype_prompt,
    "MECHANISM": fill_mechanism_prompt,
    "APASL_RELEVANCE": fill_apasl_prompt,
    "DIFFERENTIATION": fill_differentiation_prompt,
}
