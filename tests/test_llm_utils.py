"""Unit tests for llm_utils.py."""

import pytest
from src.utils.llm_utils import (
    extract_biomarkers_from_text,
    validate_llm_response,
    parse_llm_json,
)


def test_extract_biomarkers_basic():
    text = "The patient had elevated ALT and AST with significant fibrosis."
    found = extract_biomarkers_from_text(text)
    assert "alt" in found
    assert "ast" in found
    assert "fibrosis" in found


def test_extract_biomarkers_case_insensitive():
    text = "Elevated HBA1C and BMI with central adiposity."
    found = extract_biomarkers_from_text(text)
    assert "hba1c" in found
    assert "bmi" in found
    assert "waist" in found


def test_extract_biomarkers_synonyms():
    text = "Thrombocytopenia with hypertriglyceridaemia."
    found = extract_biomarkers_from_text(text)
    assert "platelets" in found
    assert "triglycerides" in found


def test_validate_no_hallucination():
    resp = {
        "parsed": {"description": "Elevated ALT and BMI drove the FIB-4 underestimation."},
        "status": "success",
    }
    result = validate_llm_response(resp)
    assert result["hallucination_detected"] is False


def test_parse_llm_json_plain():
    raw = '{"phenotype_name": "Metabolic Silent Fibroser", "confidence": "high"}'
    parsed = parse_llm_json(raw)
    assert parsed is not None
    assert parsed["phenotype_name"] == "Metabolic Silent Fibroser"


def test_parse_llm_json_with_fences():
    raw = '```json\n{"key": "value"}\n```'
    parsed = parse_llm_json(raw)
    assert parsed is not None
    assert parsed["key"] == "value"


def test_parse_llm_json_invalid():
    raw = "This is not JSON at all."
    parsed = parse_llm_json(raw)
    assert parsed is None


def test_parse_llm_json_embedded():
    raw = 'Some preamble text\n{"result": 42}\nSome postamble'
    parsed = parse_llm_json(raw)
    assert parsed is not None
    assert parsed["result"] == 42
