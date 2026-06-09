"""Unit tests for compute_scores.py."""

import numpy as np
import pandas as pd
import pytest

from src.compute_scores import (
    compute_fib4,
    compute_fib4_category,
    compute_apri,
    compute_de_ritis,
    compute_nfs,
    compute_tyg,
    compute_whtr,
    compute_lsm_category,
)


@pytest.fixture
def sample_row():
    return pd.DataFrame([{
        "RIDAGEYR": 55.0,
        "LBXSASSI": 35.0,   # AST U/L
        "LBXSATSI": 25.0,   # ALT U/L
        "LBXPLTSI": 200.0,  # Platelets
        "BMXBMI": 28.0,
        "DIQ010": 2,        # No diabetes
        "RIAGENDR": 1,      # Male
        "LBXSAL": 4.2,      # Albumin
        "BMXWAIST": 95.0,
        "BMXHT": 170.0,
        "LBXTR": 150.0,
        "LBXSGL": 90.0,
        "LUXSMED": 7.0,
        "LBXWBCSI": 6.0,
    }])


def test_fib4_calculation(sample_row):
    fib4 = compute_fib4(sample_row)
    # FIB-4 = (55 * 35) / (200 * sqrt(25)) = 1925 / 1000 = 1.925
    expected = (55 * 35) / (200 * np.sqrt(25))
    assert abs(fib4.iloc[0] - expected) < 0.001


def test_fib4_low_risk():
    df = pd.DataFrame([{
        "RIDAGEYR": 30.0, "LBXSASSI": 20.0,
        "LBXSATSI": 20.0, "LBXPLTSI": 300.0,
    }])
    fib4 = compute_fib4(df)
    cat = compute_fib4_category(fib4)
    assert cat.iloc[0] == 0


def test_fib4_high_risk():
    df = pd.DataFrame([{
        "RIDAGEYR": 75.0, "LBXSASSI": 80.0,
        "LBXSATSI": 40.0, "LBXPLTSI": 80.0,
    }])
    fib4 = compute_fib4(df)
    cat = compute_fib4_category(fib4)
    assert cat.iloc[0] == 2


def test_apri(sample_row):
    apri = compute_apri(sample_row)
    # APRI = (35/40) / 200 * 100 = 0.4375
    expected = (35.0 / 40.0) / 200.0 * 100
    assert abs(apri.iloc[0] - expected) < 0.01


def test_de_ritis(sample_row):
    dr = compute_de_ritis(sample_row)
    assert abs(dr.iloc[0] - 35.0 / 25.0) < 0.001


def test_tyg(sample_row):
    tyg = compute_tyg(sample_row)
    expected = np.log(150.0 * 90.0 / 2)
    assert abs(tyg.iloc[0] - expected) < 0.001


def test_whtr(sample_row):
    whtr = compute_whtr(sample_row)
    assert abs(whtr.iloc[0] - 95.0 / 170.0) < 0.001


def test_lsm_category():
    df = pd.DataFrame([
        {"LUXSMED": 5.0},
        {"LUXSMED": 9.0},
        {"LUXSMED": 14.0},
    ])
    cats = compute_lsm_category(df)
    assert cats.iloc[0] == 0  # No/Mild
    assert cats.iloc[1] == 1  # Significant
    assert cats.iloc[2] == 2  # Advanced


def test_fib4_no_zero_division():
    df = pd.DataFrame([{
        "RIDAGEYR": 50.0, "LBXSASSI": 30.0,
        "LBXSATSI": 0.0, "LBXPLTSI": 0.0,
    }])
    fib4 = compute_fib4(df)
    assert not np.isnan(fib4.iloc[0])
    assert not np.isinf(fib4.iloc[0])
