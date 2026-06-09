"""Unit tests for stats_utils.py."""

import numpy as np
import pandas as pd
import pytest

from src.utils.stats_utils import (
    weighted_proportion,
    compute_auroc_ci,
    sensitivity_specificity,
    youden_cutoff,
    compute_nri_idi,
)


def test_weighted_proportion_all_true():
    mask = pd.Series([True, True, True])
    weights = pd.Series([1.0, 2.0, 3.0])
    prop, n = weighted_proportion(mask, weights)
    assert prop == 1.0
    assert n == 3


def test_weighted_proportion_half():
    mask = pd.Series([True, False, True, False])
    weights = pd.Series([1.0, 1.0, 1.0, 1.0])
    prop, n = weighted_proportion(mask, weights)
    assert abs(prop - 0.5) < 0.001
    assert n == 2


def test_weighted_proportion_empty():
    mask = pd.Series([False, False])
    weights = pd.Series([1.0, 1.0])
    prop, n = weighted_proportion(mask, weights)
    assert prop == 0.0
    assert n == 0


def test_sensitivity_specificity_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    ss = sensitivity_specificity(y_true, y_pred)
    assert ss["sensitivity"] == 1.0
    assert ss["specificity"] == 1.0


def test_sensitivity_specificity_zeros():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    ss = sensitivity_specificity(y_true, y_pred)
    assert ss["sensitivity"] == 0.0
    assert ss["specificity"] == 1.0


def test_auroc_ci_random():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    result = compute_auroc_ci(y_true, y_score, n_iter=100)
    assert 0.0 < result["auroc"] < 1.0
    assert result["ci_lo"] <= result["auroc"] <= result["ci_hi"]


def test_auroc_ci_perfect():
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.array([0.1] * 50 + [0.9] * 50)
    result = compute_auroc_ci(y_true, y_score, n_iter=100)
    assert result["auroc"] == 1.0


def test_youden_cutoff():
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.concatenate([np.random.default_rng(0).uniform(0, 0.5, 50),
                              np.random.default_rng(1).uniform(0.5, 1.0, 50)])
    cutoff = youden_cutoff(y_true, y_score)
    assert 0.0 < cutoff < 1.0


def test_nri_positive():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    p_old = rng.uniform(0.2, 0.6, 200)
    p_new = p_old + 0.1 * y_true  # new model is better for events
    result = compute_nri_idi(y_true, p_old, p_new)
    assert "nri" in result
    assert "idi" in result
    assert isinstance(result["nri"], float)
