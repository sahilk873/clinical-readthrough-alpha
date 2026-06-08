"""Tests for upgraded statistical tests: Bayes factors, FDR bootstrap, structural breaks."""

import numpy as np
import pandas as pd
import pytest

from clinical_alpha.returns.statistical_tests import (
    _fdr_bootstrap_stepup,
    adjust_pvalues_multiple_testing,
    bai_perron_test,
    bayes_factor_car,
    bootstrap_clustered_car_ci,
    chow_structural_break_test,
)


@pytest.fixture
def sample_car_data():
    np.random.seed(42)
    peer = np.random.randn(30) * 0.03 + 0.01
    control = np.random.randn(30) * 0.03
    return list(peer), list(control)


def test_bayes_factor_substantial(sample_car_data):
    peer, control = sample_car_data
    result = bayes_factor_car(peer, control)
    assert "bf10" in result
    assert "bf01" in result
    assert "log_bf" in result
    assert "interpretation" in result
    assert result["bf10"] > 0
    assert result["bf01"] > 0


def test_bayes_factor_small_sample():
    result = bayes_factor_car([0.01, 0.02], [-0.01, 0.0])
    assert result["bf10"] == 1.0
    assert result["bf01"] == 1.0


def test_bayes_factor_interpretation():
    peer = [0.05] * 20 + [0.03] * 10
    control = [-0.02] * 20 + [0.01] * 10
    result = bayes_factor_car(peer, control)
    assert isinstance(result["interpretation"], str)


def test_bootstrap_clustered_car_ci():
    np.random.seed(42)
    cars = pd.Series(np.random.randn(50) * 0.02)
    clusters = pd.Series([f"cluster_{i % 5}" for i in range(50)])
    result = bootstrap_clustered_car_ci(cars, clusters, n_iterations=500)
    assert "mean" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "n_clusters" in result
    assert result["n_clusters"] == 5
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


def test_bootstrap_clustered_car_ci_small():
    cars = pd.Series([0.01, 0.02])
    clusters = pd.Series(["a", "a"])
    result = bootstrap_clustered_car_ci(cars, clusters)
    assert result["mean"] == 0.015


def test_chow_structural_break_test():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    returns = pd.Series(np.random.randn(200) * 0.02, index=dates)
    factors = pd.DataFrame(
        {
            "mkt": np.random.randn(200) * 0.01,
            "smb": np.random.randn(200) * 0.005,
        },
        index=dates,
    )

    result = chow_structural_break_test(returns, dates[100], factors)
    assert "chow_statistic" in result
    assert "p_value" in result
    assert "df1" in result
    assert "df2" in result
    assert result["df1"] > 0


def test_chow_structural_break_short_series():
    returns = pd.Series(
        np.random.randn(10), index=pd.date_range("2023-01-01", periods=10, freq="B")
    )
    factors = pd.DataFrame({"mkt": np.random.randn(10)}, index=returns.index)
    result = chow_structural_break_test(returns, returns.index[5], factors)
    assert result["chow_statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_bai_perron_test():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    returns = pd.Series(np.random.randn(n) * 0.02, index=dates)
    factors = pd.DataFrame({"mkt": np.random.randn(n) * 0.01}, index=dates)

    result = bai_perron_test(returns, factors, max_breaks=2, min_segment_length=30)
    assert "n_breaks" in result
    assert "break_dates" in result
    assert "break_stats" in result
    assert 0 <= result["n_breaks"] <= 2


def test_bai_perron_too_short():
    returns = pd.Series(
        np.random.randn(30), index=pd.date_range("2023-01-01", periods=30, freq="B")
    )
    factors = pd.DataFrame({"mkt": np.random.randn(30)}, index=returns.index)
    result = bai_perron_test(returns, factors, max_breaks=3, min_segment_length=20)
    assert result["n_breaks"] == 0


def test_adjust_pvalues_fdr_bootstrap():
    p_values = [0.001, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 0.80]
    adjusted = adjust_pvalues_multiple_testing(p_values, method="fdr_bootstrap")
    assert len(adjusted) == len(p_values)
    assert all(0 <= p <= 1.0 for p in adjusted)


def test_adjust_pvalues_unknown_method():
    with pytest.raises(Exception):
        adjust_pvalues_multiple_testing([0.05], method="unknown")


def test_fdr_bootstrap_stepup():
    p_values = np.array([0.001, 0.01, 0.02, 0.05, 0.10, 0.50])
    adjusted = _fdr_bootstrap_stepup(p_values, n_bootstrap=200)
    assert len(adjusted) == len(p_values)
    assert all(p >= 0 for p in adjusted)
    assert all(p <= 1.0 for p in adjusted)


def test_bayes_factor_identical_groups():
    peer = [0.01] * 10
    control = [0.01] * 10
    result = bayes_factor_car(peer, control)
    assert isinstance(result["bf10"], float)
