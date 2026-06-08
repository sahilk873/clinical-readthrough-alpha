"""Tests for upgraded risk model: regime-switching, copula, La-VaR."""

import numpy as np
import pandas as pd
import pytest

from clinical_alpha.risk.model import (
    copula_dependence,
    liquidity_adjusted_var,
    regime_switching_covariance,
    value_at_risk,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    returns = pd.DataFrame(
        {
            "A": np.random.randn(200) * 0.02,
            "B": np.random.randn(200) * 0.02 + 0.001,
            "C": np.random.randn(200) * 0.015 - 0.0005,
        },
        index=dates,
    )
    return returns


def test_regime_switching_covariance(sample_returns):
    result = regime_switching_covariance(sample_returns, n_regimes=2)
    assert "regime_covariances" in result
    assert "regime_weights" in result
    assert "regime_assignments" in result
    assert "regime_volatilities" in result
    assert len(result["regime_covariances"]) > 0


def test_regime_switching_covariance_single_asset():
    returns = pd.DataFrame({"A": np.random.randn(50) * 0.02})
    result = regime_switching_covariance(returns)
    assert 0 in result["regime_covariances"]


def test_regime_switching_covariance_short_series():
    returns = pd.DataFrame({"A": np.random.randn(5) * 0.02, "B": np.random.randn(5) * 0.02})
    result = regime_switching_covariance(returns)
    assert 0 in result["regime_covariances"]


def test_copula_dependence_gaussian(sample_returns):
    result = copula_dependence(sample_returns, method="gaussian")
    assert "kendall_tau" in result
    assert "upper_tail_dependence" in result
    assert "lower_tail_dependence" in result
    assert result["kendall_tau"].shape == (3, 3)
    assert np.allclose(np.diag(result["kendall_tau"]), 1.0)


def test_copula_dependence_clayton(sample_returns):
    result = copula_dependence(sample_returns, method="clayton")
    assert "theta" in result
    assert result["upper_tail_dependence"].shape == (3, 3)


def test_copula_dependence_frank(sample_returns):
    result = copula_dependence(sample_returns, method="frank")
    assert "theta" in result
    assert result["kendall_tau"].shape == (3, 3)


def test_copula_dependence_single_asset():
    returns = pd.DataFrame({"A": np.random.randn(50) * 0.02})
    result = copula_dependence(returns)
    assert np.allclose(result["kendall_tau"], np.array([[1.0]]))


def test_copula_dependence_two_assets():
    returns = pd.DataFrame(
        {
            "A": np.random.randn(100) * 0.02,
            "B": np.random.randn(100) * 0.02 + 0.5 * np.random.randn(100) * 0.02,
        }
    )
    result = copula_dependence(returns, method="gaussian")
    assert result["kendall_tau"].shape == (2, 2)


def test_liquidity_adjusted_var():
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.02)
    result = liquidity_adjusted_var(returns, avg_bid_ask_spread=0.001, confidence_level=0.95)
    assert "var" in result
    assert "liquidity_adjusted_var" in result
    assert "expected_shortfall" in result
    assert "liquidity_adjusted_es" in result
    assert "liquidity_cost_bps" in result
    assert result["liquidity_cost_bps"] > 0
    assert result["liquidity_adjusted_var"] <= result["var"]


def test_liquidity_adjusted_var_short_series():
    returns = pd.Series(np.random.randn(5) * 0.02)
    result = liquidity_adjusted_var(returns)
    assert isinstance(result["var"], float)


def test_liquidity_adjusted_var_different_spreads():
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.02)
    result_tight = liquidity_adjusted_var(returns, avg_bid_ask_spread=0.0001)
    result_wide = liquidity_adjusted_var(returns, avg_bid_ask_spread=0.01)
    assert result_wide["liquidity_cost_bps"] > result_tight["liquidity_cost_bps"]


def test_value_at_risk_all_methods():
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.02)
    for method in ["historical", "gaussian", "cornish_fisher"]:
        result = value_at_risk(returns, confidence_level=0.95, method=method)
        assert "var" in result
        assert "expected_shortfall" in result
        assert result["expected_shortfall"] <= result["var"]
