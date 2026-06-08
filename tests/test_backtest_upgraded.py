"""Tests for upgraded backtest engine: Black-Litterman, HRP, Bayesian, TC-aware."""

import numpy as np
import pandas as pd
import pytest

from clinical_alpha.backtest.engine import (
    bayesian_mean_variance_weights,
    black_litterman_weights,
    hierarchical_risk_parity,
    tc_aware_weights,
)


@pytest.fixture
def sample_covariance():
    np.random.seed(42)
    n = 5
    tickers = [f"ASSET_{i}" for i in range(n)]
    vol = np.random.uniform(0.15, 0.40, n)
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vol, vol) * corr
    return pd.DataFrame(cov, index=tickers, columns=tickers)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    n = 5
    tickers = [f"ASSET_{i}" for i in range(n)]
    returns = pd.DataFrame(
        np.random.randn(252, n) * 0.02,
        index=dates,
        columns=tickers,
    )
    return returns


def test_black_litterman_equal_prior(sample_covariance):
    weights = black_litterman_weights(sample_covariance)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 5
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)


def test_black_litterman_with_views(sample_covariance):
    views = {"ASSET_0": 0.05, "ASSET_2": -0.02}
    confidences = {"ASSET_0": 0.8, "ASSET_2": 0.6}
    weights = black_litterman_weights(
        sample_covariance,
        views=views,
        view_confidences=confidences,
    )
    assert len(weights) == 5
    assert abs(weights.sum() - 1.0) < 1e-6


def test_black_litterman_single_asset():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    weights = black_litterman_weights(cov)
    assert abs(weights["A"] - 1.0) < 1e-6


def test_black_litterman_custom_prior(sample_covariance):
    market_cap = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=sample_covariance.columns)
    weights = black_litterman_weights(sample_covariance, market_cap_weights=market_cap)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_hierarchical_risk_parity(sample_covariance):
    weights = hierarchical_risk_parity(sample_covariance)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 5
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)


def test_hierarchical_risk_parity_two_assets():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    weights = hierarchical_risk_parity(cov)
    assert len(weights) == 2
    assert abs(weights.sum() - 1.0) < 1e-6


def test_hierarchical_risk_parity_single_asset():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    weights = hierarchical_risk_parity(cov)
    assert abs(weights["A"] - 1.0) < 1e-6


def test_bayesian_mean_variance(sample_returns):
    weights = bayesian_mean_variance_weights(sample_returns)
    assert isinstance(weights, pd.Series)
    assert len(weights) == 5
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)


def test_bayesian_mean_variance_single_asset():
    returns = pd.DataFrame({"A": np.random.randn(100) * 0.02})
    weights = bayesian_mean_variance_weights(returns)
    assert abs(weights["A"] - 1.0) < 1e-6


def test_bayesian_mean_variance_prior_shrinkage(sample_returns):
    weights_low = bayesian_mean_variance_weights(sample_returns, prior_shrinkage=0.1)
    weights_high = bayesian_mean_variance_weights(sample_returns, prior_shrinkage=0.9)
    assert abs(weights_low.sum() - 1.0) < 1e-6
    assert abs(weights_high.sum() - 1.0) < 1e-6


def test_tc_aware_weights(sample_covariance):
    current = pd.Series(0.2, index=sample_covariance.columns)
    target = pd.Series([0.4, 0.3, 0.1, 0.1, 0.1], index=sample_covariance.columns)
    weights = tc_aware_weights(current, target, sample_covariance, tc_bps=10.0)
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_tc_aware_weights_single_asset():
    current = pd.Series([1.0], index=["A"])
    target = pd.Series([1.0], index=["A"])
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    weights = tc_aware_weights(current, target, cov)
    assert abs(weights["A"] - 1.0) < 1e-6


def test_all_weighting_schemes_diversify(sample_covariance):
    bw = black_litterman_weights(sample_covariance)
    hrp = hierarchical_risk_parity(sample_covariance)
    assert not bw.dropna().empty
    assert not hrp.dropna().empty
