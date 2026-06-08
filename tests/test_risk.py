"""Tests for risk model module."""

import numpy as np
import pandas as pd

from clinical_alpha.risk.model import (
    min_variance_weights,
    pca_factor_model,
    risk_decomposition,
    risk_parity_weights,
    shrinkage_covariance,
    value_at_risk,
)


class TestShrinkageCovariance:
    def test_returns_dataframe(self):
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(100, 5), columns=list("ABCDE"))
        cov = shrinkage_covariance(returns)
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (5, 5)
        assert (cov.values == cov.values.T).all()

    def test_single_asset(self):
        returns = pd.DataFrame({"A": np.random.randn(100)})
        cov = shrinkage_covariance(returns)
        assert cov.shape == (1, 1)


class TestRiskDecomposition:
    def test_returns_dict(self):
        np.random.seed(42)
        n = 5
        cov = np.random.randn(n, n)
        cov = cov.T @ cov + np.eye(n) * 0.1
        weights = np.ones(n) / n
        result = risk_decomposition(weights, cov)
        assert "portfolio_vol" in result
        assert "marginal_contributions" in result
        assert "component_contributions" in result
        assert "diversification_ratio" in result
        assert result["portfolio_vol"] > 0

    def test_zero_weights(self):
        result = risk_decomposition(np.zeros(3), np.eye(3))
        assert result["portfolio_vol"] == 0.0


class TestPCAFactorModel:
    def test_returns_dict(self):
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(200, 10), columns=[f"A{i}" for i in range(10)])
        result = pca_factor_model(returns, n_factors=3)
        assert "factors" in result
        assert "loadings" in result
        assert "explained_var" in result
        assert result["factors"].shape[1] == 3
        assert abs(result["explained_var"].sum() - 1.0) < 0.01

    def test_small_dataset(self):
        returns = pd.DataFrame(np.random.randn(5, 3), columns=list("ABC"))
        result = pca_factor_model(returns, n_factors=5)
        assert result["factors"].shape[1] <= 3


class TestValueAtRisk:
    def test_historical_var(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.02)
        result = value_at_risk(returns, confidence_level=0.95, method="historical")
        assert "var" in result
        assert "expected_shortfall" in result
        assert result["var"] < 0

    def test_gaussian_var(self):
        returns = pd.Series(np.random.randn(1000) * 0.02)
        result = value_at_risk(returns, confidence_level=0.95, method="gaussian")
        assert "var" in result

    def test_short_series(self):
        result = value_at_risk(pd.Series([0.01, 0.02]), confidence_level=0.95)
        assert result["var"] == 0.0


class TestMinVarianceWeights:
    def test_weights_sum_to_one(self):
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(100, 5), columns=list("ABCDE"))
        cov = shrinkage_covariance(returns)
        weights = min_variance_weights(cov)
        assert abs(weights.sum() - 1.0) < 0.01
        assert all(w >= 0 for w in weights)

    def test_single_asset(self):
        cov = pd.DataFrame({"A": [0.04]})
        weights = min_variance_weights(cov)
        assert abs(weights["A"] - 1.0) < 0.01


class TestRiskParityWeights:
    def test_weights_sum_to_one(self):
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(100, 5), columns=list("ABCDE"))
        cov = shrinkage_covariance(returns)
        weights = risk_parity_weights(cov)
        assert abs(weights.sum() - 1.0) < 0.01
        assert all(w >= 0 for w in weights)

    def test_single_asset(self):
        cov = pd.DataFrame({"A": [0.04]})
        weights = risk_parity_weights(cov)
        assert abs(weights["A"] - 1.0) < 0.01
