"""Tests for factor models module."""

import numpy as np
import pandas as pd

from clinical_alpha.returns.factor_models import (
    estimate_capm,
    estimate_ff3,
    estimate_ff5,
    grs_test,
    newey_west_se,
)


class TestNeweyWestSE:
    def test_returns_se_array(self):
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.array([0.001, 1.2])
        y = X @ beta + np.random.randn(n) * 0.01
        residuals = y - X @ beta
        se = newey_west_se(X, residuals)
        assert len(se) == 2
        assert all(s > 0 for s in se)


class TestEstimateCAPM:
    def test_returns_expected_keys(self):
        np.random.seed(42)
        n = 252
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        mkt = pd.Series(np.random.randn(n) * 0.01, index=dates)
        stock = pd.Series(0.0005 + 1.2 * mkt.values + np.random.randn(n) * 0.005, index=dates)
        result = estimate_capm(stock, mkt)
        assert "alpha" in result
        assert "beta" in result
        assert "alpha_tstat" in result
        assert "beta_tstat" in result
        assert "rsquared" in result
        assert abs(result["beta"] - 1.2) < 0.3
        assert result["n_obs"] > 0

    def test_short_series(self):
        stock = pd.Series([0.01, 0.02], index=range(2))
        mkt = pd.Series([0.005, 0.01], index=range(2))
        result = estimate_capm(stock, mkt)
        assert result["n_obs"] == 0


class TestEstimateFF3:
    def test_returns_expected_keys(self):
        np.random.seed(42)
        n = 252
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        mkt = pd.Series(np.random.randn(n) * 0.01, index=dates)
        smb = pd.Series(np.random.randn(n) * 0.005, index=dates)
        hml = pd.Series(np.random.randn(n) * 0.005, index=dates)
        stock = pd.Series(
            0.0005
            + 1.2 * mkt.values
            + 0.3 * smb.values
            - 0.2 * hml.values
            + np.random.randn(n) * 0.005,
            index=dates,
        )
        result = estimate_ff3(stock, mkt, smb, hml, use_newey_west=False)
        assert "alpha" in result
        assert "beta_mkt" in result
        assert "beta_smb" in result
        assert "beta_hml" in result
        assert abs(result["beta_mkt"] - 1.2) < 0.3
        assert result["n_obs"] > 0

    def test_short_series(self):
        stock = pd.Series([0.01] * 30)
        mkt = pd.Series([0.005] * 30)
        smb = pd.Series([0.001] * 30)
        hml = pd.Series([0.002] * 30)
        result = estimate_ff3(stock, mkt, smb, hml)
        assert result["n_obs"] > 0


class TestEstimateFF5:
    def test_returns_expected_keys(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        factors = {
            name: pd.Series(np.random.randn(n) * 0.01, index=dates)
            for name in ["mkt_rf", "smb", "hml", "rmw", "cma"]
        }
        stock = pd.Series(
            0.0005 + sum(np.random.randn() * factors[name].values for name in factors),
            index=dates,
        )
        result = estimate_ff5(stock, **factors, use_newey_west=False)
        assert "alpha" in result
        assert "beta_mkt" in result
        assert "rsquared" in result
        assert result["n_obs"] > 0


class TestGRSTest:
    def test_returns_dict(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        factors = pd.DataFrame({"Mkt": np.random.randn(n) * 0.01}, index=dates)
        funds = pd.DataFrame(
            {
                "A": 0.001 + 1.0 * factors["Mkt"].values + np.random.randn(n) * 0.01,
                "B": -0.001 + 0.8 * factors["Mkt"].values + np.random.randn(n) * 0.01,
            },
            index=dates,
        )
        result = grs_test(funds, factors)
        assert "grs_statistic" in result
        assert "p_value" in result
        assert result["df1"] > 0

    def test_insufficient_data(self):
        result = grs_test(pd.DataFrame(), pd.DataFrame())
        assert result["p_value"] == 1.0
