"""Tests for upgraded factor models: Carhart 4F, Fama-MacBeth, model selection."""

import numpy as np
import pandas as pd
import pytest

from clinical_alpha.returns.factor_models import (
    compare_factor_models,
    estimate_capm,
    estimate_carhart_4f,
    estimate_ff3,
    estimate_ff5,
    fama_macbeth,
)


@pytest.fixture
def sample_returns_factors():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    stock = pd.Series(np.random.randn(n) * 0.02, index=dates, name="stock")
    mkt = pd.Series(np.random.randn(n) * 0.01, index=dates, name="mkt_rf")
    smb = pd.Series(np.random.randn(n) * 0.005, index=dates, name="smb")
    hml = pd.Series(np.random.randn(n) * 0.005, index=dates, name="hml")
    mom = pd.Series(np.random.randn(n) * 0.005, index=dates, name="mom")
    rmw = pd.Series(np.random.randn(n) * 0.004, index=dates, name="rmw")
    cma = pd.Series(np.random.randn(n) * 0.004, index=dates, name="cma")
    return stock, mkt, smb, hml, mom, rmw, cma


def test_carhart_4f(sample_returns_factors):
    stock, mkt, smb, hml, mom, _, _ = sample_returns_factors
    result = estimate_carhart_4f(stock, mkt, smb, hml, mom)
    assert result["n_obs"] > 0
    assert isinstance(result["alpha"], float)
    assert isinstance(result["beta_mkt"], float)
    assert isinstance(result["beta_smb"], float)
    assert isinstance(result["beta_hml"], float)
    assert isinstance(result["beta_mom"], float)
    assert result["rsquared"] >= 0
    assert result["adj_rsquared"] >= 0


def test_carhart_4f_short_series():
    stock = pd.Series(np.random.randn(10), name="stock")
    mkt = pd.Series(np.random.randn(10), name="mkt")
    smb = pd.Series(np.random.randn(10), name="smb")
    hml = pd.Series(np.random.randn(10), name="hml")
    mom = pd.Series(np.random.randn(10), name="mom")
    result = estimate_carhart_4f(stock, mkt, smb, hml, mom)
    assert result["n_obs"] == 0


def test_fama_macbeth(sample_returns_factors):
    _, mkt, smb, hml, _, _, _ = sample_returns_factors
    n_stocks = 10
    n_periods = 200
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="B")
    returns = pd.DataFrame(
        {f"stock_{i}": np.random.randn(n_periods) * 0.02 for i in range(n_stocks)},
        index=dates,
    )
    factors = pd.DataFrame(
        {
            "mkt_rf": np.random.randn(n_periods) * 0.01,
            "smb": np.random.randn(n_periods) * 0.005,
            "hml": np.random.randn(n_periods) * 0.005,
        },
        index=dates,
    )

    result = fama_macbeth(returns, factors)
    assert result["n_assets"] == n_stocks
    assert result["n_periods"] == n_periods
    assert len(result["avg_coefs"]) == 4  # alpha + 3 factors
    assert len(result["shanken_se"]) == 4
    assert len(result["shanken_tstats"]) == 4
    assert result["avg_rsquared"] >= 0


def test_fama_macbeth_too_few_assets():
    returns = pd.DataFrame({"a": np.random.randn(50)})
    factors = pd.DataFrame({"f1": np.random.randn(50)})
    result = fama_macbeth(returns, factors)
    assert result["n_assets"] == 0


def test_compare_factor_models(sample_returns_factors):
    stock, mkt, smb, hml, mom, rmw, cma = sample_returns_factors
    factor_sets = {
        "CAPM": {"market_returns": mkt},
        "FF3": {"mkt_rf": mkt, "smb": smb, "hml": hml},
        "Carhart4F": {"mkt_rf": mkt, "smb": smb, "hml": hml, "mom": mom},
        "FF5": {"mkt_rf": mkt, "smb": smb, "hml": hml, "rmw": rmw, "cma": cma},
    }
    results = compare_factor_models(stock, factor_sets)
    assert len(results) == 4
    assert "model" in results.columns
    assert "rsquared" in results.columns
    assert "aic" in results.columns
    assert "bic" in results.columns
    assert "adj_rsquared" in results.columns
    assert all(results["rsquared"] >= 0)


def test_compare_factor_models_empty():
    stock = pd.Series(np.random.randn(100))
    results = compare_factor_models(stock, {})
    assert len(results) == 0


def test_capm_with_selection_criteria(sample_returns_factors):
    stock, mkt, _, _, _, _, _ = sample_returns_factors
    result = estimate_capm(stock, mkt)
    assert "adj_rsquared" in result
    assert "aic" in result
    assert "bic" in result
    assert isinstance(result["adj_rsquared"], float)
    assert isinstance(result["aic"], float)
    assert isinstance(result["bic"], float)


def test_ff3_with_selection_criteria(sample_returns_factors):
    stock, mkt, smb, hml, _, _, _ = sample_returns_factors
    result = estimate_ff3(stock, mkt, smb, hml)
    assert "adj_rsquared" in result
    assert "aic" in result
    assert "bic" in result


def test_ff5_with_selection_criteria(sample_returns_factors):
    stock, mkt, smb, hml, _, rmw, cma = sample_returns_factors
    result = estimate_ff5(stock, mkt, smb, hml, rmw, cma)
    assert "adj_rsquared" in result
    assert "aic" in result
    assert "bic" in result


def test_fama_macbeth_rolling(sample_returns_factors):
    _, mkt, smb, hml, _, _, _ = sample_returns_factors
    n_periods = 100
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="B")
    returns = pd.DataFrame(
        {f"s{i}": np.random.randn(n_periods) * 0.02 for i in range(5)},
        index=dates,
    )
    factors = pd.DataFrame(
        {
            "mkt_rf": mkt.values[:n_periods],
            "smb": smb.values[:n_periods],
            "hml": hml.values[:n_periods],
        },
        index=dates,
    )

    result = fama_macbeth(returns, factors, rolling_window=50)
    assert result["n_assets"] == 5
    assert len(result["avg_coefs"]) == 4
