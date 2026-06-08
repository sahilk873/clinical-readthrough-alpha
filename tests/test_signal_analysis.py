"""Tests for signal analysis module: IC, signal decay, signal-to-noise."""

import numpy as np
import pandas as pd
import pytest

from clinical_alpha.signal.analysis import (
    compute_ic_time_series,
    compute_information_coefficient,
    compute_signal_decay,
    compute_signal_noise_ratio,
    evaluate_signal_cross_sectional,
    signal_contribution_decomposition,
)


@pytest.fixture
def sample_signal_data():
    np.random.seed(42)
    n = 100
    signal = pd.Series(np.random.randn(n), name="signal")
    returns = pd.Series(signal.values * 0.5 + np.random.randn(n) * 0.1, name="fwd_return")
    return signal, returns


@pytest.fixture
def sample_panel_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    tickers = [f"STOCK_{i}" for i in range(20)]
    signals = pd.DataFrame(
        np.random.randn(50, 20),
        index=dates,
        columns=tickers,
    )
    returns = pd.DataFrame(
        signals.values * 0.3 + np.random.randn(50, 20) * 0.1,
        index=dates,
        columns=tickers,
    )
    return signals, returns


def test_information_coefficient_pearson(sample_signal_data):
    signal, returns = sample_signal_data
    result = compute_information_coefficient(signal, returns, method="pearson")
    assert "ic" in result
    assert "p_value" in result
    assert "n_obs" in result
    assert result["n_obs"] == 100
    assert -1 <= result["ic"] <= 1


def test_information_coefficient_spearman(sample_signal_data):
    signal, returns = sample_signal_data
    result = compute_information_coefficient(signal, returns, method="spearman")
    assert "ic" in result
    assert -1 <= result["ic"] <= 1


def test_information_coefficient_small_sample():
    signal = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    returns = pd.Series([0.05, 0.1, 0.15, 0.2, 0.25])
    result = compute_information_coefficient(signal, returns)
    assert result["n_obs"] == 5


def test_information_coefficient_too_few():
    signal = pd.Series([0.1])
    returns = pd.Series([0.05])
    result = compute_information_coefficient(signal, returns)
    assert result["n_obs"] == 0


def test_ic_time_series(sample_panel_data):
    signals, returns = sample_panel_data
    ic_series = compute_ic_time_series(signals, returns, min_obs=10)
    assert isinstance(ic_series, pd.Series)
    assert len(ic_series) > 0
    assert all(-1 <= v <= 1 for v in ic_series)


def test_ic_time_series_spearman(sample_panel_data):
    signals, returns = sample_panel_data
    ic_series = compute_ic_time_series(signals, returns, method="spearman", min_obs=10)
    assert len(ic_series) > 0


def test_ic_time_series_high_min_obs(sample_panel_data):
    signals, returns = sample_panel_data
    ic_series = compute_ic_time_series(signals, returns, min_obs=100)
    assert len(ic_series) == 0


def test_signal_decay(sample_signal_data):
    signal, returns = sample_signal_data
    n = len(returns)
    fwd_returns = pd.DataFrame(
        {
            "fwd_1d": returns.values,
            "fwd_5d": returns.values * 0.8 + np.random.randn(n) * 0.02,
            "fwd_10d": returns.values * 0.6 + np.random.randn(n) * 0.03,
        }
    )
    decay = compute_signal_decay(signal, fwd_returns, lags=[1, 5, 10])
    assert isinstance(decay, pd.DataFrame)
    assert len(decay) == 3
    assert "lag" in decay.columns
    assert "ic" in decay.columns
    assert all(decay["ic"].notna())


def test_signal_decay_with_series():
    signal = pd.Series(np.random.randn(50), name="signal")
    fwd = pd.Series(signal.values * 0.5 + np.random.randn(50) * 0.1, name="fwd_1d")
    decay = compute_signal_decay(signal, pd.DataFrame({"fwd_1d": fwd.values}), lags=[1])
    assert len(decay) == 1


def test_signal_noise_ratio(sample_signal_data):
    signal, returns = sample_signal_data
    result = compute_signal_noise_ratio(signal, returns)
    assert "snr" in result
    assert "information_ratio" in result
    assert "mean_signal_return" in result
    assert "top_quantile_mean" in result
    assert "bottom_quantile_mean" in result
    assert "spread" in result
    assert result["n_obs"] == 100


def test_signal_noise_ratio_small():
    signal = pd.Series(np.random.randn(4))
    returns = pd.Series(np.random.randn(4))
    result = compute_signal_noise_ratio(signal, returns)
    assert result["n_obs"] < 5


def test_evaluate_signal_cross_sectional(sample_panel_data):
    signals, returns = sample_panel_data
    result = evaluate_signal_cross_sectional(signals, returns, n_quantiles=5)
    assert "mean_ic" in result
    assert "mean_rank_ic" in result
    assert "ic_sharpe" in result
    assert "top_minus_bottom_spread" in result
    assert "quantile_1_mean_return" in result
    assert "quantile_5_mean_return" in result
    assert isinstance(result["mean_ic"], float)


def test_evaluate_signal_cross_sectional_three_quantiles(sample_panel_data):
    signals, returns = sample_panel_data
    result = evaluate_signal_cross_sectional(signals, returns, n_quantiles=3)
    assert "quantile_1_mean_return" in result
    assert "quantile_3_mean_return" in result


def test_signal_contribution_decomposition(sample_panel_data):
    signals, returns = sample_panel_data
    bench = pd.Series(np.random.randn(50) * 0.01, index=signals.index)
    result = signal_contribution_decomposition(returns, signals, bench)
    assert "alpha" in result
    assert "beta" in result
    assert "systematic_vol" in result
    assert "idiosyncratic_vol" in result
    assert "systematic_pct" in result
    assert "idiosyncratic_pct" in result


def test_signal_contribution_decomposition_single_stock():
    returns = pd.DataFrame({"A": np.random.randn(50) * 0.02})
    signals = pd.DataFrame({"A": np.random.randn(50)})
    bench = pd.Series(np.random.randn(50) * 0.01)
    result = signal_contribution_decomposition(returns, signals, bench)
    assert "alpha" in result


def test_evaluate_signal_cross_sectional_ic_sharpe(sample_panel_data):
    signals, returns = sample_panel_data
    result = evaluate_signal_cross_sectional(signals, returns)
    if result["ic_std"] > 0:
        assert result["ic_sharpe"] == result["mean_ic"] / result["ic_std"]
