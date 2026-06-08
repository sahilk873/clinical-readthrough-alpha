"""Tests for abnormal return calculations."""

import numpy as np
import pandas as pd

from clinical_alpha.returns.abnormal import (
    AbnormalReturnCalculator,
    compute_abnormal_returns_single_benchmark,
    compute_car,
    compute_market_model_ar,
    compute_regression_residual_ar,
    compute_t_statistic,
)


class TestComputeAbnormalReturns:
    def test_single_benchmark_returns_series(self):
        stock = pd.Series([0.01, 0.02, -0.01, 0.005], index=range(4))
        bench = pd.Series([0.005, 0.01, -0.005, 0.003], index=range(4))
        ar = compute_abnormal_returns_single_benchmark(stock, bench)
        assert len(ar) == 4
        assert abs(ar.iloc[0] - 0.005) < 1e-10

    def test_aligns_different_indices(self):
        stock = pd.Series([0.01, 0.02, -0.01], index=[1, 2, 3])
        bench = pd.Series([0.005, 0.01, -0.005], index=[2, 3, 4])
        ar = compute_abnormal_returns_single_benchmark(stock, bench)
        assert len(ar) == 2

    def test_market_model_returns_series(self):
        np.random.seed(42)
        n = 300
        bench = pd.Series(np.random.randn(n) * 0.01, index=range(n))
        stock = pd.Series(0.001 + 1.2 * bench.values + np.random.randn(n) * 0.005, index=range(n))
        ar = compute_market_model_ar(stock, bench, estimation_window=100)
        assert len(ar) == n
        assert not ar.isna().all()


class TestCAR:
    def test_car_calculation(self):
        ar = pd.Series([0.01, 0.02, -0.005, 0.015, -0.01], index=range(5))
        car = compute_car(ar, (0, 3))
        expected = ar.iloc[0:4].sum()
        assert abs(car - expected) < 1e-10

    def test_car_empty_series(self):
        assert compute_car(pd.Series([], dtype=float), (0, 5)) == 0.0

    def test_car_clamps_window(self):
        ar = pd.Series([0.01, 0.02], index=range(2))
        car = compute_car(ar, (-5, 10))
        assert abs(car - ar.sum()) < 1e-10


class TestTStatistic:
    def test_returns_dict(self):
        ar = pd.Series([0.01, 0.02, -0.01, 0.005, -0.005])
        result = compute_t_statistic(ar)
        assert "t_stat" in result
        assert "p_value" in result
        assert "mean" in result
        assert result["n"] == 5

    def test_single_value(self):
        ar = pd.Series([0.01])
        result = compute_t_statistic(ar)
        assert result["t_stat"] == 0.0
        assert result["p_value"] == 1.0


class TestAbnormalReturnCalculator:
    def test_compute_all_returns_dict(self, sample_returns):
        sr = sample_returns[["PFE", "MRK", "JNJ", "ABBV", "LLY"]]
        bench = sample_returns[["SPY", "XLV", "XBI"]]
        calc = AbnormalReturnCalculator(sr, bench, estimation_window=50)
        results = calc.compute_all("PFE")
        assert "spy_adjusted" in results
        assert "xlv_adjusted" in results
        assert "xbi_adjusted" in results

    def test_unknown_ticker_returns_empty(self, sample_returns):
        sr = sample_returns[["PFE"]]
        bench = sample_returns[["SPY"]]
        calc = AbnormalReturnCalculator(sr, bench)
        assert calc.compute_all("UNKNOWN") == {}

    def test_compute_window_ar(self, sample_returns):
        sr = sample_returns[["PFE"]]
        bench = sample_returns[["SPY"]]
        calc = AbnormalReturnCalculator(sr, bench, estimation_window=50)
        event_date = sample_returns.index[100]
        ar = calc.compute_window_ar("PFE", event_date, window=(-5, 10))
        assert ar is not None
        assert len(ar) > 0

    def test_compute_event_study_metrics(self, sample_returns):
        sr = sample_returns[["PFE"]]
        bench = sample_returns[["SPY"]]
        calc = AbnormalReturnCalculator(sr, bench, estimation_window=50)
        event_date = sample_returns.index[100]
        metrics = calc.compute_event_study_metrics("PFE", event_date, window=(-5, 10))
        assert "spy_adjusted" in metrics
        m = metrics["spy_adjusted"]
        assert "car" in m
        assert "t_stat" in m
        assert "p_value" in m


class TestRegressionResidual:
    def test_regression_residual(self):
        np.random.seed(42)
        n = 300
        bench = pd.DataFrame(
            {
                "SPY": pd.Series(np.random.randn(n) * 0.01, index=range(n)),
                "XLV": pd.Series(np.random.randn(n) * 0.012, index=range(n)),
            }
        )
        stock = pd.Series(
            0.001
            + 0.8 * bench["SPY"].values
            + 0.4 * bench["XLV"].values
            + np.random.randn(n) * 0.005,
            index=range(n),
        )
        ar = compute_regression_residual_ar(stock, bench, estimation_window=100)
        assert len(ar) == n
        assert not ar.isna().all()

    def test_regression_residual_fallback(self):
        stock = pd.Series([0.01, 0.02, -0.01], index=range(3))
        bench = pd.DataFrame({"SPY": pd.Series([0.005, 0.01, -0.005], index=range(3))})
        ar = compute_regression_residual_ar(stock, bench, estimation_window=100)
        assert len(ar) == 3
