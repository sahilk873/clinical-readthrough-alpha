"""Tests for data quality module."""

import numpy as np
import pandas as pd

from clinical_alpha.data.quality import (
    check_price_quality,
    detect_survivorship_bias,
    filter_quality_tickers,
)


class TestCheckPriceQuality:
    def test_passes_clean_data(self):
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        prices = pd.DataFrame(
            {"PFE": 50 * np.exp(np.random.randn(len(dates)).cumsum() * 0.02)}, index=dates
        )
        report = check_price_quality(prices, "PFE")
        assert report.passed()
        assert report.pct_observed > 0.9

    def test_missing_data_flagged(self):
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        series = pd.Series(np.nan, index=dates)
        series.iloc[:10] = 50.0
        prices = pd.DataFrame({"BAD": series}, index=dates)
        report = check_price_quality(prices, "BAD", min_observations_pct=0.8)
        assert not report.passed()
        assert len(report.errors) > 0

    def test_negative_prices_flagged(self):
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        series = pd.Series(50.0, index=dates)
        series.iloc[5] = -1.0
        prices = pd.DataFrame({"BAD": series}, index=dates)
        report = check_price_quality(prices, "BAD")
        assert not report.passed()

    def test_missing_ticker(self):
        prices = pd.DataFrame({"A": [1.0]})
        report = check_price_quality(prices, "MISSING")
        assert not report.passed()

    def test_return_outliers_flagged(self):
        dates = pd.date_range("2023-01-01", "2024-01-01", freq="B")
        series = pd.Series(50.0, index=dates)
        series.iloc[100] = 500.0  # huge jump
        series.iloc[101] = 50.0  # immediate reversion
        prices = pd.DataFrame({"A": series}, index=dates)
        report = check_price_quality(prices, "A")
        assert len(report.warnings) > 0


class TestFilterQualityTickers:
    def test_filters_low_quality(self):
        dates = pd.date_range("2023-01-01", "2024-01-01", freq="B")
        good = pd.Series(50.0 * np.exp(np.random.randn(len(dates)).cumsum() * 0.02), index=dates)
        bad = pd.Series(np.nan, index=dates)
        bad.iloc[:5] = 50.0
        prices = pd.DataFrame({"GOOD": good, "BAD": bad}, index=dates)
        returns = prices.pct_change().dropna()
        filtered_prices, filtered_returns, tickers = filter_quality_tickers(prices, returns)
        assert "GOOD" in tickers
        assert "BAD" not in tickers


class TestDetectSurvivorshipBias:
    def test_returns_dict(self):
        result = detect_survivorship_bias(["A", "B", "C"], ["A", "B", "C", "D", "E"])
        assert result["n_used"] == 3
        assert result["n_reference"] == 5
        assert result["n_missing"] == 2
