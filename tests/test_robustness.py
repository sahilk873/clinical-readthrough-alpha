"""Tests for robustness checks."""

import pandas as pd
import pytest

from clinical_alpha.robustness.checks import RobustnessChecker


class TestRobustnessChecker:
    def test_event_type_sensitivity(self, sample_graph, sample_returns):
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

        rc = RobustnessChecker(
            sample_graph, (sample_returns, stock_returns, benchmarks), all_tickers
        )
        events = pd.DataFrame(
            {
                "company_ticker": ["PFE", "MRK"],
                "event_date": ["2023-06-15", "2023-09-15"],
                "event_type": ["fda_approval", "trial_result"],
                "direction": ["positive", "positive"],
                "confidence": [0.9, 0.8],
            }
        )
        result = rc.check_event_type_sensitivity(events, window=(-5, 5))
        assert isinstance(result, pd.DataFrame)

    def test_confidence_threshold_sensitivity(self, sample_graph, sample_returns):
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

        rc = RobustnessChecker(
            sample_graph, (sample_returns, stock_returns, benchmarks), all_tickers
        )
        events = pd.DataFrame(
            {
                "company_ticker": ["PFE", "MRK"],
                "event_date": ["2023-06-15", "2023-09-15"],
                "event_type": ["fda_approval", "trial_result"],
                "direction": ["positive", "positive"],
                "confidence": [0.9, 0.8],
            }
        )
        result = rc.check_confidence_threshold_sensitivity(
            events, thresholds=[0.7, 0.8], window=(-5, 5)
        )
        assert isinstance(result, pd.DataFrame)

    def test_subsample_time_period(self, sample_graph, sample_returns):
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

        rc = RobustnessChecker(
            sample_graph, (sample_returns, stock_returns, benchmarks), all_tickers
        )
        events = pd.DataFrame(
            {
                "company_ticker": ["PFE", "MRK"],
                "event_date": ["2023-06-15", "2023-09-15"],
                "event_type": ["fda_approval", "trial_result"],
                "direction": ["positive", "positive"],
                "confidence": [0.9, 0.8],
            }
        )
        result = rc.check_subsample_time_period(events, window=(-5, 5))
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.slow
    def test_run_all_checks(self, sample_graph, sample_returns):
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

        rc = RobustnessChecker(
            sample_graph, (sample_returns, stock_returns, benchmarks), all_tickers
        )
        events = pd.DataFrame(
            {
                "company_ticker": ["PFE", "MRK"],
                "event_date": ["2023-06-15", "2023-09-15"],
                "event_type": ["fda_approval", "trial_result"],
                "direction": ["positive", "positive"],
                "confidence": [0.9, 0.8],
            }
        )
        results = rc.run_all_checks(events, window=(-2, 2))
        assert isinstance(results, dict)
        assert len(results) > 0
