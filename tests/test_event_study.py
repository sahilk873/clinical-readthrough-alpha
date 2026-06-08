"""Tests for event study engine."""

import pandas as pd

from clinical_alpha.studies.event_study import (
    EventStudy,
    aggregate_results,
    build_control_basket,
    build_matched_control_basket,
)


class TestBuildControlBasket:
    def test_excludes_event_and_peers(self):
        result = build_control_basket(["A", "B", "C", "D", "E"], "A", ["B"], n_controls=2)
        assert "A" not in result
        assert "B" not in result
        assert len(result) <= 3

    def test_no_candidates(self):
        result = build_control_basket(["A"], "A", [], n_controls=5)
        assert result == []


class TestBuildMatchedControlBasket:
    def test_falls_back_without_market_caps(self):
        result = build_matched_control_basket(["A", "B", "C", "D"], "A", ["B"], n_controls=2)
        assert len(result) <= 2

    def test_uses_market_caps(self):
        caps = {"A": 100e9, "B": 50e9, "C": 55e9, "D": 200e9}
        result = build_matched_control_basket(
            ["A", "B", "C", "D"], "A", ["B"], market_caps=caps, n_controls=1
        )
        assert "A" not in result


class TestEventStudy:
    def test_no_peers_returns_status(self, sample_graph, sample_returns):
        study = EventStudy(
            sample_graph,
            (sample_returns, sample_returns, sample_returns[["SPY", "XLV", "XBI"]]),
            [],
        )
        event = {
            "company_ticker": "UNKNOWN",
            "event_date": "2023-06-15",
            "event_type": "fda_approval",
            "direction": "positive",
            "confidence": 0.9,
        }
        result = study.run_event_study(event)
        assert result["status"] == "no_peers"

    def test_event_study_with_known_ticker(self, sample_graph, sample_returns):
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

        study = EventStudy(sample_graph, (sample_returns, stock_returns, benchmarks), all_tickers)
        event = {
            "company_ticker": "PFE",
            "event_date": "2023-06-15",
            "event_type": "fda_approval",
            "direction": "positive",
            "confidence": 0.9,
        }
        result = study.run_event_study(event, window=(-5, 5))
        assert result["status"] in ("success", "no_price_data")

    def test_aggregate_results(self):
        summary = pd.DataFrame(
            {
                "spread": [0.01, -0.005, 0.02, 0.0, 0.015],
                "p_value": [0.03, 0.5, 0.04, 0.8, 0.01],
                "peer_mean_car": [0.02, 0.01, 0.03, 0.01, 0.02],
                "control_mean_car": [0.01, 0.015, 0.01, 0.01, 0.005],
                "t_stat": [2.0, -0.5, 3.0, 0.0, 2.5],
            }
        )
        agg = aggregate_results(summary)
        assert agg["n_events"] == 5
        assert agg["positive_spread"] == 3
        assert agg["significant_events"] == 3

    def test_aggregate_empty(self):
        assert aggregate_results(pd.DataFrame()) == {}

    def test_calendar_time_portfolio(self, sample_graph, sample_returns):
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

        study = EventStudy(sample_graph, (sample_returns, stock_returns, benchmarks), all_tickers)
        events = pd.DataFrame(
            {
                "company_ticker": ["PFE", "MRK"],
                "event_date": ["2023-06-15", "2023-09-15"],
                "event_type": ["fda_approval", "trial_result"],
                "direction": ["positive", "positive"],
                "confidence": [0.9, 0.8],
            }
        )
        ct_result = study.run_calendar_time_portfolio(events, window=(-5, 5))
        assert ct_result["status"] in ("success", "no_data")
