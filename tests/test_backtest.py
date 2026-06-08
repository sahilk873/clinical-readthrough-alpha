"""Tests for backtest engine."""

import pandas as pd

from clinical_alpha.backtest.engine import ReadthroughBacktest, compute_transaction_cost


class TestTransactionCost:
    def test_linear(self):
        tc = compute_transaction_cost(0.01, model="linear", base_tc_bps=10, slippage_bps=5)
        assert tc == 15.0

    def test_quadratic(self):
        tc = compute_transaction_cost(0.50, model="quadratic", base_tc_bps=10)
        assert tc > 10

    def test_volume_based(self):
        tc = compute_transaction_cost(
            0.01, model="volume_based", base_tc_bps=10, slippage_bps=5, daily_volume_pct=0.01
        )
        assert tc > 15

    def test_unknown_model(self):
        tc = compute_transaction_cost(0.01, model="unknown")
        assert tc == 15.0  # default fallback


class TestReadthroughBacktest:
    def test_no_events_returns_status(self, sample_graph, sample_prices, sample_returns):
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        returns = sample_returns[["PFE", "MRK", "JNJ", "ABBV", "LLY"]]
        bt = ReadthroughBacktest(sample_graph, sample_prices, returns, benchmarks)
        empty_events = pd.DataFrame(
            columns=["company_ticker", "event_date", "direction", "confidence"]
        )
        result = bt.run_backtest(empty_events)
        assert result["status"] == "no_events"

    def test_backtest_with_events(self, sample_graph, sample_prices, sample_returns):
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        returns = sample_returns[["PFE", "MRK", "JNJ", "ABBV", "LLY"]]
        bt = ReadthroughBacktest(sample_graph, sample_prices, returns, benchmarks)

        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "company_ticker": ["PFE", "MRK"],
                "event_date": ["2023-06-15", "2023-09-15"],
                "event_type": ["fda_approval", "fda_approval"],
                "direction": ["positive", "positive"],
                "confidence": [0.9, 0.9],
            }
        )
        result = bt.run_backtest(events, top_k=3, holding_periods=[5, 10])
        assert result["status"] == "success"
        assert 5 in result["results_by_period"]
        assert 10 in result["results_by_period"]

    def test_summary_table(self, sample_graph, sample_prices, sample_returns):
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        returns = sample_returns[["PFE", "MRK", "JNJ", "ABBV", "LLY"]]
        bt = ReadthroughBacktest(sample_graph, sample_prices, returns, benchmarks)
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "company_ticker": ["PFE"],
                "event_date": ["2023-06-15"],
                "event_type": ["fda_approval"],
                "direction": ["positive"],
                "confidence": [0.9],
            }
        )
        result = bt.run_backtest(events, top_k=3)
        table = bt.summary_table(result)
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0

    def test_volatility_target_weighting(self, sample_graph, sample_prices, sample_returns):
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        returns = sample_returns[["PFE", "MRK", "JNJ", "ABBV", "LLY"]]
        bt = ReadthroughBacktest(
            sample_graph, sample_prices, returns, benchmarks, weighting="volatility_target"
        )
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "company_ticker": ["PFE"],
                "event_date": ["2023-06-15"],
                "event_type": ["fda_approval"],
                "direction": ["positive"],
                "confidence": [0.9],
            }
        )
        result = bt.run_backtest(events, top_k=3, holding_periods=[5])
        assert result["status"] == "success"
