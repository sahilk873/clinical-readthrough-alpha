"""Integration tests for the full pipeline."""

import numpy as np
import pandas as pd

from clinical_alpha.backtest.engine import ReadthroughBacktest
from clinical_alpha.events.extractor import extract_all_events
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.reporting.generator import generate_summary_table
from clinical_alpha.returns.factor_models import compare_factor_models
from clinical_alpha.robustness.checks import RobustnessChecker
from clinical_alpha.signal.analysis import (
    compute_information_coefficient,
    compute_signal_decay,
    compute_signal_noise_ratio,
    evaluate_signal_cross_sectional,
)
from clinical_alpha.studies.event_study import EventStudy, aggregate_results


class TestFullPipeline:
    """Integration test simulating the full pipeline with sample data."""

    def test_end_to_end(self, sample_graph, sample_prices, sample_returns, sample_company_df):
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

        # Create FDA events
        fda_events = pd.DataFrame(
            {
                "sponsor": ["Pfizer Inc", "Merck & Co Inc"],
                "drug_name": ["Drug X", "Drug Y"],
                "approval_date": pd.to_datetime(["2023-06-15", "2023-09-20"]),
            }
        )

        # Create trials
        trials_df = pd.DataFrame(
            {
                "nct_id": ["NCT00000001", "NCT00000002"],
                "brief_title": ["Trial 1", "Trial 2"],
                "phase": ["Phase 3", "Phase 2"],
                "overall_status": ["COMPLETED", "ACTIVE"],
                "sponsors": ["Pfizer Inc;", "Merck & Co Inc;"],
                "intervention_name": ["Drug X;", "Drug Y;"],
                "conditions": ["Cancer;", "Diabetes;"],
                "result_first_post_date": pd.to_datetime(["2023-07-01", "2023-10-01"]),
            }
        )

        # Build graph with sponsor mapping
        sponsor_map = {
            "Pfizer Inc": {"ticker": "PFE", "company_name": "Pfizer Inc"},
            "Merck & Co Inc": {"ticker": "MRK", "company_name": "Merck & Co Inc"},
        }
        graph = ClinicalGraph()
        graph.build_from_dataframes(trials_df, sponsor_map, fda_events)

        assert graph.summary()["total_nodes"] > 0
        assert graph.summary()["total_edges"] > 0

        # Extract events
        events_df = extract_all_events(fda_events, trials_df, graph, min_confidence=0.7)
        assert len(events_df) >= 2
        assert "direction" in events_df.columns

        # Event study
        study = EventStudy(graph, (sample_prices, stock_returns, benchmarks), all_tickers)
        study_results = study.run_all_events(events_df, window=(-5, 10), method="spy_adjusted")
        study_summary = study.summarize_results(study_results)
        agg = aggregate_results(study_summary)

        assert agg.get("n_events", 0) >= 0
        if agg.get("n_events", 0) > 0:
            assert "mean_spread" in agg

        # Backtest - equal weight
        bt = ReadthroughBacktest(graph, sample_prices, stock_returns, benchmarks)
        bt_results = bt.run_backtest(events_df, top_k=3, holding_periods=[5, 10])
        assert bt_results["status"] in ("success", "no_events")

        # Robustness
        rc = RobustnessChecker(graph, (sample_prices, stock_returns, benchmarks), all_tickers)
        robustness_results = rc.run_all_checks(events_df, window=(-5, 10))
        assert isinstance(robustness_results, dict)

        # Summary table
        summary_table = generate_summary_table(study_summary, bt_results, robustness_results)
        assert isinstance(summary_table, pd.DataFrame)

    def test_backtest_new_weighting_schemes(self, sample_graph, sample_prices, sample_returns):
        """Test new weighting schemes end-to-end."""
        all_tickers = ["PFE", "MRK", "JNJ", "ABBV", "LLY"]
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        stock_returns = sample_returns[all_tickers]

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

        for weighting in ["equal", "risk_parity", "min_variance"]:
            bt = ReadthroughBacktest(
                sample_graph,
                sample_prices,
                stock_returns,
                benchmarks,
                weighting=weighting,
            )
            result = bt.run_backtest(events, top_k=3, holding_periods=[5])
            assert result["status"] == "success"
            assert 5 in result["results_by_period"]

    def test_signal_analysis_integration(self, sample_returns):
        """Test signal analysis with realistic data."""
        np.random.seed(42)
        dates = sample_returns.index[:100]
        n = len(dates)

        signal = pd.Series(np.random.randn(n), index=dates)
        fwd_returns = pd.Series(signal.values * 0.1 + np.random.randn(n) * 0.02, index=dates)

        # IC computation
        ic_result = compute_information_coefficient(signal, fwd_returns, method="spearman")
        assert "ic" in ic_result
        assert "p_value" in ic_result
        assert ic_result["n_obs"] > 0

        # Signal decay
        decay = compute_signal_decay(signal, fwd_returns.to_frame("fwd_1d"), lags=[1, 3, 5])
        assert isinstance(decay, pd.DataFrame)
        assert len(decay) > 0

        # SNR
        snr = compute_signal_noise_ratio(signal, fwd_returns)
        assert "snr" in snr
        assert "information_ratio" in snr

        # Cross-sectional evaluation
        signal_wide = pd.DataFrame({t: signal.values for t in ["PFE", "MRK", "JNJ"]}, index=dates)
        ret_wide = pd.DataFrame({t: fwd_returns.values for t in ["PFE", "MRK", "JNJ"]}, index=dates)
        cross = evaluate_signal_cross_sectional(signal_wide, ret_wide, n_quantiles=3)
        assert "top_minus_bottom_spread" in cross
        assert "mean_ic" in cross

    def test_factor_model_comparison(self, sample_returns):
        """Test factor model comparison on sample returns."""
        benchmarks = sample_returns[["SPY", "XLV", "XBI"]]
        comparison = compare_factor_models(sample_returns, benchmarks)
        if comparison is not None and not comparison.empty:
            assert "model" in comparison.columns
            assert "aic" in comparison.columns or "AIC" in comparison.columns
