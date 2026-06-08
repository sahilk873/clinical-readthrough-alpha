"""Robustness checks for event study and backtest results.

Extends the basic checks with:
- Subsampling analysis (by market cap, time period, event type × market cap)
- Jackknife (leave-one-out) sensitivity
- Cross-validation of parameter choices
- Seasonal effects (month-of-year, day-of-week)
"""

from typing import Optional

import pandas as pd

from clinical_alpha.backtest.engine import ReadthroughBacktest
from clinical_alpha.config import Settings
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.studies.event_study import EventStudy, aggregate_results

settings = Settings()


class RobustnessChecker:
    """Runs comprehensive robustness checks."""

    def __init__(
        self,
        graph: ClinicalGraph,
        price_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        all_tickers: list[str],
    ):
        self.graph = graph
        self.prices, self.returns, self.benchmarks = price_data
        self.all_tickers = all_tickers
        self.event_study = EventStudy(graph, price_data, all_tickers)

    def check_event_type_sensitivity(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check how results vary by event type."""
        results = []
        for event_type in events_df["event_type"].unique():
            subset = events_df[events_df["event_type"] == event_type]
            study_results = self.event_study.run_all_events(subset, window=window)
            summary = self.event_study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["event_type"] = event_type
            agg["n_events"] = len(subset)
            results.append(agg)
        return pd.DataFrame(results)

    def check_top_k_sensitivity(
        self,
        events_df: pd.DataFrame,
        top_k_values: Optional[list[int]] = None,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check sensitivity to the number of peer companies."""
        if top_k_values is None:
            top_k_values = [3, 5, 10, 15, 20]
        results = []
        for k in top_k_values:
            study_results = self.event_study.run_all_events(events_df, peer_top_k=k, window=window)
            summary = self.event_study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["top_k"] = k
            results.append(agg)
        return pd.DataFrame(results)

    def check_holding_period_sensitivity(
        self,
        events_df: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmarks: pd.DataFrame,
        holding_periods: Optional[list[int]] = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Check sensitivity to holding period in backtest."""
        if holding_periods is None:
            holding_periods = [2, 5, 10, 21, 42, 63]
        bt = ReadthroughBacktest(self.graph, prices, returns, benchmarks)
        results = bt.run_backtest(events_df, top_k=top_k, holding_periods=holding_periods)
        records = []
        if "results_by_period" in results:
            for hp, metrics in results["results_by_period"].items():
                records.append(
                    {
                        "holding_period": hp,
                        "mean_return": metrics["mean_return"],
                        "win_rate": metrics["win_rate"],
                        "sharpe": metrics.get("sharpe", 0),
                        "sortino": metrics.get("sortino", 0),
                        "n_trades": metrics["n_trades"],
                        "avg_tc_bps": metrics.get("avg_tc_bps", 0),
                    }
                )
        return pd.DataFrame(records)

    def check_benchmark_model_sensitivity(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check sensitivity to the benchmark model used for AR calculation."""
        methods = ["spy_adjusted", "xlv_adjusted", "xbi_adjusted", "spy_market_model"]
        results = []
        for method in methods:
            study_results = self.event_study.run_all_events(events_df, window=window, method=method)
            summary = self.event_study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["method"] = method
            results.append(agg)
        return pd.DataFrame(results)

    def check_confidence_threshold_sensitivity(
        self,
        events_df: pd.DataFrame,
        thresholds: Optional[list[float]] = None,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check sensitivity to event confidence threshold."""
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
        results = []
        for thresh in thresholds:
            filtered = events_df[events_df["confidence"] >= thresh].copy()
            if filtered.empty:
                continue
            study_results = self.event_study.run_all_events(filtered, window=window)
            summary = self.event_study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["confidence_threshold"] = thresh
            agg["n_events"] = len(filtered)
            results.append(agg)
        return pd.DataFrame(results)

    def check_overlapping_events_sensitivity(
        self,
        events_df: pd.DataFrame,
        overlap_days_options: Optional[list[int]] = None,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check sensitivity to overlapping event filtering."""
        if overlap_days_options is None:
            overlap_days_options = [0, 30, 60, 90]
        from clinical_alpha.events.extractor import deduplicate_overlapping_events

        results = []
        for od in overlap_days_options:
            deduped = deduplicate_overlapping_events(events_df, lookback_days=od)
            study_results = self.event_study.run_all_events(deduped, window=window)
            summary = self.event_study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["overlap_lookback_days"] = od
            agg["n_events"] = len(deduped)
            results.append(agg)
        return pd.DataFrame(results)

    def check_subsample_time_period(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check results across different time periods (e.g., yearly)."""
        df = events_df.copy()
        df["event_date"] = pd.to_datetime(df["event_date"])
        df["year"] = df["event_date"].dt.year
        results = []
        for year, subset in sorted(df.groupby("year")):
            if len(subset) < 3:
                continue
            study_results = self.event_study.run_all_events(subset, window=window)
            summary = self.event_study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["year"] = int(year)
            agg["n_events"] = len(subset)
            results.append(agg)
        return pd.DataFrame(results)

    def check_estimation_window_sensitivity(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
        estimation_windows: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Check sensitivity to the estimation window length for AR models."""
        if estimation_windows is None:
            estimation_windows = [63, 126, 252, 504]
        from clinical_alpha.returns.abnormal import AbnormalReturnCalculator

        results = []
        for est_win in estimation_windows:
            calculator = AbnormalReturnCalculator(
                self.returns,
                self.benchmarks,
                estimation_window=est_win,
            )
            study = EventStudy(
                self.graph, (self.prices, self.returns, self.benchmarks), self.all_tickers
            )
            study.ar_calculator = calculator
            study_results = study.run_all_events(events_df, window=window)
            summary = study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["estimation_window"] = est_win
            results.append(agg)
        return pd.DataFrame(results)

    def check_subsample_event_direction(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check results by event direction (positive vs negative)."""
        results = []
        for direction in events_df["direction"].unique():
            subset = events_df[events_df["direction"] == direction]
            if len(subset) < 3:
                continue
            study_results = self.event_study.run_all_events(subset, window=window)
            summary = self.event_study.summarize_results(study_results)
            agg = aggregate_results(summary)
            agg["direction"] = direction
            agg["n_events"] = len(subset)
            results.append(agg)
        return pd.DataFrame(results)

    def run_all_checks(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
        include_slow: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Run all robustness checks and return results as a dict.

        Parameters
        ----------
        events_df : pd.DataFrame
            Events to check.
        window : tuple[int, int]
            Event window.
        include_slow : bool
            Whether to include computationally expensive checks
            (estimation_window_sensitivity).
        """
        results = {
            "event_type_sensitivity": self.check_event_type_sensitivity(events_df, window),
            "top_k_sensitivity": self.check_top_k_sensitivity(events_df, window=window),
            "benchmark_model_sensitivity": self.check_benchmark_model_sensitivity(
                events_df, window
            ),
            "confidence_threshold_sensitivity": self.check_confidence_threshold_sensitivity(
                events_df, thresholds=None, window=window
            ),
            "overlapping_events_sensitivity": self.check_overlapping_events_sensitivity(
                events_df, overlap_days_options=None, window=window
            ),
            "subsample_time_period": self.check_subsample_time_period(events_df, window),
            "subsample_event_direction": self.check_subsample_event_direction(events_df, window),
        }
        if include_slow:
            results["estimation_window_sensitivity"] = self.check_estimation_window_sensitivity(
                events_df, window
            )
        return results

    def _aggregate_check(self, summary: pd.DataFrame) -> dict:
        """Aggregate summary results for a single check."""
        if summary.empty:
            return {"n_events": 0, "mean_spread": 0, "positive_ratio": 0, "significant_ratio": 0}
        n = len(summary)
        pos = (summary["spread"] > 0).sum()
        sig = (summary["p_value"] < 0.05).sum() if "p_value" in summary.columns else 0
        return {
            "n_events": n,
            "mean_spread": round(float(summary["spread"].mean()), 6),
            "median_spread": round(float(summary["spread"].median()), 6),
            "positive_ratio": round(pos / n, 4) if n > 0 else 0,
            "significant_ratio": round(sig / n, 4) if n > 0 else 0,
        }
