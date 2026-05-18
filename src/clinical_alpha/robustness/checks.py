"""Robustness checks for event study and backtest results.

Checks sensitivity to:
- Event type classification
- Top-k peer selection
- Holding period
- Benchmark model choice
- Event confidence threshold
- Overlapping events
"""

from typing import Optional

import pandas as pd

from clinical_alpha.backtest.engine import ReadthroughBacktest
from clinical_alpha.config import Settings
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.studies.event_study import EventStudy

settings = Settings()


class RobustnessChecker:
    """Runs robustness checks on event study and backtest results."""

    def __init__(
        self,
        graph: ClinicalGraph,
        price_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        all_tickers: list[str],
    ):
        self.graph = graph
        self.prices, self.returns, self.benchmarks = price_data
        self.all_tickers = all_tickers
        self.event_study = EventStudy(
            graph,
            price_data,
            all_tickers,
        )

    def check_event_type_sensitivity(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check how results vary by event type."""
        results = []
        for event_type in events_df["event_type"].unique():
            subset = events_df[events_df["event_type"] == event_type]
            study_results = self.event_study.run_all_events(
                subset,
                window=window,
            )
            summary = self.event_study.summarize_results(study_results)
            agg = self._aggregate_check(summary)
            agg["event_type"] = event_type
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
            study_results = self.event_study.run_all_events(
                events_df,
                peer_top_k=k,
                window=window,
            )
            summary = self.event_study.summarize_results(study_results)
            agg = self._aggregate_check(summary)
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
                        "n_trades": metrics["n_trades"],
                    }
                )

        return pd.DataFrame(records)

    def check_benchmark_model_sensitivity(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
    ) -> pd.DataFrame:
        """Check sensitivity to the benchmark model used for AR calculation."""
        methods = ["spy_adjusted", "xlv_adjusted", "xbi_adjusted"]
        results = []

        for method in methods:
            study_results = self.event_study.run_all_events(
                events_df,
                window=window,
                method=method,
            )
            summary = self.event_study.summarize_results(study_results)
            agg = self._aggregate_check(summary)
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
            study_results = self.event_study.run_all_events(
                filtered,
                window=window,
            )
            summary = self.event_study.summarize_results(study_results)
            agg = self._aggregate_check(summary)
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
            deduped = deduplicate_overlapping_events(
                events_df,
                lookback_days=od,
            )
            study_results = self.event_study.run_all_events(
                deduped,
                window=window,
            )
            summary = self.event_study.summarize_results(study_results)
            agg = self._aggregate_check(summary)
            agg["overlap_lookback_days"] = od
            agg["n_events"] = len(deduped)
            results.append(agg)

        return pd.DataFrame(results)

    def run_all_checks(
        self,
        events_df: pd.DataFrame,
        window: tuple[int, int] = (-10, 20),
    ) -> dict[str, pd.DataFrame]:
        """Run all robustness checks and return results as a dict."""
        return {
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
        }

    def _aggregate_check(self, summary: pd.DataFrame) -> dict:
        """Aggregate summary results for a single check."""
        if summary.empty:
            return {
                "n_events": 0,
                "mean_spread": 0,
                "positive_ratio": 0,
                "significant_ratio": 0,
            }

        n = len(summary)
        pos = (summary["spread"] > 0).sum()
        sig = (summary["p_value"] < 0.05).sum() if "p_value" in summary.columns else 0

        return {
            "n_events": n,
            "mean_spread": round(float(summary["spread"].mean()), 6),
            "positive_ratio": round(pos / n, 4) if n > 0 else 0,
            "significant_ratio": round(sig / n, 4) if n > 0 else 0,
        }
