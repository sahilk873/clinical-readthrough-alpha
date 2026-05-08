"""Event study engine.

Runs event studies comparing graph-related peer baskets against
matched/random healthcare controls.
"""

import numpy as np
import pandas as pd

from clinical_alpha.config import Settings
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.returns.abnormal import (
    AbnormalReturnCalculator,
)

settings = Settings()


def build_control_basket(
    all_tickers: list[str],
    event_ticker: str,
    peer_tickers: list[str],
    n_controls: int = 20,
    random_seed: int = 42,
) -> list[str]:
    """Build a matched control basket from remaining healthcare companies.

    Excludes the event company and its identified peers.
    """
    rng = np.random.default_rng(random_seed)
    excluded = {event_ticker} | set(peer_tickers)
    candidates = [t for t in all_tickers if t not in excluded]
    n = min(n_controls, len(candidates))
    if n == 0:
        return []
    return list(rng.choice(candidates, size=n, replace=False))


class EventStudy:
    """Runs event studies for a set of events."""

    def __init__(
        self,
        graph: ClinicalGraph,
        price_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        all_tickers: list[str],
    ):
        self.graph = graph
        self.prices, self.returns, self.benchmarks = price_data
        self.all_tickers = all_tickers
        self.ar_calculator = AbnormalReturnCalculator(
            self.returns,
            self.benchmarks,
        )

    def run_event_study(
        self,
        event: dict,
        peer_top_k: int = 10,
        window: tuple[int, int] = (-10, 20),
        method: str = "spy_adjusted",
        n_controls: int = 20,
    ) -> dict:
        """Run a single event study comparing peer basket vs control basket."""
        ticker = event["company_ticker"]
        event_date = pd.Timestamp(event["event_date"])

        # Get peer basket from graph
        peer_tickers = self.graph.get_peer_basket(ticker, top_k=peer_top_k)
        if not peer_tickers:
            return {"status": "no_peers", "event": event}

        # Filter to tickers with available price data
        available_peers = [t for t in peer_tickers if t in self.returns.columns]
        if not available_peers:
            return {"status": "no_price_data", "event": event}

        # Build control basket
        control_tickers = build_control_basket(
            self.all_tickers,
            ticker,
            available_peers,
            n_controls,
        )
        if not control_tickers:
            return {"status": "no_controls", "event": event}

        # Compute peer basket CAR
        peer_cars = []
        for p in available_peers:
            metrics = self.ar_calculator.compute_event_study_metrics(p, event_date, window)
            if method in metrics:
                peer_cars.append(metrics[method]["car"])

        # Compute control basket CAR
        control_cars = []
        for c in control_tickers:
            metrics = self.ar_calculator.compute_event_study_metrics(c, event_date, window)
            if method in metrics:
                control_cars.append(metrics[method]["car"])

        if not peer_cars or not control_cars:
            return {"status": "insufficient_data", "event": event}

        peer_mean = np.mean(peer_cars)
        control_mean = np.mean(control_cars)
        spread = peer_mean - control_mean

        # T-test for difference
        t_stat, p_value = None, None
        try:
            from scipy import stats

            t_stat, p_value = stats.ttest_ind(peer_cars, control_cars)
        except Exception:
            pass

        # Compute peer basket AR series for the event window
        peer_ar_series: pd.Series | None = None
        for p in available_peers:
            ar = self.ar_calculator.compute_window_ar(p, event_date, window, method)
            if ar is not None and peer_ar_series is None:
                peer_ar_series = ar.copy()
            elif ar is not None and peer_ar_series is not None:
                peer_ar_series = peer_ar_series.add(ar, fill_value=0)

        if peer_ar_series is not None and len(available_peers) > 0:
            peer_ar_series = peer_ar_series / len(available_peers)

        return {
            "status": "success",
            "event": event,
            "peer_tickers": available_peers,
            "control_tickers": control_tickers,
            "peer_mean_car": round(float(peer_mean), 6),
            "control_mean_car": round(float(control_mean), 6),
            "spread": round(float(spread), 6),
            "t_stat": round(float(t_stat), 4) if t_stat is not None else None,
            "p_value": round(float(p_value), 4) if p_value is not None else None,
            "n_peers": len(peer_cars),
            "n_controls": len(control_cars),
            "peer_cars": peer_cars,
            "control_cars": control_cars,
            "ar_series": peer_ar_series,
        }

    def run_all_events(
        self,
        events_df: pd.DataFrame,
        peer_top_k: int = 10,
        window: tuple[int, int] = (-10, 20),
        method: str = "spy_adjusted",
        n_controls: int = 20,
    ) -> list[dict]:
        """Run event studies for all events in the DataFrame."""
        results = []
        for _, event in events_df.iterrows():
            result = self.run_event_study(
                event.to_dict(),
                peer_top_k,
                window,
                method,
                n_controls,
            )
            results.append(result)
        return results

    def summarize_results(self, results: list[dict]) -> pd.DataFrame:
        """Summarize event study results into a DataFrame."""
        records = []
        for r in results:
            if r["status"] != "success":
                continue
            e = r["event"]
            records.append(
                {
                    "event_id": e.get("event_id", ""),
                    "company_ticker": e.get("company_ticker", ""),
                    "event_type": e.get("event_type", ""),
                    "event_date": e.get("event_date", ""),
                    "direction": e.get("direction", ""),
                    "peer_mean_car": r["peer_mean_car"],
                    "control_mean_car": r["control_mean_car"],
                    "spread": r["spread"],
                    "t_stat": r["t_stat"],
                    "p_value": r["p_value"],
                    "n_peers": r["n_peers"],
                    "n_controls": r["n_controls"],
                }
            )
        return pd.DataFrame(records)


def aggregate_results(results_summary: pd.DataFrame) -> dict:
    """Aggregate event study results across all events."""
    if results_summary.empty:
        return {}

    n_events = len(results_summary)
    positive_spread = (results_summary["spread"] > 0).sum()
    negative_spread = (results_summary["spread"] < 0).sum()
    significant = (
        (results_summary["p_value"] < 0.05).sum() if "p_value" in results_summary.columns else 0
    )

    return {
        "n_events": n_events,
        "positive_spread": positive_spread,
        "negative_spread": negative_spread,
        "significant_events": significant,
        "mean_spread": round(float(results_summary["spread"].mean()), 6),
        "median_spread": round(float(results_summary["spread"].median()), 6),
        "mean_peer_car": round(float(results_summary["peer_mean_car"].mean()), 6),
        "mean_control_car": round(float(results_summary["control_mean_car"].mean()), 6),
        "positive_ratio": round(positive_spread / n_events, 4) if n_events > 0 else 0,
    }
