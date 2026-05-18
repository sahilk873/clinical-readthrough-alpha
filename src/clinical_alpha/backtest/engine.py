"""Backtest engine for readthrough strategies.

Simulates longing top-k peer companies after high-confidence positive events,
with transaction costs and multiple holding periods.
"""

from typing import Optional

import numpy as np
import pandas as pd

from clinical_alpha.config import Settings
from clinical_alpha.graph.builder import ClinicalGraph

settings = Settings()


class ReadthroughBacktest:
    """Backtests a readthrough strategy: long top-k peers after positive events."""

    def __init__(
        self,
        graph: ClinicalGraph,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmarks: pd.DataFrame,
        transaction_cost: float = 0.0010,
    ):
        self.graph = graph
        self.prices = prices
        self.returns = returns
        self.benchmarks = benchmarks
        self.transaction_cost = transaction_cost

    def run_backtest(
        self,
        events_df: pd.DataFrame,
        top_k: int = 5,
        holding_periods: Optional[list[int]] = None,
        min_confidence: float = 0.7,
    ) -> dict:
        """Run backtest over a set of events.

        For each positive-direction event:
        - Identify top-k peer companies from graph
        - Long equal-weight basket of peers
        - Hold for each specified holding period
        - Apply transaction costs

        Returns performance metrics.
        """
        base = settings.BACKTEST_HOLDING_PERIODS if holding_periods is None else holding_periods
        periods: list[int] = base if base is not None else [5, 10, 21, 63]

        results_by_period: dict[int, list[dict]] = {hp: [] for hp in periods}

        # Filter to positive events with sufficient confidence
        positive_events = events_df[
            (events_df["direction"] == "positive") & (events_df["confidence"] >= min_confidence)
        ].copy()

        if positive_events.empty:
            return {"status": "no_events"}

        for _, event in positive_events.iterrows():
            ticker = event["company_ticker"]
            event_date = pd.Timestamp(event["event_date"])

            # Get peer basket
            peer_tickers = self.graph.get_peer_basket(ticker, top_k=top_k)
            available_peers = [t for t in peer_tickers if t in self.returns.columns]
            if len(available_peers) < 1:
                continue

            for hp in periods:
                result = self._simulate_trade(
                    available_peers,
                    event_date,
                    hp,
                )
                if result is not None:
                    result["event_id"] = event.get("event_id", "")
                    result["company_ticker"] = ticker
                    result["holding_period"] = hp
                    result["top_k"] = top_k
                    results_by_period[hp].append(result)

        # Aggregate results by holding period
        aggregated = {}
        for hp, trades in results_by_period.items():
            if not trades:
                continue
            df = pd.DataFrame(trades)
            aggregated[hp] = {
                "n_trades": len(trades),
                "mean_return": round(float(df["total_return"].mean()), 6),
                "median_return": round(float(df["total_return"].median()), 6),
                "std_return": round(float(df["total_return"].std()), 6),
                "win_rate": round(float((df["total_return"] > 0).mean()), 4),
                "mean_peer_return": round(float(df["peer_return"].mean()), 6),
                "mean_benchmark_return": round(
                    float(df.get("benchmark_return", df["peer_return"]).mean()), 6
                ),
                "sharpe": (
                    round(
                        float(
                            df["total_return"].mean() / df["total_return"].std() * np.sqrt(252 / hp)
                        ),
                        4,
                    )
                    if df["total_return"].std() > 0 and hp > 0
                    else 0.0
                ),
                "max_drawdown": round(float(self._compute_max_dd(df["cumulative_return"])), 4)
                if "cumulative_return" in df.columns and not df["cumulative_return"].empty
                else 0.0,
            }

        # Compute overall stats
        all_trades = []
        for hp in periods:
            all_trades.extend(results_by_period[hp])

        return {
            "status": "success",
            "n_positive_events": len(positive_events),
            "results_by_period": aggregated,
            "total_trades": len(all_trades),
            "holding_periods": periods,
        }

    def _simulate_trade(
        self,
        peer_tickers: list[str],
        event_date: pd.Timestamp,
        holding_period: int,
    ) -> Optional[dict]:
        """Simulate a single trade: buy peers at close on event date, sell after holding period."""
        if event_date not in self.prices.index:
            try:
                idx = self.prices.index.get_indexer([event_date], method="nearest")[0]
                entry_date = self.prices.index[idx]
            except (IndexError, ValueError):
                return None
        else:
            entry_date = event_date

        try:
            entry_idx = self.prices.index.get_loc(entry_date)
        except (KeyError, IndexError):
            return None

        exit_idx = entry_idx + holding_period
        if exit_idx >= len(self.prices.index):
            return None

        exit_date = self.prices.index[exit_idx]

        # Equally weighted peer basket
        entry_prices = []
        exit_prices = []
        valid_peers = []

        for p in peer_tickers:
            if p not in self.prices.columns:
                continue
            ep = self.prices[p].iloc[entry_idx]
            xp = self.prices[p].iloc[exit_idx]
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                entry_prices.append(ep)
                exit_prices.append(xp)
                valid_peers.append(p)

        if len(valid_peers) < 1:
            return None

        # Equal-weighted basket return
        peer_returns = [(x / e - 1) for e, x in zip(entry_prices, exit_prices)]
        peer_return = np.mean(peer_returns)

        # Transaction costs (entry + exit)
        tc = 2 * self.transaction_cost
        total_return = peer_return - tc

        # Benchmark return (SPY)
        if "SPY" in self.prices.columns:
            spy_entry = self.prices["SPY"].iloc[entry_idx]
            spy_exit = self.prices["SPY"].iloc[exit_idx]
            bench_return = (spy_exit / spy_entry - 1) if spy_entry > 0 else 0
        else:
            bench_return = 0

        cumulative_return = (1 + total_return) - 1

        return {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "n_peers": len(valid_peers),
            "peer_return": round(float(peer_return), 6),
            "transaction_cost": round(float(tc), 6),
            "total_return": round(float(total_return), 6),
            "benchmark_return": round(float(bench_return), 6),
            "cumulative_return": round(float(cumulative_return), 6),
            "excess_return": round(float(total_return - bench_return), 6),
            "peers": valid_peers,
        }

    def _compute_max_dd(self, returns_series: pd.Series) -> float:
        """Compute maximum drawdown from a series of returns."""
        if returns_series.empty:
            return 0.0
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.cummax()
        dd = (cumulative - running_max) / running_max
        return abs(float(dd.min())) if not dd.empty else 0.0

    def summary_table(self, results: dict) -> pd.DataFrame:
        """Generate a summary table from backtest results."""
        records = []
        if "results_by_period" not in results:
            return pd.DataFrame()

        for hp, metrics in results["results_by_period"].items():
            records.append(
                {
                    "holding_period": f"{hp}d",
                    "n_trades": metrics["n_trades"],
                    "mean_return": metrics["mean_return"],
                    "median_return": metrics["median_return"],
                    "std_return": metrics["std_return"],
                    "win_rate": metrics["win_rate"],
                    "sharpe": metrics.get("sharpe", 0),
                    "max_dd": metrics.get("max_drawdown", 0),
                }
            )

        return pd.DataFrame(records)
