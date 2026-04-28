"""Abnormal return calculation engine.

Computes abnormal returns using:
1. SPY-adjusted (market model)
2. XLV-adjusted (healthcare sector)
3. XBI-adjusted (biotech sector)
4. Regression-residual (multi-factor model)
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from clinical_alpha.config import Settings

settings = Settings()


def compute_abnormal_returns_single_benchmark(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    estimation_window: int = 252,
) -> pd.Series:
    """Compute abnormal returns using a single benchmark (market model).

    AR = R_stock - R_benchmark
    """
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["stock", "benchmark"]
    ar = aligned["stock"] - aligned["benchmark"]
    return ar


def compute_market_model_ar(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    estimation_window: int = 252,
) -> pd.Series:
    """Compute abnormal returns using OLS market model.

    Estimates alpha and beta over the estimation window,
    then computes AR = R_stock - (alpha + beta * R_benchmark).
    """
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["stock", "bench"]

    if len(aligned) < estimation_window:
        estimation_window = len(aligned) // 2

    if len(aligned) < 30:
        return compute_abnormal_returns_single_benchmark(stock_returns, benchmark_returns)

    # Run rolling regression
    n = len(aligned)
    ar_values = np.full(n, np.nan)

    for i in range(estimation_window, n):
        est = aligned.iloc[i - estimation_window : i]
        X = np.column_stack([np.ones(estimation_window), est["bench"].values])
        y = est["stock"].values
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = beta[0] + beta[1] * aligned["bench"].iloc[i]
            ar_values[i] = aligned["stock"].iloc[i] - pred
        except Exception:
            ar_values[i] = np.nan

    result = pd.Series(ar_values, index=aligned.index)
    return result


def compute_regression_residual_ar(
    stock_returns: pd.Series,
    benchmark_returns: pd.DataFrame,
    estimation_window: int = 252,
) -> pd.Series:
    """Compute abnormal returns using multi-factor regression residuals.

    Uses multiple benchmarks (e.g., SPY, XLV, XBI) as factors.
    """
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[1] < 2:
        raise ValueError("Need at least one benchmark factor")

    if len(aligned) < estimation_window:
        estimation_window = len(aligned) // 2

    if len(aligned) < 30:
        # Fall back to simple benchmark adjustment
        return compute_abnormal_returns_single_benchmark(
            stock_returns, benchmark_returns.iloc[:, 0]
        )

    stock_col = aligned.columns[0]
    bench_cols = list(aligned.columns[1:])

    n = len(aligned)
    ar_values = np.full(n, np.nan)

    for i in range(estimation_window, n):
        est = aligned.iloc[i - estimation_window : i]
        X = np.column_stack([np.ones(estimation_window)] + [est[c].values for c in bench_cols])
        y = est[stock_col].values
        try:
            coefs = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = coefs[0] + sum(
                coefs[j + 1] * aligned[bench_cols[j]].iloc[i] for j in range(len(bench_cols))
            )
            ar_values[i] = aligned[stock_col].iloc[i] - pred
        except Exception:
            ar_values[i] = np.nan

    result = pd.Series(ar_values, index=aligned.index)
    return result


def compute_car(ar_series: pd.Series, window: tuple[int, int]) -> float:
    """Compute cumulative abnormal return over an event window.

    window is (start, end) offset from event date (e.g., (-5, 10)).
    """
    if ar_series.empty:
        return 0.0
    start, end = window
    windowed = ar_series.iloc[max(0, start) : min(len(ar_series), end + 1)]
    if windowed.empty:
        return 0.0
    return windowed.sum()


def compute_aar(ar_series: pd.Series, window: tuple[int, int]) -> float:
    """Compute average abnormal return over an event window."""
    if ar_series.empty:
        return 0.0
    start, end = window
    windowed = ar_series.iloc[max(0, start) : min(len(ar_series), end + 1)]
    if windowed.empty:
        return 0.0
    return windowed.mean()


def compute_caar(ar_matrix: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    """Compute cumulative average abnormal return across multiple securities.

    ar_matrix: DataFrame with columns = securities, index = relative days.
    """
    return ar_matrix.iloc[window[0] : window[1] + 1].cumsum().mean(axis=1)


def compute_t_statistic(ar_series: pd.Series) -> dict:
    """Compute t-statistic and p-value for abnormal return series."""
    if len(ar_series) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "mean": 0.0, "std": 0.0}

    mean = ar_series.mean()
    std = ar_series.std()
    n = len(ar_series)
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "n": n,
    }


class AbnormalReturnCalculator:
    """Computes abnormal returns using multiple methods."""

    def __init__(
        self,
        stock_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        estimation_window: int = 252,
    ):
        self.stock_returns = stock_returns
        self.benchmark_returns = benchmark_returns
        self.estimation_window = estimation_window

    def compute_all(
        self,
        ticker: str,
    ) -> dict[str, pd.Series]:
        """Compute abnormal returns using all available methods."""
        if ticker not in self.stock_returns.columns:
            return {}

        sr = self.stock_returns[ticker].dropna()
        results = {}

        # SPY-adjusted
        if "SPY" in self.benchmark_returns.columns:
            spy = self.benchmark_returns["SPY"].dropna()
            results["spy_adjusted"] = compute_abnormal_returns_single_benchmark(sr, spy)
            results["spy_market_model"] = compute_market_model_ar(sr, spy)

        # XLV-adjusted
        if "XLV" in self.benchmark_returns.columns:
            xlv = self.benchmark_returns["XLV"].dropna()
            results["xlv_adjusted"] = compute_abnormal_returns_single_benchmark(sr, xlv)

        # XBI-adjusted
        if "XBI" in self.benchmark_returns.columns:
            xbi = self.benchmark_returns["XBI"].dropna()
            results["xbi_adjusted"] = compute_abnormal_returns_single_benchmark(sr, xbi)

        # Multi-factor regression residual
        available_benchmarks = [
            c for c in ["SPY", "XLV", "XBI"] if c in self.benchmark_returns.columns
        ]
        if len(available_benchmarks) >= 2:
            bench_df = self.benchmark_returns[available_benchmarks].dropna()
            try:
                results["regression_residual"] = compute_regression_residual_ar(
                    sr, bench_df, self.estimation_window
                )
            except Exception:
                pass

        return results

    def compute_window_ar(
        self,
        ticker: str,
        event_date: pd.Timestamp,
        window: tuple[int, int] = (-10, 20),
        method: str = "spy_adjusted",
    ) -> Optional[pd.Series]:
        """Compute abnormal returns around an event date for a given window."""
        ar_methods = self.compute_all(ticker)
        if method not in ar_methods:
            return None

        ar = ar_methods[method]

        # Locate event date in the AR series
        try:
            idx = ar.index.get_indexer([event_date], method="nearest")[0]
        except (IndexError, ValueError):
            return None

        if idx < 0 or idx >= len(ar):
            return None

        start = max(0, idx + window[0])
        end = min(len(ar), idx + window[1] + 1)

        return ar.iloc[start:end]

    def compute_event_study_metrics(
        self,
        ticker: str,
        event_date: pd.Timestamp,
        window: tuple[int, int] = (-10, 20),
    ) -> dict:
        """Compute comprehensive event study metrics for a single event."""
        results = {}
        for method in ["spy_adjusted", "xlv_adjusted", "xbi_adjusted", "regression_residual"]:
            ar = self.compute_window_ar(ticker, event_date, window, method)
            if ar is not None and not ar.empty:
                car = compute_car(ar, (0, window[1]))
                aar_val = compute_aar(ar, (0, window[1]))
                pre_car = compute_car(ar, (window[0], -1))
                tstats = compute_t_statistic(ar)
                results[method] = {
                    "car": round(car, 6),
                    "aar": round(aar_val, 6),
                    "pre_event_car": round(pre_car, 6),
                    "t_stat": tstats["t_stat"],
                    "p_value": tstats["p_value"],
                    "n_obs": tstats["n"],
                }
        return results
