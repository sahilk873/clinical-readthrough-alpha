"""Abnormal return calculation engine.

Computes abnormal returns using:
1. Market-adjusted (single benchmark)
2. Market model (OLS with rolling estimation)
3. Multi-factor regression residual
4. Buy-and-hold abnormal returns (BHAR)
5. Calendar-time portfolio abnormal returns
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from clinical_alpha.config import Settings
from clinical_alpha.returns.factor_models import estimate_capm

settings = Settings()


def compute_abnormal_returns_single_benchmark(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Compute abnormal returns using a single benchmark (market adjusted).

    AR = R_stock - R_benchmark
    """
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["stock", "benchmark"]
    return aligned["stock"] - aligned["benchmark"]


def compute_market_model_ar(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    estimation_window: int = 252,
) -> pd.Series:
    """Compute abnormal returns using OLS market model with rolling estimation.

    Estimates alpha and beta over the estimation window, then:
    AR = R_stock - (alpha + beta * R_benchmark)
    """
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["stock", "bench"]

    if len(aligned) < estimation_window:
        estimation_window = len(aligned) // 2

    if len(aligned) < 30:
        return compute_abnormal_returns_single_benchmark(stock_returns, benchmark_returns)

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

    return pd.Series(ar_values, index=aligned.index)


def compute_capm_ar(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    estimation_window: int = 252,
) -> pd.Series:
    """Compute abnormal returns using CAPM with rolling beta estimation."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["stock", "bench"]

    if len(aligned) < estimation_window:
        estimation_window = len(aligned) // 2

    if len(aligned) < 30:
        return compute_abnormal_returns_single_benchmark(stock_returns, benchmark_returns)

    n = len(aligned)
    ar_values = np.full(n, np.nan)

    for i in range(estimation_window, n):
        est = aligned.iloc[i - estimation_window : i]
        capm = estimate_capm(est["stock"], est["bench"])
        beta = capm["beta"]
        alpha = capm["alpha"]
        pred = alpha + beta * aligned["bench"].iloc[i]
        ar_values[i] = aligned["stock"].iloc[i] - pred

    return pd.Series(ar_values, index=aligned.index)


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

    return pd.Series(ar_values, index=aligned.index)


def compute_car(ar_series: pd.Series, window: tuple[int, int]) -> float:
    """Cumulative abnormal return over an event window.

    window is (start, end) offset from event date (e.g., (-5, 10)).
    """
    if ar_series.empty:
        return 0.0
    start, end = window
    windowed = ar_series.iloc[max(0, start) : min(len(ar_series), end + 1)]
    if windowed.empty:
        return 0.0
    return windowed.sum()


def compute_bhar(
    prices: pd.Series,
    benchmark_prices: pd.Series,
    window: tuple[int, int],
) -> float:
    """Buy-and-hold abnormal return.

    BHAR = prod(1 + R_stock) - prod(1 + R_benchmark) over the window.
    """
    start, end = window
    if end <= start:
        return 0.0
    stock_win = prices.iloc[max(0, start) : min(len(prices), end + 1)]
    bench_win = benchmark_prices.iloc[max(0, start) : min(len(benchmark_prices), end + 1)]
    if stock_win.empty or bench_win.empty:
        return 0.0
    bhar_stock = (stock_win / stock_win.iloc[0] - 1).iloc[-1] if len(stock_win) > 1 else 0.0
    bhar_bench = (bench_win / bench_win.iloc[0] - 1).iloc[-1] if len(bench_win) > 1 else 0.0
    return bhar_stock - bhar_bench


def compute_caar(ar_matrix: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    """Cumulative average abnormal return across multiple securities.

    ar_matrix: DataFrame with columns = securities, index = relative days.
    """
    return ar_matrix.iloc[window[0] : window[1] + 1].cumsum().mean(axis=1)


def compute_t_statistic(ar_series: pd.Series) -> dict:
    """Compute t-statistic and p-value for abnormal return series."""
    if len(ar_series) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "mean": 0.0, "std": 0.0, "n": 0}

    mean = ar_series.mean()
    std = ar_series.std(ddof=1)
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


def compute_multi_window_car(
    ar_series: pd.Series,
    windows: list[tuple[int, int]],
) -> dict[str, float]:
    """Compute CARs for multiple event windows."""
    return {f"({s},{d})": round(compute_car(ar_series, (s, d)), 6) for s, d in windows}


class AbnormalReturnCalculator:
    """Computes abnormal returns using multiple methods and window configurations."""

    def __init__(
        self,
        stock_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        stock_prices: Optional[pd.DataFrame] = None,
        estimation_window: int = 252,
    ):
        self.stock_returns = stock_returns
        self.benchmark_returns = benchmark_returns
        self.stock_prices = stock_prices
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

        bench_map = {}
        if "SPY" in self.benchmark_returns.columns:
            spy = self.benchmark_returns["SPY"].dropna()
            results["spy_adjusted"] = compute_abnormal_returns_single_benchmark(sr, spy)
            results["spy_market_model"] = compute_market_model_ar(sr, spy)
            results["spy_capm"] = compute_capm_ar(sr, spy)
            bench_map["spy"] = spy

        if "XLV" in self.benchmark_returns.columns:
            xlv = self.benchmark_returns["XLV"].dropna()
            results["xlv_adjusted"] = compute_abnormal_returns_single_benchmark(sr, xlv)
            bench_map["xlv"] = xlv

        if "XBI" in self.benchmark_returns.columns:
            xbi = self.benchmark_returns["XBI"].dropna()
            results["xbi_adjusted"] = compute_abnormal_returns_single_benchmark(sr, xbi)
            bench_map["xbi"] = xbi

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
        """Extract abnormal returns around an event date for a given window."""
        ar_methods = self.compute_all(ticker)
        if method not in ar_methods:
            return None

        ar = ar_methods[method]
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
        """Compute comprehensive event study metrics across all methods and windows."""
        results = {}
        windows = [(-1, 1), (-3, 3), (-5, 5), window, (-10, 20), (-20, 40)]

        for method in [
            "spy_adjusted",
            "xlv_adjusted",
            "xbi_adjusted",
            "spy_market_model",
            "spy_capm",
            "regression_residual",
        ]:
            ar = self.compute_window_ar(ticker, event_date, window, method)
            if ar is None or ar.empty:
                continue

            pre_car = compute_car(ar, (window[0], -1))
            post_car = compute_car(ar, (0, window[1]))
            tstats = compute_t_statistic(ar)

            results[method] = {
                "car": round(post_car, 6),
                "pre_event_car": round(pre_car, 6),
                "t_stat": tstats["t_stat"],
                "p_value": tstats["p_value"],
                "n_obs": tstats["n"],
            }
            results[method].update(
                {f"car_{s}_{e}": round(compute_car(ar, (s, e)), 6) for s, e in windows}
            )

        return results
