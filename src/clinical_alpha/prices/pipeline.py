"""Price pipeline for fetching and processing daily stock prices.

Uses yfinance as the default source (MVP), with optional Polygon.io support
configured through environment variables.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from clinical_alpha.config import Settings

settings = Settings()

# Core healthcare ETFs for benchmark models
BENCHMARK_TICKERS = {
    "SPY": "SPY",  # S&P 500
    "XLV": "XLV",  # Healthcare sector
    "XBI": "XBI",  # Biotech
}


def fetch_prices_yfinance(
    tickers: list[str],
    start_date: str | datetime,
    end_date: str | datetime | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted close prices using yfinance.

    Returns a DataFrame with columns = tickers, index = date.
    """
    import yfinance as yf

    if end_date is None:
        end_date = datetime.now()

    # Download in chunks to avoid rate limits
    all_data: list[pd.DataFrame] = []
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            data = yf.download(
                chunk,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if data.empty:
                continue
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data
            if isinstance(close, pd.Series):
                close = close.to_frame()
            all_data.append(close)
        except Exception:
            continue

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, axis=1)
    combined.columns = [c.upper() for c in combined.columns]
    return combined


def fetch_benchmark_prices(
    start_date: str | datetime,
    end_date: str | datetime | None = None,
) -> pd.DataFrame:
    """Fetch benchmark ETF prices (SPY, XLV, XBI)."""
    return fetch_prices_yfinance(
        list(BENCHMARK_TICKERS.values()),
        start_date,
        end_date,
    )


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price DataFrame."""
    return np.log(prices / prices.shift(1))


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns."""
    return prices.pct_change()


def fill_missing_prices(
    prices: pd.DataFrame,
    method: str = "ffill",
    max_fill_days: int = 5,
) -> pd.DataFrame:
    """Forward-fill missing prices for up to max_fill_days."""
    if method == "ffill":
        filled = prices.ffill(limit=max_fill_days)
    else:
        filled = prices.fillna(method=method)
    return filled


def filter_low_volume_tickers(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    min_obs: int = 60,
) -> list[str]:
    """Remove tickers with insufficient price history."""
    valid = returns.columns[returns.notna().sum() >= min_obs]
    return list(valid)


def align_prices_to_benchmark(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align stock prices and benchmark prices to common dates."""
    common_dates = prices.index.intersection(benchmarks.index)
    return prices.loc[common_dates], benchmarks.loc[common_dates]


def fetch_price_data(
    tickers: list[str],
    start_date: str | datetime,
    end_date: str | datetime | None = None,
    include_benchmarks: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch prices, returns, and benchmark data.

    Returns (prices, returns, benchmark_prices).
    """
    if include_benchmarks:
        all_tickers = list(set(tickers + list(BENCHMARK_TICKERS.values())))
    else:
        all_tickers = tickers

    prices = fetch_prices_yfinance(all_tickers, start_date, end_date)
    prices = fill_missing_prices(prices)

    if include_benchmarks:
        bench_tickers = [t for t in BENCHMARK_TICKERS.values() if t in prices.columns]
        benchmark_prices = prices[bench_tickers].copy()
    else:
        benchmark_prices = pd.DataFrame()

    returns = compute_simple_returns(prices)

    return prices, returns, benchmark_prices


def get_polygon_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch prices from Polygon.io (optional higher-quality source).

    Requires polygon-api-client package and valid API key.
    """
    try:
        from polygon import RESTClient
    except ImportError:
        raise ImportError("polygon-api-client required. Install: pip install polygon-api-client")

    all_data: dict[str, pd.Series] = {}
    with RESTClient(api_key) as client:
        for ticker in tickers:
            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date,
                    to=end_date,
                    adjusted=True,
                )
                if aggs:
                    dates = [pd.Timestamp(a.timestamp, unit="ms") for a in aggs]
                    closes = [a.close for a in aggs]
                    all_data[ticker.upper()] = pd.Series(closes, index=dates)
            except Exception:
                continue

    return pd.DataFrame(all_data)
