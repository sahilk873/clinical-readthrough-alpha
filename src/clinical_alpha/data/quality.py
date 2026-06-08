"""Data quality assurance checks for price and return data."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment for a price/returns dataset."""

    ticker: str
    n_expected: int = 0
    n_observed: int = 0
    pct_observed: float = 0.0
    n_missing: int = 0
    max_consecutive_missing: int = 0
    n_zero_prices: int = 0
    n_negative_prices: int = 0
    n_outlier_returns: int = 0
    n_stale_days: int = 0
    first_date: Optional[pd.Timestamp] = None
    last_date: Optional[pd.Timestamp] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "pct_observed": round(self.pct_observed, 4),
            "max_consecutive_missing": self.max_consecutive_missing,
            "pct_outlier_returns": round(self.n_outlier_returns / max(self.n_observed, 1), 4),
            "n_warnings": len(self.warnings),
            "n_errors": len(self.errors),
            "passed": self.passed(),
        }


def check_price_quality(
    prices: pd.DataFrame,
    ticker: str,
    min_observations_pct: float = 0.80,
    min_price: float = 1.0,
    max_return_zscore: float = 6.0,
    max_missing_consecutive_days: int = 10,
) -> DataQualityReport:
    """Run comprehensive quality checks on a single ticker's price series.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with DatetimeIndex and columns = tickers.
    ticker : str
        Column name to check.
    min_observations_pct : float
        Minimum fraction of expected observations (trading days).
    min_price : float
        Minimum allowable price (filter penny stocks).
    max_return_zscore : float
        Returns beyond this many MAD from the median are flagged as outliers.
    max_missing_consecutive_days : int
        Maximum allowed consecutive missing trading days.

    Returns
    -------
    DataQualityReport
    """
    if ticker not in prices.columns:
        return DataQualityReport(
            ticker=ticker,
            errors=[f"Ticker {ticker} not found in price DataFrame"],
        )

    series = prices[ticker]
    report = DataQualityReport(ticker=ticker)

    full_idx = prices.index
    report.n_expected = len(full_idx)
    valid = series.dropna()
    report.n_observed = len(valid)
    report.pct_observed = report.n_observed / report.n_expected if report.n_expected > 0 else 0.0
    report.n_missing = report.n_expected - report.n_observed
    report.first_date = valid.index.min() if not valid.empty else None
    report.last_date = valid.index.max() if not valid.empty else None

    if report.pct_observed < min_observations_pct:
        report.errors.append(
            f"Only {report.pct_observed:.1%} of observations present "
            f"(threshold: {min_observations_pct:.1%})"
        )

    if not valid.empty:
        missing_mask = series.isna()
        if missing_mask.any():
            consec = 0
            max_consec = 0
            for m in missing_mask:
                if m:
                    consec += 1
                    max_consec = max(max_consec, consec)
                else:
                    consec = 0
            report.max_consecutive_missing = max_consec
            if max_consec > max_missing_consecutive_days:
                report.errors.append(
                    f"{max_consec} consecutive missing days "
                    f"(threshold: {max_missing_consecutive_days})"
                )

        n_zero = (valid == 0).sum()
        report.n_zero_prices = int(n_zero)
        if n_zero > 0:
            report.warnings.append(f"{n_zero} zero-price observations")

        n_neg = (valid < 0).sum()
        report.n_negative_prices = int(n_neg)
        if n_neg > 0:
            report.errors.append(f"{n_neg} negative prices")

        below_min = (valid < min_price).sum()
        if below_min > 0:
            report.warnings.append(f"{below_min} observations below minimum price ${min_price:.2f}")

        returns = valid.pct_change().dropna()
        if len(returns) > 0:
            median = returns.median()
            mad = np.median(np.abs(returns - median))
            if mad > 0:
                zscores = np.abs(returns - median) / mad
                n_outliers = int((zscores > max_return_zscore).sum())
                report.n_outlier_returns = n_outliers
                if n_outliers > 0:
                    report.warnings.append(
                        f"{n_outliers} return outliers (>{max_return_zscore:.0f} MAD)"
                    )

        stale = (valid == valid.shift(1)).sum()
        report.n_stale_days = int(stale)
        if stale > 0.01 * len(valid):
            report.warnings.append(f"{stale} stale (zero-return) days detected")

    return report


def generate_quality_report(
    prices: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Run quality checks on all tickers and return a summary DataFrame."""
    records = []
    for ticker in prices.columns:
        report = check_price_quality(prices, ticker, **kwargs)
        records.append(report.to_dict())
    return pd.DataFrame(records)


def filter_quality_tickers(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    min_observations_pct: float = 0.80,
    min_price: float = 1.0,
    max_return_zscore: float = 6.0,
    max_missing_consecutive_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Filter out low-quality tickers from price/returns data.

    Returns filtered prices, returns, and list of passed tickers.
    """
    report = generate_quality_report(
        prices,
        min_observations_pct=min_observations_pct,
        min_price=min_price,
        max_return_zscore=max_return_zscore,
        max_missing_consecutive_days=max_missing_consecutive_days,
    )
    passed = report[report["passed"]]["ticker"].tolist()
    passed_in_both = [t for t in passed if t in returns.columns]
    return prices[passed_in_both], returns[passed_in_both], passed_in_both


def detect_survivorship_bias(
    tickers_used: list[str],
    reference_universe: list[str],
) -> dict:
    """Analyze potential survivorship bias.

    Compares the set of tickers used in the study against a broader
    reference universe (e.g., all healthcare companies ever listed).

    Parameters
    ----------
    tickers_used : list[str]
        Tickers that survived the data filter.
    reference_universe : list[str]
        Broader universe of tickers (e.g., from SEC EDGAR).

    Returns
    -------
    dict with keys: n_used, n_reference, n_missing, pct_missing
    """
    used_set = set(tickers_used)
    ref_set = set(reference_universe)
    missing = ref_set - used_set
    return {
        "n_used": len(used_set),
        "n_reference": len(ref_set),
        "n_missing": len(missing),
        "pct_missing": round(len(missing) / len(ref_set), 4) if ref_set else 0.0,
    }
