"""Signal quality analysis for event-driven strategies.

Implements:
- Information Coefficient (IC) and Rank IC time series
- Signal decay analysis (half-life, autocorrelation)
- Signal-to-noise ratio
- Cross-sectional signal evaluation
- Signal contribution decomposition
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_information_coefficient(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = "pearson",
) -> dict:
    """Information Coefficient between signal and forward returns.

    Parameters
    ----------
    signal : pd.Series
        Signal values (index = tickers or event IDs).
    forward_returns : pd.Series
        Forward return values (same index).
    method : str
        'pearson' (IC) or 'spearman' (Rank IC).

    Returns
    -------
    dict with keys: ic, p_value, n_obs
    """
    aligned = pd.concat([signal, forward_returns], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return {"ic": 0.0, "p_value": 1.0, "n_obs": 0}

    if method == "spearman":
        ic, p_val = scipy_stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    else:
        ic, p_val = scipy_stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])

    return {
        "ic": float(ic),
        "p_value": float(p_val),
        "n_obs": len(aligned),
        "method": method,
    }


def compute_rank_ic(rank_ic: str = "spearman") -> dict:
    """Alias for compute_information_coefficient with spearman method."""
    return compute_information_coefficient(
        pd.Series(dtype=float), pd.Series(dtype=float), method="spearman"
    )


def compute_ic_time_series(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "pearson",
    min_obs: int = 10,
) -> pd.Series:
    """Compute IC time series (IC at each time period cross-sectionally).

    Parameters
    ----------
    signals : pd.DataFrame
        Signal values, index = dates, columns = tickers.
    forward_returns : pd.DataFrame
        Forward returns, same structure.
    method : str
        'pearson' or 'spearman'.
    min_obs : int
        Minimum observations for IC calculation.

    Returns
    -------
    pd.Series of IC values indexed by date.
    """
    aligned_signals, aligned_returns = signals.align(forward_returns, join="inner")
    ic_values = {}

    for date in aligned_signals.index:
        s = aligned_signals.loc[date].dropna()
        r = aligned_returns.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_obs:
            continue

        s_vals = s[common].values
        r_vals = r[common].values
        try:
            if method == "spearman":
                ic, _ = scipy_stats.spearmanr(s_vals, r_vals)
            else:
                ic, _ = scipy_stats.pearsonr(s_vals, r_vals)
            ic_values[date] = ic
        except (ValueError, ZeroDivisionError):
            continue

    return pd.Series(ic_values)


def compute_signal_decay(
    signal: pd.Series,
    forward_returns: pd.DataFrame,
    lags: list[int] = [1, 3, 5, 10, 21, 63],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute signal decay profile: IC at different forward horizons.

    Parameters
    ----------
    signal : pd.Series
        Signal values indexed by event/ticker.
    forward_returns : pd.DataFrame
        Forward returns at different horizons (columns = horizons).
    lags : list[int]
        Forward return horizons to test.
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    pd.DataFrame with columns: lag, ic, p_value, n_obs
    """
    records = []
    for lag in lags:
        lag_col = f"fwd_{lag}d" if lag not in forward_returns.columns else str(lag)
        if lag_col in forward_returns.columns:
            fwd = forward_returns[lag_col]
        elif isinstance(forward_returns, pd.DataFrame) and lag in forward_returns.columns:
            fwd = forward_returns[lag]
        elif isinstance(forward_returns, pd.Series):
            fwd = forward_returns
        else:
            continue

        result = compute_information_coefficient(signal, fwd, method)
        records.append(
            {
                "lag": lag,
                "ic": result["ic"],
                "p_value": result["p_value"],
                "n_obs": result["n_obs"],
            }
        )

    return pd.DataFrame(records)


def compute_signal_half_life(signal_autocorr: pd.Series) -> float:
    """Compute signal half-life from autocorrelation series.

    Half-life = log(0.5) / log(|rho|) where rho is the 1-lag autocorrelation.

    Parameters
    ----------
    signal_autocorr : pd.Series
        Autocorrelation values (index = lag). Typically from pd.Series.autocorr().

    Returns
    -------
    float estimated half-life in periods.
    """
    if signal_autocorr.empty:
        return 0.0

    rho = abs(signal_autocorr.iloc[0]) if len(signal_autocorr) > 0 else 0.0
    if rho <= 0 or rho >= 1:
        return 0.0

    half_life = np.log(0.5) / np.log(rho)
    return max(0.0, half_life)


def compute_signal_noise_ratio(
    signal: pd.Series,
    forward_returns: pd.Series,
) -> dict:
    """Signal-to-noise ratio for event signals.

    SNR = mean(signal_return) / std(signal_return) * sqrt(n)
    Also computes the Information Ratio (IR).
    """
    aligned = pd.concat([signal, forward_returns], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return {"snr": 0.0, "ir": 0.0, "mean_signal": 0.0, "std_signal": 0.0, "n_obs": 0}

    signal_values = aligned.iloc[:, 0].values
    return_values = aligned.iloc[:, 1].values

    sorted_idx = np.argsort(signal_values)
    n_top = max(1, len(sorted_idx) // 5)
    top_returns = return_values[sorted_idx[-n_top:]]
    bottom_returns = return_values[sorted_idx[:n_top]]

    spread = top_returns.mean() - bottom_returns.mean()
    pooled_std = np.sqrt((top_returns.var() + bottom_returns.var()) / 2) if n_top > 1 else 1.0

    snr = spread / pooled_std if pooled_std > 0 else 0.0
    ir = (
        np.mean(return_values) / np.std(return_values) * np.sqrt(len(return_values))
        if np.std(return_values) > 0
        else 0.0
    )

    return {
        "snr": float(snr),
        "information_ratio": float(ir),
        "mean_signal_return": float(np.mean(return_values)),
        "std_signal_return": float(np.std(return_values)),
        "top_quantile_mean": float(top_returns.mean()),
        "bottom_quantile_mean": float(bottom_returns.mean()),
        "spread": float(spread),
        "n_obs": len(aligned),
    }


def evaluate_signal_cross_sectional(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> dict:
    """Cross-sectional signal evaluation.

    Sorts stocks into quantiles by signal and computes mean return,
    hit rate, and spread between top and bottom quantiles.

    Parameters
    ----------
    signals : pd.DataFrame
        Index = dates, columns = tickers.
    forward_returns : pd.DataFrame
        Same structure.
    n_quantiles : int

    Returns
    -------
    dict with keys: quantile_means, quantile_hit_rates, top_spread,
                    ic_mean, rank_ic_mean, cum_return_top_minus_bottom
    """
    aligned_s, aligned_r = signals.align(forward_returns, join="inner")

    quantile_returns: dict[int, list[float]] = {q: [] for q in range(n_quantiles)}
    quantile_hits: dict[int, list[float]] = {q: [] for q in range(n_quantiles)}
    ic_list: list[float] = []
    rank_ic_list: list[float] = []

    for date in aligned_s.index:
        s = aligned_s.loc[date].dropna()
        r = aligned_r.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < n_quantiles * 2:
            continue

        s_vals = s[common]
        r_vals = r[common]

        try:
            ic, _ = scipy_stats.pearsonr(s_vals.values, r_vals.values)
            ic_list.append(ic)
            rank_ic, _ = scipy_stats.spearmanr(s_vals.values, r_vals.values)
            rank_ic_list.append(rank_ic)
        except (ValueError, ZeroDivisionError):
            continue

        quantile_labels = pd.qcut(
            s_vals.rank(method="first"), q=n_quantiles, labels=range(n_quantiles)
        )
        for q in range(n_quantiles):
            mask = quantile_labels == q
            if mask.any():
                quantile_returns[q].append(r_vals[mask].mean())
                quantile_hits[q].append((r_vals[mask] > 0).mean())

    result = {}
    for q in range(n_quantiles):
        result[f"quantile_{q + 1}_mean_return"] = (
            float(np.mean(quantile_returns[q])) if quantile_returns[q] else 0.0
        )
        result[f"quantile_{q + 1}_hit_rate"] = (
            float(np.mean(quantile_hits[q])) if quantile_hits[q] else 0.0
        )

    top_mean = (
        result.get("quantile_5_mean_return", 0)
        if n_quantiles == 5
        else result.get(f"quantile_{n_quantiles}_mean_return", 0)
    )
    bottom_mean = result.get("quantile_1_mean_return", 0)
    result["top_minus_bottom_spread"] = top_mean - bottom_mean
    result["mean_ic"] = float(np.mean(ic_list)) if ic_list else 0.0
    result["mean_rank_ic"] = float(np.mean(rank_ic_list)) if rank_ic_list else 0.0
    result["ic_std"] = float(np.std(ic_list)) if ic_list else 0.0
    result["ic_sharpe"] = (
        float(np.mean(ic_list) / np.std(ic_list))
        if len(ic_list) > 1 and np.std(ic_list) > 0
        else 0.0
    )

    return result


def signal_contribution_decomposition(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> dict:
    """Decompose signal-based portfolio returns into factor and idiosyncratic components.

    Uses time-series regression of portfolio returns on factors to attribute
    signal performance to systematic vs. idiosyncratic sources.
    """
    aligned = pd.concat([returns, signals, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or returns.shape[1] < 1:
        return {}

    port_returns = returns.mean(axis=1)
    bench = benchmark_returns

    X = np.column_stack([np.ones(len(bench)), bench.values])
    y = port_returns.values

    try:
        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coefs
        alpha = coefs[0]
        beta = coefs[1]
        systematic = beta * bench.values
        idiosyncratic = residuals

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "systematic_vol": float(np.std(systematic)),
            "idiosyncratic_vol": float(np.std(idiosyncratic)),
            "total_vol": float(np.std(y)),
            "systematic_pct": float(np.var(systematic) / np.var(y) * 100) if np.var(y) > 0 else 0.0,
            "idiosyncratic_pct": float(np.var(idiosyncratic) / np.var(y) * 100)
            if np.var(y) > 0
            else 0.0,
        }
    except np.linalg.LinAlgError:
        return {}
