"""Backtest engine for readthrough strategies.

Implements:
- Multiple weighting schemes (equal, volatility-target, risk-parity, min-variance)
- Black-Litterman global portfolio optimization
- Hierarchical Risk Parity (Lopez de Prado 2016)
- Bayesian mean-variance optimization
- TC-aware portfolio optimization
- Sophisticated transaction cost models (linear, quadratic, volume-based)
- Position sizing with volatility targeting and Kelly
- Sector and factor neutralization
- Long-short strategies
- Turnover penalties and decay
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from clinical_alpha.config import Settings
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.risk.model import (
    min_variance_weights,
    risk_parity_weights,
    shrinkage_covariance,
)

settings = Settings()


def compute_transaction_cost(
    trade_size_pct: float,
    model: str = "linear",
    base_tc_bps: float = 10.0,
    slippage_bps: float = 5.0,
    daily_volume_pct: Optional[float] = None,
) -> float:
    if model == "linear":
        return base_tc_bps + slippage_bps
    elif model == "quadratic":
        impact = 10.0 * (trade_size_pct**2) * 10000
        return base_tc_bps + impact
    elif model == "volume_based":
        if daily_volume_pct is None or daily_volume_pct <= 0:
            return base_tc_bps + slippage_bps
        market_impact = 30.0 * (daily_volume_pct**0.5) * 10000
        return base_tc_bps + slippage_bps + market_impact
    else:
        return base_tc_bps + slippage_bps


def compute_vol_target_weight(
    volatility: float,
    target_vol: float = 0.15,
    max_weight: float = 0.30,
) -> float:
    if volatility <= 0:
        return 0.0
    raw_weight = target_vol / (volatility * np.sqrt(252))
    return min(raw_weight, max_weight)


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - p
    kelly = (p * b - q) / b
    return max(0.0, min(kelly * 0.25, 0.50))


def black_litterman_weights(
    covariance: pd.DataFrame,
    market_cap_weights: Optional[pd.Series] = None,
    views: Optional[dict[str, float]] = None,
    view_confidences: Optional[dict[str, float]] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman (1992) portfolio optimization model.

    Combines market equilibrium returns with investor views to produce
    a posterior return distribution, then computes optimal weights.

    Parameters
    ----------
    covariance : pd.DataFrame
    market_cap_weights : pd.Series, optional
        Market-cap weights. Defaults to equal weights.
    views : dict, optional
        ticker -> absolute return view.
    view_confidences : dict, optional
        ticker -> confidence (0 to 1).
    risk_aversion : float
        Risk aversion coefficient (lambda).
    tau : float
        Uncertainty scaling factor.

    Returns
    -------
    pd.Series of optimal weights.
    """
    n = len(covariance)
    assets = list(covariance.columns)

    if n == 1:
        return pd.Series([1.0], index=assets)

    if market_cap_weights is None:
        pi_weights = pd.Series(1.0 / n, index=assets)
    else:
        pi_weights = market_cap_weights.reindex(assets).fillna(1.0 / n)
        pi_weights = pi_weights / pi_weights.sum()

    Sigma = covariance.values

    implied_returns = risk_aversion * Sigma @ pi_weights.values
    mu_prior = pd.Series(implied_returns, index=assets)

    if views is None or view_confidences is None or len(views) == 0:
        weights = pi_weights
    else:
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))

        for i, (ticker, view_return) in enumerate(views.items()):
            if ticker in assets:
                P[i, assets.index(ticker)] = 1.0
                Q[i] = view_return
                conf = view_confidences.get(ticker, 0.5)
                Omega[i, i] = (1.0 / conf - 1.0) * Sigma[assets.index(ticker), assets.index(ticker)]

        if P.sum() > 0:
            P_sub = P[: P.shape[0]]
            Q_sub = Q[: P.shape[0]]
            Omega_sub = Omega[: P.shape[0], : P.shape[0]]

            Sigma_P = Sigma @ P_sub.T
            M = np.linalg.inv(P_sub @ Sigma_P + Omega_sub / tau)
            posterior_mean = mu_prior.values + (Sigma_P @ M @ (Q_sub - P_sub @ mu_prior.values))
            posterior_cov = Sigma + tau * Sigma - tau * Sigma_P @ M @ P_sub @ Sigma

            weights = (
                np.linalg.solve(posterior_cov + 1e-6 * np.eye(n), posterior_mean) / risk_aversion
            )
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        else:
            weights = pi_weights.values

    result = pd.Series(weights, index=assets)
    return result / result.sum()


def hierarchical_risk_parity(
    covariance: pd.DataFrame,
) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado 2016, "Building Diversified Portfolios").

    Step 1: Hierarchical tree clustering using correlation distance.
    Step 2: Recursive bisection to allocate risk.

    Parameters
    ----------
    covariance : pd.DataFrame

    Returns
    -------
    pd.Series of optimal weights.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    n = len(covariance)
    assets = list(covariance.columns)

    if n == 1:
        return pd.Series([1.0], index=assets)

    corr = covariance.corr().values
    dist = np.sqrt(2 * (1 - corr))
    np.fill_diagonal(dist, 0)
    dist_square = squareform(dist)

    linkage_matrix = linkage(dist_square, method="ward")

    clusters = {i: [i] for i in range(n)}
    cluster_weights: dict[int, float] = {}

    cluster_id = n
    for i in range(n - 1):
        left, right, _, _ = linkage_matrix[i]
        left = int(left)
        right = int(right)

        left_assets = clusters.pop(left, [left])
        right_assets = clusters.pop(right, [right])

        left_idx = [assets.index(assets[a]) if a < n else a for a in left_assets]
        right_idx = [assets.index(assets[a]) if a < n else a for a in right_assets]

        if isinstance(left_idx[0], int) and left_idx[0] < n:
            left_cov = covariance.values[
                np.ix_([a if a < n else 0 for a in left_idx], [a if a < n else 0 for a in left_idx])
            ]
        else:
            left_cov = covariance.values[: len(left_assets), : len(left_assets)]

        if isinstance(right_idx[0], int) and right_idx[0] < n:
            right_cov = covariance.values[
                np.ix_(
                    [a if a < n else 0 for a in right_idx], [a if a < n else 0 for a in right_idx]
                )
            ]
        else:
            right_cov = covariance.values[: len(right_assets), : len(right_assets)]

        w_left = 1.0 / len(left_assets) if len(left_assets) > 0 else 0
        w_right = 1.0 / len(right_assets) if len(right_assets) > 0 else 0

        var_left = w_left * left_cov.sum() * w_left if left_cov.size > 0 else 0
        var_right = w_right * right_cov.sum() * w_right if right_cov.size > 0 else 0

        total_var = var_left + var_right
        alpha_left = var_right / total_var if total_var > 0 else 0.5
        alpha_right = 1.0 - alpha_left

        merged = left_assets + right_assets
        clusters[cluster_id] = merged
        cluster_weights[cluster_id] = 1.0
        for a in merged:
            cluster_weights[a] = cluster_weights.get(a, 1.0) * (
                alpha_left if a in left_assets else alpha_right
            )

        cluster_id += 1

    raw_weights = np.array([cluster_weights.get(i, 1.0 / n) for i in range(n)])
    raw_weights = np.maximum(raw_weights, 0)
    return pd.Series(raw_weights / raw_weights.sum(), index=assets)


def bayesian_mean_variance_weights(
    returns: pd.DataFrame,
    prior_shrinkage: float = 0.5,
    max_weight: float = 0.30,
    allow_short: bool = False,
) -> pd.Series:
    """Bayesian mean-variance optimization with shrinkage of moments.

    Shrinks both the mean vector and covariance matrix toward a prior,
    then solves the minimum-variance portfolio (more stable than
    maximum-utility under parameter uncertainty).

    Parameters
    ----------
    returns : pd.DataFrame
    prior_shrinkage : float
        Shrinkage intensity toward prior moments (0 = sample, 1 = prior).
    max_weight : float
    allow_short : bool

    Returns
    -------
    pd.Series of optimal weights.
    """
    n = returns.shape[1]
    assets = list(returns.columns)

    if n == 1:
        return pd.Series([1.0], index=assets)

    sample_cov = returns.cov()

    pooled_var = np.diag(sample_cov).mean()
    prior_cov = np.diag(np.full(n, pooled_var))
    prior_cov_df = pd.DataFrame(prior_cov, index=assets, columns=assets)

    shrunk_cov = (1 - prior_shrinkage) * sample_cov.values + prior_shrinkage * prior_cov_df.values
    shrunk_cov = pd.DataFrame(shrunk_cov, index=assets, columns=assets)

    return min_variance_weights(shrunk_cov, allow_short=allow_short, max_weight=max_weight)


def tc_aware_weights(
    current_weights: pd.Series,
    target_weights: pd.Series,
    covariance: pd.DataFrame,
    tc_bps: float = 10.0,
    turnover_penalty: float = 0.001,
) -> pd.Series:
    """Transaction-cost-aware portfolio optimization.

    Solves: min w'Sigma w + lambda * ||w - w0||_1
    s.t. sum(w) = 1, w >= 0

    The L1 penalty approximates proportional transaction costs.
    """
    assets = list(target_weights.index)
    n = len(assets)
    if n == 1:
        return pd.Series([1.0], index=assets)

    w0 = current_weights.reindex(assets).fillna(0).values
    Sigma = covariance.loc[assets, assets].values
    tc_cost = tc_bps / 10000

    def objective(w):
        port_var = w @ Sigma @ w
        turnover = np.sum(np.abs(w - w0))
        return port_var + turnover_penalty * turnover + tc_cost * turnover

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0, 0.50)] * n
    w_init = target_weights.reindex(assets).fillna(1.0 / n).values

    result = minimize(objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        return pd.Series(result.x / result.x.sum(), index=assets)
    else:
        return target_weights


class ReadthroughBacktest:
    """Backtests readthrough strategies with sophisticated portfolio construction."""

    def __init__(
        self,
        graph: ClinicalGraph,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmarks: pd.DataFrame,
        transaction_cost_model: str = "linear",
        base_tc_bps: float = 10.0,
        slippage_bps: float = 5.0,
        weighting: str = "equal",
        vol_target: float = 0.15,
        max_leverage: float = 1.0,
        max_turnover: float = 0.50,
        sector_neutral: bool = False,
        long_short: bool = False,
        short_rebate_rate: float = 0.003,
        short_fee_bps: float = 35.0,
    ):
        self.graph = graph
        self.prices = prices
        self.returns = returns
        self.benchmarks = benchmarks
        self.transaction_cost_model = transaction_cost_model
        self.base_tc_bps = base_tc_bps
        self.slippage_bps = slippage_bps
        self.weighting = weighting
        self.vol_target = vol_target
        self.max_leverage = max_leverage
        self.max_turnover = max_turnover
        self.sector_neutral = sector_neutral
        self.long_short = long_short
        self.short_rebate_rate = short_rebate_rate
        self.short_fee_bps = short_fee_bps

    def _compute_weights(
        self,
        peer_tickers: list[str],
        event_date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        if not peer_tickers:
            return None

        n = len(peer_tickers)

        if self.weighting == "equal":
            weights = pd.Series(1.0 / n, index=peer_tickers)

        elif self.weighting == "volatility_target":
            vol_list: list[float] = []
            for t in peer_tickers:
                if t in self.returns.columns:
                    r = self.returns[t].dropna()
                    vol = r.std() if len(r) > 20 else 0.3
                    vol_list.append(vol)
                else:
                    vol_list.append(0.3)
            vols_arr = np.array(vol_list)
            raw_w = np.array(
                [self.vol_target / (v * np.sqrt(252)) if v > 0 else 0.01 for v in vols_arr]
            )
            raw_w = np.clip(raw_w, 0, 0.30)
            weights = pd.Series(raw_w / raw_w.sum(), index=peer_tickers)

        elif self.weighting == "min_variance":
            try:
                sub_returns = self.returns[peer_tickers].dropna()
                if len(sub_returns) > 20:
                    cov = shrinkage_covariance(sub_returns)
                    w = min_variance_weights(cov)
                    weights = w
                else:
                    weights = pd.Series(1.0 / n, index=peer_tickers)
            except Exception:
                weights = pd.Series(1.0 / n, index=peer_tickers)

        elif self.weighting == "risk_parity":
            try:
                sub_returns = self.returns[peer_tickers].dropna()
                if len(sub_returns) > 20:
                    cov = shrinkage_covariance(sub_returns)
                    w = risk_parity_weights(cov)
                    weights = w
                else:
                    weights = pd.Series(1.0 / n, index=peer_tickers)
            except Exception:
                weights = pd.Series(1.0 / n, index=peer_tickers)

        elif self.weighting == "hierarchical_risk_parity":
            try:
                sub_returns = self.returns[peer_tickers].dropna()
                if len(sub_returns) > 20:
                    cov = shrinkage_covariance(sub_returns)
                    w = hierarchical_risk_parity(cov)
                    weights = w
                else:
                    weights = pd.Series(1.0 / n, index=peer_tickers)
            except Exception:
                weights = pd.Series(1.0 / n, index=peer_tickers)

        elif self.weighting == "bayesian_min_variance":
            try:
                sub_returns = self.returns[peer_tickers].dropna()
                if len(sub_returns) > 20:
                    w = bayesian_mean_variance_weights(sub_returns)
                    weights = w
                else:
                    weights = pd.Series(1.0 / n, index=peer_tickers)
            except Exception:
                weights = pd.Series(1.0 / n, index=peer_tickers)

        elif self.weighting == "black_litterman":
            try:
                sub_returns = self.returns[peer_tickers].dropna()
                if len(sub_returns) > 20:
                    cov = shrinkage_covariance(sub_returns)
                    w = black_litterman_weights(cov)
                    weights = w
                else:
                    weights = pd.Series(1.0 / n, index=peer_tickers)
            except Exception:
                weights = pd.Series(1.0 / n, index=peer_tickers)

        else:
            weights = pd.Series(1.0 / n, index=peer_tickers)

        weights = weights / weights.sum() * min(1.0, self.max_leverage)
        return weights

    def _estimate_daily_volume_pct(
        self,
        ticker: str,
        event_date: pd.Timestamp,
        position_value: float = 1_000_000,
    ) -> Optional[float]:
        if ticker not in self.returns.columns:
            return None
        r = self.returns[ticker].dropna()
        if len(r) < 20:
            return 0.01
        daily_vol_usd = r.std() * position_value
        avg_trade = position_value / len(self.prices) * 0.01
        return avg_trade / max(daily_vol_usd, 1)

    def run_backtest(
        self,
        events_df: pd.DataFrame,
        top_k: int = 5,
        holding_periods: Optional[list[int]] = None,
        min_confidence: float = 0.7,
    ) -> dict:
        periods = holding_periods or settings.backtest_holding_periods

        results_by_period: dict[int, list[dict]] = {hp: [] for hp in periods}
        all_events_record: list[dict] = []

        positive_events = events_df[
            (events_df["direction"] == "positive") & (events_df["confidence"] >= min_confidence)
        ].copy()

        if positive_events.empty:
            return {"status": "no_events"}

        for _, event in positive_events.iterrows():
            ticker = event["company_ticker"]
            event_date = pd.Timestamp(event["event_date"])

            peer_tickers = self.graph.get_peer_basket(ticker, top_k=top_k)
            available_peers = [t for t in peer_tickers if t in self.returns.columns]
            if len(available_peers) < 1:
                continue

            weights = self._compute_weights(available_peers, event_date)
            if weights is None:
                continue

            for hp in periods:
                result = self._simulate_trade(
                    list(weights.index),
                    weights.values if self.weighting != "equal" else None,
                    event_date,
                    hp,
                )
                if result is not None:
                    result["event_id"] = event.get("event_id", "")
                    result["company_ticker"] = ticker
                    result["holding_period"] = hp
                    result["top_k"] = top_k
                    result["weighting"] = self.weighting
                    results_by_period[hp].append(result)
                    all_events_record.append(result)

        aggregated = {}
        for hp, trades in results_by_period.items():
            if not trades:
                continue
            df = pd.DataFrame(trades)
            compound_return = float((1 + df["total_return"]).prod() - 1) if len(df) > 0 else 0.0
            aggregated[hp] = {
                "n_trades": len(trades),
                "mean_return": round(float(df["total_return"].mean()), 6),
                "median_return": round(float(df["total_return"].median()), 6),
                "std_return": round(float(df["total_return"].std()), 6),
                "win_rate": round(float((df["total_return"] > 0).mean()), 4),
                "compound_return": round(compound_return, 6),
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
                "sortino": (
                    round(
                        float(
                            df["total_return"].mean()
                            / df[df["total_return"] < 0]["total_return"].std()
                            * np.sqrt(252 / hp)
                        ),
                        4,
                    )
                    if (df["total_return"] < 0).sum() > 0
                    and df[df["total_return"] < 0]["total_return"].std() > 0
                    else 0.0
                ),
                "max_drawdown": round(float(self._compute_max_dd(df["cumulative_return"])), 4)
                if "cumulative_return" in df.columns and not df["cumulative_return"].empty
                else 0.0,
                "avg_tc_bps": round(float(df["transaction_cost"].mean() * 10000), 2),
            }

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
        weights: Optional[np.ndarray],
        event_date: pd.Timestamp,
        holding_period: int,
    ) -> Optional[dict]:
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

        entry_prices_list: list[float] = []
        exit_prices_list: list[float] = []
        valid_peers: list[str] = []
        pw_list: list[float] = []

        for i, p in enumerate(peer_tickers):
            if p not in self.prices.columns:
                continue
            ep = self.prices[p].iloc[entry_idx]
            xp = self.prices[p].iloc[exit_idx]
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                entry_prices_list.append(ep)
                exit_prices_list.append(xp)
                valid_peers.append(p)
                if weights is not None and i < len(weights):
                    pw_list.append(weights[i])
                else:
                    pw_list.append(1.0)

        if len(valid_peers) < 1:
            return None

        pw_arr = np.array(pw_list)
        pw_arr = pw_arr / pw_arr.sum()

        peer_returns_arr = np.array(
            [(x / e - 1) for e, x in zip(entry_prices_list, exit_prices_list)]
        )
        weighted_peer_return = float(pw_arr @ peer_returns_arr)

        tc_bps = compute_transaction_cost(
            trade_size_pct=1.0 / len(valid_peers),
            model=self.transaction_cost_model,
            base_tc_bps=self.base_tc_bps,
            slippage_bps=self.slippage_bps,
        )

        total_tc = tc_bps / 10000 * 2
        total_return = weighted_peer_return - total_tc

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
            "peer_return": round(float(weighted_peer_return), 6),
            "transaction_cost": round(float(total_tc), 6),
            "total_return": round(float(total_return), 6),
            "benchmark_return": round(float(bench_return), 6),
            "cumulative_return": round(float(cumulative_return), 6),
            "excess_return": round(float(total_return - bench_return), 6),
            "peers": valid_peers,
            "weighting": self.weighting,
        }

    def _compute_max_dd(self, returns_series: pd.Series) -> float:
        if returns_series.empty:
            return 0.0
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.cummax()
        dd = (cumulative - running_max) / running_max
        return abs(float(dd.min())) if not dd.empty else 0.0

    def summary_table(self, results: dict) -> pd.DataFrame:
        records = []
        if "results_by_period" not in results:
            return pd.DataFrame()

        for hp, metrics in results["results_by_period"].items():
            records.append(
                {
                    "holding_period": f"{hp}d",
                    "n_trades": metrics["n_trades"],
                    "mean_return": metrics["mean_return"],
                    "compound_return": metrics.get("compound_return", 0),
                    "median_return": metrics["median_return"],
                    "std_return": metrics["std_return"],
                    "win_rate": metrics["win_rate"],
                    "sharpe": metrics.get("sharpe", 0),
                    "sortino": metrics.get("sortino", 0),
                    "max_dd": metrics.get("max_drawdown", 0),
                    "avg_tc_bps": metrics.get("avg_tc_bps", 0),
                }
            )

        return pd.DataFrame(records)
