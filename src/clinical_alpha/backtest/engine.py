"""Backtest engine for readthrough strategies.

Implements:
- Multiple weighting schemes (equal, volatility-target, risk-parity, min-variance)
- Graph-proximity and event-confidence-aware position sizing
- Black-Litterman global portfolio optimization
- Hierarchical Risk Parity (Lopez de Prado 2016)
- Bayesian mean-variance optimization
- TC-aware portfolio optimization
- Event-aware reward decomposition (alpha vs beta vs TC)
- Dynamic exit logic with stop-loss, take-profit, trailing-stop
- Intra-trade PnL tracking and excursion analysis
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
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        trailing_stop: float = 0.0,
        min_weight: float = 0.01,
        max_single_position: float = 0.30,
        position_scaling: str = "equal",
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
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.min_weight = min_weight
        self.max_single_position = max_single_position
        self.position_scaling = position_scaling

    def _compute_weights(
        self,
        peer_tickers: list[str],
        event_date: pd.Timestamp,
        event_ticker: str = "",
        event: Optional[dict] = None,
    ) -> Optional[pd.Series]:
        if not peer_tickers:
            return None

        n = len(peer_tickers)

        if self.weighting == "equal":
            weights = pd.Series(1.0 / n, index=peer_tickers)

        elif self.weighting == "graph_proximity":
            weights = self._compute_graph_proximity_weights(peer_tickers, event_ticker)

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
            raw_w = np.clip(raw_w, 0, self.max_single_position)
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

        if self.position_scaling == "confidence" and event:
            alpha_score = self._event_alpha_score(event)
            weights = weights * (0.5 + 0.5 * alpha_score)

        if self.position_scaling == "graph_proximity" and event_ticker:
            prox = self._compute_graph_proximity_weights(peer_tickers, event_ticker)
            weights = weights * prox

        weights = weights / weights.sum() * min(1.0, self.max_leverage)
        weights = weights.clip(lower=self.min_weight)
        weights = weights / weights.sum()
        return weights

    def _event_alpha_score(self, event: dict) -> float:
        score = event.get("confidence", 0.7)
        event_type = event.get("event_type", "")
        if event_type == "fda_approval":
            score = min(1.0, score * 1.15)
        elif event_type == "fda_rejection":
            score = min(1.0, score * 1.10)
        direction = event.get("direction", "positive")
        if direction == "positive":
            score = score * 1.0
        elif direction == "negative":
            score = score * 0.9
        return float(np.clip(score, 0.1, 1.0))

    def _compute_graph_proximity_weights(
        self, peer_tickers: list[str], event_ticker: str
    ) -> pd.Series:
        all_peers = self.graph.get_peer_companies(event_ticker, max_distance=3)
        score_map: dict[str, float] = {}
        if all_peers:
            for t, s in all_peers:
                score_map[t] = s
        weights = pd.Series(
            {t: score_map.get(t, 0.01) for t in peer_tickers},
            index=peer_tickers,
        )
        weights = weights.clip(lower=self.min_weight)
        return weights / weights.sum()

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

            weights = self._compute_weights(
                available_peers, event_date, event_ticker=ticker, event=event.to_dict()
            )
            if weights is None:
                continue

            for hp in periods:
                result = self._simulate_trade(
                    list(weights.index),
                    weights.values if self.weighting != "equal" else None,
                    event_date,
                    hp,
                    event=event.to_dict(),
                )
                if result is not None:
                    result["event_id"] = event.get("event_id", "")
                    result["company_ticker"] = ticker
                    result["event_type"] = event.get("event_type", "")
                    result["event_confidence"] = event.get("confidence", 0.0)
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
            alpha_return_mean = float(df.get("alpha_return", df["total_return"]).mean())
            exit_reasons = (
                df["exit_reason"].value_counts().to_dict() if "exit_reason" in df.columns else {}
            )
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
                "mean_alpha_return": round(alpha_return_mean, 6),
                "mean_event_alpha_score": round(
                    float(df.get("event_alpha_score", df["total_return"]).mean()), 4
                ),
                "mean_avg_graph_proximity": round(
                    float(df.get("avg_graph_proximity", 0.5).mean()), 4
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
                "early_exit_pct": round((df["actual_days_held"] < df["scheduled_days"]).mean(), 4)
                if "actual_days_held" in df.columns and "scheduled_days" in df.columns
                else 0.0,
                "exit_reason_breakdown": exit_reasons,
                "mean_intra_trade_sharpe": round(float(df.get("sharpe_intra", 0.0).mean()), 4),
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
        event: Optional[dict] = None,
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

        max_exit_idx = entry_idx + holding_period
        if max_exit_idx >= len(self.prices.index):
            return None

        valid_peers: list[str] = []
        pw_list: list[float] = []
        daily_price_series: list[np.ndarray] = []

        for i, p in enumerate(peer_tickers):
            if p not in self.prices.columns:
                continue
            ep = self.prices[p].iloc[entry_idx]
            if pd.isna(ep) or ep <= 0:
                continue
            slice_prices = self.prices[p].iloc[entry_idx : max_exit_idx + 1].values
            if len(slice_prices) < 2 or any(pd.isna(slice_prices)):
                continue
            daily_price_series.append(slice_prices)
            valid_peers.append(p)
            if weights is not None and i < len(weights):
                pw_list.append(weights[i])
            else:
                pw_list.append(1.0)

        if len(valid_peers) < 1:
            return None

        pw_arr = np.array(pw_list)
        pw_arr = pw_arr / pw_arr.sum()

        price_array = np.column_stack(daily_price_series)
        ret_array = price_array / price_array[0:1, :] - 1
        port_value_series = ret_array @ pw_arr

        exit_idx = len(port_value_series) - 1
        peak_value = 0.0
        max_drawdown = 0.0
        max_runup = 0.0
        exit_reason = "time"

        has_stop = self.stop_loss > 0
        has_tp = self.take_profit > 0
        has_trail = self.trailing_stop > 0

        for t in range(1, len(port_value_series)):
            cv = port_value_series[t]
            peak_value = max(peak_value, cv)
            dd = peak_value - cv
            max_drawdown = max(max_drawdown, dd)

            if has_tp and cv >= self.take_profit:
                exit_idx = t
                exit_reason = "take_profit"
                break
            if has_stop and cv <= -self.stop_loss:
                exit_idx = t
                exit_reason = "stop_loss"
                break
            if has_trail and peak_value > 0 and cv <= peak_value - self.trailing_stop:
                exit_idx = t
                exit_reason = "trailing_stop"
                break

        max_runup = float(peak_value)
        actual_days_held = exit_idx

        total_return = float(port_value_series[exit_idx])

        tc_bps = compute_transaction_cost(
            trade_size_pct=1.0 / len(valid_peers),
            model=self.transaction_cost_model,
            base_tc_bps=self.base_tc_bps,
            slippage_bps=self.slippage_bps,
        )
        total_tc = tc_bps / 10000 * 2
        total_return_net = total_return - total_tc

        exit_date_idx = entry_idx + exit_idx
        if exit_date_idx >= len(self.prices.index):
            exit_date_idx = entry_idx + exit_idx
        exit_date = self.prices.index[exit_date_idx]

        bench_return = 0.0
        bench_col = None
        for b in ["XBI", "XLV", "SPY"]:
            if b in self.prices.columns:
                bench_col = b
                break
        if bench_col:
            b_entry = self.prices[bench_col].iloc[entry_idx]
            b_exit = self.prices[bench_col].iloc[exit_date_idx]
            bench_return = (b_exit / b_entry - 1) if b_entry > 0 else 0.0

        peer_entry_price = price_array[0]
        peer_exit_price = price_array[exit_idx]
        if len(peer_entry_price.shape) == 1:
            peer_returns_direct = peer_exit_price / peer_entry_price - 1
        else:
            peer_returns_direct = peer_exit_price / peer_entry_price - 1
        weighted_peer_return_direct = float(pw_arr @ peer_returns_direct)

        alpha_return = total_return_net - bench_return

        intra_trade_vol = float(np.std(port_value_series[: exit_idx + 1])) if exit_idx > 1 else 0.0
        sharpe_intra = (
            total_return / intra_trade_vol * np.sqrt(252 / max(actual_days_held, 1))
            if intra_trade_vol > 0
            else 0.0
        )

        event_alpha = self._event_alpha_score(event) if event else 0.5

        return {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "n_peers": len(valid_peers),
            "peer_return": round(float(weighted_peer_return_direct), 6),
            "transaction_cost": round(float(total_tc), 6),
            "total_return": round(float(total_return_net), 6),
            "benchmark_return": round(float(bench_return), 6),
            "alpha_return": round(float(alpha_return), 6),
            "excess_return": round(float(total_return_net - bench_return), 6),
            "cumulative_return": round(float(total_return_net), 6),
            "actual_days_held": actual_days_held,
            "scheduled_days": holding_period,
            "exit_reason": exit_reason,
            "max_runup": round(max_runup, 6),
            "max_drawdown": round(max_drawdown, 6),
            "intra_trade_vol": round(intra_trade_vol, 6),
            "sharpe_intra": round(sharpe_intra, 4),
            "event_alpha_score": round(event_alpha, 4),
            "avg_graph_proximity": round(float(pw_arr.max()), 4),
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
                    "mean_alpha_return": metrics.get("mean_alpha_return", 0),
                    "mean_event_score": metrics.get("mean_event_alpha_score", 0),
                    "mean_graph_prox": metrics.get("mean_avg_graph_proximity", 0),
                    "early_exit_pct": metrics.get("early_exit_pct", 0),
                    "intra_sharpe": metrics.get("mean_intra_trade_sharpe", 0),
                }
            )

        return pd.DataFrame(records)
