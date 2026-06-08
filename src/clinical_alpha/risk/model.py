"""Risk model for portfolio construction and risk decomposition.

Implements:
- Shrinkage covariance estimation (Ledoit-Wolf)
- Risk decomposition (marginal, component, diversification ratio)
- Factor risk model (PCA-based statistical factors)
- Value-at-Risk and Expected Shortfall
- Regime-switching covariance estimation (2-state)
- Copula-based dependence modeling
- Liquidity-adjusted VaR (La-VaR)
"""

import numpy as np
import pandas as pd
from scipy.special import erfinv

from clinical_alpha.exceptions import RiskError


def shrinkage_covariance(
    returns: pd.DataFrame,
    shrinkage_lambda: float = 0.5,
) -> pd.DataFrame:
    n, p = returns.shape
    if p < 2:
        return returns.cov()

    sample_cov = returns.cov().values
    volatilities = np.sqrt(np.diag(sample_cov))

    if (volatilities == 0).any():
        return pd.DataFrame(sample_cov, index=returns.columns, columns=returns.columns)

    correlation = returns.corr().values
    avg_correlation = (correlation.sum().sum() - p) / (p * (p - 1)) if p > 1 else 0.0
    target_cov = np.outer(volatilities, volatilities) * avg_correlation
    np.fill_diagonal(target_cov, volatilities**2)

    if not shrinkage_lambda:
        shrinkage_lambda_opt = 0.5
    else:
        shrinkage_lambda_opt = shrinkage_lambda

    shrunk = shrinkage_lambda_opt * target_cov + (1 - shrinkage_lambda_opt) * sample_cov

    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


def regime_switching_covariance(
    returns: pd.DataFrame,
    n_regimes: int = 2,
    random_seed: int = 42,
) -> dict:
    """Regime-switching covariance via Gaussian mixture separation.

    Estimates separate covariance matrices for each regime using
    a volatility-based clustering heuristic (low-vol / high-vol regimes).

    Parameters
    ----------
    returns : pd.DataFrame
    n_regimes : int
        Number of regimes (default 2: low/high vol).
    random_seed : int

    Returns
    -------
    dict with keys: regime_covariances, regime_weights, regime_assignments
    """
    from sklearn.cluster import KMeans

    if returns.shape[1] < 2 or len(returns) < 20:
        full_cov = returns.cov()
        return {
            "regime_covariances": {0: full_cov},
            "regime_weights": {0: 1.0},
            "regime_assignments": pd.Series(0, index=returns.index),
            "regime_volatilities": {0: returns.std().mean()},
        }

    port_vol = returns.std(axis=1).values.reshape(-1, 1)
    n_regimes = min(n_regimes, len(returns) // 10)
    if n_regimes < 2:
        full_cov = returns.cov()
        return {
            "regime_covariances": {0: full_cov},
            "regime_weights": {0: 1.0},
            "regime_assignments": pd.Series(0, index=returns.index),
            "regime_volatilities": {0: returns.std().mean()},
        }

    kmeans = KMeans(n_clusters=n_regimes, random_state=random_seed, n_init=5)
    labels = kmeans.fit_predict(port_vol)
    assignments = pd.Series(labels, index=returns.index)

    regime_covs = {}
    regime_weights = {}
    regime_vols = {}
    for regime in range(n_regimes):
        mask = assignments == regime
        if mask.sum() > 10:
            regime_covs[regime] = returns[mask].cov()
            regime_weights[regime] = mask.sum() / len(returns)
            regime_vols[regime] = float(returns[mask].std().mean())
        else:
            regime_covs[regime] = returns.cov()
            regime_weights[regime] = 0.0
            regime_vols[regime] = float(returns.std().mean())

    return {
        "regime_covariances": regime_covs,
        "regime_weights": regime_weights,
        "regime_assignments": assignments,
        "regime_volatilities": regime_vols,
    }


def copula_dependence(
    returns: pd.DataFrame,
    method: str = "gaussian",
) -> dict:
    """Copula-based dependence measures beyond linear correlation.

    Parameters
    ----------
    returns : pd.DataFrame
    method : str
        'gaussian', 'clayton', or 'frank'

    Returns
    -------
    dict with keys: kendall_tau, upper_tail_dependence, lower_tail_dependence
    """
    n_assets = returns.shape[1]
    if n_assets < 2:
        return {
            "kendall_tau": np.array([[1.0]]),
            "upper_tail_dependence": np.array([[1.0]]),
            "lower_tail_dependence": np.array([[1.0]]),
        }

    kendall_tau = returns.corr(method="kendall").values

    if method == "gaussian":
        theta = np.sin(kendall_tau * np.pi / 2)
        upper_tail = np.zeros_like(theta)
        lower_tail = np.zeros_like(theta)
    elif method == "clayton":
        denom = np.clip(1 - kendall_tau, 0.001, None)
        theta = 2 * kendall_tau / denom
        theta = np.clip(theta, 0.001, 50)
        upper_tail = np.zeros_like(theta)
        lower_tail = 2.0 ** (-1.0 / theta)
    elif method == "frank":
        denom = np.clip(1 - kendall_tau, 0.001, None)
        theta = np.where(kendall_tau != 0, 2.0 * kendall_tau / denom, 0.0)
        upper_tail = np.zeros_like(theta)
        lower_tail = np.zeros_like(theta)
    else:
        theta = np.sin(kendall_tau * np.pi / 2)
        upper_tail = np.zeros_like(theta)
        lower_tail = np.zeros_like(theta)

    np.fill_diagonal(upper_tail, 1.0)
    np.fill_diagonal(lower_tail, 1.0)

    return {
        "kendall_tau": kendall_tau,
        "theta": theta,
        "upper_tail_dependence": upper_tail,
        "lower_tail_dependence": lower_tail,
        "method": method,
    }


def risk_decomposition(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> dict:
    if np.abs(weights).sum() == 0:
        return {
            "portfolio_vol": 0.0,
            "marginal_contributions": np.zeros(len(weights)),
            "component_contributions": np.zeros(len(weights)),
            "pct_contributions": np.zeros(len(weights)),
            "diversification_ratio": 0.0,
        }

    portfolio_var = weights @ covariance @ weights
    portfolio_vol = np.sqrt(max(portfolio_var, 1e-16))

    marginal = (
        (covariance @ weights) / portfolio_vol if portfolio_vol > 0 else np.zeros(len(weights))
    )
    component = weights * marginal
    pct_contrib = component / component.sum() if component.sum() > 0 else component

    weighted_vol = np.sqrt(np.diag(covariance)) @ np.abs(weights)
    div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0

    return {
        "portfolio_vol": float(portfolio_vol),
        "marginal_contributions": marginal,
        "component_contributions": component,
        "pct_contributions": pct_contrib,
        "diversification_ratio": float(div_ratio),
    }


def pca_factor_model(
    returns: pd.DataFrame,
    n_factors: int = 5,
) -> dict:
    n_factors = min(n_factors, returns.shape[1], returns.shape[0] - 1)
    if n_factors < 1:
        return {
            "factors": pd.DataFrame(),
            "loadings": np.array([]),
            "explained_var": np.array([]),
            "cumulative_var": np.array([]),
        }

    standardized = (returns - returns.mean()) / returns.std()
    standardized = standardized.fillna(0)

    X = standardized.values
    X_centered = X - X.mean(axis=0)

    n, p = X_centered.shape
    if n < p:
        eigvals, eigvecs = np.linalg.eigh(X_centered @ X_centered.T)
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_idx][:n_factors]
        U = eigvecs[:, sorted_idx][:, :n_factors]
        loadings = (X_centered.T @ U) / np.sqrt(np.maximum(eigvals * (n - 1), 1e-16))
        factors = pd.DataFrame(
            U * np.sqrt(np.maximum(eigvals, 0)),
            index=returns.index,
        )
    else:
        cov = (X_centered.T @ X_centered) / (n - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_idx][:n_factors]
        loadings = eigvecs[:, sorted_idx][:, :n_factors]
        factors = pd.DataFrame(
            X_centered @ loadings,
            index=returns.index,
        )

    total_var = eigvals.sum() if eigvals.sum() > 0 else 1.0
    explained_var = eigvals / total_var
    cumulative_var = np.cumsum(explained_var)

    return {
        "factors": factors,
        "loadings": loadings,
        "explained_var": explained_var,
        "cumulative_var": cumulative_var,
    }


def value_at_risk(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> dict:
    values = returns.values.flatten() if isinstance(returns, pd.DataFrame) else returns.values
    values = values[~np.isnan(values)]

    if len(values) < 10:
        return {
            "var": 0.0,
            "expected_shortfall": 0.0,
            "method": method,
            "confidence_level": confidence_level,
        }

    alpha = 1 - confidence_level

    if method == "historical":
        var = np.percentile(values, 100 * alpha)
        es = values[values <= var].mean() if (values <= var).sum() > 0 else var

    elif method == "gaussian":
        mu = values.mean()
        sigma = values.std()
        var = mu + sigma * np.sqrt(2) * erfinv(2 * alpha - 1)
        es = (
            mu - sigma * np.exp(-0.5 * ((var - mu) / sigma) ** 2) / (alpha * np.sqrt(2 * np.pi))
            if sigma > 0
            else var
        )

    elif method == "cornish_fisher":
        mu = values.mean()
        sigma = values.std()
        skew = pd.Series(values).skew()
        kurt = pd.Series(values).kurtosis()
        z_alpha = np.sqrt(2) * erfinv(2 * alpha - 1)
        z_cf = (
            z_alpha
            + (skew / 6) * (z_alpha**2 - 1)
            + (kurt / 24) * (z_alpha**3 - 3 * z_alpha)
            - (skew**2 / 36) * (2 * z_alpha**3 - 5 * z_alpha)
        )
        var = mu + sigma * z_cf
        es = (
            mu - sigma * np.exp(-0.5 * z_cf**2) / (alpha * np.sqrt(2 * np.pi)) if sigma > 0 else var
        )

    else:
        raise RiskError(f"Unknown VaR method: {method}")

    return {
        "var": float(var),
        "expected_shortfall": float(es),
        "method": method,
        "confidence_level": confidence_level,
    }


def liquidity_adjusted_var(
    returns: pd.Series,
    avg_bid_ask_spread: float = 0.001,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
) -> dict:
    """Liquidity-adjusted Value-at-Risk (La-VaR).

    Adjusts VaR for illiquidity using the exogenous spread approach
    (Bangia et al. 1999, Jorion 2000).

    La-VaR = VaR + 0.5 * spread * position_value
    """
    var_result = value_at_risk(returns, confidence_level, method="historical")
    base_var = abs(var_result["var"])

    liquidity_cost = 0.5 * avg_bid_ask_spread * np.sqrt(holding_period_days)
    la_var = base_var + liquidity_cost
    la_es = abs(var_result["expected_shortfall"]) + liquidity_cost

    return {
        "var": float(-base_var),
        "liquidity_adjusted_var": float(-la_var),
        "expected_shortfall": float(-var_result["expected_shortfall"]),
        "liquidity_adjusted_es": float(-la_es),
        "liquidity_cost_bps": float(liquidity_cost * 10000),
        "bid_ask_spread": avg_bid_ask_spread,
        "confidence_level": confidence_level,
        "holding_period_days": holding_period_days,
    }


def min_variance_weights(
    covariance: pd.DataFrame,
    allow_short: bool = False,
    max_weight: float = 0.30,
) -> pd.Series:
    n = len(covariance)
    assets = covariance.columns

    if n == 1:
        return pd.Series([1.0], index=assets)

    from scipy.optimize import minimize

    def portfolio_var(w):
        return w @ covariance.values @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0, max_weight)] * n if not allow_short else [(-max_weight, max_weight)] * n

    w0 = np.ones(n) / n
    result = minimize(portfolio_var, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        return pd.Series(result.x, index=assets)
    else:
        return pd.Series(w0, index=assets)


def risk_parity_weights(
    covariance: pd.DataFrame,
    max_weight: float = 0.30,
) -> pd.Series:
    n = len(covariance)
    assets = covariance.columns

    if n == 1:
        return pd.Series([1.0], index=assets)

    from scipy.optimize import minimize

    def risk_parity_obj(w):
        w = np.maximum(w, 0)
        w = w / w.sum()
        port_var = w @ covariance.values @ w
        if port_var <= 0:
            return 1e10
        mrc = covariance.values @ w / np.sqrt(port_var)
        rc = w * mrc
        target_rc = port_var**0.5 / n
        return np.sum((rc - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n
    result = minimize(risk_parity_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        return pd.Series(result.x / result.x.sum(), index=assets)
    else:
        return pd.Series(w0, index=assets)
