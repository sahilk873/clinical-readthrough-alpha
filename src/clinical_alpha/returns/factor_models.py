"""Factor model estimation for abnormal return calculation.

Implements:
- CAPM with rolling OLS beta estimation
- Fama-French 3-factor model
- Fama-French 5-factor model
- Carhart 4-factor (momentum) model
- Fama-MacBeth (1973) two-pass cross-sectional regression
- Newey-West heteroskedasticity and autocorrelation consistent (HAC) standard errors
- GRS (Gibbons, Ross, Shanken 1989) test for model efficiency
- Model selection criteria (AIC, BIC, adjusted R^2)
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def newey_west_se(
    X: np.ndarray,
    residuals: np.ndarray,
    lags: Optional[int] = None,
) -> np.ndarray:
    n = len(residuals)
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))

    XpX_inv = np.linalg.inv(X.T @ X)
    kernel = np.ones(lags + 1)

    Omega = np.zeros((X.shape[1], X.shape[1]))
    for t in range(n):
        xt = X[t, :].reshape(-1, 1)
        Omega += (residuals[t] ** 2) * (xt @ xt.T)

    for lag in range(1, lags + 1):
        weight = kernel[lag] * (1 - lag / (lags + 1))
        for t in range(lag, n):
            xt = X[t, :].reshape(-1, 1)
            xt_lag = X[t - lag, :].reshape(1, -1)
            Omega += weight * residuals[t] * residuals[t - lag] * (xt @ xt_lag + xt_lag.T @ xt.T)

    cov_matrix = XpX_inv @ Omega @ XpX_inv
    return np.sqrt(np.diag(cov_matrix))


def _ols_fit(
    y: np.ndarray,
    X: np.ndarray,
    use_newey_west: bool = True,
) -> dict:
    n, k = X.shape
    try:
        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coefs
        mse = np.sum(residuals**2) / (n - k)
        if use_newey_west:
            se = newey_west_se(X, residuals)
        else:
            var_cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(var_cov))
        tstats = np.where(se > 0, coefs / se, 0.0)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        rsquared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_rsquared = 1 - (1 - rsquared) * (n - 1) / max(n - k, 1)
        aic = n * np.log(ss_res / max(n, 1)) + 2 * k if ss_res > 0 else -np.inf
        bic = n * np.log(ss_res / max(n, 1)) + k * np.log(n) if ss_res > 0 else -np.inf
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        return {
            "coefs": coefs,
            "se": se,
            "tstats": tstats,
            "rsquared": rsquared,
            "adj_rsquared": adj_rsquared,
            "aic": aic,
            "bic": bic,
            "aicc": aicc,
            "n_obs": n,
            "residuals": residuals,
        }
    except np.linalg.LinAlgError:
        return {
            "coefs": np.zeros(k),
            "se": np.zeros(k),
            "tstats": np.zeros(k),
            "rsquared": 0.0,
            "adj_rsquared": 0.0,
            "aic": 0.0,
            "bic": 0.0,
            "aicc": 0.0,
            "n_obs": n,
            "residuals": np.full(n, np.nan),
        }


def estimate_capm(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: pd.Series | float = 0.0,
) -> dict:
    if isinstance(risk_free_rate, (int, float)):
        rf = pd.Series(risk_free_rate, index=stock_returns.index)
    else:
        rf = risk_free_rate

    aligned = pd.concat([stock_returns, market_returns, rf], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "alpha_tstat": 0.0,
            "beta_tstat": 0.0,
            "rsquared": 0.0,
            "n_obs": 0,
        }

    excess_stock = aligned.iloc[:, 0] - aligned.iloc[:, 2]
    excess_market = aligned.iloc[:, 1] - aligned.iloc[:, 2]

    X = np.column_stack([np.ones(len(excess_market)), excess_market.values])
    y = excess_stock.values

    result = _ols_fit(y, X, use_newey_west=True)
    if result["n_obs"] < 10:
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "alpha_tstat": 0.0,
            "beta_tstat": 0.0,
            "rsquared": 0.0,
            "n_obs": 0,
        }

    return {
        "alpha": float(result["coefs"][0]),
        "beta": float(result["coefs"][1]),
        "alpha_tstat": float(result["tstats"][0]),
        "beta_tstat": float(result["tstats"][1]),
        "rsquared": float(result["rsquared"]),
        "adj_rsquared": float(result["adj_rsquared"]),
        "aic": float(result["aic"]),
        "bic": float(result["bic"]),
        "n_obs": result["n_obs"],
    }


def estimate_ff3(
    stock_returns: pd.Series,
    mkt_rf: pd.Series,
    smb: pd.Series,
    hml: pd.Series,
    risk_free_rate: pd.Series | float = 0.0,
    use_newey_west: bool = True,
) -> dict:
    if isinstance(risk_free_rate, (int, float)):
        rf = pd.Series(risk_free_rate, index=stock_returns.index)
    else:
        rf = risk_free_rate

    aligned = pd.concat([stock_returns, mkt_rf, smb, hml, rf], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        return {
            k: 0.0
            for k in [
                "alpha",
                "beta_mkt",
                "beta_smb",
                "beta_hml",
                "alpha_tstat",
                "mkt_tstat",
                "smb_tstat",
                "hml_tstat",
                "rsquared",
                "adj_rsquared",
                "aic",
                "bic",
                "n_obs",
            ]
        }

    excess_stock = aligned.iloc[:, 0] - aligned.iloc[:, 4]
    factors = aligned.iloc[:, 1:4]

    X = np.column_stack([np.ones(len(factors)), factors.values])
    y = excess_stock.values

    result = _ols_fit(y, X, use_newey_west)
    if result["n_obs"] < 20:
        return {
            k: 0.0
            for k in [
                "alpha",
                "beta_mkt",
                "beta_smb",
                "beta_hml",
                "alpha_tstat",
                "mkt_tstat",
                "smb_tstat",
                "hml_tstat",
                "rsquared",
                "adj_rsquared",
                "aic",
                "bic",
                "n_obs",
            ]
        }

    return {
        "alpha": float(result["coefs"][0]),
        "beta_mkt": float(result["coefs"][1]),
        "beta_smb": float(result["coefs"][2]),
        "beta_hml": float(result["coefs"][3]),
        "alpha_tstat": float(result["tstats"][0]),
        "mkt_tstat": float(result["tstats"][1]),
        "smb_tstat": float(result["tstats"][2]),
        "hml_tstat": float(result["tstats"][3]),
        "rsquared": float(result["rsquared"]),
        "adj_rsquared": float(result["adj_rsquared"]),
        "aic": float(result["aic"]),
        "bic": float(result["bic"]),
        "n_obs": result["n_obs"],
    }


def estimate_carhart_4f(
    stock_returns: pd.Series,
    mkt_rf: pd.Series,
    smb: pd.Series,
    hml: pd.Series,
    mom: pd.Series,
    risk_free_rate: pd.Series | float = 0.0,
    use_newey_west: bool = True,
) -> dict:
    """Carhart (1997) 4-factor model adding momentum to FF3.

    R_i - R_f = a + b1*(Mkt-Rf) + b2*SMB + b3*HML + b4*MOM
    """
    if isinstance(risk_free_rate, (int, float)):
        rf = pd.Series(risk_free_rate, index=stock_returns.index)
    else:
        rf = risk_free_rate

    aligned = pd.concat([stock_returns, mkt_rf, smb, hml, mom, rf], axis=1, join="inner").dropna()
    if len(aligned) < 25:
        return {
            k: 0.0
            for k in [
                "alpha",
                "beta_mkt",
                "beta_smb",
                "beta_hml",
                "beta_mom",
                "alpha_tstat",
                "mkt_tstat",
                "smb_tstat",
                "hml_tstat",
                "mom_tstat",
                "rsquared",
                "adj_rsquared",
                "aic",
                "bic",
                "n_obs",
            ]
        }

    excess_stock = aligned.iloc[:, 0] - aligned.iloc[:, 5]
    factors = aligned.iloc[:, 1:5]

    X = np.column_stack([np.ones(len(factors)), factors.values])
    y = excess_stock.values

    result = _ols_fit(y, X, use_newey_west)
    if result["n_obs"] < 25:
        return {
            k: 0.0
            for k in [
                "alpha",
                "beta_mkt",
                "beta_smb",
                "beta_hml",
                "beta_mom",
                "alpha_tstat",
                "mkt_tstat",
                "smb_tstat",
                "hml_tstat",
                "mom_tstat",
                "rsquared",
                "adj_rsquared",
                "aic",
                "bic",
                "n_obs",
            ]
        }

    return {
        "alpha": float(result["coefs"][0]),
        "beta_mkt": float(result["coefs"][1]),
        "beta_smb": float(result["coefs"][2]),
        "beta_hml": float(result["coefs"][3]),
        "beta_mom": float(result["coefs"][4]),
        "alpha_tstat": float(result["tstats"][0]),
        "mkt_tstat": float(result["tstats"][1]),
        "smb_tstat": float(result["tstats"][2]),
        "hml_tstat": float(result["tstats"][3]),
        "mom_tstat": float(result["tstats"][4]),
        "rsquared": float(result["rsquared"]),
        "adj_rsquared": float(result["adj_rsquared"]),
        "aic": float(result["aic"]),
        "bic": float(result["bic"]),
        "n_obs": result["n_obs"],
    }


def estimate_ff5(
    stock_returns: pd.Series,
    mkt_rf: pd.Series,
    smb: pd.Series,
    hml: pd.Series,
    rmw: pd.Series,
    cma: pd.Series,
    risk_free_rate: pd.Series | float = 0.0,
    use_newey_west: bool = True,
) -> dict:
    if isinstance(risk_free_rate, (int, float)):
        rf = pd.Series(risk_free_rate, index=stock_returns.index)
    else:
        rf = risk_free_rate

    aligned = pd.concat(
        [stock_returns, mkt_rf, smb, hml, rmw, cma, rf], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 30:
        return {
            k: 0.0
            for k in [
                "alpha",
                "beta_mkt",
                "beta_smb",
                "beta_hml",
                "beta_rmw",
                "beta_cma",
                "alpha_tstat",
                "mkt_tstat",
                "smb_tstat",
                "hml_tstat",
                "rmw_tstat",
                "cma_tstat",
                "rsquared",
                "adj_rsquared",
                "aic",
                "bic",
                "n_obs",
            ]
        }

    excess_stock = aligned.iloc[:, 0] - aligned.iloc[:, 6]
    factors = aligned.iloc[:, 1:6]

    X = np.column_stack([np.ones(len(factors)), factors.values])
    y = excess_stock.values

    result = _ols_fit(y, X, use_newey_west)
    if result["n_obs"] < 30:
        return {
            k: 0.0
            for k in [
                "alpha",
                "beta_mkt",
                "beta_smb",
                "beta_hml",
                "beta_rmw",
                "beta_cma",
                "alpha_tstat",
                "mkt_tstat",
                "smb_tstat",
                "hml_tstat",
                "rmw_tstat",
                "cma_tstat",
                "rsquared",
                "adj_rsquared",
                "aic",
                "bic",
                "n_obs",
            ]
        }

    return {
        "alpha": float(result["coefs"][0]),
        "beta_mkt": float(result["coefs"][1]),
        "beta_smb": float(result["coefs"][2]),
        "beta_hml": float(result["coefs"][3]),
        "beta_rmw": float(result["coefs"][4]),
        "beta_cma": float(result["coefs"][5]),
        "alpha_tstat": float(result["tstats"][0]),
        "mkt_tstat": float(result["tstats"][1]),
        "smb_tstat": float(result["tstats"][2]),
        "hml_tstat": float(result["tstats"][3]),
        "rmw_tstat": float(result["tstats"][4]),
        "cma_tstat": float(result["tstats"][5]),
        "rsquared": float(result["rsquared"]),
        "adj_rsquared": float(result["adj_rsquared"]),
        "aic": float(result["aic"]),
        "bic": float(result["bic"]),
        "n_obs": result["n_obs"],
    }


def fama_macbeth(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    rolling_window: Optional[int] = None,
) -> dict:
    """Fama-MacBeth (1973) two-pass cross-sectional regression.

    First pass: N time-series regressions (one per asset).
    Second pass: T cross-sectional regressions (one per time period),
    with Shanken (1992) standard error correction.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (columns = assets, index = dates).
    factors : pd.DataFrame
        Factor returns (columns = factors, index = dates).
    rolling_window : int, optional
        Rolling window length for time-varying betas.

    Returns
    -------
    dict with keys: avg_coefs, shanken_se, shanken_tstats,
                    avg_rsquared, n_assets, n_periods
    """
    aligned = pd.concat([returns, factors], axis=1, join="inner").dropna()
    if aligned.empty:
        return {
            "avg_coefs": np.array([]),
            "shanken_se": np.array([]),
            "shanken_tstats": np.array([]),
            "avg_rsquared": 0.0,
            "n_assets": 0,
            "n_periods": 0,
        }

    n_assets = returns.shape[1]
    n_factors = factors.shape[1]
    n_periods = len(aligned)

    if n_assets < 2 or n_periods < 10:
        return {
            "avg_coefs": np.array([]),
            "shanken_se": np.array([]),
            "shanken_tstats": np.array([]),
            "avg_rsquared": 0.0,
            "n_assets": 0,
            "n_periods": 0,
        }

    asset_cols = list(returns.columns)
    factor_cols = list(factors.columns)
    n = len(aligned)

    if rolling_window is not None and rolling_window < n:
        beta_slices = []
        for start in range(0, n - rolling_window + 1):
            end = start + rolling_window
            sub = aligned.iloc[start:end]
            betas = []
            for asset in asset_cols:
                y = sub[asset].values
                X = np.column_stack([np.ones(rolling_window), sub[factor_cols].values])
                try:
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    betas.append(coef)
                except np.linalg.LinAlgError:
                    betas.append(np.zeros(n_factors + 1))
            beta_slices.append(np.array(betas))
        beta_estimates = np.nanmean(beta_slices, axis=0)
    else:
        beta_estimates = np.zeros((n_assets, n_factors + 1))
        for i, asset in enumerate(asset_cols):
            y = aligned[asset].values
            X = np.column_stack([np.ones(n), aligned[factor_cols].values])
            try:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                beta_estimates[i] = coef
            except np.linalg.LinAlgError:
                beta_estimates[i] = np.zeros(n_factors + 1)

    gamma_t = np.zeros((n_periods, n_factors + 1))
    r2_list = []
    for t in range(n_periods):
        y_cs = aligned[asset_cols].values[t, :]
        X_cs = beta_estimates
        try:
            gamma = np.linalg.lstsq(X_cs, y_cs, rcond=None)[0]
            gamma_t[t] = gamma
            pred = X_cs @ gamma
            ss_res = np.sum((y_cs - pred) ** 2)
            ss_tot = np.sum((y_cs - y_cs.mean()) ** 2)
            r2_list.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        except np.linalg.LinAlgError:
            gamma_t[t] = np.zeros(n_factors + 1)

    avg_coefs = gamma_t.mean(axis=0)
    T = n_periods
    Gamma = np.cov(gamma_t, rowvar=False)
    naive_se = np.sqrt(np.diag(Gamma) / T)

    mu_f = aligned[factor_cols].mean().values
    Sigma_f = aligned[factor_cols].cov().values
    Sigma_f_inv = np.linalg.inv(Sigma_f) if Sigma_f.shape[0] > 0 else np.eye(1)

    shanken_factor = 1 + mu_f @ Sigma_f_inv @ mu_f
    shanken_se = naive_se * np.sqrt(shanken_factor)
    shanken_tstats = np.where(shanken_se > 0, avg_coefs / shanken_se, 0.0)

    return {
        "avg_coefs": avg_coefs,
        "naive_se": naive_se,
        "naive_tstats": np.where(naive_se > 0, avg_coefs / naive_se, 0.0),
        "shanken_se": shanken_se,
        "shanken_tstats": shanken_tstats,
        "avg_rsquared": float(np.mean(r2_list)) if r2_list else 0.0,
        "n_assets": n_assets,
        "n_periods": n_periods,
    }


def compare_factor_models(
    stock_returns: pd.Series,
    factor_sets: dict[str, dict[str, pd.Series]],
    risk_free_rate: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """Compare multiple factor models on the same asset using model selection criteria.

    Parameters
    ----------
    stock_returns : pd.Series
    factor_sets : dict
        e.g. {"CAPM": {"mkt_rf": mkt}, "FF3": {"mkt_rf": mkt, "smb": s, "hml": h}}
    risk_free_rate : pd.Series or float

    Returns
    -------
    pd.DataFrame with rows = models, columns = metrics
    """
    records = []
    for model_name, factors in factor_sets.items():
        if model_name == "CAPM":
            result = estimate_capm(
                stock_returns, factors.get("market_returns", factors.get("mkt_rf")), risk_free_rate
            )
        elif model_name == "FF3":
            result = estimate_ff3(
                stock_returns, factors["mkt_rf"], factors["smb"], factors["hml"], risk_free_rate
            )
        elif model_name == "Carhart4F":
            result = estimate_carhart_4f(
                stock_returns,
                factors["mkt_rf"],
                factors["smb"],
                factors["hml"],
                factors["mom"],
                risk_free_rate,
            )
        elif model_name == "FF5":
            result = estimate_ff5(
                stock_returns,
                factors["mkt_rf"],
                factors["smb"],
                factors["hml"],
                factors["rmw"],
                factors["cma"],
                risk_free_rate,
            )
        else:
            continue
        records.append(
            {
                "model": model_name,
                "rsquared": result.get("rsquared", 0),
                "adj_rsquared": result.get("adj_rsquared", 0),
                "aic": result.get("aic", 0),
                "bic": result.get("bic", 0),
                "alpha_tstat": result.get("alpha_tstat", 0),
                "n_obs": result.get("n_obs", 0),
            }
        )
    return pd.DataFrame(records)


def grs_test(
    fund_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> dict:
    aligned = pd.concat([fund_returns, factor_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"grs_statistic": 0.0, "p_value": 1.0, "df1": 0, "df2": 0}

    n = len(aligned)
    n_assets = fund_returns.shape[1]
    n_factors = factor_returns.shape[1]

    if n_assets < 2 or n_factors < 1:
        return {"grs_statistic": 0.0, "p_value": 1.0, "df1": 0, "df2": 0}

    y = aligned.iloc[:, :n_assets].values
    X = np.column_stack([np.ones(n), aligned.iloc[:, n_assets:].values])

    try:
        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coefs
        alphas = coefs[0, :]

        sigma_hat = (residuals.T @ residuals) / (n - n_factors - 1)
        Omega_hat = np.atleast_2d(np.cov(aligned.iloc[:, n_assets:].values, rowvar=False))
        mu_f = aligned.iloc[:, n_assets:].mean().values

        Omega_inv = np.linalg.inv(Omega_hat) if Omega_hat.shape[0] > 0 else np.eye(1)
        sigma_inv = np.linalg.inv(sigma_hat) if sigma_hat.shape[0] > 0 else np.eye(1)

        theta = mu_f @ Omega_inv @ mu_f
        grs_stat = ((n - n_factors - 1) / n_assets) * (alphas @ sigma_inv @ alphas) / (1 + theta)
        p_val = 1 - scipy_stats.f.cdf(grs_stat, n_assets, n - n_factors - 1)

        return {
            "grs_statistic": float(grs_stat),
            "p_value": float(p_val),
            "df1": n_assets,
            "df2": n - n_factors - 1,
        }
    except np.linalg.LinAlgError:
        return {"grs_statistic": 0.0, "p_value": 1.0, "df1": 0, "df2": 0}
