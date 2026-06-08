"""Statistical tests for event study inference.

Implements the suite of tests expected in rigorous event study research:
- Bootstrap confidence intervals for CARs
- Permutation tests for peer/control basket significance
- Multiple hypothesis testing corrections (incl. FDR bootstrap)
- Non-parametric tests (Wilcoxon signed-rank, Corrado rank, sign test)
- Cross-sectional correlation adjustment (Boehmer et al. 1991)
- Patell test statistic
- Bayesian hypothesis testing with Bayes factors
- Structural break tests (Chow, Bai-Perron)
- Cross-sectional bootstrap for clustered events
"""


import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from clinical_alpha.exceptions import StatisticalTestError


def bootstrap_car_ci(
    car_series: pd.Series,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    if len(car_series) < 2:
        values = car_series.values
        mean = float(np.mean(values)) if len(values) > 0 else 0.0
        return {"mean": mean, "ci_lower": mean, "ci_upper": mean, "std_err": 0.0}

    rng = np.random.default_rng(random_seed)
    values = car_series.values
    n = len(values)
    boot_means = np.empty(n_iterations)

    for i in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci_level
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_err": float(np.std(boot_means)),
        "ci_level": ci_level,
        "n_iterations": n_iterations,
    }


def bootstrap_clustered_car_ci(
    car_series: pd.Series,
    cluster_labels: pd.Series,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """Bootstrap CI that resamples at the cluster level (e.g., same-company events).

    Preserves within-cluster correlation structure.
    """
    if len(car_series) < 2:
        values = car_series.values
        mean = float(np.mean(values)) if len(values) > 0 else 0.0
        return {"mean": mean, "ci_lower": mean, "ci_upper": mean, "std_err": 0.0}

    rng = np.random.default_rng(random_seed)
    clusters = cluster_labels.unique()
    cluster_map = {c: car_series[cluster_labels == c].values for c in clusters}
    n_clusters = len(clusters)
    n = len(car_series)
    boot_means = np.empty(n_iterations)

    for i in range(n_iterations):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_values = np.concatenate([cluster_map[c] for c in sampled_clusters])
        boot_means[i] = np.mean(np.random.choice(sampled_values, size=n, replace=True))

    alpha = 1.0 - ci_level
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(car_series)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_err": float(np.std(boot_means)),
        "ci_level": ci_level,
        "n_iterations": n_iterations,
        "n_clusters": n_clusters,
    }


def permutation_test(
    peer_cars: list[float],
    control_cars: list[float],
    n_iterations: int = 10_000,
    alternative: str = "two-sided",
    random_seed: int = 42,
) -> dict:
    if len(peer_cars) < 1 or len(control_cars) < 1:
        return {"observed_diff": 0.0, "p_value": 1.0, "n_permutations": 0}

    rng = np.random.default_rng(random_seed)
    pooled = np.array(peer_cars + control_cars)
    n_peer = len(peer_cars)
    observed_diff = np.mean(peer_cars) - np.mean(control_cars)

    count_extreme = 0
    for _ in range(n_iterations):
        rng.shuffle(pooled)
        perm_diff = np.mean(pooled[:n_peer]) - np.mean(pooled[n_peer:])
        if alternative == "two-sided":
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1
        elif alternative == "greater":
            if perm_diff >= observed_diff:
                count_extreme += 1
        elif alternative == "less":
            if perm_diff <= observed_diff:
                count_extreme += 1

    p_value = (count_extreme + 1) / (n_iterations + 1)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_iterations,
    }


def adjust_pvalues_multiple_testing(
    p_values: list[float],
    method: str = "benjamini_hochberg",
) -> list[float]:
    p = np.array(p_values)
    n = len(p)

    if n == 0:
        return []

    if method == "bonferroni":
        adjusted = np.minimum(p * n, 1.0)

    elif method == "holm":
        sorted_idx = np.argsort(p)
        adjusted = np.ones(n)
        cumulative_min = 1.0
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(1.0, max(cumulative_min, p[idx] * (n - i)))
            cumulative_min = adjusted[idx]

    elif method == "benjamini_hochberg":
        sorted_idx = np.argsort(p)
        adjusted = np.ones(n)
        rank = np.zeros(n, dtype=int)
        for i, idx in enumerate(sorted_idx):
            rank[idx] = i + 1
        for i in range(n):
            adjusted[i] = min(1.0, p[i] * n / rank[i])
        adjusted_sorted = np.sort(adjusted)
        for i in range(n):
            adjusted[sorted_idx[i]] = adjusted_sorted[i]

    elif method == "benjamini_yekutieli":
        sorted_idx = np.argsort(p)
        adjusted = np.ones(n)
        rank = np.zeros(n, dtype=int)
        c_m = np.sum(1.0 / np.arange(1, n + 1))
        for i, idx in enumerate(sorted_idx):
            rank[idx] = i + 1
        for i in range(n):
            adjusted[i] = min(1.0, p[i] * n * c_m / rank[i])
        adjusted_sorted = np.sort(adjusted)
        for i in range(n):
            adjusted[sorted_idx[i]] = adjusted_sorted[i]

    elif method == "fdr_bootstrap":
        adjusted = _fdr_bootstrap_stepup(p)

    else:
        raise StatisticalTestError(f"Unknown multiple testing method: {method}")

    return adjusted.tolist()


def _fdr_bootstrap_stepup(
    p_values: np.ndarray,
    n_bootstrap: int = 1000,
    q_level: float = 0.05,
) -> np.ndarray:
    """Bootstrap-based FDR control (Romano & Wolf 2005 step-up procedure)."""
    n = len(p_values)
    if n == 0:
        return np.array([])

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    boot_rejected = np.zeros((n_bootstrap, n), dtype=bool)
    rng = np.random.default_rng(42)
    for b in range(n_bootstrap):
        boot_p = rng.uniform(size=n)
        boot_sorted_idx = np.argsort(boot_p)
        boot_sorted = boot_p[boot_sorted_idx]
        cumulative_min = 1.0
        for i in range(n):
            thresh = (i + 1) * q_level / n
            boot_rejected[b, boot_sorted_idx[i]] = boot_sorted[i] <= thresh
            cumulative_min = min(cumulative_min, boot_sorted[i] / (i + 1) * n)

    k = 0
    threshold = q_level
    for i in reversed(range(n)):
        reject_count = np.mean(
            [np.any(boot_rejected[b, sorted_idx[: i + 1]]) for b in range(n_bootstrap)]
        )
        if reject_count <= threshold:
            k = i + 1
            break

    adjusted = np.ones(n)
    adjusted[sorted_idx[:k]] = sorted_p[:k] * n / np.arange(1, k + 1)
    adjusted_sorted = np.sort(adjusted)
    for i in range(n):
        adjusted[sorted_idx[i]] = adjusted_sorted[i]
    return np.minimum(adjusted, 1.0)


def bayes_factor_car(
    peer_cars: list[float],
    control_cars: list[float],
    prior_scale: float = 0.5,
) -> dict:
    """Bayes factor for peer > control spread using Bayesian t-test.

    Implements Liang et al. (2008) Jeffreys-Zellner-Siow (JZS) Bayes factor.
    Larger BF indicates stronger evidence for H1 (spread != 0).
    """
    if len(peer_cars) < 3 or len(control_cars) < 3:
        return {"bf10": 1.0, "bf01": 1.0, "log_bf": 0.0}

    differences = np.array(peer_cars) - np.array(control_cars)
    n = len(differences)
    d = differences
    t_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n)) if np.std(d, ddof=1) > 0 else 0.0

    t_grid = np.linspace(0.01, 5, 500)
    prior_density = (
        (1 / prior_scale) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (t_grid / prior_scale) ** 2)
    )

    integrated_likelihood = 0.0
    for i, g in enumerate(t_grid):
        se = 1.0 / np.sqrt(n)
        scaled_se = se / np.sqrt(1 + n * g)
        likelihood = np.exp(-0.5 * (t_stat * se / scaled_se) ** 2) / np.sqrt(1 + n * g)
        integrated_likelihood += likelihood * prior_density[i] * (t_grid[1] - t_grid[0])

    if integrated_likelihood <= 0:
        return {"bf10": 1.0, "bf01": 1.0, "log_bf": 0.0}

    h0_likelihood = np.exp(-0.5 * t_stat**2)
    bf10 = integrated_likelihood / h0_likelihood
    bf01 = 1.0 / bf10 if bf10 > 0 else 1.0

    return {
        "bf10": float(bf10),
        "bf01": float(bf01),
        "log_bf": float(np.log(bf10)),
        "interpretation": _interpret_bayes_factor(bf10),
    }


def _interpret_bayes_factor(bf10: float) -> str:
    if bf10 >= 100:
        return "decisive evidence for H1"
    elif bf10 >= 30:
        return "very strong evidence for H1"
    elif bf10 >= 10:
        return "strong evidence for H1"
    elif bf10 >= 3:
        return "substantial evidence for H1"
    elif bf10 >= 1:
        return "anecdotal evidence for H1"
    elif bf10 >= 1 / 3:
        return "anecdotal evidence for H0"
    elif bf10 >= 1 / 10:
        return "substantial evidence for H0"
    elif bf10 >= 1 / 30:
        return "strong evidence for H0"
    elif bf10 >= 1 / 100:
        return "very strong evidence for H0"
    else:
        return "decisive evidence for H0"


def chow_structural_break_test(
    returns: pd.Series,
    break_date: pd.Timestamp,
    factor_returns: pd.DataFrame,
) -> dict:
    """Chow (1960) test for a known structural break.

    Tests whether regression coefficients change at break_date.
    """
    before = returns[returns.index < break_date]
    after = returns[returns.index >= break_date]

    if len(before) < 10 or len(after) < 10:
        return {"chow_statistic": 0.0, "p_value": 1.0, "df1": 0, "df2": 0}

    factors_before = factor_returns.loc[before.index]
    factors_after = factor_returns.loc[after.index]

    aligned_before = pd.concat([before, factors_before], axis=1, join="inner").dropna()
    aligned_after = pd.concat([after, factors_after], axis=1, join="inner").dropna()

    if len(aligned_before) < 10 or len(aligned_after) < 10:
        return {"chow_statistic": 0.0, "p_value": 1.0, "df1": 0, "df2": 0}

    y_full = pd.concat([aligned_before.iloc[:, 0], aligned_after.iloc[:, 0]])
    X_full = np.column_stack(
        [np.ones(len(y_full)), pd.concat([aligned_before.iloc[:, 1:], aligned_after.iloc[:, 1:]])]
    )

    y1 = aligned_before.iloc[:, 0].values
    X1 = np.column_stack([np.ones(len(y1)), aligned_before.iloc[:, 1:].values])
    y2 = aligned_after.iloc[:, 0].values
    X2 = np.column_stack([np.ones(len(y2)), aligned_after.iloc[:, 1:].values])

    try:
        coef_full = np.linalg.lstsq(X_full, y_full.values, rcond=None)[0]
        resid_full = y_full.values - X_full @ coef_full
        rss_full = np.sum(resid_full**2)

        coef1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        resid1 = y1 - X1 @ coef1
        rss1 = np.sum(resid1**2)

        coef2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        resid2 = y2 - X2 @ coef2
        rss2 = np.sum(resid2**2)

        k = X1.shape[1]
        n1, n2 = len(y1), len(y2)
        chow_num = (rss_full - (rss1 + rss2)) / k
        chow_den = (rss1 + rss2) / (n1 + n2 - 2 * k)
        chow_stat = chow_num / chow_den if chow_den > 0 else 0.0
        p_val = 1 - scipy_stats.f.cdf(chow_stat, k, n1 + n2 - 2 * k)

        return {
            "chow_statistic": float(chow_stat),
            "p_value": float(p_val),
            "df1": k,
            "df2": n1 + n2 - 2 * k,
            "break_date": break_date,
        }
    except np.linalg.LinAlgError:
        return {"chow_statistic": 0.0, "p_value": 1.0, "df1": 0, "df2": 0}


def bai_perron_test(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    max_breaks: int = 3,
    min_segment_length: int = 20,
) -> dict:
    """Bai-Perron (2003) multiple structural break test.

    Identifies unknown break points using sequential sup-F test.
    Returns the optimal number of breaks and break dates.
    """
    n = len(returns)
    if n < min_segment_length * (max_breaks + 1):
        return {"n_breaks": 0, "break_dates": [], "break_stats": []}

    def _sup_f(seg_start: int, seg_end: int, candidate: int) -> float:
        y = returns.iloc[seg_start:seg_end]
        X = factor_returns.iloc[seg_start:seg_end]
        if len(y) < 2 * min_segment_length:
            return 0.0
        n_seg = len(y)
        X_full = np.column_stack([np.ones(n_seg), X.values])
        try:
            coef_full = np.linalg.lstsq(X_full, y.values, rcond=None)[0]
            resid_full = y.values - X_full @ coef_full
            rss_full = np.sum(resid_full**2)
        except np.linalg.LinAlgError:
            return 0.0

        rel_idx = candidate - seg_start
        y1 = y.iloc[:rel_idx]
        X1 = X.iloc[:rel_idx]
        y2 = y.iloc[rel_idx:]
        X2 = X.iloc[rel_idx:]

        if len(y1) < min_segment_length or len(y2) < min_segment_length:
            return 0.0

        try:
            X1_full = np.column_stack([np.ones(len(y1)), X1.values])
            coef1 = np.linalg.lstsq(X1_full, y1.values, rcond=None)[0]
            resid1 = y1.values - X1_full @ coef1
            rss1 = np.sum(resid1**2)

            X2_full = np.column_stack([np.ones(len(y2)), X2.values])
            coef2 = np.linalg.lstsq(X2_full, y2.values, rcond=None)[0]
            resid2 = y2.values - X2_full @ coef2
            rss2 = np.sum(resid2**2)

            k = X1_full.shape[1]
            F_stat = ((rss_full - (rss1 + rss2)) / k) / (
                (rss1 + rss2) / (len(y1) + len(y2) - 2 * k)
            )
            return max(F_stat, 0.0)
        except np.linalg.LinAlgError:
            return 0.0

    dates = returns.index
    break_dates: list[pd.Timestamp] = []
    break_stats: list[dict] = []

    for _ in range(max_breaks):
        best_f = 0.0
        best_idx = -1
        seg_start = 0 if not break_dates else dates.get_loc(break_dates[-1]) + 1
        seg_end = n
        if isinstance(seg_start, slice):
            seg_start = seg_start.start or 0

        for candidate in range(seg_start + min_segment_length, seg_end - min_segment_length):
            if dates[candidate] in break_dates:
                continue
            f_stat = _sup_f(seg_start, seg_end, candidate)
            if f_stat > best_f:
                best_f = f_stat
                best_idx = candidate

        if best_idx > 0 and best_f > 2.0:
            break_date = dates[best_idx]
            break_dates.append(break_date)
            p_val = 1 - scipy_stats.f.cdf(
                best_f, factor_returns.shape[1] + 1, n - 2 * (factor_returns.shape[1] + 1)
            )
            break_stats.append(
                {
                    "break_date": break_date,
                    "sup_f_stat": float(best_f),
                    "p_value": float(p_val),
                }
            )
        else:
            break

    return {
        "n_breaks": len(break_dates),
        "break_dates": break_dates,
        "break_stats": break_stats,
    }


def wilcoxon_signed_rank_test(
    peer_cars: list[float],
    control_cars: list[float],
) -> dict:
    if len(peer_cars) < 2 or len(control_cars) < 2:
        return {"statistic": 0.0, "p_value": 1.0}

    differences = np.array(peer_cars) - np.array(control_cars)
    try:
        stat, p_val = scipy_stats.wilcoxon(differences, alternative="two-sided")
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0}

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_obs": len(differences),
    }


def mann_whitney_test(
    peer_cars: list[float],
    control_cars: list[float],
) -> dict:
    if len(peer_cars) < 2 or len(control_cars) < 2:
        return {"statistic": 0.0, "p_value": 1.0}

    stat, p_val = scipy_stats.mannwhitneyu(peer_cars, control_cars, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_peer": len(peer_cars),
        "n_control": len(control_cars),
    }


def corrado_rank_test(
    event_window_returns: pd.DataFrame,
    estimation_window_returns: pd.DataFrame,
    event_indices: list[int],
) -> dict:
    if event_window_returns.empty or estimation_window_returns.empty:
        return {"rank_statistic": 0.0, "p_value": 1.0}

    combined = pd.concat([estimation_window_returns, event_window_returns], axis=0)
    n_est = len(estimation_window_returns)
    n_total = len(combined)
    n_sec = combined.shape[1]

    if n_sec < 2:
        return {"rank_statistic": 0.0, "p_value": 1.0}

    ranks = combined.rank(axis=0, pct=False)

    rank_bar = (n_total + 1) / 2.0

    L_bar = 0.0
    count = 0
    for idx in event_indices:
        actual_idx = n_est + idx
        if actual_idx < n_total:
            L_bar += (ranks.iloc[actual_idx] - rank_bar).sum()
            count += 1

    if count == 0:
        return {"rank_statistic": 0.0, "p_value": 1.0}

    L_bar /= count * n_sec

    sd_sq = (1.0 / n_total) * ((ranks - rank_bar) ** 2).sum().sum() / n_sec
    if sd_sq <= 0:
        return {"rank_statistic": 0.0, "p_value": 1.0}

    t_rank = L_bar / np.sqrt(sd_sq)
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(t_rank)))

    return {
        "rank_statistic": float(t_rank),
        "p_value": float(p_val),
    }


def generalized_sign_test(
    peer_cars: list[float],
    control_cars: list[float],
    p_0: float = 0.5,
) -> dict:
    n = len(peer_cars)
    if n < 2:
        return {"n_positive": 0, "n_total": n, "p_value": 1.0}

    differences = np.array(peer_cars) - np.array(control_cars)
    n_pos = int((differences > 0).sum())

    p_val = 2 * min(
        scipy_stats.binom.cdf(n_pos, n, p_0),
        1 - scipy_stats.binom.cdf(n_pos - 1, n, p_0),
    )

    return {
        "n_positive": n_pos,
        "n_total": n,
        "observed_ratio": round(n_pos / n, 4),
        "p_value": float(p_val),
    }


def boehmer_test(
    event_window_ar: pd.DataFrame,
    estimation_window_ar: pd.DataFrame,
    event_indices: list[int],
) -> dict:
    if event_window_ar.empty or estimation_window_ar.empty:
        return {"boehmer_statistic": 0.0, "p_value": 1.0}

    n_est = len(estimation_window_ar)
    n_sec = event_window_ar.shape[1]

    if n_sec < 2 or n_est < 10:
        return {"boehmer_statistic": 0.0, "p_value": 1.0}

    est_sigma = estimation_window_ar.std(ddof=1)
    if (est_sigma == 0).any():
        return {"boehmer_statistic": 0.0, "p_value": 1.0}

    sar_sum = 0.0
    count = 0
    for idx in event_indices:
        actual_idx = idx
        if 0 <= actual_idx < event_window_ar.shape[0]:
            ar = event_window_ar.iloc[actual_idx]
            sar = ar / est_sigma
            sar_sum += sar.sum()
            count += 1

    if count == 0:
        return {"boehmer_statistic": 0.0, "p_value": 1.0}

    sar_mean = sar_sum / (count * n_sec)
    cross_sectional_var = 0.0
    for idx in event_indices:
        actual_idx = idx
        if 0 <= actual_idx < event_window_ar.shape[0]:
            ar = event_window_ar.iloc[actual_idx]
            sar = ar / est_sigma
            cross_sectional_var += ((sar.mean() - sar_mean) ** 2).sum()

    cross_sectional_var /= (count * n_sec) * (count * n_sec - 1) if count * n_sec > 1 else 1

    if cross_sectional_var <= 0:
        return {"boehmer_statistic": 0.0, "p_value": 1.0}

    t_stat = sar_mean / np.sqrt(cross_sectional_var)
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(t_stat)))

    return {
        "boehmer_statistic": float(t_stat),
        "p_value": float(p_val),
        "n_events": count,
        "n_securities": n_sec,
    }


def patell_test(
    event_window_ar: pd.DataFrame,
    estimation_window_ar: pd.DataFrame,
    event_indices: list[int],
) -> dict:
    if event_window_ar.empty or estimation_window_ar.empty:
        return {"patell_statistic": 0.0, "p_value": 1.0}

    n_est = len(estimation_window_ar)
    n_sec = event_window_ar.shape[1]

    if n_sec < 2 or n_est < 10:
        return {"patell_statistic": 0.0, "p_value": 1.0}

    est_var = estimation_window_ar.var(ddof=1)
    if (est_var == 0).any():
        return {"patell_statistic": 0.0, "p_value": 1.0}

    total_standardized = 0.0
    for idx in event_indices:
        actual_idx = idx
        if 0 <= actual_idx < event_window_ar.shape[0]:
            ar = event_window_ar.iloc[actual_idx]
            standardized = ar / np.sqrt(est_var)
            total_standardized += standardized.sum()

    n_test_days = len(event_indices)
    denominator = np.sqrt(n_sec * n_test_days * (n_est - 2) / (n_est - 4)) if n_est > 4 else 1.0
    if denominator == 0:
        return {"patell_statistic": 0.0, "p_value": 1.0}

    z_stat = total_standardized / denominator
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

    return {
        "patell_statistic": float(z_stat),
        "p_value": float(p_val),
        "n_events": len(event_indices) * n_sec,
    }
