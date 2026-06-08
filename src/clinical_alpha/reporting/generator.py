"""Report and figure generation for clinical-alpha research.

Generates publication-quality tables, figures, and LaTeX output.
Includes performance attribution, factor decomposition, and professional reporting.
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clinical_alpha.config import Settings

settings = Settings()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11


def save_table(df: pd.DataFrame, filename: str, directory: Optional[Path] = None):
    out_dir = directory or settings.table_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / filename, index=False)


def save_table_latex(df: pd.DataFrame, filename: str, directory: Optional[Path] = None):
    out_dir = directory or settings.table_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    latex_str = df.to_latex(index=False, float_format="%.4f", na_rep="--")
    (out_dir / filename).write_text(latex_str)


def plot_ar_comparison(
    peer_ar_series: pd.Series,
    control_ar_series: pd.Series,
    title: str = "Abnormal Returns: Peers vs Controls",
    filename: str = "ar_comparison.png",
    directory: Optional[Path] = None,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
):
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(
        peer_ar_series.index,
        peer_ar_series.values,
        label="Peer Basket",
        linewidth=2,
        color="steelblue",
    )
    ax.plot(
        control_ar_series.index,
        control_ar_series.values,
        label="Control Basket",
        linewidth=2,
        alpha=0.7,
        color="coral",
    )

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            peer_ar_series.index,
            ci_lower.values,
            ci_upper.values,
            alpha=0.2,
            color="steelblue",
            label="95% CI (Peers)",
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="black", linestyle=":", alpha=0.5, label="Event Day")
    ax.set_xlabel("Relative Trading Day")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_car_distribution(
    results_summary: pd.DataFrame,
    title: str = "Cumulative Abnormal Return Distribution",
    filename: str = "car_distribution.png",
    directory: Optional[Path] = None,
):
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if "peer_mean_car" in results_summary.columns:
        peer_cars = results_summary["peer_mean_car"].dropna()
        axes[0].hist(peer_cars, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
        axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        axes[0].axvline(
            x=peer_cars.mean(),
            color="darkblue",
            linestyle=":",
            linewidth=1.5,
            label=f"Mean: {peer_cars.mean():.4f}",
        )
        axes[0].set_xlabel("CAR")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Peer Basket CARs")
        axes[0].legend()

    if "control_mean_car" in results_summary.columns:
        control_cars = results_summary["control_mean_car"].dropna()
        axes[1].hist(control_cars, bins=30, alpha=0.7, color="coral", edgecolor="white")
        axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        axes[1].axvline(
            x=control_cars.mean(),
            color="darkred",
            linestyle=":",
            linewidth=1.5,
            label=f"Mean: {control_cars.mean():.4f}",
        )
        axes[1].set_xlabel("CAR")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Control Basket CARs")
        axes[1].legend()

    if "spread" in results_summary.columns:
        spreads = results_summary["spread"].dropna()
        axes[2].hist(spreads, bins=30, alpha=0.7, color="purple", edgecolor="white")
        axes[2].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        axes[2].axvline(
            x=spreads.mean(),
            color="darkviolet",
            linestyle=":",
            linewidth=1.5,
            label=f"Mean: {spreads.mean():.4f}",
        )
        axes[2].set_xlabel("Peer - Control Spread")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Spread Distribution")
        axes[2].legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_event_study_results(
    results_summary: pd.DataFrame,
    filename: str = "event_study_results.png",
    directory: Optional[Path] = None,
):
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if results_summary.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].hist(
        results_summary["spread"].dropna(), bins=25, alpha=0.7, color="purple", edgecolor="white"
    )
    axes[0, 0].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_xlabel("Peer - Control CAR Spread")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Spread Distribution")

    if "event_type" in results_summary.columns:
        results_summary.boxplot(column="spread", by="event_type", ax=axes[0, 1])
        axes[0, 1].set_title("Spread by Event Type")
        axes[0, 1].set_xlabel("")
        axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    else:
        axes[0, 1].text(0.5, 0.5, "No event type data", ha="center", transform=axes[0, 1].transAxes)

    if "peer_mean_car" in results_summary.columns and "control_mean_car" in results_summary.columns:
        axes[0, 2].scatter(
            results_summary["peer_mean_car"],
            results_summary["control_mean_car"],
            alpha=0.6,
            c="steelblue",
            edgecolors="white",
        )
        min_val = min(
            results_summary["peer_mean_car"].min(), results_summary["control_mean_car"].min()
        )
        max_val = max(
            results_summary["peer_mean_car"].max(), results_summary["control_mean_car"].max()
        )
        axes[0, 2].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
        axes[0, 2].set_xlabel("Peer Basket CAR")
        axes[0, 2].set_ylabel("Control Basket CAR")
        axes[0, 2].set_title("Peer vs Control CAR")
    else:
        axes[0, 2].text(
            0.5, 0.5, "No peer/control CAR data", ha="center", transform=axes[0, 2].transAxes
        )

    if "p_value" in results_summary.columns:
        p_values = results_summary["p_value"].dropna()
        axes[1, 0].hist(p_values, bins=25, alpha=0.7, color="green", edgecolor="white")
        axes[1, 0].axvline(x=0.05, color="red", linestyle="--", linewidth=1.5, label="p=0.05")
        axes[1, 0].set_xlabel("P-value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title(
            f"P-value Distribution ({'%.1f' % (p_values < 0.05).mean() * 100}% significant)"
        )
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "No p-value data", ha="center", transform=axes[1, 0].transAxes)

    if "t_stat" in results_summary.columns:
        t_stats = results_summary["t_stat"].dropna()
        axes[1, 1].hist(t_stats, bins=25, alpha=0.7, color="orange", edgecolor="white")
        axes[1, 1].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        axes[1, 1].set_xlabel("T-statistic")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("T-statistic Distribution")
    else:
        axes[1, 1].text(0.5, 0.5, "No t-stat data", ha="center", transform=axes[1, 1].transAxes)

    if "event_date" in results_summary.columns and "spread" in results_summary.columns:
        dates = pd.to_datetime(results_summary["event_date"])
        spreads = results_summary["spread"]
        axes[1, 2].scatter(dates, spreads, alpha=0.6, c="purple", edgecolors="white")
        axes[1, 2].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[1, 2].set_xlabel("Event Date")
        axes[1, 2].set_ylabel("Spread")
        axes[1, 2].set_title("Spread Over Time")
        fig.autofmt_xdate()
    else:
        axes[1, 2].text(
            0.5, 0.5, "No time series data", ha="center", transform=axes[1, 2].transAxes
        )

    fig.suptitle("Event Study Results", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_checks(
    robustness_results: dict[str, pd.DataFrame],
    filename: str = "robustness_checks.png",
    directory: Optional[Path] = None,
):
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_checks = len(robustness_results)
    if n_checks == 0:
        return

    fig, axes = plt.subplots(n_checks, 1, figsize=(16, 4 * n_checks))
    if n_checks == 1:
        axes = [axes]

    for ax, (check_name, df) in zip(axes, robustness_results.items()):
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(check_name.replace("_", " ").title())
            continue

        param_cols = [
            c
            for c in df.columns
            if c
            not in [
                "n_events",
                "mean_spread",
                "median_spread",
                "positive_ratio",
                "significant_ratio",
                "median_t_stat",
                "median_p_value",
            ]
        ]
        param_col = param_cols[0] if param_cols else None

        if param_col and "mean_spread" in df.columns:
            ax.plot(
                df[param_col].astype(str),
                df["mean_spread"],
                marker="o",
                linewidth=2,
                color="steelblue",
                markersize=8,
            )
            if "median_spread" in df.columns:
                ax.plot(
                    df[param_col].astype(str),
                    df["median_spread"],
                    marker="s",
                    linewidth=1.5,
                    color="coral",
                    markersize=6,
                    linestyle="--",
                    alpha=0.7,
                )
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel(param_col.replace("_", " ").title())
            ax.set_ylabel("Spread")
            ax.set_title(f"{check_name.replace('_', ' ').title()} — Spread Sensitivity")
            ax.legend(["Mean Spread", "Median Spread"], loc="upper left")

        if "n_events" in df.columns:
            ax2 = ax.twinx()
            ax2.bar(df[param_col].astype(str), df["n_events"], alpha=0.2, color="gray", width=0.6)
            ax2.set_ylabel("N Events", color="gray")

        if "significant_ratio" in df.columns:
            for i, (_, row) in enumerate(df.iterrows()):
                ax.annotate(
                    f"sig={row['significant_ratio']:.0%}",
                    (df[param_col].astype(str).iloc[i], row["mean_spread"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

    fig.suptitle("Robustness Checks", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_backtest_results(
    backtest_results: dict,
    filename: str = "backtest_results.png",
    directory: Optional[Path] = None,
):
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if "results_by_period" not in backtest_results:
        return

    periods = backtest_results["results_by_period"]
    if not periods:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    hp_values = sorted(periods.keys())
    mean_returns = [periods[hp].get("mean_return", 0) for hp in hp_values]
    win_rates = [periods[hp].get("win_rate", 0) for hp in hp_values]
    sharpes = [periods[hp].get("sharpe", 0) for hp in hp_values]
    n_trades = [periods[hp].get("n_trades", 0) for hp in hp_values]
    compound_returns = [periods[hp].get("compound_return", 0) for hp in hp_values]
    dds = [periods[hp].get("max_drawdown", 0) for hp in hp_values]

    labels = [f"{hp}d" for hp in hp_values]

    axes[0, 0].bar(labels, mean_returns, color="steelblue", edgecolor="white")
    axes[0, 0].axhline(y=0, color="red", linestyle="--")
    axes[0, 0].set_title("Mean Return by Holding Period")
    axes[0, 0].set_ylabel("Mean Return")

    axes[0, 1].bar(labels, win_rates, color="green", alpha=0.7, edgecolor="white")
    axes[0, 1].axhline(y=0.5, color="red", linestyle="--", label="50%")
    axes[0, 1].set_title("Win Rate by Holding Period")
    axes[0, 1].set_ylabel("Win Rate")
    axes[0, 1].legend()

    axes[0, 2].bar(labels, sharpes, color="purple", alpha=0.7, edgecolor="white")
    axes[0, 2].axhline(y=0, color="red", linestyle="--")
    axes[0, 2].set_title("Sharpe Ratio by Holding Period")
    axes[0, 2].set_ylabel("Sharpe Ratio")

    axes[1, 0].bar(labels, compound_returns, color="darkblue", alpha=0.7, edgecolor="white")
    axes[1, 0].axhline(y=0, color="red", linestyle="--")
    axes[1, 0].set_title("Compound Return by Holding Period")
    axes[1, 0].set_ylabel("Compound Return")

    axes[1, 1].bar(labels, dds, color="red", alpha=0.6, edgecolor="white")
    axes[1, 1].set_title("Max Drawdown by Holding Period")
    axes[1, 1].set_ylabel("Max Drawdown")

    axes[1, 2].bar(labels, n_trades, color="orange", alpha=0.7, edgecolor="white")
    axes[1, 2].set_title("Number of Trades by Holding Period")
    axes[1, 2].set_ylabel("N Trades")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Backtest Results", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calendar_time_results(
    calendar_time_result: dict,
    filename: str = "calendar_time_results.png",
    directory: Optional[Path] = None,
):
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if "daily_ar_series" not in calendar_time_result:
        return

    daily_ar = calendar_time_result["daily_ar_series"]
    cumulative_ar = calendar_time_result.get("cumulative_ar_series", daily_ar.cumsum())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(daily_ar.index, daily_ar.values, linewidth=1, color="steelblue", alpha=0.7)
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Daily Abnormal Return")
    axes[0].set_title("Calendar-Time Portfolio: Daily Abnormal Returns")
    n_days = calendar_time_result.get("n_days", 0)
    mean_ar = calendar_time_result.get("mean_daily_ar", 0)
    t_stat = calendar_time_result.get("t_stat", 0)
    axes[0].text(
        0.02,
        0.95,
        f"Mean AR: {mean_ar:.6f}  |  t-stat: {t_stat:.3f}  |  N days: {n_days}",
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    axes[1].plot(cumulative_ar.index, cumulative_ar.values, linewidth=2, color="darkgreen")
    axes[1].axhline(y=0, color="red", linestyle="--")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Abnormal Return")
    axes[1].set_title("Calendar-Time Portfolio: Cumulative Abnormal Return")

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 252,
    filename: str = "rolling_beta.png",
    directory: Optional[Path] = None,
):
    """Plot rolling beta estimate with confidence bands."""
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < window:
        return

    betas = []
    beta_upper = []
    beta_lower = []
    dates = aligned.index[window:]

    for i in range(window, len(aligned)):
        y = aligned.iloc[i - window : i, 0].values
        x = aligned.iloc[i - window : i, 1].values
        X = np.column_stack([np.ones(window), x])
        try:
            coefs = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ coefs
            mse = np.sum(residuals**2) / (window - 2)
            se_beta = np.sqrt(mse / np.sum((x - x.mean()) ** 2))
            betas.append(coefs[1])
            beta_upper.append(coefs[1] + 1.96 * se_beta)
            beta_lower.append(coefs[1] - 1.96 * se_beta)
        except np.linalg.LinAlgError:
            betas.append(np.nan)
            beta_upper.append(np.nan)
            beta_lower.append(np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, betas, color="steelblue", linewidth=1.5, label="Rolling Beta")
    ax.fill_between(dates, beta_lower, beta_upper, alpha=0.2, color="steelblue", label="95% CI")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Beta=1")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Beta")
    ax.set_title(f"Rolling {window}-Day Beta Estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_brinson_attribution(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    sector_weights_portfolio: pd.Series,
    sector_weights_benchmark: pd.Series,
    sector_returns_portfolio: pd.DataFrame,
    sector_returns_benchmark: pd.DataFrame,
) -> dict:
    """Brinson (1986) performance attribution decomposition.

    Decomposes active return into:
    - Allocation effect: sector weight differences
    - Selection effect: within-sector stock selection
    - Interaction effect: combined weight × selection

    Parameters
    ----------
    portfolio_returns : pd.Series
    benchmark_returns : pd.Series
    sector_weights_portfolio : pd.Series
        Portfolio sector weights (index = sectors).
    sector_weights_benchmark : pd.Series
        Benchmark sector weights.
    sector_returns_portfolio : pd.DataFrame
        Portfolio returns by sector (columns = sectors).
    sector_returns_benchmark : pd.DataFrame
        Benchmark returns by sector.

    Returns
    -------
    dict with keys: allocation_effect, selection_effect, interaction_effect,
                    total_effect, active_return
    """
    common_sectors = sector_weights_portfolio.index.intersection(sector_weights_benchmark.index)
    if len(common_sectors) == 0:
        return {
            "allocation_effect": 0.0,
            "selection_effect": 0.0,
            "interaction_effect": 0.0,
            "total_effect": 0.0,
            "active_return": 0.0,
        }

    w_p = sector_weights_portfolio[common_sectors]
    w_b = sector_weights_benchmark[common_sectors]
    w_p = w_p / w_p.sum()
    w_b = w_b / w_b.sum()

    r_p = sector_returns_portfolio[common_sectors].mean()
    r_b = sector_returns_benchmark[common_sectors].mean()

    allocation = (w_p - w_b) * r_b
    selection = w_b * (r_p - r_b)
    interaction = (w_p - w_b) * (r_p - r_b)

    return {
        "allocation_effect": float(allocation.sum()),
        "selection_effect": float(selection.sum()),
        "interaction_effect": float(interaction.sum()),
        "total_effect": float(allocation.sum() + selection.sum() + interaction.sum()),
        "active_return": float(portfolio_returns.mean() - benchmark_returns.mean()),
        "sector_details": pd.DataFrame(
            {
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
            }
        ),
    }


def plot_brinson_attribution(
    attribution_result: dict,
    filename: str = "brinson_attribution.png",
    directory: Optional[Path] = None,
):
    """Plot Brinson attribution decomposition."""
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sector_df = attribution_result.get("sector_details", pd.DataFrame())
    if sector_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sector_df.plot(kind="bar", ax=axes[0])
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("Sector-Level Attribution")
    axes[0].set_ylabel("Contribution")
    axes[0].set_xlabel("Sector")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    effects = pd.DataFrame(
        {
            "effect": ["Allocation", "Selection", "Interaction", "Total"],
            "value": [
                attribution_result.get("allocation_effect", 0),
                attribution_result.get("selection_effect", 0),
                attribution_result.get("interaction_effect", 0),
                attribution_result.get("total_effect", 0),
            ],
        }
    )
    colors = ["steelblue", "green", "orange", "purple"]
    axes[1].bar(effects["effect"], effects["value"], color=colors, edgecolor="white")
    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Attribution Effect Summary")
    axes[1].set_ylabel("Contribution")

    fig.suptitle("Brinson Performance Attribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factor_exposure_decomposition(
    factor_loadings: pd.DataFrame,
    filename: str = "factor_exposure.png",
    directory: Optional[Path] = None,
):
    """Plot rolling factor exposure decomposition."""
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if factor_loadings.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    factor_loadings.plot(ax=ax, linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Factor Loading")
    ax.set_title("Rolling Factor Exposure Decomposition")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_heatmap(
    robustness_results: dict[str, pd.DataFrame],
    filename: str = "robustness_heatmap.png",
    directory: Optional[Path] = None,
):
    out_dir = directory or settings.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    for check_name, df in robustness_results.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            param_cols = [
                c
                for c in df.columns
                if c
                not in [
                    "n_events",
                    "mean_spread",
                    "median_spread",
                    "positive_ratio",
                    "significant_ratio",
                ]
            ]
            param_val = str(row[param_cols[0]]) if param_cols else ""
            summary_records.append(
                {
                    "check": check_name.replace("_", " ").title(),
                    "parameter": param_val,
                    "mean_spread": row.get("mean_spread", 0),
                    "significant_ratio": row.get("significant_ratio", 0),
                    "n_events": row.get("n_events", 0),
                }
            )

    if not summary_records:
        return

    summary_df = pd.DataFrame(summary_records)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(summary_records) * 0.4)))

    for ax, metric in zip(axes, ["mean_spread", "significant_ratio"]):
        pivot = summary_df.pivot_table(
            index="check", columns="parameter", values=metric, aggfunc="first"
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn" if metric == "mean_spread" else "Blues",
            center=0 if metric == "mean_spread" else None,
            ax=ax,
            cbar_kws={"label": metric.replace("_", " ").title()},
        )
        ax.set_title(f"Robustness: {metric.replace('_', ' ').title()}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Robustness Check Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_summary_table(
    event_study_summary: pd.DataFrame,
    backtest_results: dict,
    robustness_results: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    records = []

    study_agg = {}
    if not event_study_summary.empty:
        n = len(event_study_summary)
        study_agg = {
            "n_events": n,
            "mean_spread": round(float(event_study_summary["spread"].mean()), 6),
            "median_spread": round(float(event_study_summary["spread"].median()), 6),
            "positive_spread": int((event_study_summary["spread"] > 0).sum()),
            "significant_p05": int((event_study_summary["p_value"] < 0.05).sum())
            if "p_value" in event_study_summary.columns
            else 0,
            "significant_p01": int((event_study_summary["p_value"] < 0.01).sum())
            if "p_value" in event_study_summary.columns
            else 0,
            "mean_peer_car": round(float(event_study_summary["peer_mean_car"].mean()), 6),
            "mean_control_car": round(float(event_study_summary["control_mean_car"].mean()), 6),
            "positive_ratio": round((event_study_summary["spread"] > 0).mean(), 4),
        }

    if study_agg:
        for key, val in study_agg.items():
            records.append({"metric": f"event_study_{key}", "value": val})

    if "results_by_period" in backtest_results:
        for hp, metrics in backtest_results["results_by_period"].items():
            for key in [
                "mean_return",
                "compound_return",
                "win_rate",
                "sharpe",
                "sortino",
                "max_drawdown",
                "n_trades",
                "avg_tc_bps",
            ]:
                records.append({"metric": f"backtest_{hp}d_{key}", "value": metrics.get(key, 0)})

    if robustness_results:
        for check_name, df in robustness_results.items():
            if not df.empty and "mean_spread" in df.columns:
                spread_range = df["mean_spread"].max() - df["mean_spread"].min()
                records.append(
                    {
                        "metric": f"robustness_{check_name}_spread_range",
                        "value": round(float(spread_range), 6),
                    }
                )
            if not df.empty and "significant_ratio" in df.columns:
                sig_range = df["significant_ratio"].max() - df["significant_ratio"].min()
                records.append(
                    {
                        "metric": f"robustness_{check_name}_sig_range",
                        "value": round(float(sig_range), 4),
                    }
                )

    return pd.DataFrame(records)
