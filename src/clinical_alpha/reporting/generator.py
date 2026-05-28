"""Report and figure generation for clinical-alpha research."""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from clinical_alpha.config import Settings

settings = Settings()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11


def save_table(df: pd.DataFrame, filename: str, directory: Optional[Path] = None):
    """Save a DataFrame as a CSV table in the reports/tables directory."""
    out_dir = directory or settings.TABLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)


def plot_ar_comparison(
    peer_ar_series: pd.Series,
    control_ar_series: pd.Series,
    title: str = "Abnormal Returns: Peers vs Controls",
    filename: str = "ar_comparison.png",
    directory: Optional[Path] = None,
):
    """Plot peer basket AR vs control basket AR."""
    out_dir = directory or settings.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(peer_ar_series.index, peer_ar_series.values, label="Peer Basket", linewidth=2)
    ax.plot(
        control_ar_series.index,
        control_ar_series.values,
        label="Control Basket",
        linewidth=2,
        alpha=0.7,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Relative Day")
    ax.set_ylabel("Abnormal Return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_car_distribution(
    results_summary: pd.DataFrame,
    title: str = "Cumulative Abnormal Return Distribution",
    filename: str = "car_distribution.png",
    directory: Optional[Path] = None,
):
    """Plot distribution of peer and control CARs."""
    out_dir = directory or settings.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "peer_mean_car" in results_summary.columns:
        axes[0].hist(
            results_summary["peer_mean_car"].dropna(), bins=30, alpha=0.7, color="steelblue"
        )
        axes[0].axvline(x=0, color="red", linestyle="--")
        axes[0].set_xlabel("CAR")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Peer Basket CARs")

    if "control_mean_car" in results_summary.columns:
        axes[1].hist(
            results_summary["control_mean_car"].dropna(), bins=30, alpha=0.7, color="coral"
        )
        axes[1].axvline(x=0, color="red", linestyle="--")
        axes[1].set_xlabel("CAR")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Control Basket CARs")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_event_study_results(
    results_summary: pd.DataFrame,
    filename: str = "event_study_results.png",
    directory: Optional[Path] = None,
):
    """Plot event study spread results."""
    out_dir = directory or settings.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if results_summary.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Spread histogram
    axes[0, 0].hist(results_summary["spread"].dropna(), bins=25, alpha=0.7, color="purple")
    axes[0, 0].axvline(x=0, color="red", linestyle="--")
    axes[0, 0].set_xlabel("Peer - Control CAR Spread")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Spread Distribution")

    # Spread by event type
    if "event_type" in results_summary.columns:
        results_summary.boxplot(column="spread", by="event_type", ax=axes[0, 1])
        axes[0, 1].set_title("Spread by Event Type")
        axes[0, 1].set_xlabel("")
    else:
        axes[0, 1].text(0.5, 0.5, "No event type data", ha="center", transform=axes[0, 1].transAxes)

    # Scatter: peer vs control
    if "peer_mean_car" in results_summary.columns and "control_mean_car" in results_summary.columns:
        axes[1, 0].scatter(
            results_summary["peer_mean_car"],
            results_summary["control_mean_car"],
            alpha=0.6,
        )
        min_val = min(
            results_summary["peer_mean_car"].min(), results_summary["control_mean_car"].min()
        )
        max_val = max(
            results_summary["peer_mean_car"].max(), results_summary["control_mean_car"].max()
        )
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
        axes[1, 0].set_xlabel("Peer Basket CAR")
        axes[1, 0].set_ylabel("Control Basket CAR")
        axes[1, 0].set_title("Peer vs Control CAR")
    else:
        axes[1, 0].text(
            0.5, 0.5, "No peer/control CAR data", ha="center", transform=axes[1, 0].transAxes
        )

    # Significance
    if "p_value" in results_summary.columns:
        axes[1, 1].hist(results_summary["p_value"].dropna(), bins=25, alpha=0.7, color="green")
        axes[1, 1].axvline(x=0.05, color="red", linestyle="--", label="p=0.05")
        axes[1, 1].set_xlabel("P-value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("P-value Distribution")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No p-value data", ha="center", transform=axes[1, 1].transAxes)

    fig.suptitle("Event Study Results")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_robustness_checks(
    robustness_results: dict[str, pd.DataFrame],
    filename: str = "robustness_checks.png",
    directory: Optional[Path] = None,
):
    """Plot robustness check results."""
    out_dir = directory or settings.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_checks = len(robustness_results)
    fig, axes = plt.subplots(n_checks, 1, figsize=(14, 4 * n_checks))
    if n_checks == 1:
        axes = [axes]

    for ax, (check_name, df) in zip(axes, robustness_results.items()):
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            ax.set_title(check_name)
            continue

        # Determine the varying parameter column
        param_cols = [
            c
            for c in df.columns
            if c not in ["n_events", "mean_spread", "positive_ratio", "significant_ratio"]
        ]
        param_col = param_cols[0] if param_cols else None

        if param_col and "mean_spread" in df.columns:
            ax.plot(df[param_col].astype(str), df["mean_spread"], marker="o", linewidth=2)
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel(param_col)
            ax.set_ylabel("Mean Spread")
            ax.set_title(f"{check_name} - Spread Sensitivity")

        # Add n_events as secondary
        if "n_events" in df.columns:
            ax2 = ax.twinx()
            ax2.bar(df[param_col].astype(str), df["n_events"], alpha=0.3, color="gray")
            ax2.set_ylabel("N Events", color="gray")

    fig.suptitle("Robustness Checks", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_backtest_results(
    backtest_results: dict,
    filename: str = "backtest_results.png",
    directory: Optional[Path] = None,
):
    """Plot backtest results by holding period."""
    out_dir = directory or settings.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if "results_by_period" not in backtest_results:
        return

    periods = backtest_results["results_by_period"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Prepare data
    hp_values = sorted(periods.keys())
    mean_returns = [periods[hp]["mean_return"] for hp in hp_values]
    win_rates = [periods[hp]["win_rate"] for hp in hp_values]
    sharpes = [periods[hp].get("sharpe", 0) for hp in hp_values]
    n_trades = [periods[hp]["n_trades"] for hp in hp_values]

    labels = [f"{hp}d" for hp in hp_values]

    axes[0, 0].bar(labels, mean_returns, color="steelblue")
    axes[0, 0].axhline(y=0, color="red", linestyle="--")
    axes[0, 0].set_title("Mean Return by Holding Period")
    axes[0, 0].set_ylabel("Mean Return")

    axes[0, 1].bar(labels, win_rates, color="green", alpha=0.7)
    axes[0, 1].axhline(y=0.5, color="red", linestyle="--", label="50%")
    axes[0, 1].set_title("Win Rate by Holding Period")
    axes[0, 1].set_ylabel("Win Rate")
    axes[0, 1].legend()

    axes[1, 0].bar(labels, sharpes, color="purple", alpha=0.7)
    axes[1, 0].axhline(y=0, color="red", linestyle="--")
    axes[1, 0].set_title("Sharpe Ratio by Holding Period")
    axes[1, 0].set_ylabel("Sharpe Ratio")

    axes[1, 1].bar(labels, n_trades, color="orange", alpha=0.7)
    axes[1, 1].set_title("Number of Trades by Holding Period")
    axes[1, 1].set_ylabel("N Trades")

    fig.suptitle("Backtest Results")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def generate_summary_table(
    event_study_summary: pd.DataFrame,
    backtest_results: dict,
    robustness_results: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Generate a comprehensive summary table of all results."""
    records = []

    # Event study summary
    study_agg = {}
    if not event_study_summary.empty:
        n = len(event_study_summary)
        study_agg = {
            "n_events": n,
            "mean_spread": round(float(event_study_summary["spread"].mean()), 6),
            "positive_spread": int((event_study_summary["spread"] > 0).sum()),
            "significant_events": int((event_study_summary["p_value"] < 0.05).sum())
            if "p_value" in event_study_summary.columns
            else 0,
            "mean_peer_car": round(float(event_study_summary["peer_mean_car"].mean()), 6),
            "mean_control_car": round(float(event_study_summary["control_mean_car"].mean()), 6),
        }

    if study_agg:
        for key, val in study_agg.items():
            records.append({"metric": f"event_study_{key}", "value": val})

    # Backtest summary
    if "results_by_period" in backtest_results:
        for hp, metrics in backtest_results["results_by_period"].items():
            for key in ["mean_return", "win_rate", "sharpe", "n_trades"]:
                records.append(
                    {
                        "metric": f"backtest_{hp}d_{key}",
                        "value": metrics.get(key, 0),
                    }
                )

    # Robustness summary
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

    return pd.DataFrame(records)
