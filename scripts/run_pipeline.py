"""Full pipeline runner for clinical-alpha research.

Executes the complete research pipeline with caching, logging,
and phase-based execution.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_alpha.backtest.engine import ReadthroughBacktest
from clinical_alpha.clinical_trials.fetcher import fetch_studies_for_companies
from clinical_alpha.config import Settings
from clinical_alpha.data.quality import generate_quality_report
from clinical_alpha.events.extractor import extract_all_events
from clinical_alpha.fda.fetcher import fetch_fda_approval_events
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.logging import logger, setup_logging
from clinical_alpha.matching.normalizer import create_matched_sponsor_map
from clinical_alpha.prices.pipeline import fetch_price_data
from clinical_alpha.reporting.generator import (
    generate_summary_table,
    plot_backtest_results,
    plot_calendar_time_results,
    plot_event_study_results,
    plot_robustness_checks,
    plot_robustness_heatmap,
    save_table,
    save_table_latex,
)
from clinical_alpha.returns.factor_models import compare_factor_models
from clinical_alpha.robustness.checks import RobustnessChecker
from clinical_alpha.sec.edgar import fetch_sec_healthcare_universe
from clinical_alpha.signal.analysis import (
    compute_information_coefficient,
    compute_signal_decay,
    evaluate_signal_cross_sectional,
)
from clinical_alpha.studies.event_study import EventStudy, aggregate_results
from clinical_alpha.universe.builder import save_universe

settings = Settings()
setup_logging(level=settings.log_level, log_file=settings.report_dir / "pipeline.log")


def phase(step: int, total: int, name: str):
    logger.info("\u2500" * 60)
    logger.info(f"[{step}/{total}] {name}")
    logger.info("\u2500" * 60)


def run_pipeline():
    """Run the full clinical-alpha research pipeline."""

    # Phase 1: Build universe
    phase(1, 13, "Building Healthcare Universe")
    healthcare_universe = fetch_sec_healthcare_universe()
    save_universe(healthcare_universe, str(settings.processed_dir / "healthcare_universe.parquet"))
    logger.info(f"Universe: {len(healthcare_universe)} healthcare companies")

    # Phase 2: Fetch clinical trials
    phase(2, 13, "Fetching Clinical Trials")
    company_names = healthcare_universe["name"].dropna().unique().tolist()[:50]
    trials_df = fetch_studies_for_companies(company_names, max_per_company=100)
    trials_df.to_parquet(settings.processed_dir / "clinical_trials.parquet", index=False)
    logger.info(f"Trials: {len(trials_df)} trials fetched")

    # Phase 3: Fetch FDA data
    phase(3, 13, "Fetching FDA Approval Data")
    fda_events = fetch_fda_approval_events(max_records=settings.max_fda_fetch)
    if fda_events is not None and not fda_events.empty:
        fda_events.to_parquet(settings.processed_dir / "fda_approvals.parquet", index=False)
        logger.info(f"FDA records: {len(fda_events)}")
    else:
        fda_events = __import__("pandas").DataFrame()
        fda_events.to_parquet(settings.processed_dir / "fda_approvals.parquet", index=False)
        logger.info("No FDA data fetched")

    # Phase 4: Normalize and match sponsors
    phase(4, 13, "Matching Sponsors to Companies")
    sponsor_names = trials_df["sponsors"].dropna().unique().tolist()
    sponsor_names = [s.strip() for s in " ".join(sponsor_names).split(";") if s.strip()][:200]
    sponsor_map = create_matched_sponsor_map(
        sponsor_names,
        healthcare_universe,
        threshold=0.75,
        low_conf_export_path=str(settings.sample_dir / "low_confidence_matches.csv"),
    )
    matched = {k: v for k, v in sponsor_map.items() if v is not None}
    logger.info(f"Matched: {len(matched)} / {len(sponsor_names)} sponsors")

    # Phase 5: Build graph
    phase(5, 13, "Building Heterogeneous Graph")
    graph = ClinicalGraph()
    graph.build_from_dataframes(trials_df, sponsor_map, fda_events)
    gsum = graph.summary()
    logger.info(f"Graph: {gsum['total_nodes']} nodes, {gsum['total_edges']} edges")

    # Phase 6: Extract events
    phase(6, 13, "Extracting High-Confidence Events")
    events_df = extract_all_events(fda_events, trials_df, graph, min_confidence=0.7)
    events_df.to_parquet(settings.processed_dir / "events.parquet", index=False)
    logger.info(f"Events: {len(events_df)} extracted")

    if events_df.empty:
        logger.warning("No events extracted \u2014 cannot proceed with event study.")
        return

    # Phase 7: Fetch price data
    phase(7, 13, "Fetching Price Data")
    universe_tickers = healthcare_universe["ticker"].dropna().unique().tolist()
    min_date = str(pd.to_datetime(events_df["event_date"]).min() - pd.Timedelta(days=365))
    max_date = str(pd.to_datetime(events_df["event_date"]).max() + pd.Timedelta(days=365))
    prices, returns, benchmarks = fetch_price_data(
        universe_tickers,
        start_date=min_date,
        end_date=max_date,
    )

    # Data quality filtering
    logger.info(f"Raw tickers: {len(prices.columns)}")
    quality_report = generate_quality_report(prices)
    passed_tickers = quality_report[quality_report["passed"]]["ticker"].tolist()
    returns = returns[[t for t in passed_tickers if t in returns.columns]]
    prices = prices[[t for t in passed_tickers if t in prices.columns]]
    logger.info(f"After quality filter: {len(prices.columns)} tickers")

    # Phase 8: Event study
    phase(8, 13, "Running Event Study")
    all_tickers = list(returns.columns)
    study = EventStudy(graph, (prices, returns, benchmarks), all_tickers)

    # Main event study
    study_results = study.run_all_events(events_df)
    study_summary = study.summarize_results(study_results)
    save_table(study_summary, "event_study_summary.csv")
    save_table_latex(study_summary.head(20), "event_study_summary.tex")
    agg = aggregate_results(study_summary)
    logger.info(f"Event Study: {agg}")

    # Calendar-time portfolio
    ct_results = study.run_calendar_time_portfolio(events_df)
    logger.info(
        f"Calendar-Time AR: mean={ct_results.get('mean_daily_ar')}, "
        f"t-stat={ct_results.get('t_stat')}"
    )

    # Phase 9: Factor model comparison
    phase(9, 13, "Running Factor Model Comparison")
    try:
        fm_comparison = compare_factor_models(returns, benchmarks)
        if fm_comparison is not None and not fm_comparison.empty:
            save_table(fm_comparison, "factor_model_comparison.csv")
            logger.info(f"Factor model comparison: {len(fm_comparison)} models")
    except Exception as e:
        logger.warning(f"Factor model comparison skipped: {e}")

    # Phase 10: Signal analysis
    phase(10, 13, "Running Signal Analysis")
    try:
        if not returns.empty and len(events_df) > 0:
            signal_values = pd.Series(
                events_df["confidence"].values,
                index=pd.to_datetime(events_df["event_date"]),
            )
            ic, ic_pvalue = compute_information_coefficient(
                signal_values,
                returns.reindex(signal_values.index).iloc[:, 0]
                if len(returns.columns) > 0
                else pd.Series(dtype=float),
                method="spearman",
            )
            logger.info(f"Signal IC: {ic:.4f} (p={ic_pvalue:.4f})")

            mean_fwd = (
                returns.reindex(signal_values.index).mean(axis=1)
                if len(returns.columns) > 0
                else pd.Series(dtype=float)
            )
            decay_df = pd.DataFrame(
                {"fwd_1d": mean_fwd, "fwd_5d": mean_fwd, "fwd_21d": mean_fwd}, index=mean_fwd.index
            )
            decay = compute_signal_decay(signal_values, decay_df, lags=[1, 5, 21])
            if decay is not None and not decay.empty:
                save_table(decay.to_frame(name="ic"), "signal_decay.csv")

            signal_df = (
                pd.DataFrame(
                    {t: signal_values.values for t in returns.columns[:3]},
                    index=signal_values.index,
                )
                if len(returns.columns) >= 3
                else pd.DataFrame()
            )
            ret_subset = returns.reindex(signal_values.index)
            if not signal_df.empty and not ret_subset.empty:
                cross_result = evaluate_signal_cross_sectional(
                    signal_df,
                    ret_subset[signal_df.columns],
                    n_quantiles=settings.signal_n_quantiles,
                )
                if cross_result:
                    pd.DataFrame([cross_result]).to_csv(
                        settings.report_dir / "tables/signal_quantiles.csv", index=False
                    )
    except Exception as e:
        logger.warning(f"Signal analysis skipped: {e}")

    # Phase 11: Backtest
    phase(11, 13, "Running Backtest")
    bt = ReadthroughBacktest(
        graph,
        prices,
        returns,
        benchmarks,
        transaction_cost_model=settings.backtest_transaction_cost_model,
        base_tc_bps=settings.backtest_base_tc_bps,
        slippage_bps=settings.backtest_slippage_bps,
        weighting=settings.backtest_weighting,
        vol_target=settings.backtest_vol_target,
        long_short=settings.backtest_long_short,
    )
    bt_results = bt.run_backtest(events_df)
    bt_table = bt.summary_table(bt_results)
    save_table(bt_table, "backtest_summary.csv")
    logger.info(f"Backtest: {bt_results.get('n_positive_events', 0)} events traded")

    # Phase 12: Robustness checks
    phase(12, 13, "Running Robustness Checks")
    rc = RobustnessChecker(graph, (prices, returns, benchmarks), all_tickers)
    robustness_results = rc.run_all_checks(events_df)

    for check_name, df in robustness_results.items():
        if not df.empty:
            save_table(df, f"robustness_{check_name}.csv")

    # Phase 13: Generate reports and figures
    phase(13, 13, "Generating Reports and Figures")

    plot_event_study_results(study_summary)
    plot_backtest_results(bt_results)
    plot_robustness_checks(robustness_results)
    plot_robustness_heatmap(robustness_results)

    if ct_results.get("daily_ar_series") is not None:
        plot_calendar_time_results(ct_results)

    summary_table = generate_summary_table(study_summary, bt_results, robustness_results)
    save_table(summary_table, "full_summary.csv")
    save_table_latex(summary_table, "full_summary.tex")

    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Event study: {agg}")
    logger.info(f"Calendar-time AR: {ct_results.get('mean_daily_ar')}")
    logger.info(f"Reports saved to {settings.report_dir}")

    return {
        "universe_size": len(healthcare_universe),
        "n_trials": len(trials_df),
        "n_events": len(events_df),
        "event_study": agg,
        "calendar_time_ar": ct_results.get("mean_daily_ar"),
        "calendar_time_tstat": ct_results.get("t_stat"),
        "backtest": {
            str(k): v.get("sharpe", 0) for k, v in bt_results.get("results_by_period", {}).items()
        },
        "robustness": {k: len(v) for k, v in robustness_results.items()},
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(json.dumps(results, indent=2, default=str))
