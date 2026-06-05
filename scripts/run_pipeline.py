"""Full pipeline runner for clinical-alpha research."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_alpha.backtest.engine import ReadthroughBacktest
from clinical_alpha.clinical_trials.fetcher import fetch_studies_for_companies
from clinical_alpha.config import Settings
from clinical_alpha.events.extractor import extract_all_events
from clinical_alpha.fda.fetcher import fetch_fda_approval_events
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.matching.normalizer import create_matched_sponsor_map
from clinical_alpha.prices.pipeline import fetch_price_data
from clinical_alpha.reporting.generator import (
    generate_summary_table,
    plot_backtest_results,
    plot_event_study_results,
    plot_robustness_checks,
    save_table,
)
from clinical_alpha.robustness.checks import RobustnessChecker
from clinical_alpha.sec.edgar import fetch_sec_healthcare_universe
from clinical_alpha.studies.event_study import EventStudy, aggregate_results
from clinical_alpha.universe.builder import save_universe

settings = Settings()


def run_pipeline():
    """Run the full clinical-alpha pipeline."""
    # Phase 2: Build universe
    print("[1/8] Building healthcare universe...")
    healthcare_universe = fetch_sec_healthcare_universe()
    save_universe(healthcare_universe, str(settings.PROCESSED_DIR / "healthcare_universe.parquet"))
    print(f"  -> {len(healthcare_universe)} healthcare companies")

    # Phase 3: Fetch clinical trials
    print("[2/8] Fetching clinical trials...")
    company_names = healthcare_universe["name"].dropna().unique().tolist()[:50]
    trials_df = fetch_studies_for_companies(company_names, max_per_company=100)
    trials_df.to_parquet(settings.PROCESSED_DIR / "clinical_trials.parquet", index=False)
    print(f"  -> {len(trials_df)} trials fetched")

    # Phase 4: Fetch FDA data
    print("[3/8] Fetching FDA approval data...")
    fda_events = fetch_fda_approval_events(max_records=settings.MAX_FDA_FETCH)
    if fda_events is not None and not fda_events.empty:
        fda_events.to_parquet(settings.PROCESSED_DIR / "fda_approvals.parquet", index=False)
        print(f"  -> {len(fda_events)} FDA approval records")
    else:
        fda_events = __import__("pandas").DataFrame()
        print("  -> No FDA data fetched (working with trials only)")
        __import__("pandas").DataFrame().to_parquet(
            settings.PROCESSED_DIR / "fda_approvals.parquet", index=False
        )

    # Phase 5: Normalize and match
    print("[4/8] Matching sponsors to companies...")
    sponsor_names = trials_df["sponsors"].dropna().unique().tolist()
    sponsor_names = [s.strip() for s in " ".join(sponsor_names).split(";") if s.strip()][:200]
    sponsor_map = create_matched_sponsor_map(
        sponsor_names,
        healthcare_universe,
        threshold=0.75,
        low_conf_export_path=str(settings.SAMPLE_DIR / "low_confidence_matches.csv"),
    )
    matched = {k: v for k, v in sponsor_map.items() if v is not None}
    print(f"  -> {len(matched)} / {len(sponsor_names)} sponsors matched")

    # Phase 6: Build graph
    print("[5/8] Building heterogeneous graph...")
    graph = ClinicalGraph()
    graph.build_from_dataframes(trials_df, sponsor_map, fda_events)
    print(f"  -> Graph: {graph.summary()}")

    # Phase 7: Extract events
    print("[6/8] Extracting high-confidence events...")
    events_df = extract_all_events(fda_events, trials_df, graph, min_confidence=0.7)
    events_df.to_parquet(settings.PROCESSED_DIR / "events.parquet", index=False)
    print(f"  -> {len(events_df)} events extracted")

    # Phase 8: Fetch prices
    print("[7/8] Fetching price data...")
    universe_tickers = healthcare_universe["ticker"].dropna().unique().tolist()
    min_date = events_df["event_date"].min() if not events_df.empty else "2020-01-01"
    max_date = events_df["event_date"].max() if not events_df.empty else "2024-12-31"
    prices, returns, benchmarks = fetch_price_data(
        universe_tickers,
        start_date=min_date,
        end_date=max_date,
    )
    print(f"  -> {len(prices.columns)} tickers with price data")

    # Phase 9-10: Event study
    print("[8/8] Running event study...")
    study = EventStudy(graph, (prices, returns, benchmarks), universe_tickers)
    study_results = study.run_all_events(events_df)
    study_summary = study.summarize_results(study_results)
    save_table(study_summary, "event_study_summary.csv")
    agg = aggregate_results(study_summary)
    print(f"  -> Study results: {agg}")

    # Phase 11: Backtest
    print("[9] Running backtest...")
    bt = ReadthroughBacktest(graph, prices, returns, benchmarks)
    bt_results = bt.run_backtest(events_df)
    bt_table = bt.summary_table(bt_results)
    save_table(bt_table, "backtest_summary.csv")
    print(f"  -> Backtest complete: {bt_results.get('n_positive_events', 0)} events traded")

    # Phase 12: Robustness checks
    print("[10] Running robustness checks...")
    rc = RobustnessChecker(graph, (prices, returns, benchmarks), universe_tickers)
    robustness_results = rc.run_all_checks(events_df)

    # Phase 13: Generate reports and figures
    print("[11] Generating reports and figures...")
    plot_event_study_results(study_summary)
    plot_backtest_results(bt_results)
    plot_robustness_checks(robustness_results)

    summary_table = generate_summary_table(study_summary, bt_results, robustness_results)
    save_table(summary_table, "full_summary.csv")

    print("\n=== Pipeline Complete ===")
    print(f"Event study: {agg}")
    print(f"Backtest holding periods: {list(bt_results.get('results_by_period', {}).keys())}")
    print(f"Reports saved to {settings.REPORT_DIR}")

    return {
        "universe_size": len(healthcare_universe),
        "n_trials": len(trials_df),
        "n_events": len(events_df),
        "event_study": agg,
        "backtest": bt_results,
        "robustness": {k: len(v) for k, v in robustness_results.items()},
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(json.dumps(results, indent=2, default=str))
