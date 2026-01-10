# Clinical Trial Readthrough Alpha

**A quant research platform that tests whether clinical-trial and FDA events create readthrough effects in related public healthcare companies.**

## Overview

This repository implements a heterogeneous graph-based approach to measuring readthrough effects from clinical trial results and FDA approval events. The core hypothesis is that when a company announces clinical trial results or receives FDA approval/rejection, the market impact spills over to peer companies with similar drugs, indications, or trial phases.

## Architecture

```
src/clinical_alpha/
├── config.py              # Central configuration
├── universe/builder.py    # Company universe from SEC EDGAR
├── clinical_trials/       # ClinicalTrials.gov API v2 client
├── fda/                   # Drugs@FDA data fetcher
├── sec/                   # SEC EDGAR ticker/CIK mapping
├── prices/                # Price pipeline (yfinance + Polygon)
├── matching/              # Fuzzy name/drug normalizer
├── graph/                 # Heterogeneous graph builder
├── events/                # High-confidence event extraction
├── returns/               # Abnormal return calculation
├── studies/               # Event study engine
├── backtest/              # Backtest engine
├── robustness/            # Robustness checks
└── reporting/             # Report/table/figure generator
```

## Data Sources

| Source | API | Purpose |
|--------|-----|---------|
| ClinicalTrials.gov | REST API v2 | Trial records, sponsors, interventions, conditions |
| FDA (Drugs@FDA) | Downloadable files | Approval events, drug metadata |
| SEC EDGAR | APIs | Ticker/CIK/company name mapping |
| yfinance | Python lib | Daily stock prices (MVP) |
| Polygon.io | REST API (optional) | Higher-quality price data |

## Methodology

1. **Universe Construction**: Map public healthcare companies (SEC EDGAR → tickers/CIKs)
2. **Graph Construction**: Build heterogeneous graph with COMPANY, TRIAL, DRUG, INDICATION, PHASE, EVENT nodes and weighted edges
3. **Event Detection**: Extract FDA approval dates and trial result posting dates as high-confidence event anchors
4. **Peer Identification**: Form peer baskets from graph proximity, excluding the event company
5. **Abnormal Returns**: Compute SPY-adjusted, XLV-adjusted, XBI-adjusted, and regression-residual abnormal returns
6. **Event Study**: Compare graph-based peer baskets against matched/random control baskets
7. **Backtest**: Long top-k peers after positive events with transaction costs
8. **Robustness**: Multiple checks on event type, parameters, overlapping events

## Key Constraints

- Trial success/failure is **never** overclaimed — only sourced explicitly from source data
- ClinicalTrials.gov is primarily a **graph-construction source**, not a real-time event feed
- FDA approvals and trial result posting dates are **higher-confidence event anchors**
- All low-confidence fuzzy matches are **exported for review**, not silently accepted
- All results **compare peer baskets to control baskets**
- Final results include **transaction costs and robustness checks**

## Progress

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Repo setup |
| 2 | ✅ | Universe builder |
| 3 | ✅ | ClinicalTrials.gov fetcher |
| 4 | ✅ | FDA data fetcher |
| 5 | ✅ | Normalization/matching |
| 6 | ✅ | Graph builder |
| 7 | ✅ | Event extraction |
| 8 | ✅ | Price pipeline |
| 9 | ✅ | Abnormal return engine |
| 10 | ✅ | Event study |
| 11 | ✅ | Backtest |
| 12 | ✅ | Robustness checks |
| 13 | ✅ | Reports/README |
| 14 | ✅ | Tests |

## Quick Start

```bash
# Install
make install

# Run full pipeline
make run-pipeline

# Run tests
make test

# Generate sample output
make sample-output

# Generate research report
make generate-report
```

## Configuration

Copy `.env.example` to `.env` and configure:

- `POLYGON_API_KEY`: Optional Polygon.io API key
- `DATA_DIR`: Data storage location
- `MAX_TRIAL_FETCH`: Maximum trials to fetch (default 5000)
- `MAX_FDA_FETCH`: Maximum FDA records to fetch (default 1000)

## Outputs

- `reports/tables/`: Summary statistics and event study tables
- `reports/figures/`: Abnormal return plots and robustness charts
- `data/samples/`: Sample output files for each pipeline stage
- `research_report.md`: Full research writeup

## License

MIT
