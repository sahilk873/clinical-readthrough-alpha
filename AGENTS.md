# AGENTS.md - Clinical Alpha development guide for AI agents

## Commands
- `make install` - Install with dev dependencies
- `make test` - Run pytest with coverage
- `make lint` - Run ruff linter
- `make typecheck` - Run mypy type checker
- `make run-pipeline` - Execute full pipeline
- `make fetch-all` - Fetch all raw data sources
- `make generate-report` - Generate research report
- `make sample-output` - Generate sample output files
- `make quality` - Run lint + typecheck + tests

## Code conventions
- Type hints required for all function signatures
- Use `pydantic` where validation is needed, dataclasses for simple data
- NetworkX for graph operations
- pandas/NumPy for data manipulation
- httpx for HTTP requests
- All API calls have try/except with graceful fallbacks
- No pipeline logic in notebooks
- Functions are single-purpose and composable
- Use `Settings` from `config.py` for all configuration

## Architecture
- `src/clinical_alpha/` - Core library
  - `config.py` - Central Settings class (reads .env)
  - `universe/builder.py` - Company universe construction
  - `sec/edgar.py` - SEC EDGAR API client
  - `clinical_trials/fetcher.py` - ClinicalTrials.gov API v2
  - `fda/fetcher.py` - FDA Drugs@FDA data
  - `matching/normalizer.py` - Fuzzy name matching
  - `graph/builder.py` - Heterogeneous graph
  - `graph/models.py` - Node/Edge types
  - `events/extractor.py` - Event extraction
  - `prices/pipeline.py` - Price data pipeline
  - `returns/abnormal.py` - Abnormal return calc
  - `studies/event_study.py` - Event study engine
  - `backtest/engine.py` - Backtest engine
  - `robustness/checks.py` - Robustness checks
  - `reporting/generator.py` - Reports/figures
- `scripts/` - Pipeline orchestration scripts
- `tests/` - pytest tests
- `reports/` - Generated output (tables + figures)

## Key constraints
- Do not overclaim trial success/failure
- ClinicalTrials.gov is graph source, not event feed
- FDA approvals = higher confidence event anchors
- Export low-confidence fuzzy matches for review
- Always compare peer baskets to control baskets
- Include transaction costs and robustness checks
