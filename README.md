# Clinical Trial Readthrough Alpha

**A research framework that tests whether clinical-trial and FDA events create readthrough effects in related public healthcare companies.**

Built with institutional-grade methodology — this is not a toy project.

---

## The Hypothesis

> When a company announces clinical trial results or receives an FDA decision, does the market impact **spill over** to peer companies with similar drugs, indications, or trial phases?

This is the **readthrough effect** — a well-known phenomenon in healthcare investing where information about one company is used to revalue related companies. This project provides a rigorous, reproducible framework to test and quantify this effect.

## What Makes This Different

| Dimension | Typical Quant Project | This Project |
|-----------|----------------------|--------------|
| **Data** | Clean CSV, one source | Messy real-world APIs, multi-source reconciliation |
| **Methodology** | Single t-test | Bootstrap, permutation, Fama-French, GRS, multiple-testing correction, Bayesian tests, structural breaks |
| **Peer Selection** | Fixed industry labels | Heterogeneous graph with shortest-path proximity |
| **Controls** | Random selection | Matched on market cap, propensity score, graph distance |
| **Factor Models** | CAPM only | CAPM, FF3, FF5, Carhart 4F, Fama-MacBeth two-pass |
| **Portfolio Construction** | Equal weight, no costs | Vol-targeting, risk-parity, min-variance, HRP, Black-Litterman, Bayesian MV, transaction-cost-aware |
| **Risk Model** | Simple cov | Shrinkage, PCA factor model, regime-switching covariance, copula dependence, liquidity-adjusted VaR |
| **Backtest** | Equal weight, no costs | Vol-targeting, risk-parity, volume-based TC, sector neutral |
| **Signal Analysis** | None | IC, rank IC, signal decay, quantile breakdown, contribution decomposition |
| **Performance Attribution** | None | Brinson attribution, factor exposure decomposition, rolling beta |
| **Robustness** | 1-2 checks | 8+ checks: subsampling, jackknife, cross-validation, seasonality, structural breaks |
| **Engineering** | Notebook | Package with CI, type hints, tests, logging, pydantic validation |

---

## Architecture

```
src/clinical_alpha/
├── config.py              # Pydantic-validated settings (hierarchical)
├── exceptions.py          # Custom exception hierarchy
├── logging.py             # Structured logging
├── data/
│   └── quality.py         # Data quality checks, survivorship bias detection
├── universe/builder.py    # Company universe from SEC EDGAR
├── clinical_trials/       # ClinicalTrials.gov API v2 client
├── fda/                   # Drugs@FDA data fetcher
├── sec/                   # SEC EDGAR ticker/CIK mapping
├── prices/                # Price pipeline (yfinance + Polygon)
├── matching/              # Fuzzy name/drug normalizer
├── signal/                # Information coefficient, decay, quantile analysis
├── graph/                 # Heterogeneous graph builder
├── events/                # High-confidence event extraction
├── returns/
│   ├── abnormal.py        # AR, CAR, BHAR, multi-window, calendar-time
│   ├── factor_models.py   # CAPM, FF3, FF5, Carhart 4F, Fama-MacBeth, Newey-West, GRS, model selection
│   └── statistical_tests.py # Bootstrap, permutation, Corrado, Boehmer, Patell, Bayes factors, FDR bootstrap, Bai-Perron
├── studies/
│   └── event_study.py     # Event study with calendar-time portfolio
├── risk/
│   └── model.py           # Shrinkage cov, PCA factors, regime-switching cov, copula dependence, La-VaR, risk parity, min variance
├── backtest/
│   └── engine.py          # Multi-weighting, HRP, Black-Litterman, Bayesian MV, TC-aware, volume TC, vol-targeting, long-short
├── robustness/
│   └── checks.py          # 8+ sensitivity checks, subsampling
└── reporting/
    └── generator.py       # Publication-quality tables/figures, Brinson attribution, factor decomposition
```

## Data Sources

| Source | API | Purpose |
|--------|-----|---------|
| ClinicalTrials.gov | REST API v2 | Trial records, sponsors, interventions, conditions |
| FDA (Drugs@FDA) | Downloadable files | Approval events, drug metadata |
| SEC EDGAR | APIs | Ticker/CIK/company name mapping |
| yfinance | Python lib | Daily stock prices (MVP) |
| Polygon.io | REST API (optional) | Higher-quality price data |
| Ken French Data Library | (auto) | Fama-French factors (via pandas-datareader) |

---

## Methodology

### 1. Universe Construction
Build a comprehensive set of publicly traded healthcare companies from SEC EDGAR, mapping tickers to CIKs and company names.

### 2. Heterogeneous Graph
Build a graph with **6 node types** and **7 edge types**:
- `COMPANY` — `SPONSOR` → `TRIAL`
- `TRIAL` — `INTERVENTION` → `DRUG`
- `TRIAL` — `INDICATION` → `CONDITION`
- `TRIAL` — `PHASE` → `PHASE`
- `COMPANY` — `COMPANY_SIMILARITY` → `COMPANY`

Peer companies are identified via **shortest weighted path distance** through drugs, indications, and trial phases.

### 3. Event Detection
Two event sources, each with explicit confidence scores:
- **FDA approvals** (confidence = 0.9) — high certainty
- **Trial result postings** (confidence = 0.75) — moderate certainty

### 4. Abnormal Return Models
| Model | Description |
|-------|-------------|
| Market Adjusted | R_i - R_m (simple, transparent) |
| Market Model | OLS rolling alpha + beta × R_m |
| CAPM | Rolling beta estimation |
| Multi-Factor Regression | Residual from SPY + XLV + XBI regression |
| Fama-French 3-Factor | α from Mkt + SMB + HML regression |
| Fama-French 5-Factor | α from Mkt + SMB + HML + RMW + CMA regression |
| Carhart 4-Factor | α from Mkt + SMB + HML + MOM regression |
| Fama-MacBeth (1973) | Two-pass cross-sectional regression with Shanken SE correction |

### 5. Statistical Tests
| Test | Purpose |
|------|---------|
| **Standard t-test** | Parametric difference in means |
| **Mann-Whitney U** | Non-parametric group comparison |
| **Wilcoxon Signed-Rank** | Non-parametric paired test |
| **Permutation Test** | Distribution-free significance |
| **Bootstrap CI** | Percentile confidence intervals |
| **Corrado Rank Test** | Robust to event-induced variance |
| **Boehmer et al.** | Cross-sectional correlation adjustment |
| **Patell Test** | Standardized abnormal return test |
| **Generalized Sign Test** | Proportion of positive spreads |
| **Multiple Testing Correction** | Bonferroni, Holm, Benjamini-Hochberg, FDR bootstrap |
| **Bayes Factor (JZS)** | Bayesian hypothesis testing for CAR |
| **Clustered Bootstrap** | Handles within-company correlation |
| **Chow Test** | Structural break at known date |
| **Bai-Perron Test** | Multiple unknown structural break detection |
| **GRS Test** | Gibbons-Ross-Shanken test for joint alpha significance |

### 6. Portfolio Construction
| Scheme | Description |
|--------|-------------|
| Equal Weight | Baseline, transparent |
| Volatility Targeting | Position size = target_vol / asset_vol |
| Risk Parity | Equal risk contribution |
| Minimum Variance | Global minimum variance portfolio |
| Hierarchical Risk Parity (2016) | Lopez de Prado tree-based risk parity |
| Black-Litterman (1992) | Bayesian views + market equilibrium |
| Bayesian Mean-Variance | Shrinkage-aware posterior optimization |
| TC-Aware Optimization | L1 turnover penalty for transaction cost minimization |

### 7. Signal Analysis
| Metric | Description |
|--------|-------------|
| Information Coefficient | Pearson/spearman rank correlation between signal and forward return |
| IC Time Series | Track IC stability over rolling windows |
| Signal Decay | How IC decays across increasing forward horizons |
| Signal-to-Noise Ratio | Mean(IC) / std(IC) |
| Quantile Evaluation | Spread between top and bottom quantile portfolios |
| Contribution Decomposition | Decompose signal into factor contributions |

### 8. Risk Model
| Method | Description |
|--------|-------------|
| Shrinkage Covariance | Ledoit-Wolf / constant correlation shrinkage |
| PCA Factor Model | Principal components as statistical risk factors |
| Regime-Switching Covariance | KMeans-clustered volatility regimes with regime-weighted cov |
| Copula Dependence | Gaussian, Clayton, Frank copulas with tail dependence |
| Liquidity-Adjusted VaR | Bangia et al. (1999) approach adding liquidity cost to VaR |

### 9. Performance Attribution
| Method | Description |
|--------|-------------|
| Brinson Attribution (1986) | Asset allocation + stock selection effects |
| Rolling Beta | Time-varying market exposure |
| Factor Exposure Decomposition | Decompose returns into factor contributions |

### 10. Transaction Costs
| Model | Formula |
|-------|---------|
| Linear | base_tc + slippage (fixed bps) |
| Quadratic | base_tc + 10 × (trade_size_pct)² × 10000 |
| Volume-Based | base_tc + slippage + 30 × (% ADV)^0.5 × 10000 |

---

## Quick Start

```bash
# Install with development dependencies
make install

# Run the full pipeline
make run-pipeline

# Run tests (excluding slow robustness checks)
make test

# Run all tests including slow ones
make test-full

# Lint and type check
make lint
make typecheck

# Generate sample output
make sample-output

# Generate research report
make generate-report

# Run quality gate (lint + typecheck + test)
make quality
```

## Configuration

Copy `.env.example` to `.env`. Key settings:

```bash
# API Keys
POLYGON_API_KEY=

# Signal Analysis
SIGNAL_MIN_OBS=10
SIGNAL_N_QUANTILES=5
SIGNAL_MAX_LAG=63
SIGNAL_METHOD=pearson

# Black-Litterman
BLACK_LITTERMAN_TAU=0.05
BLACK_LITTERMAN_RISK_AVERSION=2.5

# Bayesian Priors
BAYES_PRIOR_SCALE=0.5
```

All settings are validated via pydantic in `config.py`. Override any setting with an environment variable.

---

## Results & Interpretation

The output lives in `reports/`:

```
reports/
├── tables/
│   ├── event_study_summary.csv    # Per-event CARs and statistics
│   ├── backtest_summary.csv       # Backtest by holding period
│   ├── robustness_*.csv           # Sensitivity analysis results
│   ├── factor_model_comparison.csv # Model selection metrics
│   ├── signal_analysis.csv        # IC, decay, quantile metrics
│   └── full_summary.csv           # Master summary table
└── figures/
    ├── ar_comparison.png          # Peer vs Control AR
    ├── car_distribution.png       # CAR histograms
    ├── event_study_results.png    # Comprehensive study plot
    ├── backtest_results.png       # Backtest performance
    ├── robustness_checks.png      # Sensitivity grid
    ├── robustness_heatmap.png     # Summary heatmap
    ├── calendar_time_results.png  # Calendar-time portfolio
    ├── rolling_beta.png           # Time-varying market exposure
    ├── brinson_attribution.png    # Performance attribution
    ├── factor_exposure.png        # Factor decomposition
    └── signal_decay.png           # IC decay profile
```

### Interpreting the Event Study

- **Spread**: `mean(peer_CAR) - mean(control_CAR)`. Positive = peer basket outperformed control around events.
- **P-value**: From a two-sample t-test. Low p-values (< 0.05) suggest the spread is not due to random chance.
- **Permutation p-value**: More robust alternative — shuffles group labels to build a null distribution.
- **Bootstrap CI**: If the 95% CI excludes zero, we have evidence of a genuine readthrough effect.
- **Bayes Factor**: BF > 3 provides positive evidence, BF > 10 provides strong evidence.

### Interpreting the Backtest

- **Sharpe ratio**: Risk-adjusted return. Above 0.5 is notable for a long-only strategy.
- **Sortino ratio**: Downside-risk-adjusted return.
- **Win rate**: Fraction of trades with positive excess return.
- **Max drawdown**: Peak-to-trough decline during the backtest.
- **Weighting schemes**: Compare equal, risk-parity, HRP, Black-Litterman, and TC-aware results.

---

## Key Highlights

> **Built a research platform analyzing clinical-trial readthrough effects across 500+ healthcare equities using a heterogeneous graph-based peer identification methodology.**
>
> **Designed and implemented the full research pipeline: multi-source data ingestion (ClinicalTrials.gov, FDA, SEC EDGAR), fuzzy entity resolution, NetworkX graph construction, event study engine with 15+ statistical tests (bootstrap, permutation, Fama-French, GRS, Boehmer, Patell, Corrado rank, Bayes factors, Bai-Perron), factor models (CAPM, FF3, FF5, Carhart 4F, Fama-MacBeth), and portfolio construction with vol-targeting, risk-parity, Hierarchical Risk Parity, Black-Litterman, and volume-based transaction costs.**
>
> **Implemented advanced risk modeling including regime-switching covariance, copula dependence, liquidity-adjusted VaR, and shrinkage covariance estimation — with signal analysis (IC decay, quantile spreads) and Brinson performance attribution.**
>
> **Achieved comprehensive robustness validation across 8+ sensitivity dimensions including subsampling, multiple-testing corrections (Bonferroni, Holm, BH, FDR bootstrap), structural break detection, and cross-validation — ensuring honest reporting of statistical significance.**
>
> **Engineered the system with pydantic-validated configuration, custom exception hierarchy, structured logging, data quality checks, and 190+ tests — demonstrating production-grade Python engineering.**

---

## Limitations (Honest)

1. **Survivorship bias**: We only see companies that survived to appear in the data. Delisted, acquired, or bankrupt peers are missing.
2. **Look-ahead bias**: Fuzzy matching uses current company names. We mitigate with explicit event-date-based filtering.
3. **Small sample sizes**: Certain event types (FDA rejections, specific phases) have limited observations.
4. **No NLP**: We don't parse trial result descriptions — we use post dates as proxies.
5. **Confounding events**: Multiple events near the same date can contaminate results (handled via overlap filtering).
6. **Market impact**: Backtest assumes prices are unaffected by our trades (reasonable for small sizes, not for institutional scale).
7. **Data source limitations**: ClinicalTrials.gov is not designed as a real-time event feed; posting dates can lag by weeks.
8. **Regime-switching simplicity**: Volatility clustering via KMeans is a simplification — more sophisticated HMM or Markov-switching models could improve regime detection.
9. **Copula calibration**: Limited sample sizes can make tail-dependence estimates unreliable.
10. **Black-Litterman views**: Prior views are heuristic; systematic view construction from fundamentals would be more rigorous.

---

## Project Status

| Component | Status |
|-----------|--------|
| Universe builder | ✅ |
| ClinicalTrials.gov fetcher | ✅ |
| FDA data fetcher | ✅ |
| Fuzzy name matching | ✅ |
| Graph construction | ✅ |
| Event extraction | ✅ |
| Price pipeline | ✅ |
| **Abnormal returns (AR/CAR/BHAR)** | ✅ |
| **Factor models (CAPM/FF3/FF5/Carhart4F/Fama-MacBeth)** | ✅ |
| **Statistical tests (15+ tests)** | ✅ |
| **Bayesian hypothesis testing** | ✅ |
| **Structural break detection (Chow/Bai-Perron)** | ✅ |
| **Event study (calendar-time)** | ✅ |
| **Backtest (multi-weighting: equal/vol-parity/risk-parity/min-var/HRP/BL/Bayesian/TC-aware)** | ✅ |
| **Risk model (shrinkage/PCA/regime-switching/copula/La-VaR)** | ✅ |
| **Signal analysis (IC/decay/quantiles)** | ✅ |
| **Performance attribution (Brinson)** | ✅ |
| **Data quality checks** | ✅ |
| **Robustness (8+ dimensions)** | ✅ |
| **Reporting & visualization** | ✅ |
| **Tests (190+ unit + integration)** | ✅ |
| **CI/CD (GitHub Actions)** | ✅ |

---

## License

MIT
