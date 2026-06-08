# Clinical Trial Readthrough Alpha: Research Report

## Executive Summary

This research investigates whether clinical trial events and FDA regulatory decisions create measurable **readthrough effects** in related publicly traded healthcare companies. Using a heterogeneous graph-based approach to identify peer companies based on shared drugs, indications, and trial phases, we construct peer baskets and measure abnormal returns around high-confidence event dates with rigorous statistical methodology — including bootstrap confidence intervals, permutation tests, Fama-French factor models, Bayesian hypothesis testing, structural break detection, and multiple-hypothesis corrections.

**Key question**: When a company announces a clinical trial result or FDA decision, does the market revalue peer companies with related exposures?

## Methodology

### Data Sources
- **SEC EDGAR**: Company universe with ticker/CIK mappings (~500+ healthcare companies)
- **ClinicalTrials.gov (API v2)**: Trial records with sponsors, interventions, conditions, phases, result posting dates
- **FDA Drugs@FDA**: Approval events and drug metadata
- **yfinance**: Daily stock prices (default; Polygon.io optional)
- **Fama-French factors**: Via Ken French Data Library

### Graph Construction
We build a **heterogeneous graph** with 6 node types and 7 edge types:

| Node Type | Description |
|-----------|-------------|
| COMPANY | Public healthcare issuers |
| TRIAL | Clinical trial records (NCT IDs) |
| DRUG | Intervention/drug names |
| INDICATION | Medical conditions studied |
| PHASE | Trial phases (Phase 1–4) |
| EVENT | FDA approvals and trial result postings |

Edges capture sponsor relationships, drug interventions, indication mappings, phase assignments, and company similarity (computed from shared graph proximity).

**Peer identification**: For each event company, we compute shortest weighted path distances to all other company nodes. Shorter distance = stronger peer relationship. Peers are ranked by proximity score and the top-k form the peer basket.

### Event Identification

| Source | Confidence | Description |
|--------|-----------|-------------|
| FDA approval dates | 0.90 | High certainty — regulatory action with clear market impact |
| Trial result postings | 0.75 | Moderate — posting dates may lag actual results |

Events are classified as **positive** (FDA approvals, trial results) or **negative** (FDA rejections).

### Abnormal Return Models

We compute abnormal returns using multiple methods to ensure robustness:

1. **Market-adjusted**: AR = R_i - R_m (simple, transparent)
2. **Market model**: AR = R_i - (α + β × R_m) from OLS rolling estimation
3. **CAPM**: Rolling beta from market model regression
4. **Multi-factor regression residual**: Residual from SPY + XLV + XBI regression
5. **Fama-French 3-factor**: α from Mkt-RF + SMB + HML regression
6. **Fama-French 5-factor**: α from Mkt-RF + SMB + HML + RMW + CMA regression
7. **Carhart 4-factor**: α from Mkt-RF + SMB + HML + MOM regression
8. **Fama-MacBeth (1973)**: Two-pass cross-sectional regression with Shanken standard error correction

### Factor Model Selection
We provide AIC, BIC, AICc, and adjusted R² for all factor models, plus a `compare_factor_models()` function that ranks models by selection criteria. The GRS (Gibbons-Ross-Shanken) test evaluates whether alphas are jointly zero across test assets.

### Statistical Tests

To avoid overclaiming significance, we implement a battery of tests:

| Test | Type | What It Tests |
|------|------|---------------|
| Two-sample t-test | Parametric | Difference in mean CAR (peer vs control) |
| Mann-Whitney U | Non-parametric | Stochastic dominance of one group |
| Wilcoxon signed-rank | Non-parametric | Paired differences |
| Permutation test | Distribution-free | Group label randomization |
| Bootstrap CI | Resampling | Percentile confidence interval for mean CAR |
| Corrado rank test | Non-parametric | Rank-based event window significance |
| Boehmer et al. (1991) | Parametric | Cross-sectional correlation adjustment |
| Patell (1976) | Parametric | Standardized abnormal return test |
| Generalized sign test | Non-parametric | Proportion of positive spreads |
| Multiple testing correction | Adjustment | Bonferroni, Holm, Benjamini-Hochberg, FDR bootstrap |
| JZS Bayes Factor | Bayesian | Evidence for/against readthrough effect |
| Clustered bootstrap | Resampling | Within-company correlation robust CI |
| Chow structural break | Parametric | Known-date break in CAR series |
| Bai-Perron (2003) | Econometric | Multiple unknown structural breaks |
| GRS test | Multivariate | Joint alpha significance across assets |

### Portfolio Construction

The backtest simulates a long strategy: buy top-k graph peers after positive events.

| Weighting Scheme | Description |
|-----------------|-------------|
| Equal weight | 1/n allocation |
| Volatility targeting | Position size = target_vol / asset_vol (cap 30%) |
| Risk parity | Equal contribution to portfolio risk |
| Minimum variance | Global minimum variance portfolio |
| Hierarchical Risk Parity (2016) | Lopez de Prado tree-based clustering and recursive bisection |
| Black-Litterman (1992) | Combine market equilibrium with investor views |
| Bayesian Mean-Variance | Shrinkage-aware posterior optimization |
| TC-Aware Optimization | L1 turnover penalty minimizing net return impact |

**Transaction costs**: Linear (10bps base + 5bps slippage), quadratic, or volume-based (% of ADV).

### Signal Analysis

We evaluate the predictive power of event signals:

| Metric | Description |
|--------|-------------|
| Information Coefficient (IC) | Pearson/Spearman correlation between signal and forward return |
| IC Time Series | Rolling IC to assess signal stability |
| Signal Decay Profile | IC across increasing forward horizons (1d to 63d) |
| Signal-to-Noise Ratio | Mean IC / std IC |
| Quantile Spread | Return spread between top and bottom signal quintiles |
| Contribution Decomposition | Decompose signal into constituent factor contributions |

### Risk Model

| Method | Description |
|--------|-------------|
| Shrinkage Covariance | Ledoit-Wolf / constant correlation shrinkage |
| PCA Factor Model | Principal components as statistical risk factors |
| Regime-Switching Covariance | KMeans-clustered volatility regimes; regime-weighted covariance blending |
| Copula Dependence | Gaussian, Clayton, and Frank copulas with tail dependence coefficients |
| Liquidity-Adjusted VaR (La-VaR) | Bangia et al. (1999) adjustment: VaR + 0.5 × (bid-ask spread) |

### Calendar-Time Portfolio Approach

To address cross-sectional correlation in event-time analysis, we also implement the **calendar-time portfolio** method (Fama 1998). On each calendar day, we form a portfolio of all securities that experienced an event within the event window, and track the average abnormal return forward. This produces a single time-series of portfolio returns, avoiding the assumption of independence across events.

### Performance Attribution

| Method | Description |
|--------|-------------|
| Brinson Attribution (1986) | Decompose excess return into allocation effect, selection effect, and interaction effect |
| Rolling Beta | 60-day rolling market exposure visualization |
| Factor Exposure Decomposition | Regression-based decomposition into factor contributions |

## Results

### Event Study

| Metric | Value |
|--------|-------|
| Number of Events | 0* |
| Mean Peer-Control Spread | — |
| Positive Spread Ratio | — |
| Significant Events (p<0.05) | — |
| Bootstrap CI (95%) | — |
| Bayes Factor | — |

*Results will populate after running `make run-pipeline` with real API data.

### Backtest

| Holding Period | Mean Return | Win Rate | Sharpe | Sortino | Max DD |
|----------------|------------|----------|--------|---------|--------|
| — | — | — | — | — | — |

### Calendar-Time Portfolio

| Metric | Value |
|--------|-------|
| Mean Daily AR | — |
| Cumulative AR | — |
| T-statistic | — |

### Robustness Checks

Results are tested across:

1. **Event type** (FDA approvals vs trial results)
2. **Top-k selection** (k=3 to k=20)
3. **Holding period** (2d to 126d)
4. **Benchmark model** (SPY, XLV, XBI, market model, CAPM)
5. **Confidence threshold** (0.6 to 0.95)
6. **Overlap filtering** (0d to 90d lookback)
7. **Time period subsamples** (by year)
8. **Event direction** (positive vs negative)
9. **Estimation window** (63d to 504d)

## Interpretation Guide

### If Spread > 0 and p < 0.05
The peer basket significantly outperformed the control basket around events. This is consistent with a **positive readthrough effect**: the market revalues peer companies when their related peers experience events.

### If Spread ≈ 0
No detectable readthrough effect. Possible explanations:
- The graph-based peer identification does not capture economically meaningful relationships
- The market efficiently prices in expected outcomes before the event date
- Information leakage or anticipation eliminates the event-date impact

### If Bootstrap CI excludes 0
Stronger evidence — the confidence interval suggests the effect is not driven by outliers or small sample issues.

### If Bayes Factor > 3
Positive Bayesian evidence for the readthrough effect. BF > 10 indicates strong evidence.

## Limitations (Honest)

1. **Survivorship bias**: Our universe contains only companies that survived to appear in SEC EDGAR and have current price data. Delisted, acquired, or bankrupt peers are excluded.
2. **Look-ahead bias**: Company names used for fuzzy matching reflect current names. We mitigate by matching at the CIK level where possible.
3. **Small sample sizes**: FDA rejections and certain trial phases have limited observations, reducing statistical power.
4. **No NLP on trial results**: We do not parse the actual outcome (positive/negative) of trial result postings — all trial results are classified as "positive" by default.
5. **Confounding events**: Multiple events near the same date can contaminate results. We handle this via overlap filtering (default 90-day lookback).
6. **Market impact**: Backtests assume our trading does not move prices. Reasonable for small sizes, unrealistic at institutional scale.
7. **Data source latency**: ClinicalTrials.gov posts results when submitted by sponsors, not when results are actually announced. This can introduce date misalignment.
8. **No shorting capacity**: The primary strategy is long-only. Shorting FDA rejections around negative events could increase alpha capture.
9. **Hypothesis fishing**: Multiple testing across event windows, peer definitions, and model specifications requires correction (we apply Benjamini-Hochberg and FDR bootstrap where appropriate).
10. **Regime-switching simplification**: KMeans-based volatility clustering is a first-order approximation; HMM or Markov-switching models would capture richer dynamics.
11. **Black-Litterman view subjectivity**: Event-based views are heuristic; systematic view construction from fundamentals or factor models would be more rigorous.

## Future Work

- **NLP on trial result descriptions**: Parse primary endpoint outcomes to classify trial results as genuinely positive/negative
- **FDA advisory committee dates**: Additional high-confidence event source
- **Drug mechanism-of-action similarity**: Add MOA edges to the graph for finer-grained peer identification
- **Long-short strategies**: Short negative-event peer baskets
- **Cross-sectional asset pricing**: Fama-MacBeth regressions of event exposures
- **Real-time implementation**: Stream events as they occur, with daily portfolio rebalancing
- **Transaction cost optimization**: Implement implementation shortfall and VWAP benchmarks
- **Machine learning peer scores**: Learn edge weights from historical readthrough co-movement
- **Bayesian hierarchical event study**: Pool information across event types with partial pooling
- **Regime-aware backtesting**: Condition portfolio construction on volatility regime
- **ML-based signal combination**: Combine multiple signals using ensemble methods

## Appendix: Graph Construction Detail

The heterogeneous graph is implemented using NetworkX's MultiDiGraph. Node IDs follow the pattern `{NODETYPE}::{key}` (e.g., `COMPANY::PFE`, `DRUG::drug x`). Edge weights for SPONSOR, INTERVENTION, INDICATION, and PHASE edges are all set to 1.0. COMPANY_SIMILARITY edges can be added from an external similarity matrix with configurable threshold (default 0.3).

Peer distance is computed as:
```
score = 1 / (1 + shortest_path_length)
```
where shortest_path_length is the weighted graph distance between two COMPANY nodes. Companies are ranked by score and the top-k form the peer basket (excluding the event company itself).

## Appendix: New Methodologies

### Fama-MacBeth Two-Pass Regression
First-pass: time-series regressions of each asset on factors to obtain betas. Second-pass: cross-sectional regression of average returns on betas to obtain factor risk prices. Standard errors are corrected using the Shanken (1992) adjustment for generated regressors. Supports rolling window estimation for time-varying betas.

### Black-Litterman Model
Combines market equilibrium returns (from reverse-optimization of a benchmark portfolio) with investor views (from event signals). The posterior return distribution blends these two sources weighted by the confidence parameter τ (tau). The resulting expected returns feed into a mean-variance optimizer.

### Hierarchical Risk Parity
Uses single-linkage hierarchical clustering of the correlation matrix to build a tree structure, then allocates capital via recursive bisection (top-down inverse-variance weighting within each cluster). This avoids the instabilities of full-covariance inversion.

### Regime-Switching Covariance
Clusters historical returns into K volatility regimes using KMeans. Blends the current-regime covariance matrix with the long-run covariance matrix to produce a forward-looking estimate that adapts to the current volatility environment.

### Bai-Perron Structural Break Test
Sequentially tests for multiple unknown breakpoints in a linear time series using the sup-F statistic. The number of breaks is selected via BIC or sequential hypothesis testing. Useful for detecting changes in the readthrough effect over time.
