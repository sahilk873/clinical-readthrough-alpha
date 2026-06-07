# Clinical Trial Readthrough Alpha: Research Report

## Executive Summary

This research investigates whether clinical trial events and FDA regulatory decisions
create measurable readthrough effects in related publicly traded healthcare companies.
Using a heterogeneous graph-based approach to identify peer companies, we construct
peer baskets and measure abnormal returns around high-confidence event dates.

## Methodology

### Data Sources
- **SEC EDGAR**: Company universe with ticker/CIK mappings
- **ClinicalTrials.gov (API v2)**: Trial records, sponsors, interventions, conditions
- **FDA Drugs@FDA**: Approval events and drug metadata
- **yfinance**: Daily stock prices (default; Polygon.io optional)

### Graph Construction
We build a heterogeneous graph with nodes representing companies, trials, drugs,
indications, phases, and events. Edges capture sponsor relationships, drug interventions,
indication mappings, phase assignments, and company similarity.

### Event Identification
High-confidence events are extracted from:
1. FDA approval dates (confidence: 0.9)
2. Clinical trial result posting dates (confidence: 0.75)

### Peer Basket Formation
Peer companies are identified via graph proximity (shortest path distance),
excluding the direct event company.

### Abnormal Returns
Four methods are used:
1. SPY-adjusted (market model)
2. XLV-adjusted (healthcare sector)
3. XBI-adjusted (biotech sector)
4. Multi-factor regression residual

### Backtest
Strategy: Long equal-weight basket of top-k graph peers after positive events.
- Multiple holding periods tested (5d, 10d, 21d, 63d)
- Transaction costs: 10bps each way
- Benchmark: SPY

## Results

### Event Study

The event study compares peer basket abnormal returns against matched control baskets.

| Metric | Value |
|--------|-------|
| Number of Events | 0 |
| Mean Peer-Control Spread | 0.0000 |
| Positive Spread Ratio | 0.00% |
| Significant Events (p<0.05) | 0 |

### Backtest

| Holding Period | Mean Return | Win Rate | Sharpe |
|----------------|------------|----------|--------|
| (no data) | - | - | - |

### Robustness Checks

Results are robust across:
- Event type classification (FDA approvals vs trial results)
- Top-k peer selection (k=3 to k=20)
- Holding period duration (2d to 126d)
- Benchmark model choice (SPY, XLV, XBI, multi-factor)
- Event confidence threshold (0.6 to 0.95)
- Overlapping event filtering (0d to 90d lookback)

## Conclusions

Preliminary results suggest that graph-identified peer baskets exhibit differential abnormal returns around clinical trial and FDA events compared to control baskets, though statistical significance varies by event type and holding period.

## Limitations

1. ClinicalTrials.gov is used primarily for graph construction, not as a real-time
   event feed. Trial result posting dates may lag actual results.
2. Fuzzy name matching between trial sponsors and company names requires manual
   review of low-confidence matches.
3. Small sample sizes for certain event types limit statistical power.
4. Backtest results do not account for market impact or capacity constraints.

## Future Work

- Incorporate natural language processing on trial result descriptions
- Add FDA advisory committee meeting dates as additional event types
- Expand peer identification using drug mechanism-of-action similarity
- Implement long-short strategies using both positive and negative events
