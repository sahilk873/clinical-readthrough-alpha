"""High-confidence event extraction from FDA approvals and trial result postings.

Extracts FDA approval events and trial result posting dates as
higher-confidence event anchors for event studies.
"""


import pandas as pd

from clinical_alpha.config import Settings
from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.graph.models import EdgeType, NodeType

settings = Settings()


def extract_fda_approval_events(
    fda_events: pd.DataFrame,
    graph: ClinicalGraph,
    min_confidence: float = 0.7,
) -> pd.DataFrame:
    """Extract high-confidence FDA approval events linked to graph companies.

    Returns DataFrame with columns:
        event_id, company_ticker, event_type, event_date, drug_name, confidence, source
    """
    records = []

    for _, event in fda_events.iterrows():
        sponsor = str(event.get("sponsor", ""))
        drug_name = str(event.get("drug_name", ""))
        approval_date = event.get("approval_date")

        if not sponsor or pd.isna(approval_date):
            continue

        # Find matching company in graph
        companies = graph.get_nodes_by_type(NodeType.COMPANY)
        matched_company = None
        for co_id in companies:
            co_name = graph.graph.nodes[co_id].get("name", "")
            if sponsor.lower() in co_name.lower() or co_name.lower() in sponsor.lower():
                matched_company = co_id
                break

        if matched_company is None:
            continue

        ticker = graph.graph.nodes[matched_company].get("ticker", "")
        date_str = (
            str(approval_date.date()) if hasattr(approval_date, "date") else str(approval_date)
        )

        confidence = 0.9  # FDA approvals are high confidence
        if min_confidence and confidence < min_confidence:
            continue

        records.append(
            {
                "event_id": f"FDA_{drug_name}_{ticker}_{date_str}",
                "company_ticker": ticker,
                "company_node_id": matched_company,
                "event_type": "fda_approval",
                "event_date": date_str,
                "drug_name": drug_name,
                "confidence": confidence,
                "source": "FDA Drugs@FDA",
            }
        )

    return pd.DataFrame(records)


def extract_trial_result_events(
    trials_df: pd.DataFrame,
    graph: ClinicalGraph,
    min_confidence: float = 0.7,
) -> pd.DataFrame:
    """Extract trial result posting events from ClinicalTrials.gov.

    Uses result_first_post_date as the event anchor date.
    """
    records = []

    for _, trial in trials_df.iterrows():
        nct_id = trial.get("nct_id", "")
        result_date = trial.get("result_first_post_date")

        if not nct_id or not result_date or pd.isna(result_date):
            continue

        # Find the trial node in graph
        trial_id = f"TRIAL::{nct_id}"
        if not graph.graph.has_node(trial_id):
            continue

        # Find sponsor company for this trial
        for u, v, data in graph.graph.edges(data=True):
            if v == trial_id and data.get("edge_type") == EdgeType.SPONSOR:
                company_id = u
                ticker = graph.graph.nodes[company_id].get("ticker", "")
                date_str = (
                    str(result_date.date()) if hasattr(result_date, "date") else str(result_date)
                )

                # Determine confidence based on result posting presence
                confidence = 0.75  # Trial result posting is moderately high confidence

                if min_confidence and confidence < min_confidence:
                    continue

                records.append(
                    {
                        "event_id": f"TRIAL_RESULT_{nct_id}_{ticker}_{date_str}",
                        "company_ticker": ticker,
                        "company_node_id": company_id,
                        "event_type": "trial_result",
                        "event_date": date_str,
                        "drug_name": trial.get("intervention_name", ""),
                        "confidence": confidence,
                        "source": "ClinicalTrials.gov",
                    }
                )

    return pd.DataFrame(records)


def classify_event_direction(event: dict) -> str:
    """Classify the expected direction of an event.

    Returns 'positive', 'negative', or 'neutral'.
    This is a conservative classifier that only marks
    high-certainty positive/negative events.
    """
    event_type = event.get("event_type", "")

    if event_type == "fda_approval":
        return "positive"
    elif event_type == "fda_rejection":
        return "negative"
    elif event_type == "trial_result":
        return "positive"  # Default assumption; can be refined
    return "neutral"


def extract_all_events(
    fda_events: pd.DataFrame,
    trials_df: pd.DataFrame,
    graph: ClinicalGraph,
    min_confidence: float = 0.7,
) -> pd.DataFrame:
    """Extract all high-confidence events from all sources.

    Combined DataFrame with all events deduplicated.
    """
    fda_records = extract_fda_approval_events(fda_events, graph, min_confidence)
    trial_records = extract_trial_result_events(trials_df, graph, min_confidence)

    combined = pd.concat([fda_records, trial_records], ignore_index=True)
    combined["direction"] = combined.apply(classify_event_direction, axis=1)

    # Deduplicate
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["event_id"])
        combined = combined.sort_values("event_date").reset_index(drop=True)

    return combined


def filter_high_confidence_events(
    events: pd.DataFrame,
    min_confidence: float = 0.8,
) -> pd.DataFrame:
    """Filter events to only those above a confidence threshold."""
    return events[events["confidence"] >= min_confidence].copy()


def deduplicate_overlapping_events(
    events: pd.DataFrame,
    company_col: str = "company_ticker",
    date_col: str = "event_date",
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Remove events for the same company within a lookback window.

    Keeps the highest confidence event when overlap is detected.
    """
    if events.empty:
        return events

    df = events.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([company_col, date_col])

    to_keep = []
    for _, group in df.groupby(company_col):
        group = group.sort_values(date_col)
        keep_indices = []
        last_date = None
        for idx, row in group.iterrows():
            if last_date is None or (row[date_col] - last_date).days > lookback_days:
                keep_indices.append(idx)
                last_date = row[date_col]
            else:
                # Keep higher confidence
                current_best = group.loc[keep_indices[-1]]
                if row["confidence"] > current_best["confidence"]:
                    keep_indices[-1] = idx
                    last_date = row[date_col]
        to_keep.extend(keep_indices)

    return df.loc[to_keep].reset_index(drop=True)
