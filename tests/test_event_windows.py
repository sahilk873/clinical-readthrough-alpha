"""Tests for event window extraction and processing."""

import pandas as pd

from clinical_alpha.events.extractor import (
    classify_event_direction,
    deduplicate_overlapping_events,
    extract_all_events,
    filter_high_confidence_events,
)


class TestClassifyEventDirection:
    def test_fda_approval_is_positive(self):
        assert classify_event_direction({"event_type": "fda_approval"}) == "positive"

    def test_fda_rejection_is_negative(self):
        assert classify_event_direction({"event_type": "fda_rejection"}) == "negative"

    def test_trial_result_is_positive(self):
        assert classify_event_direction({"event_type": "trial_result"}) == "positive"

    def test_unknown_is_neutral(self):
        assert classify_event_direction({"event_type": "unknown"}) == "neutral"


class TestFilterHighConfidence:
    def test_filters_below_threshold(self):
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2", "E3"],
                "confidence": [0.6, 0.8, 0.95],
            }
        )
        filtered = filter_high_confidence_events(events, min_confidence=0.8)
        assert len(filtered) == 2
        assert all(filtered["confidence"] >= 0.8)

    def test_all_kept_when_threshold_low(self):
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "confidence": [0.5, 0.6],
            }
        )
        filtered = filter_high_confidence_events(events, min_confidence=0.4)
        assert len(filtered) == 2

    def test_empty_when_threshold_high(self):
        events = pd.DataFrame(
            {
                "event_id": ["E1"],
                "confidence": [0.5],
            }
        )
        filtered = filter_high_confidence_events(events, min_confidence=0.9)
        assert filtered.empty


class TestDeduplicateOverlapping:
    def test_keeps_separate_events(self):
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "company_ticker": ["PFE", "MRK"],
                "event_date": ["2023-01-01", "2023-06-01"],
                "confidence": [0.8, 0.8],
            }
        )
        deduped = deduplicate_overlapping_events(events, lookback_days=90)
        assert len(deduped) == 2

    def test_deduplicates_close_events_same_company(self):
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "company_ticker": ["PFE", "PFE"],
                "event_date": ["2023-01-01", "2023-02-01"],
                "confidence": [0.8, 0.9],
            }
        )
        deduped = deduplicate_overlapping_events(events, lookback_days=90)
        # E2 has higher confidence, so E1 should be dropped
        assert len(deduped) == 1
        assert deduped.iloc[0]["event_id"] == "E2"

    def test_keeps_distant_events(self):
        events = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "company_ticker": ["PFE", "PFE"],
                "event_date": ["2023-01-01", "2023-06-01"],
                "confidence": [0.8, 0.8],
            }
        )
        deduped = deduplicate_overlapping_events(events, lookback_days=30)
        assert len(deduped) == 2


class TestExtractAllEvents:
    def test_requires_graph_with_companies(self):
        from clinical_alpha.graph.builder import ClinicalGraph

        graph = ClinicalGraph()
        graph.add_company_node("PFE", name="Pfizer Inc")

        fda = pd.DataFrame(
            {
                "sponsor": ["Pfizer Inc"],
                "drug_name": ["Drug X"],
                "approval_date": pd.to_datetime(["2023-06-01"]),
            }
        )
        trials = pd.DataFrame()

        events = extract_all_events(fda, trials, graph, min_confidence=0.7)
        assert len(events) >= 1
        assert events.iloc[0]["event_type"] == "fda_approval"

    def test_returns_empty_for_no_data(self):
        from clinical_alpha.graph.builder import ClinicalGraph

        graph = ClinicalGraph()
        graph.add_company_node("PFE", name="Pfizer Inc")

        empty_fda = pd.DataFrame(columns=["sponsor", "drug_name", "approval_date"])
        empty_trials = pd.DataFrame(columns=["nct_id", "result_first_post_date"])

        events = extract_all_events(empty_fda, empty_trials, graph, min_confidence=0.7)
        assert events.empty
