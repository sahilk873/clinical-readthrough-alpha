"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_alpha.graph.builder import ClinicalGraph


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = ClinicalGraph()

    companies = {
        "PFE": "Pfizer Inc",
        "MRK": "Merck & Co Inc",
        "JNJ": "Johnson & Johnson",
        "ABBV": "AbbVie Inc",
        "LLY": "Eli Lilly and Co",
    }
    for ticker, name in companies.items():
        graph.add_company_node(ticker, name=name)

    for i in range(5):
        nct_id = f"NCT{i + 1:08d}"
        graph.add_trial_node(nct_id, title=f"Trial {i + 1}", phase="Phase 3", status="COMPLETED")

    for drug in ["Drug X", "Drug Y", "Therapy Z"]:
        graph.add_drug_node(drug)

    for cond in ["Cancer", "Diabetes", "Autoimmune"]:
        graph.add_indication_node(cond)

    for phase in ["Phase 3", "Phase 2", "Phase 1"]:
        graph.add_phase_node(phase)

    co_ids = [f"COMPANY::{t}" for t in companies]
    trial_ids = [f"TRIAL::NCT{i + 1:08d}" for i in range(5)]
    drug_ids = ["DRUG::drug x", "DRUG::drug y", "DRUG::therapy z"]
    ind_ids = ["INDICATION::cancer", "INDICATION::diabetes", "INDICATION::autoimmune"]
    phase_ids = ["PHASE::Phase 3", "PHASE::Phase 2", "PHASE::Phase 1"]

    for i in range(3):
        graph.add_sponsor_edge(co_ids[i], trial_ids[i])
    for i in range(3):
        graph.add_intervention_edge(trial_ids[i], drug_ids[i])
    for i in range(3):
        graph.add_indication_edge(trial_ids[i], ind_ids[i])
    for i in range(3):
        graph.add_phase_edge(trial_ids[i], phase_ids[0])

    similarity_df = pd.DataFrame(
        {
            "ticker_a": ["PFE", "PFE", "MRK", "MRK", "JNJ", "JNJ"],
            "ticker_b": ["MRK", "JNJ", "PFE", "JNJ", "PFE", "MRK"],
            "similarity": [0.6, 0.4, 0.6, 0.5, 0.4, 0.5],
        }
    )
    graph.add_company_similarity_edges(similarity_df, threshold=0.2)

    return graph


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
    np.random.seed(42)
    n = len(dates)

    prices = pd.DataFrame(
        {
            "PFE": 50 * np.exp(np.random.randn(n).cumsum() * 0.02),
            "MRK": 80 * np.exp(np.random.randn(n).cumsum() * 0.02),
            "JNJ": 150 * np.exp(np.random.randn(n).cumsum() * 0.015),
            "ABBV": 100 * np.exp(np.random.randn(n).cumsum() * 0.018),
            "LLY": 200 * np.exp(np.random.randn(n).cumsum() * 0.022),
            "SPY": 400 * np.exp(np.random.randn(n).cumsum() * 0.01),
            "XLV": 120 * np.exp(np.random.randn(n).cumsum() * 0.012),
            "XBI": 80 * np.exp(np.random.randn(n).cumsum() * 0.018),
        },
        index=dates,
    )

    return prices


@pytest.fixture
def sample_returns(sample_prices):
    """Create sample returns from prices."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_company_df():
    """Create sample company DataFrame for matching tests."""
    return pd.DataFrame(
        {
            "ticker": ["PFE", "MRK", "JNJ", "ABBV", "LLY"],
            "name": [
                "Pfizer Inc",
                "Merck & Co Inc",
                "Johnson & Johnson",
                "AbbVie Inc",
                "Eli Lilly and Co",
            ],
            "cik": ["0001", "0002", "0003", "0004", "0005"],
            "exchange": ["NYSE"] * 5,
        }
    )
