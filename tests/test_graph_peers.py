"""Tests for graph peer basket generation."""

import pandas as pd

from clinical_alpha.graph.builder import ClinicalGraph
from clinical_alpha.graph.models import EdgeType, NodeType, normalize_phase


class TestClinicalGraph:
    def test_add_company_node(self, sample_graph):
        sample_graph.add_company_node("TEST", name="Test Corp")
        node_id = "COMPANY::TEST"
        assert sample_graph.graph.has_node(node_id)
        assert sample_graph.graph.nodes[node_id]["ticker"] == "TEST"

    def test_add_trial_node(self, sample_graph):
        sample_graph.add_trial_node("NCT99999999", title="Test Trial")
        node_id = "TRIAL::NCT99999999"
        assert sample_graph.graph.has_node(node_id)
        assert sample_graph.graph.nodes[node_id]["nct_id"] == "NCT99999999"

    def test_add_drug_node(self, sample_graph):
        sample_graph.add_drug_node("Test Drug")
        node_id = "DRUG::test drug"
        assert sample_graph.graph.has_node(node_id)

    def test_get_nodes_by_type(self, sample_graph):
        companies = sample_graph.get_nodes_by_type(NodeType.COMPANY)
        assert len(companies) > 0
        trials = sample_graph.get_nodes_by_type(NodeType.TRIAL)
        assert len(trials) > 0

    def test_get_edges_by_type(self, sample_graph):
        sponsor_edges = sample_graph.get_edges_by_type(EdgeType.SPONSOR)
        assert len(sponsor_edges) > 0

    def test_graph_summary(self, sample_graph):
        summary = sample_graph.summary()
        assert summary["total_nodes"] > 0
        assert summary["total_edges"] > 0
        assert "COMPANY" in summary["node_counts"]
        assert "TRIAL" in summary["node_counts"]


class TestPeerBasket:
    def test_get_peer_companies(self, sample_graph):
        peers = sample_graph.get_peer_companies("PFE")
        assert isinstance(peers, list)
        if peers:
            assert len(peers[0]) == 2  # (ticker, score)

    def test_get_peer_basket(self, sample_graph):
        basket = sample_graph.get_peer_basket("PFE", top_k=3)
        assert isinstance(basket, list)
        assert all(isinstance(t, str) for t in basket)

    def test_empty_for_unknown_company(self, sample_graph):
        basket = sample_graph.get_peer_basket("UNKNOWN")
        assert basket == []

    def test_excludes_self(self, sample_graph):
        basket = sample_graph.get_peer_basket("PFE", top_k=10)
        assert "PFE" not in basket

    def test_peer_distance_ranking(self, sample_graph):
        peers = sample_graph.get_peer_companies("PFE", max_distance=2)
        if len(peers) >= 2:
            # Higher score = closer peer
            assert peers[0][1] >= peers[1][1]


class TestGraphBuildFromDataframes:
    def test_build_adds_nodes(self):
        graph = ClinicalGraph()
        trials = pd.DataFrame(
            {
                "nct_id": ["NCT00000001"],
                "brief_title": ["Test Trial"],
                "phase": ["Phase 3"],
                "overall_status": ["COMPLETED"],
                "sponsors": [""],
                "intervention_name": ["Drug X"],
                "conditions": ["Cancer"],
            }
        )
        graph.build_from_dataframes(trials, {})
        assert len(graph.graph.nodes) >= 4  # TRIAL, DRUG, INDICATION, PHASE

    def test_build_with_sponsor_mapping(self):
        graph = ClinicalGraph()
        trials = pd.DataFrame(
            {
                "nct_id": ["NCT00000001"],
                "brief_title": ["Test Trial"],
                "phase": ["Phase 3"],
                "overall_status": ["COMPLETED"],
                "sponsors": ["Pfizer Inc"],
                "intervention_name": ["Drug X"],
                "conditions": ["Cancer"],
            }
        )
        sponsor_map = {"Pfizer Inc": {"ticker": "PFE", "company_name": "Pfizer Inc"}}
        graph.build_from_dataframes(trials, sponsor_map)
        companies = graph.get_nodes_by_type(NodeType.COMPANY)
        assert len(companies) >= 1


class TestNormalizePhase:
    def test_normalize_phase_3(self):
        assert normalize_phase("Phase 3") == "Phase 3"
        assert normalize_phase("phase 3") == "Phase 3"

    def test_normalize_phase_1_2(self):
        assert normalize_phase("Phase 1/Phase 2") == "Phase 1/2"

    def test_normalize_unknown(self):
        assert normalize_phase("N/A") == "Unknown"
        assert normalize_phase("") == "Unknown"
