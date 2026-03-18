"""Heterogeneous graph builder.

Constructs a graph with COMPANY, TRIAL, DRUG, INDICATION, PHASE, and EVENT nodes.
Adds weighted edges for sponsor, intervention, indication, phase, event,
and company similarity.
"""

from typing import Optional

import networkx as nx
import pandas as pd

from clinical_alpha.config import Settings
from clinical_alpha.graph.models import (
    EdgeType,
    NodeType,
    normalize_phase,
)

settings = Settings()


class ClinicalGraph:
    """Heterogeneous graph for clinical trial readthrough analysis."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_counter: dict[NodeType, int] = {}

    def _make_node_id(self, node_type: NodeType, key: str) -> str:
        """Create a unique node ID."""
        return f"{node_type.value}::{key}"

    def add_company_node(self, ticker: str, name: str = "", cik: str = "") -> str:
        node_id = self._make_node_id(NodeType.COMPANY, ticker)
        self.graph.add_node(
            node_id,
            node_type=NodeType.COMPANY,
            ticker=ticker,
            name=name,
            cik=cik,
        )
        return node_id

    def add_trial_node(
        self, nct_id: str, title: str = "", phase: str = "", status: str = ""
    ) -> str:
        node_id = self._make_node_id(NodeType.TRIAL, nct_id)
        self.graph.add_node(
            node_id,
            node_type=NodeType.TRIAL,
            nct_id=nct_id,
            title=title,
            phase=phase,
            status=status,
        )
        return node_id

    def add_drug_node(self, drug_name: str) -> str:
        norm_name = drug_name.lower().strip()
        node_id = self._make_node_id(NodeType.DRUG, norm_name)
        self.graph.add_node(
            node_id,
            node_type=NodeType.DRUG,
            name=drug_name,
            normalized=norm_name,
        )
        return node_id

    def add_indication_node(self, condition: str) -> str:
        norm_cond = condition.lower().strip()
        node_id = self._make_node_id(NodeType.INDICATION, norm_cond)
        self.graph.add_node(
            node_id,
            node_type=NodeType.INDICATION,
            name=condition,
            normalized=norm_cond,
        )
        return node_id

    def add_phase_node(self, phase: str) -> str:
        norm_phase = normalize_phase(phase)
        node_id = self._make_node_id(NodeType.PHASE, norm_phase)
        self.graph.add_node(
            node_id,
            node_type=NodeType.PHASE,
            name=norm_phase,
        )
        return node_id

    def add_event_node(
        self, event_id: str, event_type: str = "", date: str = "", confidence: float = 1.0
    ) -> str:
        node_id = self._make_node_id(NodeType.EVENT, event_id)
        self.graph.add_node(
            node_id,
            node_type=NodeType.EVENT,
            event_type=event_type,
            date=date,
            confidence=confidence,
        )
        return node_id

    def add_edge(
        self, source_id: str, target_id: str, edge_type: EdgeType, weight: float = 1.0, **props
    ):
        self.graph.add_edge(
            source_id,
            target_id,
            key=edge_type.value,
            edge_type=edge_type,
            weight=weight,
            **props,
        )

    # --- Convenience builders ---

    def add_sponsor_edge(self, company_id: str, trial_id: str, weight: float = 1.0):
        self.add_edge(company_id, trial_id, EdgeType.SPONSOR, weight=weight)

    def add_intervention_edge(self, trial_id: str, drug_id: str, weight: float = 1.0):
        self.add_edge(trial_id, drug_id, EdgeType.INTERVENTION, weight=weight)

    def add_indication_edge(self, trial_id: str, indication_id: str, weight: float = 1.0):
        self.add_edge(trial_id, indication_id, EdgeType.INDICATION, weight=weight)

    def add_phase_edge(self, trial_id: str, phase_id: str, weight: float = 1.0):
        self.add_edge(trial_id, phase_id, EdgeType.PHASE, weight=weight)

    def add_event_edge(self, node_id: str, event_id: str, weight: float = 1.0):
        self.add_edge(node_id, event_id, EdgeType.EVENT, weight=weight)

    def add_company_similarity_edge(self, company_a: str, company_b: str, weight: float):
        self.add_edge(company_a, company_b, EdgeType.COMPANY_SIMILARITY, weight=weight)

    # --- Query methods ---

    def get_node(self, node_id: str) -> Optional[dict]:
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> list[str]:
        return [
            n for n, attrs in self.graph.nodes(data=True) if attrs.get("node_type") == node_type
        ]

    def get_edges_by_type(self, edge_type: EdgeType) -> list[tuple]:
        """Get all edges of a given type."""
        results = []
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if data.get("edge_type") == edge_type:
                results.append((u, v, data))
        return results

    def get_peer_companies(
        self,
        company_ticker: str,
        max_distance: int = 3,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Get peer companies from graph proximity.

        Uses weighted graph distance to find related companies.
        Returns list of (ticker, proximity_score) tuples.
        """
        company_id = self._make_node_id(NodeType.COMPANY, company_ticker)
        if not self.graph.has_node(company_id):
            return []

        # Compute shortest path distances to other companies
        companies = self.get_nodes_by_type(NodeType.COMPANY)
        peers = []
        for other_id in companies:
            if exclude_self and other_id == company_id:
                continue
            try:
                dist = nx.shortest_path_length(
                    self.graph, source=company_id, target=other_id, weight="weight"
                )
                if dist <= max_distance:
                    other_ticker = self.graph.nodes[other_id].get("ticker", "")
                    score = 1.0 / (1.0 + dist)
                    peers.append((other_ticker, score))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        peers.sort(key=lambda x: x[1], reverse=True)
        return peers

    def get_peer_basket(
        self,
        company_ticker: str,
        top_k: int = 10,
        max_distance: int = 3,
    ) -> list[str]:
        """Get top-k peer companies by graph proximity."""
        peers = self.get_peer_companies(company_ticker, max_distance=max_distance)
        return [p[0] for p in peers[:top_k]]

    def build_from_dataframes(
        self,
        trials_df: pd.DataFrame,
        sponsor_map: dict[str, Optional[dict]],
        fda_events: Optional[pd.DataFrame] = None,
    ):
        """Build graph from trial data and sponsor mappings."""
        companies_added: set[str] = set()
        drugs_added: set[str] = set()
        indications_added: set[str] = set()
        phases_added: set[str] = set()

        for _, trial in trials_df.iterrows():
            nct_id = trial.get("nct_id", "")
            if not nct_id:
                continue

            trial_id = self.add_trial_node(
                nct_id=nct_id,
                title=trial.get("brief_title", ""),
                phase=trial.get("phase", ""),
                status=trial.get("overall_status", ""),
            )

            # Sponsor edges
            sponsors_raw = str(trial.get("sponsors", ""))
            for s in sponsors_raw.split(";"):
                s = s.strip()
                if not s:
                    continue
                matched = sponsor_map.get(s)
                if matched and matched.get("ticker"):
                    ticker = matched["ticker"]
                    if ticker not in companies_added:
                        self.add_company_node(
                            ticker=ticker,
                            name=matched.get("company_name", ""),
                        )
                        companies_added.add(ticker)
                    co_id = self._make_node_id(NodeType.COMPANY, ticker)
                    self.add_sponsor_edge(co_id, trial_id)

            # Drug/intervention edges
            interventions = str(trial.get("intervention_name", ""))
            for drug in interventions.split(";"):
                drug = drug.strip()
                if not drug:
                    continue
                drug_norm = drug.lower().strip()
                if drug_norm not in drugs_added:
                    self.add_drug_node(drug)
                    drugs_added.add(drug_norm)
                drug_id = self._make_node_id(NodeType.DRUG, drug_norm)
                self.add_intervention_edge(trial_id, drug_id)

            # Indication edges
            conditions = str(trial.get("conditions", ""))
            for cond in conditions.split(";"):
                cond = cond.strip()
                if not cond:
                    continue
                cond_norm = cond.lower().strip()
                if cond_norm not in indications_added:
                    self.add_indication_node(cond)
                    indications_added.add(cond_norm)
                ind_id = self._make_node_id(NodeType.INDICATION, cond_norm)
                self.add_indication_edge(trial_id, ind_id)

            # Phase edges
            phase = str(trial.get("phase", ""))
            if phase:
                norm_phase = normalize_phase(phase)
                if norm_phase not in phases_added:
                    self.add_phase_node(phase)
                    phases_added.add(norm_phase)
                phase_id = self._make_node_id(NodeType.PHASE, norm_phase)
                self.add_phase_edge(trial_id, phase_id)

        return self

    def add_company_similarity_edges(self, similarity_df: pd.DataFrame, threshold: float = 0.3):
        """Add company similarity edges from a similarity matrix.

        similarity_df should have columns: ticker_a, ticker_b, similarity
        """
        for _, row in similarity_df.iterrows():
            weight = row.get("similarity", 0)
            if weight < threshold:
                continue
            co_a_id = self._make_node_id(NodeType.COMPANY, row["ticker_a"])
            co_b_id = self._make_node_id(NodeType.COMPANY, row["ticker_b"])
            self.add_company_similarity_edge(co_a_id, co_b_id, weight)

    def summary(self) -> dict:
        """Return a summary of the graph."""
        node_counts: dict[str, int] = {}
        edge_counts: dict[str, int] = {}

        for _, attrs in self.graph.nodes(data=True):
            nt = (
                attrs.get("node_type", "UNKNOWN").value
                if isinstance(attrs.get("node_type"), NodeType)
                else str(attrs.get("node_type", "UNKNOWN"))
            )
            node_counts[nt] = node_counts.get(nt, 0) + 1

        for _, _, _, data in self.graph.edges(data=True, keys=True):
            et = (
                data.get("edge_type", "UNKNOWN").value
                if isinstance(data.get("edge_type"), EdgeType)
                else str(data.get("edge_type", "UNKNOWN"))
            )
            edge_counts[et] = edge_counts.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_counts": node_counts,
            "edge_counts": edge_counts,
        }

    def export_fuzzy_matches_report(self) -> pd.DataFrame:
        """Generate a report of low-confidence matches for manual review.

        Returns DataFrame with edges where weight < 0.75
        """
        records = []
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            w = data.get("weight", 1.0)
            if w < 0.75:
                records.append(
                    {
                        "source": u,
                        "target": v,
                        "edge_type": str(data.get("edge_type", "")),
                        "weight": w,
                    }
                )
        return pd.DataFrame(records)
