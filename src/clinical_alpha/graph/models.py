"""Graph node and edge type definitions for the heterogeneous graph."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    COMPANY = "COMPANY"
    TRIAL = "TRIAL"
    DRUG = "DRUG"
    INDICATION = "INDICATION"
    PHASE = "PHASE"
    EVENT = "EVENT"


class EdgeType(str, Enum):
    SPONSOR = "SPONSOR"  # COMPANY → TRIAL
    INTERVENTION = "INTERVENTION"  # TRIAL → DRUG
    INDICATION = "INDICATION"  # TRIAL → INDICATION
    PHASE = "PHASE"  # TRIAL → PHASE
    EVENT = "EVENT"  # COMPANY → EVENT / TRIAL → EVENT
    COMPANY_SIMILARITY = "COMPANY_SIMILARITY"  # COMPANY → COMPANY
    DRUG_SIMILARITY = "DRUG_SIMILARITY"  # DRUG → DRUG


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.node_id == other.node_id


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type.value))

    def __eq__(self, other):
        return (
            isinstance(other, Edge)
            and self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.edge_type == other.edge_type
        )


# Standardized clinical trial phases
PHASE_MAPPING = {
    "Phase 1": "Phase 1",
    "Phase 1/Phase 2": "Phase 1/2",
    "Phase 2": "Phase 2",
    "Phase 2/Phase 3": "Phase 2/3",
    "Phase 3": "Phase 3",
    "Phase 4": "Phase 4",
    "N/A": "Unknown",
}


def normalize_phase(phase_str: str) -> str:
    """Normalize a phase string to a standard value."""
    sorted_keys = sorted(PHASE_MAPPING.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key.lower() in phase_str.lower():
            return PHASE_MAPPING[key]
    return "Unknown"
