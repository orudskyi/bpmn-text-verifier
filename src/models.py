"""Shared Pydantic data models for the BPMN-Text Conformance Checker.

These models define the **contract** between all pipeline modules.
Every module must produce and consume these exact types.

Module outputs:
    - Module A → BPMNGraph, list[TextFragment]
    - Agent 1  → list[Mapping]
    - Agent 2  → list[DeclareConstraint]
    - Module C → VerificationResult
    - Full pipeline → ConformanceReport
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── BPMN Graph (output of Module A) ──────────────────────────────────────────


class BPMNNodeType(str, Enum):
    """Enumeration of supported BPMN 2.0 node types."""

    START_EVENT = "start_event"
    END_EVENT = "end_event"
    INTERMEDIATE_EVENT = "intermediate_event"
    TASK = "task"
    EXCLUSIVE_GATEWAY = "exclusive_gateway"
    INCLUSIVE_GATEWAY = "inclusive_gateway"
    PARALLEL_GATEWAY = "parallel_gateway"
    EVENT_BASED_GATEWAY = "event_based_gateway"


class BPMNNode(BaseModel):
    """A single node in the BPMN process graph.

    Attributes:
        id: Unique BPMN element identifier (e.g. ``"Task_0vaxgaa"``).
        type: Semantic type of the node.
        name: Human-readable label from the BPMN diagram (e.g. ``"Clarify shipment method"``).
        lane: Swim-lane the node belongs to (e.g. ``"Secretary"``); empty string if none.
    """

    id: str
    type: BPMNNodeType
    name: str = ""
    lane: str = ""


class BPMNEdge(BaseModel):
    """A directed sequence flow (edge) connecting two BPMN nodes.

    Attributes:
        id: Unique identifier of the sequence flow (e.g. ``"SequenceFlow_023hzxi"``).
        source: ID of the source node.
        target: ID of the target node.
        name: Optional condition label on the flow (e.g. ``"yes"``, ``"no"``).
    """

    id: str
    source: str
    target: str
    name: str = ""


class BPMNGraph(BaseModel):
    """Complete structural representation of a BPMN process.

    Attributes:
        process_name: Human-readable name of the process.
        nodes: All BPMN flow elements (events, tasks, gateways).
        edges: All sequence flows between nodes.
        lanes: Mapping of swim-lane name → list of node IDs in that lane.
    """

    process_name: str
    nodes: list[BPMNNode]
    edges: list[BPMNEdge]
    lanes: dict[str, list[str]] = {}


# ── Text Fragments (output of Module A) ──────────────────────────────────────


class TextFragment(BaseModel):
    """A single meaningful sentence extracted from the process description.

    Attributes:
        id: Unique fragment identifier (e.g. ``"frag_01"``).
        text: The raw sentence text.
        sentence_index: Zero-based position of this sentence in the original document.
    """

    id: str
    text: str
    sentence_index: int


# ── Mapping (output of Agent 1) ───────────────────────────────────────────────


class Mapping(BaseModel):
    """A semantic link between a text fragment and a BPMN node.

    Attributes:
        fragment_id: ID of the source TextFragment.
        fragment_text: Raw text of the matched fragment.
        node_id: ID of the matched BPMNNode.
        node_name: Human-readable name of the matched BPMNNode.
        confidence: Model confidence score in [0.0, 1.0].
    """

    fragment_id: str
    fragment_text: str
    node_id: str
    node_name: str
    confidence: float = Field(ge=0.0, le=1.0)


# ── DECLARE Constraints (output of Agent 2) ───────────────────────────────────


class DeclareTemplate(str, Enum):
    """Supported DECLARE constraint templates.

    See: Pesic & van der Aalst (2006) — *A Declarative Approach for Flexible
    Business Processes Management*.
    """

    EXISTENCE = "existence"                   # A must occur at least once
    ABSENCE = "absence"                       # A must NOT occur
    RESPONSE = "response"                     # if A then eventually B
    PRECEDENCE = "precedence"                 # B only if A occurred before it
    SUCCESSION = "succession"                 # A then B and B only after A
    CHAIN_RESPONSE = "chain_response"         # if A then immediately B (next step)
    CHAIN_PRECEDENCE = "chain_precedence"     # B only if A was immediately before it
    NOT_COEXISTENCE = "not_coexistence"       # A and B never both occur in same trace
    CHOICE = "choice"                         # A or B must occur (at least one)
    EXCLUSIVE_CHOICE = "exclusive_choice"     # exactly one of A or B must occur


class DeclareConstraint(BaseModel):
    """A formal DECLARE constraint extracted from a text fragment.

    Attributes:
        id: Unique constraint identifier (e.g. ``"constr_01"``).
        template: The DECLARE template type.
        activity_a: Name of the primary BPMN activity (matches BPMNNode.name).
        activity_b: Name of the secondary activity; empty string for unary templates.
        source_text: The text fragment that implies this constraint.
        condition: Optional guard condition (e.g. ``"if large amounts"``).
    """

    id: str
    template: DeclareTemplate
    activity_a: str
    activity_b: str = ""
    source_text: str
    condition: str = ""


# ── Verification Result (output of Module C) ──────────────────────────────────


class Violation(BaseModel):
    """A single constraint violation found during conformance checking.

    Attributes:
        constraint_id: ID of the violated DeclareConstraint.
        constraint_description: Human-readable description of the violated rule.
        trace: Counter-example execution trace as a list of activity names.
        explanation: Natural language explanation filled by Agent 3 (Module D).
    """

    constraint_id: str
    constraint_description: str
    trace: list[str]
    explanation: str = ""


class VerificationResult(BaseModel):
    """Aggregated result of the symbolic conformance verification step.

    Attributes:
        is_conformant: True iff zero constraints were violated.
        total_constraints: Total number of constraints checked.
        satisfied: Number of constraints that passed.
        violated: Number of constraints that failed.
        violations: Detailed list of all violations found.
    """

    is_conformant: bool
    total_constraints: int
    satisfied: int
    violated: int
    violations: list[Violation]


# ── Final Report (output of full pipeline) ────────────────────────────────────


class ConformanceReport(BaseModel):
    """Complete conformance-checking report produced by the full pipeline.

    Attributes:
        process_name: Human-readable name of the analysed process.
        bpmn_file: Path or filename of the BPMN source file.
        text_file: Path or filename of the text description source file.
        graph_summary: High-level statistics about the BPMN graph
            (e.g. ``{"nodes": 10, "edges": 12, "lanes": 3}``).
        mappings: All text-to-BPMN element mappings produced by Agent 1.
        constraints: All DECLARE constraints extracted by Agent 2.
        verification: Full verification result from Module C.
    """

    process_name: str
    bpmn_file: str
    text_file: str
    graph_summary: dict
    mappings: list[Mapping]
    constraints: list[DeclareConstraint]
    verification: VerificationResult
