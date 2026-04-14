"""Tests for Module B — Mapper Agent (TASK_03).

Tests split into two groups:
  - Unit tests (no API key required): test helpers, validation, formatting.
  - Integration tests (require GOOGLE_API_KEY): marked with @pytest.mark.integration.

Run only unit tests:
    uv run --extra dev pytest tests/test_mapper.py -m "not integration"

Run all tests (including integration):
    uv run --extra dev pytest tests/test_mapper.py
"""

import pytest

from src.models import BPMNEdge, BPMNGraph, BPMNNode, BPMNNodeType, Mapping
from src.module_a.bpmn_parser import parse_bpmn
from src.module_a.text_loader import load_text
from src.module_b.mapper_agent import (
    MappingList,
    _format_fragments,
    _format_lanes,
    _format_nodes,
    _validate_mappings,
    map_text_to_bpmn,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

DISPATCH_BPMN = "data/dispatch/Dispatch-of-goods.bpmn"
DISPATCH_TXT = "data/dispatch/DispatchDescription.txt"


@pytest.fixture
def dispatch_graph() -> BPMNGraph:
    """Return the parsed Dispatch-of-goods BPMNGraph."""
    return parse_bpmn(DISPATCH_BPMN)


@pytest.fixture
def dispatch_fragments():
    """Return the loaded Dispatch text fragments."""
    return load_text(DISPATCH_TXT)


@pytest.fixture
def minimal_graph() -> BPMNGraph:
    """Tiny BPMNGraph with 2 nodes for unit testing."""
    return BPMNGraph(
        process_name="Test Process",
        nodes=[
            BPMNNode(id="Task_A", type=BPMNNodeType.TASK, name="Pack goods", lane="Warehouse"),
            BPMNNode(id="Task_B", type=BPMNNodeType.TASK, name="Ship goods", lane="Logistics"),
        ],
        edges=[
            BPMNEdge(id="Flow_1", source="Task_A", target="Task_B"),
        ],
        lanes={
            "Warehouse": ["Task_A"],
            "Logistics": ["Task_B"],
        },
    )


# ── Unit tests: _format_nodes ─────────────────────────────────────────────────


def test_format_nodes_contains_ids(minimal_graph):
    """Formatted node string must include all node IDs."""
    result = _format_nodes(minimal_graph)
    assert "Task_A" in result
    assert "Task_B" in result


def test_format_nodes_contains_names(minimal_graph):
    """Formatted node string must include all node names."""
    result = _format_nodes(minimal_graph)
    assert "Pack goods" in result
    assert "Ship goods" in result


def test_format_nodes_contains_lanes(minimal_graph):
    """Formatted node string must include lane hints."""
    result = _format_nodes(minimal_graph)
    assert "Warehouse" in result
    assert "Logistics" in result


def test_format_nodes_no_lane(dispatch_graph):
    """Nodes without a lane assignment must not produce '[lane: ]' artefacts."""
    result = _format_nodes(dispatch_graph)
    assert "[lane: ]" not in result


# ── Unit tests: _format_lanes ─────────────────────────────────────────────────


def test_format_lanes_with_lanes(minimal_graph):
    """Lane formatting must list all lane names."""
    result = _format_lanes(minimal_graph)
    assert "Warehouse" in result
    assert "Logistics" in result


def test_format_lanes_no_lanes():
    """Single-pool graph with no lanes should return the no-lanes placeholder."""
    empty_graph = BPMNGraph(process_name="Solo", nodes=[], edges=[], lanes={})
    result = _format_lanes(empty_graph)
    assert "no lanes" in result.lower()


# ── Unit tests: _format_fragments ─────────────────────────────────────────────


def test_format_fragments_contains_ids(dispatch_fragments):
    """Formatted fragment string must include all fragment IDs."""
    result = _format_fragments(dispatch_fragments)
    for f in dispatch_fragments:
        assert f.id in result


def test_format_fragments_contains_texts(dispatch_fragments):
    """Formatted fragment string must include fragment text snippets."""
    result = _format_fragments(dispatch_fragments)
    for f in dispatch_fragments:
        # At least the first word of each fragment text should appear.
        assert f.text.split()[0] in result


# ── Unit tests: _validate_mappings ────────────────────────────────────────────


def test_validate_mappings_valid(minimal_graph):
    """Valid node references must pass through unchanged."""
    valid = [
        Mapping(
            fragment_id="frag_01",
            fragment_text="pack the goods",
            node_id="Task_A",
            node_name="Pack goods",
            confidence=0.9,
        )
    ]
    result = _validate_mappings(valid, minimal_graph)
    assert result[0].confidence == 0.9
    assert result[0].node_id == "Task_A"


def test_validate_mappings_none_is_valid(minimal_graph):
    """node_id='NONE' must be treated as explicitly valid (no-match marker)."""
    none_mapping = [
        Mapping(
            fragment_id="frag_02",
            fragment_text="something unrelated",
            node_id="NONE",
            node_name="",
            confidence=0.0,
        )
    ]
    result = _validate_mappings(none_mapping, minimal_graph)
    assert result[0].node_id == "NONE"
    assert result[0].confidence == 0.0


def test_validate_mappings_unknown_node_zeroed(minimal_graph):
    """A mapping referencing a non-existent node_id gets confidence=0.0."""
    bad = [
        Mapping(
            fragment_id="frag_03",
            fragment_text="some text",
            node_id="Task_DOES_NOT_EXIST",
            node_name="Ghost task",
            confidence=0.8,
        )
    ]
    result = _validate_mappings(bad, minimal_graph)
    assert result[0].confidence == 0.0


def test_validate_mappings_preserves_order(minimal_graph):
    """Validation must not reorder the mappings list."""
    mappings = [
        Mapping(fragment_id=f"frag_{i:02d}", fragment_text="x", node_id="Task_A",
                node_name="Pack goods", confidence=0.5)
        for i in range(1, 6)
    ]
    result = _validate_mappings(mappings, minimal_graph)
    assert [m.fragment_id for m in result] == [m.fragment_id for m in mappings]


# ── Integration tests (require GOOGLE_API_KEY) ────────────────────────────────

QUOTA_ERRORS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")


def _is_quota_error(exc: Exception) -> bool:
    """Return True if the exception looks like an API quota / rate-limit error."""
    msg = str(exc).lower()
    return any(kw.lower() in msg for kw in QUOTA_ERRORS)



@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_mapping_count(dispatch_graph, dispatch_fragments):
    """Mapper must return at least one mapping per fragment."""
    try:
        mappings = await map_text_to_bpmn(dispatch_graph, dispatch_fragments)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    assert len(mappings) > 0
    assert len(mappings) == len(dispatch_fragments)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_mapping_high_confidence(dispatch_graph, dispatch_fragments):
    """At least 5 mappings must have confidence >= 0.7."""
    try:
        mappings = await map_text_to_bpmn(dispatch_graph, dispatch_fragments)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    high_conf = [m for m in mappings if m.confidence >= 0.7]
    assert len(high_conf) >= 5, (
        f"Only {len(high_conf)} high-confidence mappings: "
        + str([(m.fragment_id, m.node_name, m.confidence) for m in mappings])
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_clarify_shipment_mapped(dispatch_graph, dispatch_fragments):
    """'Clarify shipment method' task must be identified by the mapper."""
    try:
        mappings = await map_text_to_bpmn(dispatch_graph, dispatch_fragments)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    clarify = [m for m in mappings if m.node_id == "Task_0vaxgaa"]
    assert len(clarify) == 1, (
        "Expected exactly 1 mapping to Task_0vaxgaa (Clarify shipment method)"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_mapping_node_ids_valid(dispatch_graph, dispatch_fragments):
    """All non-NONE node_ids must reference actual nodes in the graph."""
    try:
        mappings = await map_text_to_bpmn(dispatch_graph, dispatch_fragments)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    known_ids = {n.id for n in dispatch_graph.nodes}
    invalid = [
        m for m in mappings
        if m.node_id != "NONE" and m.node_id not in known_ids
    ]
    assert invalid == [], f"Invalid node references: {[(m.fragment_id, m.node_id) for m in invalid]}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_mapping_confidence_range(dispatch_graph, dispatch_fragments):
    """Confidence values must always be in [0.0, 1.0]."""
    try:
        mappings = await map_text_to_bpmn(dispatch_graph, dispatch_fragments)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    out_of_range = [m for m in mappings if not (0.0 <= m.confidence <= 1.0)]
    assert out_of_range == [], f"Out-of-range confidence: {out_of_range}"
