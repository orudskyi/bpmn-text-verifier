"""Tests for Module A — BPMN parser (TASK_01).

Uses the two provided sample processes as fixtures:
  - data/dispatch/Dispatch-of-goods.bpmn
  - data/recourse/Recourse.bpmn
"""

import pytest

from src.models import BPMNNodeType
from src.module_a.bpmn_parser import parse_bpmn


# ── Fixtures ──────────────────────────────────────────────────────────────────

DISPATCH_FILE = "data/dispatch/Dispatch-of-goods.bpmn"
RECOURSE_FILE = "data/recourse/Recourse.bpmn"


# ── Dispatch tests ────────────────────────────────────────────────────────────


def test_dispatch_process_name():
    """Process name must be non-empty and come from the participant element."""
    graph = parse_bpmn(DISPATCH_FILE)
    assert graph.process_name != ""
    # The participant name contains "Dispatch"
    assert "Dispatch" in graph.process_name


def test_dispatch_task_count():
    """Dispatch process should contain exactly 7 tasks."""
    graph = parse_bpmn(DISPATCH_FILE)
    tasks = [n for n in graph.nodes if n.type == BPMNNodeType.TASK]
    assert len(tasks) == 7, f"Expected 7 tasks, got {len(tasks)}: {[t.name for t in tasks]}"


def test_dispatch_lane_count():
    """Dispatch process should have exactly 3 swim-lanes."""
    graph = parse_bpmn(DISPATCH_FILE)
    assert len(graph.lanes) == 3
    assert "Secretary" in graph.lanes
    assert "Logistics" in graph.lanes
    assert "Warehouse" in graph.lanes


def test_dispatch_lane_secretary_tasks():
    """Secretary lane must contain the expected task IDs."""
    graph = parse_bpmn(DISPATCH_FILE)
    secretary_ids = set(graph.lanes["Secretary"])
    assert "Task_0vaxgaa" in secretary_ids   # Clarify shipment method
    assert "Task_0e6hvnj" in secretary_ids   # Get 3 offers
    assert "Task_0s79ile" in secretary_ids   # Select logistic company and place order
    assert "Task_0jsoxba" in secretary_ids   # Write package label


def test_dispatch_specific_nodes():
    """Key named nodes must be present with correct types."""
    graph = parse_bpmn(DISPATCH_FILE)
    by_id = {n.id: n for n in graph.nodes}

    assert by_id["StartEvent_1"].type == BPMNNodeType.START_EVENT
    assert by_id["StartEvent_1"].name == "Ship goods"

    assert by_id["Task_0vaxgaa"].type == BPMNNodeType.TASK
    assert by_id["Task_0vaxgaa"].lane == "Secretary"

    assert by_id["ParallelGateway_02fgrfq"].type == BPMNNodeType.PARALLEL_GATEWAY

    assert by_id["ExclusiveGateway_1mpgzhg"].type == BPMNNodeType.EXCLUSIVE_GATEWAY

    assert by_id["EndEvent_1fx9yp3"].type == BPMNNodeType.END_EVENT
    assert by_id["EndEvent_1fx9yp3"].lane == "Warehouse"


def test_dispatch_no_newlines_in_names():
    """No node or edge name should contain a raw newline character."""
    graph = parse_bpmn(DISPATCH_FILE)
    for node in graph.nodes:
        assert "\n" not in node.name, f"Node {node.id!r} name contains newline: {node.name!r}"
    for edge in graph.edges:
        assert "\n" not in edge.name, f"Edge {edge.id!r} name contains newline: {edge.name!r}"


def test_dispatch_edges_reference_existing_nodes():
    """Every sequence-flow source and target must reference a known node."""
    graph = parse_bpmn(DISPATCH_FILE)
    node_ids = {n.id for n in graph.nodes}
    for edge in graph.edges:
        assert edge.source in node_ids, f"Edge {edge.id}: unknown source {edge.source!r}"
        assert edge.target in node_ids, f"Edge {edge.id}: unknown target {edge.target!r}"


def test_dispatch_edge_conditions():
    """Named sequence flows (yes/no conditions) must preserve their labels."""
    graph = parse_bpmn(DISPATCH_FILE)
    by_id = {e.id: e for e in graph.edges}
    # "yes" branch from ExclusiveGateway_1mpgzhg → Task_0e6hvnj
    assert by_id["SequenceFlow_1xv6wk4"].name == "yes"
    # "no" branch from ExclusiveGateway_1mpgzhg → InclusiveGateway_0p2e5vq
    assert by_id["SequenceFlow_0iu9po7"].name == "no"


# ── Recourse tests ────────────────────────────────────────────────────────────


def test_recourse_task_count():
    """Recourse process should contain exactly 9 tasks (three share the name 'close case').

    Note: The development plan listed 8, but the actual BPMN file has 9 tasks.
    """
    graph = parse_bpmn(RECOURSE_FILE)
    tasks = [n for n in graph.nodes if n.type == BPMNNodeType.TASK]
    assert len(tasks) == 9, f"Expected 9 tasks, got {len(tasks)}: {[t.name for t in tasks]}"


def test_recourse_event_based_gateway():
    """Recourse process should have exactly 1 event-based gateway."""
    graph = parse_bpmn(RECOURSE_FILE)
    ebg = [n for n in graph.nodes if n.type == BPMNNodeType.EVENT_BASED_GATEWAY]
    assert len(ebg) == 1
    assert ebg[0].id == "EventBasedGateway_0qdxz70"


def test_recourse_no_lanes():
    """Recourse is a single-participant process with no swim-lanes."""
    graph = parse_bpmn(RECOURSE_FILE)
    assert len(graph.lanes) == 0


def test_recourse_intermediate_events():
    """Recourse should contain exactly 3 intermediate catch events."""
    graph = parse_bpmn(RECOURSE_FILE)
    ie = [n for n in graph.nodes if n.type == BPMNNodeType.INTERMEDIATE_EVENT]
    assert len(ie) == 3


def test_recourse_edges_reference_existing_nodes():
    """Every sequence-flow in the Recourse process must reference known nodes."""
    graph = parse_bpmn(RECOURSE_FILE)
    node_ids = {n.id for n in graph.nodes}
    for edge in graph.edges:
        assert edge.source in node_ids, f"Edge {edge.id}: unknown source {edge.source!r}"
        assert edge.target in node_ids, f"Edge {edge.id}: unknown target {edge.target!r}"


# ── Error handling tests ──────────────────────────────────────────────────────


def test_file_not_found():
    """parse_bpmn should raise FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        parse_bpmn("data/nonexistent.bpmn")


def test_invalid_xml(tmp_path):
    """parse_bpmn should raise ValueError for malformed XML."""
    bad_file = tmp_path / "bad.bpmn"
    bad_file.write_text("<unclosed>")
    with pytest.raises(ValueError, match="Invalid XML"):
        parse_bpmn(str(bad_file))
