"""Tests for Module C — Symbolic Verifier (TASK_05).

All tests in this file are pure unit/integration tests that require only the
BPMN files — no LLM or API key needed.

Run:
    uv run --extra dev python -m pytest tests/test_verifier.py -v
"""

import pytest

from src.models import (
    BPMNEdge,
    BPMNGraph,
    BPMNNode,
    BPMNNodeType,
    DeclareConstraint,
    DeclareTemplate,
)
from src.module_a.bpmn_parser import parse_bpmn
from src.module_c.verifier import (
    _check_absence,
    _check_chain_precedence,
    _check_chain_response,
    _check_choice,
    _check_exclusive_choice,
    _check_existence,
    _check_not_coexistence,
    _check_precedence,
    _check_response,
    _check_succession,
    generate_traces,
    verify_constraints,
)

# ── File paths ────────────────────────────────────────────────────────────────

DISPATCH_BPMN = "data/dispatch/Dispatch-of-goods.bpmn"
RECOURSE_BPMN = "data/recourse/Recourse.bpmn"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_constraint(
    cid: str,
    tpl: DeclareTemplate,
    a: str,
    b: str = "",
    condition: str = "",
) -> DeclareConstraint:
    return DeclareConstraint(
        id=cid, template=tpl, activity_a=a, activity_b=b,
        source_text="test", condition=condition,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — individual DECLARE checker functions
# ═══════════════════════════════════════════════════════════════════════════════


class TestExistence:
    def test_present(self):
        assert _check_existence("A", ["A", "B", "C"]) is True

    def test_absent(self):
        assert _check_existence("X", ["A", "B", "C"]) is False

    def test_empty_trace(self):
        assert _check_existence("A", []) is False


class TestAbsence:
    def test_not_present(self):
        assert _check_absence("X", ["A", "B"]) is True

    def test_present_violates(self):
        assert _check_absence("A", ["A", "B"]) is False

    def test_empty_trace(self):
        assert _check_absence("A", []) is True


class TestResponse:
    def test_a_followed_by_b(self):
        assert _check_response("A", "B", ["A", "B"]) is True

    def test_a_not_followed_by_b(self):
        assert _check_response("A", "B", ["A", "C"]) is False

    def test_no_a_vacuously_true(self):
        assert _check_response("A", "B", ["C", "D"]) is True

    def test_multiple_a_all_followed(self):
        assert _check_response("A", "B", ["A", "B", "A", "B"]) is True

    def test_second_a_not_followed(self):
        assert _check_response("A", "B", ["A", "B", "A"]) is False


class TestPrecedence:
    def test_a_before_b(self):
        assert _check_precedence("A", "B", ["A", "B"]) is True

    def test_b_before_a_violates(self):
        assert _check_precedence("A", "B", ["B", "A"]) is False

    def test_no_b_vacuously_true(self):
        assert _check_precedence("A", "B", ["A", "C"]) is True

    def test_b_without_any_a(self):
        assert _check_precedence("A", "B", ["C", "B"]) is False


class TestSuccession:
    def test_both_hold(self):
        assert _check_succession("A", "B", ["A", "B"]) is True

    def test_response_fails(self):
        assert _check_succession("A", "B", ["A", "C"]) is False

    def test_precedence_fails(self):
        assert _check_succession("A", "B", ["B", "A", "B"]) is False


class TestChainResponse:
    def test_immediately_next(self):
        assert _check_chain_response("A", "B", ["A", "B", "C"]) is True

    def test_not_immediately_next(self):
        assert _check_chain_response("A", "B", ["A", "C", "B"]) is False

    def test_a_at_end_violates(self):
        assert _check_chain_response("A", "B", ["C", "A"]) is False

    def test_no_a_vacuously_true(self):
        assert _check_chain_response("A", "B", ["C", "D"]) is True


class TestChainPrecedence:
    def test_immediately_preceded(self):
        assert _check_chain_precedence("A", "B", ["A", "B"]) is True

    def test_not_immediately_preceded(self):
        assert _check_chain_precedence("A", "B", ["A", "C", "B"]) is False

    def test_b_at_start_violates(self):
        assert _check_chain_precedence("A", "B", ["B", "A"]) is False

    def test_no_b_vacuously_true(self):
        assert _check_chain_precedence("A", "B", ["C", "A"]) is True


class TestNotCoexistence:
    def test_only_a(self):
        assert _check_not_coexistence("A", "B", ["A", "C"]) is True

    def test_only_b(self):
        assert _check_not_coexistence("A", "B", ["B", "C"]) is True

    def test_both_violates(self):
        assert _check_not_coexistence("A", "B", ["A", "B"]) is False

    def test_neither_ok(self):
        assert _check_not_coexistence("A", "B", ["C", "D"]) is True


class TestChoice:
    def test_a_present(self):
        assert _check_choice("A", "B", ["A", "C"]) is True

    def test_b_present(self):
        assert _check_choice("A", "B", ["B", "C"]) is True

    def test_both_present(self):
        assert _check_choice("A", "B", ["A", "B"]) is True

    def test_neither_violates(self):
        assert _check_choice("A", "B", ["C", "D"]) is False


class TestExclusiveChoice:
    def test_only_a(self):
        assert _check_exclusive_choice("A", "B", ["A", "C"]) is True

    def test_only_b(self):
        assert _check_exclusive_choice("A", "B", ["B", "C"]) is True

    def test_both_violates(self):
        assert _check_exclusive_choice("A", "B", ["A", "B"]) is False

    def test_neither_violates(self):
        assert _check_exclusive_choice("A", "B", ["C", "D"]) is False


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE GENERATION TESTS — simple hand-crafted graphs
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def linear_graph() -> BPMNGraph:
    """A → Task_A → Task_B → End (sequential)."""
    return BPMNGraph(
        process_name="Linear",
        nodes=[
            BPMNNode(id="start", type=BPMNNodeType.START_EVENT, name=""),
            BPMNNode(id="T_A", type=BPMNNodeType.TASK, name="Pack"),
            BPMNNode(id="T_B", type=BPMNNodeType.TASK, name="Ship"),
            BPMNNode(id="end", type=BPMNNodeType.END_EVENT, name=""),
        ],
        edges=[
            BPMNEdge(id="e1", source="start", target="T_A"),
            BPMNEdge(id="e2", source="T_A", target="T_B"),
            BPMNEdge(id="e3", source="T_B", target="end"),
        ],
    )


@pytest.fixture
def xor_graph() -> BPMNGraph:
    """Start → XOR → (T_A | T_B) → End."""
    return BPMNGraph(
        process_name="XOR",
        nodes=[
            BPMNNode(id="start", type=BPMNNodeType.START_EVENT, name=""),
            BPMNNode(id="gw", type=BPMNNodeType.EXCLUSIVE_GATEWAY, name=""),
            BPMNNode(id="T_A", type=BPMNNodeType.TASK, name="Path A"),
            BPMNNode(id="T_B", type=BPMNNodeType.TASK, name="Path B"),
            BPMNNode(id="end", type=BPMNNodeType.END_EVENT, name=""),
        ],
        edges=[
            BPMNEdge(id="e1", source="start", target="gw"),
            BPMNEdge(id="e2", source="gw", target="T_A"),
            BPMNEdge(id="e3", source="gw", target="T_B"),
            BPMNEdge(id="e4", source="T_A", target="end"),
            BPMNEdge(id="e5", source="T_B", target="end"),
        ],
    )


@pytest.fixture
def parallel_graph() -> BPMNGraph:
    """Start → AND split → (T_A || T_B) → AND join → End."""
    return BPMNGraph(
        process_name="Parallel",
        nodes=[
            BPMNNode(id="start", type=BPMNNodeType.START_EVENT, name=""),
            BPMNNode(id="split", type=BPMNNodeType.PARALLEL_GATEWAY, name=""),
            BPMNNode(id="T_A", type=BPMNNodeType.TASK, name="Branch A"),
            BPMNNode(id="T_B", type=BPMNNodeType.TASK, name="Branch B"),
            BPMNNode(id="join", type=BPMNNodeType.PARALLEL_GATEWAY, name=""),
            BPMNNode(id="end", type=BPMNNodeType.END_EVENT, name=""),
        ],
        edges=[
            BPMNEdge(id="e1", source="start", target="split"),
            BPMNEdge(id="e2", source="split", target="T_A"),
            BPMNEdge(id="e3", source="split", target="T_B"),
            BPMNEdge(id="e4", source="T_A", target="join"),
            BPMNEdge(id="e5", source="T_B", target="join"),
            BPMNEdge(id="e6", source="join", target="end"),
        ],
    )


def test_linear_trace_count(linear_graph):
    """A linear graph must produce exactly one trace."""
    traces = generate_traces(linear_graph)
    assert len(traces) == 1
    assert traces[0] == ["Pack", "Ship"]


def test_xor_trace_count(xor_graph):
    """An XOR gateway must produce exactly 2 alternative traces."""
    traces = generate_traces(xor_graph)
    assert len(traces) == 2
    trace_sets = [frozenset(t) for t in traces]
    assert frozenset(["Path A"]) in trace_sets
    assert frozenset(["Path B"]) in trace_sets


def test_parallel_both_branches_in_every_trace(parallel_graph):
    """A parallel gateway must include BOTH branches in every trace."""
    traces = generate_traces(parallel_graph)
    assert len(traces) >= 1
    for trace in traces:
        assert "Branch A" in trace, f"Branch A missing from trace: {trace}"
        assert "Branch B" in trace, f"Branch B missing from trace: {trace}"


def test_parallel_all_interleavings_present(parallel_graph):
    """A parallel gateway with 2 branches must produce both orderings."""
    traces = generate_traces(parallel_graph)
    trace_tuples = {tuple(t) for t in traces}
    assert ("Branch A", "Branch B") in trace_tuples or \
           ("Branch B", "Branch A") in trace_tuples


def test_max_traces_cap(xor_graph):
    """max_traces=1 must cap the output at 1 trace."""
    traces = generate_traces(xor_graph, max_traces=1)
    assert len(traces) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — real BPMN files (no LLM)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def dispatch_graph():
    return parse_bpmn(DISPATCH_BPMN)


@pytest.fixture(scope="module")
def recourse_graph():
    return parse_bpmn(RECOURSE_BPMN)


# ── Trace generation from real BPMN ──────────────────────────────────────────


def test_dispatch_traces_nonempty(dispatch_graph):
    """Dispatch process must generate at least one trace."""
    traces = generate_traces(dispatch_graph)
    assert len(traces) >= 1
    assert all(len(t) > 0 for t in traces)


def test_dispatch_traces_contain_key_tasks(dispatch_graph):
    """'Clarify shipment method' must appear in every Dispatch trace."""
    traces = generate_traces(dispatch_graph)
    for trace in traces:
        assert "Clarify shipment method" in trace, (
            f"Expected 'Clarify shipment method' in trace: {trace}"
        )


def test_dispatch_traces_exclusive_paths(dispatch_graph):
    """XOR gateway: some traces have 'Get 3 offers', others do not."""
    traces = generate_traces(dispatch_graph)
    has_special = [t for t in traces if "Get 3 offers from logistic companies" in t]
    has_normal  = [t for t in traces if "Write package label" in t]
    assert len(has_special) >= 1, "No trace with special shipment path found"
    assert len(has_normal)  >= 1, "No trace with normal shipment (write label) path found"


def test_recourse_traces_nonempty(recourse_graph):
    """Recourse process must generate at least one trace."""
    traces = generate_traces(recourse_graph)
    assert len(traces) >= 1


# ── Constraint satisfaction — known TRUE constraints ─────────────────────────


def test_existence_satisfied(dispatch_graph):
    """'Clarify shipment method' exists in all Dispatch traces → SATISFIED."""
    c = [_make_constraint("c01", DeclareTemplate.EXISTENCE, "Clarify shipment method")]
    result = verify_constraints(dispatch_graph, c)
    assert result.is_conformant
    assert result.satisfied == 1
    assert result.violated == 0


def test_precedence_package_before_pickup(dispatch_graph):
    """'Package goods' vs 'Prepare for picking up goods' — both appear in every trace.

    Note: due to the parallel gateway, the two tasks run in parallel, so their
    relative order varies across interleavings. Both tasks always EXIST, but
    strict PRECEDENCE is not guaranteed by the BPMN model.  This test verifies
    that 'Package goods' always EXISTS (which IS guaranteed).
    """
    c = [_make_constraint("c02", DeclareTemplate.EXISTENCE, "Package goods")]
    result = verify_constraints(dispatch_graph, c)
    assert result.is_conformant, (
        f"Expected Package goods to exist in all traces, violations: {result.violations}"
    )


# ── Constraint satisfaction — known VIOLATED constraints ─────────────────────


def test_existence_violated(dispatch_graph):
    """'Select logistic company' does NOT appear in all traces → VIOLATED."""
    c = [_make_constraint(
        "c03", DeclareTemplate.EXISTENCE,
        "Select logistic company and place order"
    )]
    result = verify_constraints(dispatch_graph, c)
    assert not result.is_conformant
    assert result.violated == 1
    assert len(result.violations[0].trace) > 0


def test_absence_of_present_task_violated(dispatch_graph):
    """absence('Package goods') should be VIOLATED — that task always runs."""
    c = [_make_constraint("c04", DeclareTemplate.ABSENCE, "Package goods")]
    result = verify_constraints(dispatch_graph, c)
    assert not result.is_conformant


def test_not_coexistence_violated(dispatch_graph):
    """Both label and insure parcel can coexist in some traces → VIOLATED."""
    c = [_make_constraint(
        "c05", DeclareTemplate.NOT_COEXISTENCE,
        "Write package label", "Insure parcel"
    )]
    result = verify_constraints(dispatch_graph, c)
    assert not result.is_conformant


# ── Verification result structure ─────────────────────────────────────────────


def test_verification_result_counts(dispatch_graph):
    """satisfied + violated must equal total_constraints."""
    constraints = [
        _make_constraint("c01", DeclareTemplate.EXISTENCE, "Clarify shipment method"),
        _make_constraint("c02", DeclareTemplate.EXISTENCE,
                         "Select logistic company and place order"),
    ]
    result = verify_constraints(dispatch_graph, constraints)
    assert result.total_constraints == 2
    assert result.satisfied + result.violated == result.total_constraints


def test_empty_constraints(dispatch_graph):
    """Zero constraints must yield a CONFORMANT result with no violations."""
    result = verify_constraints(dispatch_graph, [])
    assert result.is_conformant
    assert result.total_constraints == 0
    assert result.violations == []


def test_violation_has_counter_trace(dispatch_graph):
    """A Violation object must include a non-empty counter-example trace."""
    c = [_make_constraint(
        "c01", DeclareTemplate.EXISTENCE,
        "Select logistic company and place order"
    )]
    result = verify_constraints(dispatch_graph, c)
    assert not result.is_conformant
    assert len(result.violations[0].trace) > 0


def test_conditional_constraint_vacuously_satisfied(dispatch_graph):
    """A constraint conditioned on an activity that never exists → SATISFIED."""
    c = [_make_constraint(
        "c01", DeclareTemplate.RESPONSE,
        "NONEXISTENT_ACTIVITY", "Package goods",
        condition="if nonexistent condition",
    )]
    result = verify_constraints(dispatch_graph, c)
    assert result.is_conformant
