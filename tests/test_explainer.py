"""Tests for Module D — Explainer Agent (TASK_06).

Tests split into two groups:
  - Unit tests (no API): helpers, formatting, fallback logic.
  - Integration tests (require GOOGLE_API_KEY): @pytest.mark.integration.

Run only unit tests:
    python -m pytest tests/test_explainer.py -m "not integration"
"""

import pytest

from src.models import (
    BPMNGraph,
    BPMNNode,
    BPMNNodeType,
    DeclareConstraint,
    DeclareTemplate,
    VerificationResult,
    Violation,
)
from src.module_a.bpmn_parser import parse_bpmn
from src.module_d.explainer_agent import (
    ExplanationList,
    _constraint_by_id,
    _fallback_explanation,
    _format_violations,
    explain_violations,
)

# ── File paths ────────────────────────────────────────────────────────────────

DISPATCH_BPMN = "data/dispatch/Dispatch-of-goods.bpmn"

# ── Quota-error detection ─────────────────────────────────────────────────────

_QUOTA_KW = ("429", "resource_exhausted", "quota", "rate limit")


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _QUOTA_KW)


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def dispatch_graph() -> BPMNGraph:
    return parse_bpmn(DISPATCH_BPMN)


@pytest.fixture
def sample_violation() -> Violation:
    return Violation(
        constraint_id="constr_01",
        constraint_description="existence(Select logistic company and place order)",
        trace=[
            "Clarify shipment method",
            "Write package label",
            "Package goods",
            "Prepare for picking up goods",
        ],
        explanation="",
    )


@pytest.fixture
def sample_constraint() -> DeclareConstraint:
    return DeclareConstraint(
        id="constr_01",
        template=DeclareTemplate.EXISTENCE,
        activity_a="Select logistic company and place order",
        activity_b="",
        source_text="she selects one of them",
        condition="",
    )


@pytest.fixture
def sample_result(sample_violation) -> VerificationResult:
    return VerificationResult(
        is_conformant=False,
        total_constraints=1,
        satisfied=0,
        violated=1,
        violations=[sample_violation],
    )


@pytest.fixture
def sample_constraints(sample_constraint) -> list[DeclareConstraint]:
    return [sample_constraint]


# ── Unit tests: _constraint_by_id ─────────────────────────────────────────────


def test_constraint_by_id_indexes_correctly(sample_constraint):
    """Index must map constraint IDs to constraint objects."""
    index = _constraint_by_id([sample_constraint])
    assert "constr_01" in index
    assert index["constr_01"] is sample_constraint


def test_constraint_by_id_empty():
    """Empty input must produce an empty dict."""
    assert _constraint_by_id([]) == {}


def test_constraint_by_id_multiple():
    """Multiple constraints must all be indexed."""
    cs = [
        DeclareConstraint(id=f"c{i:02d}", template=DeclareTemplate.EXISTENCE,
                          activity_a="A", source_text="t")
        for i in range(5)
    ]
    index = _constraint_by_id(cs)
    assert len(index) == 5
    for i in range(5):
        assert f"c{i:02d}" in index


# ── Unit tests: _format_violations ────────────────────────────────────────────


def test_format_violations_contains_id(sample_violation, sample_constraint):
    """Formatted output must include the constraint ID."""
    index = {sample_constraint.id: sample_constraint}
    result = _format_violations([sample_violation], index)
    assert "constr_01" in result


def test_format_violations_contains_trace(sample_violation, sample_constraint):
    """Formatted output must include all activity names from the counter-example."""
    index = {sample_constraint.id: sample_constraint}
    result = _format_violations([sample_violation], index)
    for activity in sample_violation.trace:
        assert activity in result


def test_format_violations_contains_source_text(sample_violation, sample_constraint):
    """Formatted output must include the source_text from the constraint."""
    index = {sample_constraint.id: sample_constraint}
    result = _format_violations([sample_violation], index)
    assert "she selects one of them" in result


def test_format_violations_multiple_separated(sample_violation, sample_constraint):
    """Multiple violations must be separated by the '---' delimiter."""
    v2 = sample_violation.model_copy(update={"constraint_id": "constr_02"})
    c2 = sample_constraint.model_copy(update={"id": "constr_02"})
    index = {sample_constraint.id: sample_constraint, "constr_02": c2}
    result = _format_violations([sample_violation, v2], index)
    assert "---" in result


def test_format_violations_unknown_constraint(sample_violation):
    """Violations whose constraint is not in the index must still be formatted."""
    result = _format_violations([sample_violation], {})
    assert "constr_01" in result   # ID must appear
    assert "Clarify" in result     # trace must appear


def test_format_violations_empty_trace():
    """A violation with an empty trace must show the empty-trace placeholder."""
    v = Violation(
        constraint_id="c01",
        constraint_description="existence(A)",
        trace=[],
        explanation="",
    )
    result = _format_violations([v], {})
    assert "empty trace" in result.lower()


# ── Unit tests: _fallback_explanation ─────────────────────────────────────────


def test_fallback_includes_activity_name(sample_violation, sample_constraint):
    """Fallback must mention the constrained activity name."""
    index = {sample_constraint.id: sample_constraint}
    explanation = _fallback_explanation(sample_violation, index)
    assert "Select logistic company and place order" in explanation


def test_fallback_includes_trace_excerpt(sample_violation, sample_constraint):
    """Fallback must include at least part of the counter-example trace."""
    index = {sample_constraint.id: sample_constraint}
    explanation = _fallback_explanation(sample_violation, index)
    # At least one activity from the trace should appear
    assert any(act in explanation for act in sample_violation.trace)


def test_fallback_no_constraint_in_index(sample_violation):
    """Fallback with missing constraint must not raise, must still produce text."""
    explanation = _fallback_explanation(sample_violation, {})
    assert len(explanation) > 10
    assert "constr_01" in explanation or "existence" in explanation.lower()


def test_fallback_long_trace_truncated():
    """Fallback must truncate very long traces rather than listing all steps."""
    v = Violation(
        constraint_id="c01",
        constraint_description="existence(A)",
        trace=[f"Task_{i}" for i in range(20)],
        explanation="",
    )
    explanation = _fallback_explanation(v, {})
    # Should not dump all 20 tasks
    assert explanation.count("Task_") <= 10


# ── Unit tests: explain_violations (no violations path) ───────────────────────


@pytest.mark.asyncio
async def test_explain_violations_no_violations(dispatch_graph, sample_constraints):
    """If there are no violations, the result must be returned unchanged."""
    empty_result = VerificationResult(
        is_conformant=True,
        total_constraints=1,
        satisfied=1,
        violated=0,
        violations=[],
    )
    updated = await explain_violations(empty_result, sample_constraints, dispatch_graph)
    assert updated is empty_result   # same object — no LLM call made


# ── Integration tests (require GOOGLE_API_KEY) ────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explanation_nonempty(sample_result, sample_constraints, dispatch_graph):
    """Explanation must be a non-empty string after the LLM call."""
    try:
        updated = await explain_violations(sample_result, sample_constraints, dispatch_graph)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    assert updated.violations[0].explanation != ""
    assert len(updated.violations[0].explanation) > 20


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explanation_mentions_activity(sample_result, sample_constraints, dispatch_graph):
    """Explanation must mention the constrained activity or 'select'."""
    try:
        updated = await explain_violations(sample_result, sample_constraints, dispatch_graph)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    explanation = updated.violations[0].explanation.lower()
    assert "select" in explanation or "logistic" in explanation, (
        f"Expected activity mention in: {updated.violations[0].explanation!r}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explanation_violation_count_preserved(
    sample_result, sample_constraints, dispatch_graph
):
    """The number of violations must not change after explanation."""
    try:
        updated = await explain_violations(sample_result, sample_constraints, dispatch_graph)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    assert len(updated.violations) == len(sample_result.violations)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explanation_other_fields_unchanged(
    sample_result, sample_constraints, dispatch_graph
):
    """is_conformant, satisfied, violated must not change after explanation."""
    try:
        updated = await explain_violations(sample_result, sample_constraints, dispatch_graph)
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    assert updated.is_conformant == sample_result.is_conformant
    assert updated.satisfied == sample_result.satisfied
    assert updated.violated == sample_result.violated
