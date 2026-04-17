"""Tests for TASK_07 — LangGraph Orchestrator.

Tests split into:
  - Unit tests (no API): pipeline build, graph structure, initial_state,
    synchronous node logic (parse, load, verify, build_report).
  - Integration tests (require GOOGLE_API_KEY): @pytest.mark.integration.

Run only unit tests:
    python -m pytest tests/test_orchestrator.py -m "not integration"
"""

import pytest

from src.models import (
    BPMNGraph,
    ConformanceReport,
    DeclareConstraint,
    DeclareTemplate,
    VerificationResult,
    Violation,
)
from src.module_a.bpmn_parser import parse_bpmn
from src.orchestrator import (
    PipelineState,
    _build_report_node,
    _load_text_node,
    _parse_bpmn_node,
    _should_explain,
    _verify_node,
    build_pipeline,
    initial_state,
)

# ── File paths ────────────────────────────────────────────────────────────────

DISPATCH_BPMN = "data/dispatch/Dispatch-of-goods.bpmn"
DISPATCH_TXT  = "data/dispatch/DispatchDescription.txt"
RECOURSE_BPMN = "data/recourse/Recourse.bpmn"
RECOURSE_TXT  = "data/recourse/RecourseDesription.txt"   # intentional typo in filename

# ── Quota-error helper ────────────────────────────────────────────────────────

_QUOTA_KW = ("429", "resource_exhausted", "quota", "rate limit")


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _QUOTA_KW)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — no API calls
# ═══════════════════════════════════════════════════════════════════════════════


# ── initial_state ─────────────────────────────────────────────────────────────


def test_initial_state_contains_paths():
    """initial_state must set both file paths."""
    s = initial_state("a.bpmn", "b.txt")
    assert s["bpmn_file_path"] == "a.bpmn"
    assert s["text_file_path"] == "b.txt"


def test_initial_state_defaults_empty():
    """initial_state must initialise all list/None fields to empty."""
    s = initial_state("a.bpmn", "b.txt")
    assert s["graph"] is None
    assert s["fragments"] == []
    assert s["mappings"] == []
    assert s["constraints"] == []
    assert s["verification"] is None
    assert s["report"] is None
    assert s["errors"] == []


# ── build_pipeline ────────────────────────────────────────────────────────────


def test_build_pipeline_returns_compiled():
    """build_pipeline must return a compiled graph object."""
    pipeline = build_pipeline()
    # LangGraph compiled graphs expose ainvoke / invoke
    assert hasattr(pipeline, "ainvoke")
    assert hasattr(pipeline, "invoke")


# ── _parse_bpmn_node ──────────────────────────────────────────────────────────


def test_parse_bpmn_node_success():
    """Successful parse must populate the 'graph' key."""
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    result = _parse_bpmn_node(s)
    assert "graph" in result
    assert isinstance(result["graph"], BPMNGraph)
    assert result["graph"].process_name != ""


def test_parse_bpmn_node_failure():
    """Non-existent file must add an error string, not raise."""
    s = initial_state("nonexistent.bpmn", "dummy.txt")
    result = _parse_bpmn_node(s)
    assert "errors" in result
    assert len(result["errors"]) >= 1
    assert "BPMN parsing failed" in result["errors"][0]


# ── _load_text_node ───────────────────────────────────────────────────────────


def test_load_text_node_success():
    """Successful load must populate the 'fragments' key."""
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    result = _load_text_node(s)
    assert "fragments" in result
    assert len(result["fragments"]) >= 5


def test_load_text_node_failure():
    """Non-existent text file must add an error string, not raise."""
    s = initial_state(DISPATCH_BPMN, "nonexistent.txt")
    result = _load_text_node(s)
    assert "errors" in result
    assert "Text loading failed" in result["errors"][0]


# ── _verify_node ──────────────────────────────────────────────────────────────


def test_verify_node_no_constraints():
    """With no constraints, verify must return a conformant result."""
    graph = parse_bpmn(DISPATCH_BPMN)
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["graph"] = graph
    s["constraints"] = []
    result = _verify_node(s)
    assert "verification" in result
    assert result["verification"].is_conformant is True
    assert result["verification"].total_constraints == 0


def test_verify_node_missing_graph():
    """If graph is None, verify must return an empty conformant result."""
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["graph"] = None
    result = _verify_node(s)
    assert "verification" in result
    assert result["verification"].is_conformant is True


def test_verify_node_detects_violation():
    """Verify must detect a known violation in the Dispatch process."""
    from src.models import DeclareConstraint, DeclareTemplate
    graph = parse_bpmn(DISPATCH_BPMN)
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["graph"] = graph
    s["constraints"] = [
        DeclareConstraint(
            id="c01",
            template=DeclareTemplate.EXISTENCE,
            activity_a="Select logistic company and place order",
            source_text="test",
        )
    ]
    result = _verify_node(s)
    assert result["verification"].is_conformant is False
    assert result["verification"].violated == 1


# ── _should_explain ────────────────────────────────────────────────────────────


def test_should_explain_with_violations():
    """Router must return 'explain' when there are violations."""
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["verification"] = VerificationResult(
        is_conformant=False, total_constraints=1,
        satisfied=0, violated=1,
        violations=[Violation(
            constraint_id="c01",
            constraint_description="existence(X)",
            trace=["A", "B"],
            explanation="",
        )],
    )
    assert _should_explain(s) == "explain"


def test_should_explain_without_violations():
    """Router must return 'build_report' when there are no violations."""
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["verification"] = VerificationResult(
        is_conformant=True, total_constraints=2,
        satisfied=2, violated=0, violations=[],
    )
    assert _should_explain(s) == "build_report"


def test_should_explain_no_verification():
    """Router must return 'build_report' if verification is None."""
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["verification"] = None
    assert _should_explain(s) == "build_report"


# ── _build_report_node ────────────────────────────────────────────────────────


def test_build_report_node_structure():
    """Built report must have correct process name, file paths, and counts."""
    graph = parse_bpmn(DISPATCH_BPMN)
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["graph"] = graph
    s["mappings"] = []
    s["constraints"] = []
    s["verification"] = VerificationResult(
        is_conformant=True, total_constraints=0,
        satisfied=0, violated=0, violations=[],
    )
    result = _build_report_node(s)
    report: ConformanceReport = result["report"]

    assert report.process_name == graph.process_name
    assert report.bpmn_file == DISPATCH_BPMN
    assert report.text_file == DISPATCH_TXT
    assert report.graph_summary["nodes"] == len(graph.nodes)
    assert report.graph_summary["edges"] == len(graph.edges)
    assert report.graph_summary["tasks"] >= 7


def test_build_report_node_is_pydantic():
    """Built report must be a valid ConformanceReport Pydantic model."""
    graph = parse_bpmn(DISPATCH_BPMN)
    s = initial_state(DISPATCH_BPMN, DISPATCH_TXT)
    s["graph"] = graph
    s["mappings"] = []
    s["constraints"] = []
    s["verification"] = VerificationResult(
        is_conformant=True, total_constraints=0,
        satisfied=0, violated=0, violations=[],
    )
    result = _build_report_node(s)
    report = result["report"]
    # Should be serialisable without errors
    json_str = report.model_dump_json()
    assert "process_name" in json_str


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (require GOOGLE_API_KEY)
# ═══════════════════════════════════════════════════════════════════════════════


def _empty_pipeline_state(bpmn: str, txt: str) -> dict:
    return initial_state(bpmn, txt)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_dispatch():
    """Full pipeline must complete without errors and produce a valid report."""
    try:
        pipeline = build_pipeline()
        result = await pipeline.ainvoke(_empty_pipeline_state(DISPATCH_BPMN, DISPATCH_TXT))
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise

    assert not result.get("errors"), f"Pipeline errors: {result['errors']}"
    report = result.get("report")
    assert report is not None
    assert report.process_name != ""
    assert len(report.mappings) > 0
    assert len(report.constraints) > 0
    assert report.verification is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_recourse():
    """Full pipeline must work for the Recourse process too."""
    try:
        pipeline = build_pipeline()
        result = await pipeline.ainvoke(_empty_pipeline_state(RECOURSE_BPMN, RECOURSE_TXT))
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise

    assert not result.get("errors"), f"Pipeline errors: {result['errors']}"
    report = result.get("report")
    assert report is not None
    assert len(report.mappings) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_report_is_json_serialisable():
    """The final report must be serialisable to valid JSON."""
    import json
    try:
        pipeline = build_pipeline()
        result = await pipeline.ainvoke(_empty_pipeline_state(DISPATCH_BPMN, DISPATCH_TXT))
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise

    report = result.get("report")
    if report:
        json_str = report.model_dump_json()
        parsed = json.loads(json_str)
        assert "process_name" in parsed
        assert "verification" in parsed
