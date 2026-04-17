"""Tests for Module B — Formalizer Agent (TASK_04).

Tests split into two groups:
  - Unit tests (no API key required): helpers, validation, formatting.
  - Integration tests (require GOOGLE_API_KEY): marked @pytest.mark.integration.

Run only unit tests:
    uv run --extra dev pytest tests/test_formalizer.py -m "not integration"

Run all (including integration):
    uv run --extra dev pytest tests/test_formalizer.py
"""

import pytest

from src.models import (
    BPMNEdge,
    BPMNGraph,
    BPMNNode,
    BPMNNodeType,
    DeclareConstraint,
    DeclareTemplate,
    Mapping,
)
from src.module_a.bpmn_parser import parse_bpmn
from src.module_a.text_loader import load_text
from src.module_b.formalizer_agent import (
    ConstraintList,
    _format_activity_names,
    _format_full_text,
    _format_mappings,
    _validate_constraints,
    extract_constraints,
)

# ── File paths ────────────────────────────────────────────────────────────────

DISPATCH_BPMN = "data/dispatch/Dispatch-of-goods.bpmn"
DISPATCH_TXT = "data/dispatch/DispatchDescription.txt"

# ── Quota-error detection (mirrors test_mapper.py) ────────────────────────────

QUOTA_KEYWORDS = ("429", "resource_exhausted", "quota", "rate limit")


def _is_quota_error(exc: Exception) -> bool:
    """Return True if the exception indicates an API quota / rate-limit error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in QUOTA_KEYWORDS)


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def dispatch_graph() -> BPMNGraph:
    """Parsed Dispatch-of-goods BPMNGraph."""
    return parse_bpmn(DISPATCH_BPMN)


@pytest.fixture
def dispatch_fragments():
    """Loaded Dispatch text fragments."""
    return load_text(DISPATCH_TXT)


@pytest.fixture
def minimal_graph() -> BPMNGraph:
    """Tiny BPMNGraph with 3 named tasks for unit testing."""
    return BPMNGraph(
        process_name="Test Process",
        nodes=[
            BPMNNode(id="T_A", type=BPMNNodeType.TASK, name="Pack goods", lane="Warehouse"),
            BPMNNode(id="T_B", type=BPMNNodeType.TASK, name="Ship goods", lane="Logistics"),
            BPMNNode(id="T_C", type=BPMNNodeType.TASK, name="Close case", lane=""),
            BPMNNode(id="GW_1", type=BPMNNodeType.EXCLUSIVE_GATEWAY, name="", lane=""),
        ],
        edges=[BPMNEdge(id="F1", source="T_A", target="T_B")],
        lanes={"Warehouse": ["T_A"], "Logistics": ["T_B"]},
    )


@pytest.fixture
def sample_mappings() -> list[Mapping]:
    """A small set of pre-made mappings for unit-testing."""
    return [
        Mapping(
            fragment_id="frag_01",
            fragment_text="pack the goods",
            node_id="T_A",
            node_name="Pack goods",
            confidence=0.95,
        ),
        Mapping(
            fragment_id="frag_02",
            fragment_text="ship the goods",
            node_id="T_B",
            node_name="Ship goods",
            confidence=0.9,
        ),
        Mapping(
            fragment_id="frag_03",
            fragment_text="something unrelated",
            node_id="NONE",
            node_name="",
            confidence=0.0,
        ),
    ]


# ── Unit tests: _format_activity_names ───────────────────────────────────────


def test_format_activity_names_includes_named_nodes(minimal_graph):
    """All named nodes must appear in the activity list."""
    result = _format_activity_names(minimal_graph)
    assert "Pack goods" in result
    assert "Ship goods" in result
    assert "Close case" in result


def test_format_activity_names_excludes_unnamed_nodes(minimal_graph):
    """Unnamed gateway nodes must NOT appear in the activity list."""
    result = _format_activity_names(minimal_graph)
    # The gateway has name="" — should not produce an empty "- " line
    lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
    assert all(ln != "-" for ln in lines), f"Empty name found in: {lines}"


def test_format_activity_names_full_graph(dispatch_graph):
    """All 7 task names from Dispatch must appear in the formatted list."""
    result = _format_activity_names(dispatch_graph)
    for name in [
        "Clarify shipment method",
        "Get 3 offers from logistic companies",
        "Select logistic company and place order",
        "Write package label",
        "Insure parcel",
        "Package goods",
        "Prepare for picking up goods",
    ]:
        assert name in result, f"Expected task name {name!r} in activity list"


# ── Unit tests: _format_full_text ─────────────────────────────────────────────


def test_format_full_text_joins_fragments(dispatch_fragments):
    """Full text must join all fragment texts into a single string."""
    result = _format_full_text(dispatch_fragments)
    for f in dispatch_fragments:
        assert f.text in result


def test_format_full_text_ends_with_period(dispatch_fragments):
    """Reconstructed text should end with a period."""
    result = _format_full_text(dispatch_fragments)
    assert result.endswith(".")


def test_format_full_text_no_empty_sentences():
    """Empty fragment list must produce a single period (empty text)."""
    from src.models import TextFragment
    result = _format_full_text([TextFragment(id="frag_01", text="", sentence_index=0)])
    # Joining one empty string with "." separator + trailing "." → "."
    assert "." in result


# ── Unit tests: _format_mappings ─────────────────────────────────────────────


def test_format_mappings_includes_confirmed(sample_mappings):
    """Mappings with non-NONE node_id must appear in the output."""
    result = _format_mappings(sample_mappings)
    assert "Pack goods" in result
    assert "Ship goods" in result


def test_format_mappings_excludes_none(sample_mappings):
    """Mappings with node_id='NONE' must NOT appear in the mapping table."""
    result = _format_mappings(sample_mappings)
    assert "something unrelated" not in result


def test_format_mappings_empty():
    """Empty mapping list must return the no-mappings placeholder."""
    result = _format_mappings([])
    assert "no mappings" in result.lower()


# ── Unit tests: _validate_constraints ────────────────────────────────────────


def _make_constraint(
    cid: str,
    template: DeclareTemplate,
    a: str,
    b: str = "",
) -> DeclareConstraint:
    return DeclareConstraint(
        id=cid, template=template, activity_a=a, activity_b=b,
        source_text="test text", condition="",
    )


def test_validate_constraints_valid(minimal_graph):
    """Valid constraints must pass through and keep correct IDs."""
    constraints = [
        _make_constraint("constr_01", DeclareTemplate.EXISTENCE, "Pack goods"),
        _make_constraint("constr_02", DeclareTemplate.RESPONSE, "Pack goods", "Ship goods"),
    ]
    result = _validate_constraints(constraints, minimal_graph)
    assert len(result) == 2
    assert result[0].id == "constr_01"
    assert result[1].id == "constr_02"


def test_validate_constraints_invalid_a_discarded(minimal_graph):
    """A constraint with an unknown activity_a must be discarded."""
    bad = [_make_constraint("constr_01", DeclareTemplate.EXISTENCE, "Unknown Activity")]
    result = _validate_constraints(bad, minimal_graph)
    assert result == []


def test_validate_constraints_invalid_b_discarded(minimal_graph):
    """A constraint with an unknown activity_b must be discarded."""
    bad = [
        _make_constraint("constr_01", DeclareTemplate.RESPONSE, "Pack goods", "Ghost Task")
    ]
    result = _validate_constraints(bad, minimal_graph)
    assert result == []


def test_validate_constraints_reindexes_after_filter(minimal_graph):
    """IDs must be re-indexed sequentially after invalid constraints are removed."""
    constraints = [
        _make_constraint("constr_01", DeclareTemplate.EXISTENCE, "Unknown"),   # removed
        _make_constraint("constr_02", DeclareTemplate.EXISTENCE, "Pack goods"),  # kept → 01
        _make_constraint("constr_03", DeclareTemplate.EXISTENCE, "Ship goods"),  # kept → 02
    ]
    result = _validate_constraints(constraints, minimal_graph)
    assert len(result) == 2
    assert result[0].id == "constr_01"
    assert result[1].id == "constr_02"


def test_validate_constraints_empty_b_is_ok(minimal_graph):
    """Unary constraints with activity_b='' must pass validation."""
    c = [_make_constraint("constr_01", DeclareTemplate.EXISTENCE, "Pack goods", "")]
    result = _validate_constraints(c, minimal_graph)
    assert len(result) == 1


# ── Integration tests (require GOOGLE_API_KEY) ────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_constraints_minimum_count(dispatch_graph, dispatch_fragments):
    """Must return at least 3 constraints for the Dispatch process."""
    try:
        constraints = await extract_constraints(dispatch_graph, dispatch_fragments, [])
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    assert len(constraints) >= 3, (
        f"Expected ≥3 constraints, got {len(constraints)}: "
        + str([(c.id, c.template.value, c.activity_a) for c in constraints])
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_constraints_valid_activity_names(dispatch_graph, dispatch_fragments):
    """All activity_a / activity_b must be valid BPMN node names."""
    try:
        constraints = await extract_constraints(dispatch_graph, dispatch_fragments, [])
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    valid_names = {n.name for n in dispatch_graph.nodes if n.name.strip()}
    for c in constraints:
        assert c.activity_a in valid_names, (
            f"Constraint {c.id}: activity_a={c.activity_a!r} not in BPMN graph"
        )
        if c.activity_b:
            assert c.activity_b in valid_names, (
                f"Constraint {c.id}: activity_b={c.activity_b!r} not in BPMN graph"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_constraints_has_response(dispatch_graph, dispatch_fragments):
    """At least one RESPONSE constraint must be identified."""
    try:
        constraints = await extract_constraints(dispatch_graph, dispatch_fragments, [])
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    responses = [c for c in constraints if c.template == DeclareTemplate.RESPONSE]
    assert len(responses) >= 1, "Expected at least one response constraint"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_constraints_ids_sequential(dispatch_graph, dispatch_fragments):
    """Constraint IDs must be sequential: constr_01, constr_02, …"""
    try:
        constraints = await extract_constraints(dispatch_graph, dispatch_fragments, [])
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    for i, c in enumerate(constraints, start=1):
        assert c.id == f"constr_{i:02d}", (
            f"Expected constr_{i:02d}, got {c.id!r}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_constraints_source_text_nonempty(dispatch_graph, dispatch_fragments):
    """Every constraint must cite a non-empty source_text."""
    try:
        constraints = await extract_constraints(dispatch_graph, dispatch_fragments, [])
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"API quota exceeded: {exc}")
        raise
    for c in constraints:
        assert c.source_text.strip() != "", (
            f"Constraint {c.id} has empty source_text"
        )
