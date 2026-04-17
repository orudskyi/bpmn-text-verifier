"""LangGraph Orchestrator for the BPMN-Text Conformance Checker.

Wires all pipeline modules together into a single directed acyclic graph:

    parse_bpmn ──▶ load_text ──▶ map_text ──▶ formalize
         ──▶ verify ──[violations?]──▶ explain ──▶ build_report

Each node in the graph corresponds to one module or agent:

* ``parse_bpmn``   — Module A: BPMN XML parser
* ``load_text``    — Module A: text loader / clause splitter
* ``map_text``     — Module B, Agent 1: text-to-BPMN mapper (LLM)
* ``formalize``    — Module B, Agent 2: DECLARE constraint extractor (LLM)
* ``verify``       — Module C: symbolic verifier (no LLM)
* ``explain``      — Module D, Agent 3: violation explainer (LLM, conditional)
* ``build_report`` — assembles the final ConformanceReport

Typical usage::

    import asyncio
    from src.orchestrator import build_pipeline

    pipeline = build_pipeline()
    result = asyncio.run(pipeline.ainvoke({
        "bpmn_file_path": "data/dispatch/Dispatch-of-goods.bpmn",
        "text_file_path": "data/dispatch/DispatchDescription.txt",
        "fragments": [], "mappings": [], "constraints": [],
        "graph": None, "verification": None, "report": None, "errors": [],
    }))
    print(result["report"].model_dump_json(indent=2))
"""

import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.models import (
    BPMNGraph,
    BPMNNodeType,
    ConformanceReport,
    DeclareConstraint,
    Mapping,
    TextFragment,
    VerificationResult,
)
from src.module_a.bpmn_parser import parse_bpmn
from src.module_a.text_loader import load_text
from src.module_b.formalizer_agent import extract_constraints
from src.module_b.mapper_agent import map_text_to_bpmn
from src.module_c.verifier import verify_constraints
from src.module_d.explainer_agent import explain_violations

logger = logging.getLogger(__name__)


# ── Pipeline state ────────────────────────────────────────────────────────────


class PipelineState(TypedDict):
    """Mutable state shared across all LangGraph pipeline nodes.

    Each node reads fields it needs and writes back only the fields it
    produces.  LangGraph merges the returned dicts with the existing state.

    Attributes:
        bpmn_file_path: Path to the BPMN 2.0 XML file.
        text_file_path: Path to the natural-language process description.
        graph: Parsed BPMNGraph (available from step 1 onwards).
        fragments: Text fragments (available from step 2 onwards).
        mappings: Text-to-BPMN mappings from Agent 1.
        constraints: DECLARE constraints from Agent 2.
        verification: Verification result from Module C.
        report: Final ConformanceReport (populated at the end).
        errors: Accumulated error messages; pipeline continues on soft errors.
    """

    bpmn_file_path: str
    text_file_path: str
    graph: Optional[BPMNGraph]
    fragments: list[TextFragment]
    mappings: list[Mapping]
    constraints: list[DeclareConstraint]
    verification: Optional[VerificationResult]
    report: Optional[ConformanceReport]
    errors: list[str]


# ── Node functions ────────────────────────────────────────────────────────────


def _parse_bpmn_node(state: PipelineState) -> dict:
    """Module A: Parse BPMN XML into a BPMNGraph.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with ``graph`` key on success, or ``errors`` on failure.
    """
    try:
        graph = parse_bpmn(state["bpmn_file_path"])
        logger.info("[parse_bpmn] Parsed %r: %d nodes, %d edges",
                    graph.process_name, len(graph.nodes), len(graph.edges))
        return {"graph": graph}
    except Exception as exc:
        msg = f"BPMN parsing failed: {exc}"
        logger.error("[parse_bpmn] %s", msg)
        return {"errors": state.get("errors", []) + [msg]}


def _load_text_node(state: PipelineState) -> dict:
    """Module A: Load text description and split into semantic fragments.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with ``fragments`` key on success, or ``errors`` on failure.
    """
    try:
        fragments = load_text(state["text_file_path"])
        logger.info("[load_text] Loaded %d fragment(s) from %r",
                    len(fragments), state["text_file_path"])
        return {"fragments": fragments}
    except Exception as exc:
        msg = f"Text loading failed: {exc}"
        logger.error("[load_text] %s", msg)
        return {"errors": state.get("errors", []) + [msg]}


async def _map_text_node(state: PipelineState) -> dict:
    """Module B, Agent 1: Map text fragments to BPMN nodes via Gemini.

    Skipped (returns empty mappings) if the graph or fragments are absent.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with ``mappings`` key on success, or ``errors`` on failure.
    """
    if not state.get("graph") or not state.get("fragments"):
        logger.warning("[map_text] Skipping — missing graph or fragments")
        return {"mappings": []}
    try:
        mappings = await map_text_to_bpmn(state["graph"], state["fragments"])
        logger.info("[map_text] Produced %d mapping(s)", len(mappings))
        return {"mappings": mappings}
    except Exception as exc:
        msg = f"Mapping failed: {exc}"
        logger.error("[map_text] %s", msg)
        return {"errors": state.get("errors", []) + [msg], "mappings": []}


async def _formalize_node(state: PipelineState) -> dict:
    """Module B, Agent 2: Extract DECLARE constraints via Gemini.

    Skipped (returns empty constraints) if the graph or fragments are absent.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with ``constraints`` key on success, or ``errors`` on failure.
    """
    if not state.get("graph") or not state.get("fragments"):
        logger.warning("[formalize] Skipping — missing graph or fragments")
        return {"constraints": []}
    try:
        constraints = await extract_constraints(
            state["graph"], state["fragments"], state.get("mappings", [])
        )
        logger.info("[formalize] Extracted %d constraint(s)", len(constraints))
        return {"constraints": constraints}
    except Exception as exc:
        msg = f"Formalization failed: {exc}"
        logger.error("[formalize] %s", msg)
        return {"errors": state.get("errors", []) + [msg], "constraints": []}


def _verify_node(state: PipelineState) -> dict:
    """Module C: Symbolic verification (no LLM).

    Skipped (returns conformant empty result) if the graph or constraints
    are absent.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with ``verification`` key.
    """
    if not state.get("graph"):
        logger.warning("[verify] Skipping — missing graph")
        return {
            "verification": VerificationResult(
                is_conformant=True, total_constraints=0,
                satisfied=0, violated=0, violations=[]
            )
        }
    try:
        result = verify_constraints(
            state["graph"],
            state.get("constraints", []),
            bpmn_file_path=state["bpmn_file_path"],
        )
        logger.info(
            "[verify] %s — %d satisfied, %d violated",
            "CONFORMANT" if result.is_conformant else "NON-CONFORMANT",
            result.satisfied,
            result.violated,
        )
        return {"verification": result}
    except Exception as exc:
        msg = f"Verification failed: {exc}"
        logger.error("[verify] %s", msg)
        return {"errors": state.get("errors", []) + [msg]}


async def _explain_node(state: PipelineState) -> dict:
    """Module D, Agent 3: Generate natural-language violation explanations.

    Only called when there are violations.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with updated ``verification`` (explanations filled in).
    """
    try:
        updated = await explain_violations(
            state["verification"], state.get("constraints", []), state["graph"]
        )
        logger.info("[explain] Explanations generated for %d violation(s)",
                    len(updated.violations))
        return {"verification": updated}
    except Exception as exc:
        msg = f"Explanation failed: {exc}"
        logger.error("[explain] %s", msg)
        return {"errors": state.get("errors", []) + [msg]}


def _build_report_node(state: PipelineState) -> dict:
    """Assemble the final ConformanceReport from accumulated state.

    Args:
        state: Current pipeline state (all fields should be populated).

    Returns:
        Dict with ``report`` key.
    """
    graph = state["graph"]
    verification = state.get("verification") or VerificationResult(
        is_conformant=True, total_constraints=0,
        satisfied=0, violated=0, violations=[]
    )

    report = ConformanceReport(
        process_name=graph.process_name,
        bpmn_file=state["bpmn_file_path"],
        text_file=state["text_file_path"],
        graph_summary={
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "lanes": len(graph.lanes),
            "tasks": sum(1 for n in graph.nodes if n.type == BPMNNodeType.TASK),
        },
        mappings=state.get("mappings", []),
        constraints=state.get("constraints", []),
        verification=verification,
    )
    logger.info(
        "[build_report] Report assembled for %r — conformant=%s",
        report.process_name,
        report.verification.is_conformant,
    )
    return {"report": report}


# ── Routing ───────────────────────────────────────────────────────────────────


def _should_explain(state: PipelineState) -> str:
    """Conditional routing: send to 'explain' only when violations exist.

    Args:
        state: Current pipeline state.

    Returns:
        ``"explain"`` if there are violated constraints, else ``"build_report"``.
    """
    v = state.get("verification")
    if v and v.violated > 0:
        logger.debug("[routing] %d violation(s) → explain", v.violated)
        return "explain"
    logger.debug("[routing] No violations → build_report")
    return "build_report"


# ── Public factory ────────────────────────────────────────────────────────────


def build_pipeline():
    """Build and compile the LangGraph conformance-checking pipeline.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to be invoked with
        :py:meth:`ainvoke` or :py:meth:`invoke`.

    Graph topology::

        START
          └── parse_bpmn
                └── load_text
                      └── map_text
                            └── formalize
                                  └── verify
                                        ├─[violations]──▶ explain ──▶ build_report
                                        └─[none]────────────────────▶ build_report
                                                                           └── END
    """
    workflow: StateGraph = StateGraph(PipelineState)

    # Register nodes
    workflow.add_node("parse_bpmn",   _parse_bpmn_node)
    workflow.add_node("load_text",    _load_text_node)
    workflow.add_node("map_text",     _map_text_node)
    workflow.add_node("formalize",    _formalize_node)
    workflow.add_node("verify",       _verify_node)
    workflow.add_node("explain",      _explain_node)
    workflow.add_node("build_report", _build_report_node)

    # Sequential edges
    workflow.add_edge(START,        "parse_bpmn")
    workflow.add_edge("parse_bpmn", "load_text")
    workflow.add_edge("load_text",  "map_text")
    workflow.add_edge("map_text",   "formalize")
    workflow.add_edge("formalize",  "verify")

    # Conditional fork: explain only when violations exist
    workflow.add_conditional_edges(
        "verify",
        _should_explain,
        {"explain": "explain", "build_report": "build_report"},
    )

    workflow.add_edge("explain",      "build_report")
    workflow.add_edge("build_report", END)

    return workflow.compile()


# ── Helper: initial state factory ─────────────────────────────────────────────


def initial_state(bpmn_file_path: str, text_file_path: str) -> PipelineState:
    """Create a clean initial state dict for a pipeline run.

    Args:
        bpmn_file_path: Path to the BPMN 2.0 XML file.
        text_file_path: Path to the natural-language process description.

    Returns:
        A fully-initialised :class:`PipelineState` dict.
    """
    return PipelineState(
        bpmn_file_path=bpmn_file_path,
        text_file_path=text_file_path,
        graph=None,
        fragments=[],
        mappings=[],
        constraints=[],
        verification=None,
        report=None,
        errors=[],
    )
