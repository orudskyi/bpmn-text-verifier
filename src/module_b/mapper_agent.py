"""Module B — Mapper Agent (Agent 1).

Uses Gemini LLM to find semantic correspondences between text fragments
extracted from a process description and BPMN flow nodes.

A single LLM call receives the full context (all fragments + all nodes) to
allow the model to disambiguate using global context and lane/role information.

Typical usage::

    import asyncio
    from src.module_a.bpmn_parser import parse_bpmn
    from src.module_a.text_loader import load_text
    from src.module_b.mapper_agent import map_text_to_bpmn

    graph     = parse_bpmn("data/dispatch/Dispatch-of-goods.bpmn")
    fragments = load_text("data/dispatch/DispatchDescription.txt")
    mappings  = asyncio.run(map_text_to_bpmn(graph, fragments))
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.config import settings
from src.models import BPMNGraph, Mapping, TextFragment

logger = logging.getLogger(__name__)

# ── Structured-output wrapper ─────────────────────────────────────────────────


class MappingList(BaseModel):
    """Wrapper so the LLM can return a typed list of Mapping objects."""

    mappings: list[Mapping]


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in Business Process Model and Notation (BPMN).
Your task: match each text fragment to the BPMN element it describes.

Rules:
1. Each text fragment maps to exactly ONE BPMN node (task, event, or gateway).
2. Some fragments may describe conditions (gateway labels) rather than tasks — \
map them to the corresponding gateway node.
3. Some fragments may not map to any node — assign node_id = "NONE" and confidence = 0.0.
4. Confidence scale: 1.0 = certain match, 0.5 = likely but ambiguous, 0.0 = no match.
5. Use semantic similarity, not just keyword overlap.
   Example: "the secretary clarifies who will do the shipping" → "Clarify shipment method"
6. Use lane/role information to disambiguate when the same action could belong \
to different participants.
   Example: "logistics department head insures the parcel" → look at the Logistics lane.\
"""

_USER_PROMPT = """\
## BPMN Nodes:
{nodes_formatted}

## Lanes (swim-lane → node IDs):
{lanes_formatted}

## Text Fragments to map:
{fragments_formatted}

Return a JSON object with a single key "mappings" containing an array.
Each element must have exactly these fields:
  - fragment_id   : the fragment ID string (e.g. "frag_01")
  - fragment_text : the original fragment text
  - node_id       : BPMN element ID from the list above, or "NONE"
  - node_name     : name of the matched BPMN element, or "" for NONE
  - confidence    : float between 0.0 and 1.0\
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _USER_PROMPT)]
)


# ── Formatting helpers ────────────────────────────────────────────────────────


def _format_nodes(graph: BPMNGraph) -> str:
    """Render BPMN nodes as a numbered list for the prompt.

    Args:
        graph: Parsed BPMN graph.

    Returns:
        Multi-line string suitable for insertion into the prompt.
    """
    lines: list[str] = []
    for node in graph.nodes:
        lane_hint = f"  [lane: {node.lane}]" if node.lane else ""
        lines.append(f"- id={node.id!r}  type={node.type.value}  name={node.name!r}{lane_hint}")
    return "\n".join(lines)


def _format_lanes(graph: BPMNGraph) -> str:
    """Render lane assignments for the prompt.

    Args:
        graph: Parsed BPMN graph.

    Returns:
        Multi-line string or ``"(no lanes)"`` for single-pool processes.
    """
    if not graph.lanes:
        return "(no lanes — single participant process)"
    lines: list[str] = []
    for lane_name, node_ids in graph.lanes.items():
        lines.append(f"- {lane_name!r}: {node_ids}")
    return "\n".join(lines)


def _format_fragments(fragments: list[TextFragment]) -> str:
    """Render text fragments as a numbered list for the prompt.

    Args:
        fragments: Fragments produced by the text loader.

    Returns:
        Multi-line string suitable for insertion into the prompt.
    """
    lines = [f'- id={f.id!r}  text={f.text!r}' for f in fragments]
    return "\n".join(lines)


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_mappings(
    mappings: list[Mapping],
    graph: BPMNGraph,
) -> list[Mapping]:
    """Cross-check returned mappings against the actual graph node IDs.

    Mappings whose ``node_id`` is not ``"NONE"`` and does not appear in the
    graph receive a warning and have their confidence set to 0.0.

    Args:
        mappings: Raw mappings returned by the LLM chain.
        graph: The source BPMN graph used for validation.

    Returns:
        The same list with invalid node references corrected.
    """
    known_ids = {n.id for n in graph.nodes} | {"NONE"}
    validated: list[Mapping] = []
    for m in mappings:
        if m.node_id not in known_ids:
            logger.warning(
                "Mapping for %r references unknown node_id %r — setting confidence=0.0",
                m.fragment_id,
                m.node_id,
            )
            m = m.model_copy(update={"confidence": 0.0})
        validated.append(m)
    return validated


# ── Public API ────────────────────────────────────────────────────────────────


async def map_text_to_bpmn(
    graph: BPMNGraph,
    fragments: list[TextFragment],
) -> list[Mapping]:
    """Map text fragments to BPMN nodes using Gemini LLM.

    Sends a single LLM call containing all fragments and all nodes so the
    model can use global context for disambiguation.  Retries up to
    ``settings.llm_max_retries`` times on failure, increasing temperature
    slightly on each retry.

    Args:
        graph: Parsed BPMN process graph (output of Module A).
        fragments: List of text fragments from the process description
            (output of Module A).

    Returns:
        List of :class:`~src.models.Mapping` objects linking each fragment
        to a BPMN node.  Fragments with no clear match receive
        ``node_id="NONE"`` and ``confidence=0.0``.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    settings.validate()

    nodes_fmt = _format_nodes(graph)
    lanes_fmt = _format_lanes(graph)
    frags_fmt = _format_fragments(fragments)

    last_error: Exception | None = None

    for attempt in range(settings.llm_max_retries + 1):
        # Increase temperature slightly on retries to escape stuck outputs.
        temperature = settings.gemini_temperature if attempt == 0 else 0.1 * attempt
        logger.info(
            "Mapper attempt %d/%d (temperature=%.1f) for process %r",
            attempt + 1,
            settings.llm_max_retries + 1,
            temperature,
            graph.process_name,
        )

        try:
            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                temperature=temperature,
                google_api_key=settings.google_api_key,
            )
            chain = _PROMPT | llm.with_structured_output(MappingList)
            result: MappingList = await chain.ainvoke(
                {
                    "nodes_formatted": nodes_fmt,
                    "lanes_formatted": lanes_fmt,
                    "fragments_formatted": frags_fmt,
                }
            )
            mappings = _validate_mappings(result.mappings, graph)
            logger.info(
                "Mapper returned %d mappings (%d high-confidence ≥0.7)",
                len(mappings),
                sum(1 for m in mappings if m.confidence >= 0.7),
            )
            return mappings

        except Exception as exc:  # noqa: BLE001
            logger.warning("Mapper attempt %d failed: %s", attempt + 1, exc)
            last_error = exc

    raise RuntimeError(
        f"Mapper agent failed after {settings.llm_max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
