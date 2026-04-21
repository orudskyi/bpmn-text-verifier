"""Module B — Formalizer Agent (Agent 2).

Uses Gemini LLM to extract formal behavioral constraints from a natural-language
process description.  Constraints are expressed as **DECLARE templates** — a
finite, well-defined set of patterns that can be mechanically verified against a
BPMN process model by Module C.

The agent receives three inputs to provide full context:

* ``BPMNGraph`` — supplies the vocabulary of valid activity names; the LLM is
  constrained to choose from this list.
* ``list[TextFragment]`` — the original description, used to cite source text.
* ``list[Mapping]`` — text-to-BPMN groundings produced by Agent 1; used to
  anchor free-text phrases to exact BPMN node names.

Typical usage::

    import asyncio
    from src.module_a.bpmn_parser import parse_bpmn
    from src.module_a.text_loader import load_text
    from src.module_b.mapper_agent import map_text_to_bpmn
    from src.module_b.formalizer_agent import extract_constraints

    graph     = parse_bpmn("data/dispatch/Dispatch-of-goods.bpmn")
    fragments = load_text("data/dispatch/DispatchDescription.txt")
    mappings  = asyncio.run(map_text_to_bpmn(graph, fragments))
    constraints = asyncio.run(extract_constraints(graph, fragments, mappings))
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.config import settings
from src.models import (
    BPMNGraph,
    DeclareConstraint,
    DeclareTemplate,
    Mapping,
    TextFragment,
)
from src.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


# ── Structured-output wrapper ─────────────────────────────────────────────────


class ConstraintList(BaseModel):
    """Wrapper so the LLM can return a typed list of DeclareConstraint objects."""

    constraints: list[DeclareConstraint]


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in business process formalization.
Your task: read a natural-language process description and extract formal
constraints using DECLARE templates.

DECLARE templates you can use:
  existence(A)           — A must happen at least once
  absence(A)             — A must never happen
  response(A, B)         — if A occurs, B must eventually occur after it
  precedence(A, B)       — B can only occur if A has occurred before it
  succession(A, B)       — response(A,B) AND precedence(B,A) combined
  chain_response(A, B)   — if A occurs, B must occur immediately next
  chain_precedence(A, B) — B can only occur immediately after A
  not_coexistence(A, B)  — A and B cannot both occur in the same execution
  choice(A, B)           — at least one of A or B must occur
  exclusive_choice(A, B) — exactly one of A or B must occur (not both)

Rules:
1. Use ONLY activity names from the "Available BPMN Activities" list as \
activity_a and activity_b.
2. Extract ONLY constraints that are explicitly stated or strongly implied by the text.
3. Do NOT invent constraints not supported by the text.
4. For conditional constraints (e.g., "if large amounts"), record the condition \
in the condition field.
5. Each constraint must cite the source_text fragment that implies it.
6. Be conservative — it is better to return 5 correct constraints than 15 with \
false positives.
7. Use chain_response / chain_precedence sparingly — only when the text explicitly \
says "immediately after", "right after", "directly follows".
8. Assign sequential IDs: "constr_01", "constr_02", …
9. For unary templates (existence, absence), leave activity_b as an empty string "".

Typical text-to-constraint patterns:
  "X clarifies... then Y selects..."      → response(X, Y)
  "X is always done"                      → existence(X)
  "if A then B must happen"              → response(A, B)
  "B only after A"                        → precedence(A, B)
  "either A or B (but not both)"         → exclusive_choice(A, B)
  "at least one of A or B"               → choice(A, B)
  "in the meantime X can be done"        → existence(X)  [parallel, no ordering]
  "A and B happen at the same time"      → skip (co-occurrence, not a DECLARE pattern)\
"""

_USER_PROMPT = """\
## Available BPMN Activities (use ONLY these names):
{activity_names_list}

## Full Process Description:
{full_text}

## Text-to-BPMN Mappings (for reference / grounding):
{mappings_formatted}

Extract DECLARE constraints from the process description.
Return a JSON object with a single key "constraints" containing an array.
Each element must have exactly these fields:
  - id          : sequential string "constr_01", "constr_02", …
  - template    : one of the DECLARE template names listed above
  - activity_a  : BPMN activity name (must be from the Available list)
  - activity_b  : BPMN activity name (must be from the Available list), \
or "" for unary templates
  - source_text : the text fragment that implies this constraint
  - condition   : guard condition string (e.g. "if large amounts"), or "" if unconditional\
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _USER_PROMPT)]
)


# ── Formatting helpers ────────────────────────────────────────────────────────


def _format_activity_names(graph: BPMNGraph) -> str:
    """Render the list of valid BPMN activity names for the prompt.

    Only includes named nodes (non-empty name strings) so the LLM has a clean
    vocabulary to choose from.

    Args:
        graph: Parsed BPMN graph.

    Returns:
        Numbered list of activity names, one per line.
    """
    names = [n.name for n in graph.nodes if n.name.strip()]
    return "\n".join(f"  - {name}" for name in names)


def _format_full_text(fragments: list[TextFragment]) -> str:
    """Reconstruct the original process description from its fragments.

    Args:
        fragments: Text fragments produced by the text loader.

    Returns:
        Full text as a single paragraph (sentences joined by ". ").
    """
    return ". ".join(f.text for f in fragments) + "."


def _format_mappings(mappings: list[Mapping]) -> str:
    """Render the text-to-BPMN mappings as a reference table for the prompt.

    Args:
        mappings: Mappings produced by Agent 1.

    Returns:
        Multi-line string or ``"(no mappings provided)"``.
    """
    if not mappings:
        return "(no mappings provided)"
    lines = [
        f'  - {m.fragment_id}: "{m.fragment_text}" → "{m.node_name}" '
        f"(conf={m.confidence:.2f})"
        for m in mappings
        if m.node_id != "NONE"
    ]
    return "\n".join(lines) if lines else "(no confirmed mappings)"


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_constraints(
    constraints: list[DeclareConstraint],
    graph: BPMNGraph,
) -> list[DeclareConstraint]:
    """Validate returned constraints against the graph's activity vocabulary.

    Constraints whose ``activity_a`` or ``activity_b`` does not appear in the
    set of known node names are logged and removed from the list.

    Args:
        constraints: Raw constraint list returned by the LLM chain.
        graph: The source BPMN graph.

    Returns:
        Filtered list containing only valid constraints.
    """
    valid_names = {n.name for n in graph.nodes if n.name.strip()}
    validated: list[DeclareConstraint] = []

    for c in constraints:
        if c.activity_a not in valid_names:
            logger.warning(
                "Constraint %r references unknown activity_a %r — discarding",
                c.id,
                c.activity_a,
            )
            continue
        if c.activity_b and c.activity_b not in valid_names:
            logger.warning(
                "Constraint %r references unknown activity_b %r — discarding",
                c.id,
                c.activity_b,
            )
            continue
        validated.append(c)

    discarded = len(constraints) - len(validated)
    if discarded:
        logger.warning("Discarded %d constraint(s) with invalid activity names", discarded)

    # Re-index IDs to keep them sequential after filtering
    for i, c in enumerate(validated, start=1):
        validated[i - 1] = c.model_copy(update={"id": f"constr_{i:02d}"})

    return validated


# ── Public API ────────────────────────────────────────────────────────────────


async def extract_constraints(
    graph: BPMNGraph,
    fragments: list[TextFragment],
    mappings: list[Mapping],
) -> list[DeclareConstraint]:
    """Extract DECLARE constraints from text using Gemini LLM.

    Sends a single LLM call containing the full process description, the
    vocabulary of valid BPMN activity names, and the text-to-BPMN mappings
    produced by Agent 1.  Retries up to ``settings.llm_max_retries`` times
    on failure, increasing temperature slightly on each retry.

    Args:
        graph: Parsed BPMN process graph (for activity name vocabulary).
        fragments: Text fragments from the process description.
        mappings: Text-to-BPMN mappings from Agent 1 (for grounding).

    Returns:
        List of :class:`~src.models.DeclareConstraint` objects extracted from
        the text.  Only constraints whose activity names match the BPMN graph
        vocabulary are returned.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    settings.validate()

    activities_fmt = _format_activity_names(graph)
    full_text = _format_full_text(fragments)
    mappings_fmt = _format_mappings(mappings)

    last_error: Exception | None = None

    for attempt in range(settings.llm_max_retries + 1):
        temperature = settings.gemini_temperature if attempt == 0 else 0.1 * attempt
        logger.info(
            "Formalizer attempt %d/%d (temperature=%.1f) for process %r",
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
            chain = _PROMPT | llm.with_structured_output(ConstraintList)
            await rate_limiter.wait()
            result: ConstraintList = await chain.ainvoke(
                {
                    "activity_names_list": activities_fmt,
                    "full_text": full_text,
                    "mappings_formatted": mappings_fmt,
                }
            )
            constraints = _validate_constraints(result.constraints, graph)
            logger.info(
                "Formalizer returned %d valid constraint(s) (%d templates used: %s)",
                len(constraints),
                len({c.template for c in constraints}),
                sorted({c.template.value for c in constraints}),
            )
            return constraints

        except Exception as exc:  # noqa: BLE001
            logger.warning("Formalizer attempt %d failed: %s", attempt + 1, exc)
            last_error = exc

    raise RuntimeError(
        f"Formalizer agent failed after {settings.llm_max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
