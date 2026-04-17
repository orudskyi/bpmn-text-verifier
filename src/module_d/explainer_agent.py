"""Module D — Explainer Agent (Agent 3).

Takes the raw verification result produced by Module C (constraint violations +
counter-example traces) and generates human-readable explanations.  This bridges
the gap between formal verification output and what a business analyst can
understand.

All violations are batched into a **single LLM call** to minimise API usage.
The agent returns an updated :class:`~src.models.VerificationResult` with the
``explanation`` field filled in for every :class:`~src.models.Violation`.

Typical usage::

    import asyncio
    from src.module_d.explainer_agent import explain_violations

    updated_result = asyncio.run(
        explain_violations(verification_result, constraints, graph)
    )
    for v in updated_result.violations:
        print(v.explanation)
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
    VerificationResult,
    Violation,
)

logger = logging.getLogger(__name__)

# ── Template → plain-English meaning ─────────────────────────────────────────

_TEMPLATE_MEANINGS: dict[DeclareTemplate, str] = {
    DeclareTemplate.EXISTENCE:
        "Activity A must occur in every execution",
    DeclareTemplate.ABSENCE:
        "Activity A must never occur",
    DeclareTemplate.RESPONSE:
        "Whenever A happens, B must happen after it",
    DeclareTemplate.PRECEDENCE:
        "B can only happen if A has happened before it",
    DeclareTemplate.SUCCESSION:
        "A and B must always occur together, A before B",
    DeclareTemplate.CHAIN_RESPONSE:
        "Whenever A happens, B must happen immediately after",
    DeclareTemplate.CHAIN_PRECEDENCE:
        "B can only happen immediately after A",
    DeclareTemplate.NOT_COEXISTENCE:
        "A and B can never both happen in the same execution",
    DeclareTemplate.CHOICE:
        "At least one of A or B must happen",
    DeclareTemplate.EXCLUSIVE_CHOICE:
        "Exactly one of A or B must happen, not both",
}


# ── Structured-output wrapper ─────────────────────────────────────────────────


class ExplanationItem(BaseModel):
    """A single violation explanation returned by the LLM."""

    constraint_id: str
    explanation: str


class ExplanationList(BaseModel):
    """Wrapper for the batched LLM response."""

    explanations: list[ExplanationItem]


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a business process analyst explaining verification results to a \
non-technical audience.

For each violation you receive:
1. A DECLARE constraint that was violated (what the text description says should happen).
2. A counter-example trace — a valid execution path in the BPMN model that breaks the rule.
3. The original text fragment that implies the constraint.

For each violation, write exactly 2-3 plain English sentences that:
  - State what the text description says should happen.
  - Describe what the BPMN model actually allows (using the counter-example trace).
  - Explain why this is a problem or inconsistency.

Rules:
- Use concrete activity names from the trace and the constraint.
- Do NOT use formal notation (no LTL, no DECLARE template names like "existence" or "response").
- Write as if explaining to a business analyst who knows the process but not formal methods.
- Keep each explanation under 80 words.
- Be actionable: the reader should understand what needs to be fixed.\
"""

_USER_PROMPT = """\
Process: {process_name}

Violations to explain:
{violations_formatted}

Return a JSON object with a single key "explanations" containing an array.
Each element must have:
  - constraint_id : the constraint ID string (e.g. "constr_01")
  - explanation   : 2-3 plain-English sentences explaining the violation\
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _USER_PROMPT)]
)


# ── Formatting helpers ────────────────────────────────────────────────────────


def _constraint_by_id(
    constraints: list[DeclareConstraint],
) -> dict[str, DeclareConstraint]:
    """Index constraints by their ``id`` field for fast lookup.

    Args:
        constraints: Full constraint list from Agent 2.

    Returns:
        Dictionary mapping constraint IDs to constraint objects.
    """
    return {c.id: c for c in constraints}


def _format_violations(
    violations: list[Violation],
    constraint_index: dict[str, DeclareConstraint],
) -> str:
    """Render all violations as a prompt-ready block.

    Args:
        violations: Violation objects from the VerificationResult.
        constraint_index: Pre-built ID → DeclareConstraint lookup.

    Returns:
        Multi-line string with each violation described in context.
    """
    blocks: list[str] = []
    for v in violations:
        c = constraint_index.get(v.constraint_id)
        if c is None:
            # Fallback if constraint not in the index (e.g. partial pipeline)
            meaning = v.constraint_description
            source_text = ""
            template_label = v.constraint_description
        else:
            meaning = _TEMPLATE_MEANINGS.get(c.template, c.template.value)
            # Substitute A / B placeholders with actual activity names
            meaning = (
                meaning
                .replace("Activity A", f'"{c.activity_a}"')
                .replace("A", f'"{c.activity_a}"', 1)
                .replace("B", f'"{c.activity_b}"', 1)
                if c.activity_b
                else meaning.replace("Activity A", f'"{c.activity_a}"')
                           .replace("A", f'"{c.activity_a}"')
            )
            source_text = c.source_text
            template_label = (
                f"{c.template.value}({c.activity_a}"
                + (f", {c.activity_b}" if c.activity_b else "")
                + ")"
            )

        trace_str = " → ".join(v.trace) if v.trace else "(empty trace)"

        block = (
            f"ID: {v.constraint_id}\n"
            f"Constraint: {template_label}\n"
            f"Meaning: {meaning}\n"
            f"Source text: \"{source_text}\"\n"
            f"Counter-example trace: {trace_str}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


# ── Public API ────────────────────────────────────────────────────────────────


async def explain_violations(
    result: VerificationResult,
    constraints: list[DeclareConstraint],
    graph: BPMNGraph,
) -> VerificationResult:
    """Add natural-language explanations to verification violations.

    If there are no violations, the result is returned unchanged immediately
    without making any LLM call.

    All violations are batched into a single Gemini call.  Retries up to
    ``settings.llm_max_retries`` times on failure with increasing temperature.

    Args:
        result: Verification result with violations (``explanation`` field empty).
        constraints: Full list of DECLARE constraints for context lookup.
        graph: BPMN graph (used for process name in the prompt).

    Returns:
        Updated :class:`~src.models.VerificationResult` with the ``explanation``
        field filled for each ``Violation``.  If a violation's constraint ID
        cannot be found in the *explanations* response, a fallback explanation
        is generated locally.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    if not result.violations:
        logger.info("No violations to explain — returning result unchanged.")
        return result

    settings.validate()

    constraint_index = _constraint_by_id(constraints)
    violations_fmt = _format_violations(result.violations, constraint_index)

    logger.info(
        "Explaining %d violation(s) for process %r",
        len(result.violations),
        graph.process_name,
    )

    last_error: Exception | None = None

    for attempt in range(settings.llm_max_retries + 1):
        temperature = settings.gemini_temperature if attempt == 0 else 0.1 * attempt
        logger.info(
            "Explainer attempt %d/%d (temperature=%.1f)",
            attempt + 1,
            settings.llm_max_retries + 1,
            temperature,
        )

        try:
            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                temperature=temperature,
                google_api_key=settings.google_api_key,
            )
            chain = _PROMPT | llm.with_structured_output(ExplanationList)
            response: ExplanationList = await chain.ainvoke(
                {
                    "process_name": graph.process_name,
                    "violations_formatted": violations_fmt,
                }
            )

            # Build a lookup: constraint_id → explanation text
            explanation_map: dict[str, str] = {
                item.constraint_id: item.explanation
                for item in response.explanations
            }

            # Update violation objects with the generated explanations
            updated_violations: list[Violation] = []
            for v in result.violations:
                explanation = explanation_map.get(v.constraint_id, "")
                if not explanation:
                    logger.warning(
                        "No explanation returned for constraint %r — using fallback",
                        v.constraint_id,
                    )
                    explanation = _fallback_explanation(v, constraint_index)
                updated_violations.append(
                    v.model_copy(update={"explanation": explanation})
                )

            logger.info(
                "Explainer filled %d/%d explanations",
                sum(1 for v in updated_violations if v.explanation),
                len(updated_violations),
            )

            return result.model_copy(update={"violations": updated_violations})

        except Exception as exc:  # noqa: BLE001
            logger.warning("Explainer attempt %d failed: %s", attempt + 1, exc)
            last_error = exc

    raise RuntimeError(
        f"Explainer agent failed after {settings.llm_max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )


def _fallback_explanation(
    violation: Violation,
    constraint_index: dict[str, DeclareConstraint],
) -> str:
    """Generate a simple template-based explanation without an LLM call.

    Used when the LLM fails to return an explanation for a specific violation.

    Args:
        violation: The violation that needs an explanation.
        constraint_index: Pre-built ID → DeclareConstraint lookup.

    Returns:
        A short, readable English sentence describing the violation.
    """
    c = constraint_index.get(violation.constraint_id)
    trace_str = " → ".join(violation.trace[:5])
    if len(violation.trace) > 5:
        trace_str += " → ..."

    if c is None:
        return (
            f"The constraint '{violation.constraint_description}' is violated. "
            f"Counter-example trace: {trace_str}."
        )

    meaning = _TEMPLATE_MEANINGS.get(c.template, c.template.value)
    return (
        f"The text implies: {meaning.replace('A', repr(c.activity_a), 1)}. "
        f"However, the BPMN model allows a path where this is not the case: "
        f"{trace_str}."
    )
