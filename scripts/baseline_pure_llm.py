"""Pure-LLM baseline for BPMN-text conformance checking.

This baseline sends the BPMN model + text description to Gemini in a single
prompt and asks it to find all inconsistencies directly — NO formal
verification, NO Petri nets, NO DECLARE constraints.

This is Baseline 1 from the paper (Section IV.C).

Usage::

    # Single process:
    python scripts/baseline_pure_llm.py \
        data/dispatch/Dispatch-of-goods.bpmn \
        data/dispatch/DispatchDescription.txt \
        -o results/baselines/dispatch_pure_llm.json

    # Compare with ground truth:
    python scripts/evaluate.py results/baselines/dispatch_pure_llm.json \
        data/ground_truth/dispatch_mapping.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# ── Add project root to path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.models import (
    BPMNNodeType,
    ConformanceReport,
    DeclareConstraint,
    DeclareTemplate,
    Mapping,
    VerificationResult,
    Violation,
)
from src.module_a.bpmn_parser import parse_bpmn
from src.module_a.text_loader import load_text

logger = logging.getLogger(__name__)


# ── LLM response models ─────────────────────────────────────────────────────


class LLMInconsistency(BaseModel):
    """A single inconsistency found by the LLM."""

    description: str = Field(description="What the inconsistency is")
    text_fragment: str = Field(description="The relevant text fragment")
    bpmn_element: str = Field(
        default="", description="The relevant BPMN element name, if any"
    )
    severity: str = Field(
        default="medium", description="low / medium / high"
    )
    type: str = Field(
        default="behavioral",
        description="structural / behavioral / resource",
    )


class LLMAnalysisResult(BaseModel):
    """Full analysis result from the pure-LLM approach."""

    inconsistencies: list[LLMInconsistency] = Field(default_factory=list)
    summary: str = Field(default="", description="Overall assessment")


# ── Baseline implementation ──────────────────────────────────────────────────


async def run_pure_llm_baseline(
    bpmn_file_path: str,
    text_file_path: str,
) -> dict:
    """Run the pure-LLM baseline on a BPMN + text pair.

    The LLM receives the full BPMN structure and the full text, and is asked
    to find all inconsistencies in a single call.  No formal verification is
    performed.

    Args:
        bpmn_file_path: Path to the BPMN 2.0 XML file.
        text_file_path: Path to the text description file.

    Returns:
        A dict compatible with ConformanceReport.model_dump() so that
        evaluate.py can process it identically.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate

    settings.validate()

    # ── Parse inputs (same as hybrid) ──
    graph = parse_bpmn(bpmn_file_path)
    fragments = load_text(text_file_path)

    # Format BPMN info for the prompt
    tasks_str = "\n".join(
        f"  - {n.name} (id: {n.id}, lane: {n.lane or 'none'})"
        for n in graph.nodes
        if n.type == BPMNNodeType.TASK
    )
    gateways_str = "\n".join(
        f"  - {n.name or n.id} (type: {n.type.value})"
        for n in graph.nodes
        if "gateway" in n.type.value
    )
    edges_str = "\n".join(
        f"  - {e.source} → {e.target}"
        + (f' [{e.name}]' if e.name else '')
        for e in graph.edges
    )
    text_str = "\n".join(f"  [{f.id}] {f.text}" for f in fragments)

    # ── Single LLM call ──
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
        google_api_key=settings.google_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a business process expert. You are given a BPMN process model "
            "and a natural-language description of the same process. Your task is to "
            "find ALL inconsistencies between them.\n\n"
            "An inconsistency is any place where the text says something different "
            "from what the BPMN model allows. Types of inconsistencies:\n"
            "- structural: a step exists in text but not in model, or vice versa\n"
            "- behavioral: the ordering or conditions differ\n"
            "- resource: different roles/actors are mentioned\n\n"
            "Be thorough but precise. Only report real inconsistencies, not "
            "stylistic differences or paraphrasing.",
        ),
        (
            "human",
            "## BPMN Model: {process_name}\n\n"
            "### Tasks:\n{tasks}\n\n"
            "### Gateways:\n{gateways}\n\n"
            "### Sequence Flows:\n{edges}\n\n"
            "### Lanes: {lanes}\n\n"
            "## Text Description:\n{text}\n\n"
            "Find all inconsistencies between the text and the BPMN model.",
        ),
    ])

    chain = prompt | llm.with_structured_output(LLMAnalysisResult)

    try:
        result: LLMAnalysisResult = await chain.ainvoke({
            "process_name": graph.process_name,
            "tasks": tasks_str,
            "gateways": gateways_str,
            "edges": edges_str,
            "lanes": ", ".join(graph.lanes.keys()) if graph.lanes else "none",
            "text": text_str,
        })
    except Exception as exc:
        logger.error("LLM baseline call failed: %s", exc)
        result = LLMAnalysisResult(
            inconsistencies=[],
            summary=f"LLM call failed: {exc}",
        )

    # ── Convert to ConformanceReport-compatible format ──
    # Create dummy mappings (the baseline doesn't do formal mapping)
    mappings = []
    for f in fragments:
        # Try to find a matching BPMN node by name similarity
        best_node = None
        for n in graph.nodes:
            if n.type == BPMNNodeType.TASK and n.name:
                # Simple keyword overlap
                f_words = set(f.text.lower().split())
                n_words = set(n.name.lower().split())
                if f_words & n_words:
                    best_node = n
                    break
        if best_node:
            mappings.append({
                "fragment_id": f.id,
                "fragment_text": f.text,
                "node_id": best_node.id,
                "node_name": best_node.name,
                "confidence": 0.5,  # Low confidence for keyword matching
            })

    # Convert inconsistencies to violations
    violations = []
    for i, inc in enumerate(result.inconsistencies):
        violations.append({
            "constraint_id": f"llm_inc_{i+1:02d}",
            "constraint_description": f"[{inc.type}] {inc.description}",
            "trace": [],
            "explanation": inc.description,
        })

    report = {
        "process_name": graph.process_name,
        "bpmn_file": bpmn_file_path,
        "text_file": text_file_path,
        "graph_summary": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "lanes": len(graph.lanes),
            "tasks": sum(1 for n in graph.nodes if n.type == BPMNNodeType.TASK),
        },
        "mappings": mappings,
        "constraints": [],  # Pure LLM doesn't extract formal constraints
        "verification": {
            "is_conformant": len(violations) == 0,
            "total_constraints": len(violations),
            "satisfied": 0,
            "violated": len(violations),
            "violations": violations,
        },
        "baseline_method": "pure_llm",
        "llm_raw_output": {
            "inconsistencies": [inc.model_dump() for inc in result.inconsistencies],
            "summary": result.summary,
        },
    }

    return report


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run the pure-LLM baseline for BPMN-text conformance."
    )
    parser.add_argument("bpmn_file", help="Path to BPMN 2.0 XML file.")
    parser.add_argument("text_file", help="Path to text description file.")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save result JSON to this path.",
    )
    args = parser.parse_args()

    report = asyncio.run(run_pure_llm_baseline(args.bpmn_file, args.text_file))

    # Print summary
    n_inc = len(report["llm_raw_output"]["inconsistencies"])
    print(f"\n{'='*60}")
    print(f"  PURE-LLM BASELINE: {report['process_name']}")
    print(f"{'='*60}")
    print(f"  Inconsistencies found: {n_inc}")
    for i, inc in enumerate(report["llm_raw_output"]["inconsistencies"]):
        print(f"    {i+1}. [{inc['type']}] {inc['description']}")
    print(f"  Summary: {report['llm_raw_output']['summary']}")
    print(f"{'='*60}\n")

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
