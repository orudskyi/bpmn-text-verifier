"""Mutation-based testing for BPMN-text conformance checking.

Creates systematic mutations of BPMN models (introducing known
inconsistencies with the text) and measures whether the hybrid pipeline
and the pure-LLM baseline detect them.

Produces the data for Table II in the paper.

Mutation types:
    M1: Delete a task (text describes it, model doesn't have it)
    M2: Add a spurious task (model has a step not in the text)
    M3: Swap ordering of two tasks (model contradicts described order)
    M4: Remove a gateway branch (model restricts possibilities described in text)
    M5: Change lane assignment (role mismatch between text and model)

Two measurement arms (see docs/ROADMAP.md T1.9):

    frozen (default) — Constraints are extracted from the TEXT exactly ONCE,
        against the unmutated model, and that fixed set is then verified
        (Module C only, no LLM) against each of the five mutated BPMN files.
        Only the BPMN model varies between mutants, so a change in the
        violation count can only be caused by the structural mutation, not
        by re-extraction variance. This is the arm that measures the
        symbolic verifier.

    reextract (--reextract) — The previous behaviour: the full hybrid
        pipeline, including a fresh Formalizer call, is re-run on every
        mutant. Kept only for side-by-side comparison; the observed delta
        here mixes structural change with LLM extraction variance and
        should not be read as a verifier result.

Usage::

    python scripts/mutate.py data/dispatch/Dispatch-of-goods.bpmn \
        data/dispatch/DispatchDescription.txt -o results/mutations/dispatch/

    # old (pre-T1.9) behaviour, for comparison:
    python scripts/mutate.py data/dispatch/Dispatch-of-goods.bpmn \
        data/dispatch/DispatchDescription.txt -o results/mutations/dispatch/ --reextract
"""

import argparse
import asyncio
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import DeclareConstraint, Mapping  # noqa: E402
from src.module_a.bpmn_parser import parse_bpmn  # noqa: E402
from src.module_c.verifier import verify_constraints  # noqa: E402

logger = logging.getLogger(__name__)

NSMAP = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Gemini free/paid limits: 15 RPM. src/rate_limiter.py already enforces a
# 14 RPM floor inside every mapper/formalizer/explainer call, so the pause
# below is needed only around the pure-LLM baseline, which bypasses that
# limiter entirely (docs/AUDIT.md §6.5) and has no other pacing.

PAUSE_BETWEEN_RUNS = 20  # seconds, applied around unprotected baseline calls


async def _rate_limit_pause(label: str = ""):
    """Sleep before an unprotected (non-rate-limited) API call."""
    if label:
        print(f"    ⏳ Rate limit pause ({PAUSE_BETWEEN_RUNS}s) before {label}...")
    await asyncio.sleep(PAUSE_BETWEEN_RUNS)


def generation_params_meta() -> dict:
    """Report the exact generation parameters the pipeline passes to Gemini.

    Read from ``src.config.settings`` and the LLM call sites
    (``mapper_agent.py``, ``formalizer_agent.py``, ``explainer_agent.py``,
    all of the form ``ChatGoogleGenerativeAI(model=..., temperature=...,
    google_api_key=...)``) rather than asserted from memory, per R#3.
    """
    from src.config import settings

    return {
        "model": settings.gemini_model,
        "temperature_first_attempt": settings.gemini_temperature,
        "temperature_retry_schedule": "0.1 * attempt_index for attempts after the first "
                                       "(identical in mapper_agent.py, formalizer_agent.py, "
                                       "explainer_agent.py)",
        "llm_max_retries": settings.llm_max_retries,
        "note": "Only model/temperature/google_api_key are passed to ChatGoogleGenerativeAI(...); "
                "top_p, top_k and max_output_tokens are not set anywhere in src/ and take "
                "langchain-google-genai / Gemini API provider defaults. Output is constrained "
                "via .with_structured_output(...), not via a generation parameter.",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# --dry-run stubs: replace the mapper/formalizer LLM call sites with
# deterministic, network-free functions matching the real signatures, so
# the frozen-arm wiring can be verified with zero Gemini API calls. The
# pure-LLM baseline has no such stub, so --dry-run forces --no-baseline.
# ══════════════════════════════════════════════════════════════════════════════


async def _dry_run_map_text_to_bpmn(graph, fragments):
    from src.models import BPMNNodeType, Mapping

    tasks = [n for n in graph.nodes if n.type == BPMNNodeType.TASK]
    if not tasks:
        return []
    return [
        Mapping(
            fragment_id=f.id, fragment_text=f.text,
            node_id=tasks[i % len(tasks)].id, node_name=tasks[i % len(tasks)].name,
            confidence=0.42,
        )
        for i, f in enumerate(fragments)
    ]


async def _dry_run_extract_constraints(graph, fragments, mappings):
    from src.models import DeclareConstraint, DeclareTemplate

    tasks = [n for n in graph.nodes if n.type.value == "task"]
    if len(tasks) < 2:
        return []
    return [
        DeclareConstraint(
            id="dryrun_01", template=DeclareTemplate.EXISTENCE,
            activity_a=tasks[0].name, source_text="[dry-run stub — no LLM call made]",
        ),
        DeclareConstraint(
            id="dryrun_02", template=DeclareTemplate.PRECEDENCE,
            activity_a=tasks[0].name, activity_b=tasks[-1].name,
            source_text="[dry-run stub — no LLM call made]",
        ),
    ]


@contextmanager
def _mock_llm_agents():
    with patch("src.module_b.mapper_agent.map_text_to_bpmn", _dry_run_map_text_to_bpmn), \
         patch("src.module_b.formalizer_agent.extract_constraints", _dry_run_extract_constraints):
        yield


# ══════════════════════════════════════════════════════════════════════════════
# MUTATION GENERATORS
# ══════════════════════════════════════════════════════════════════════════════


def _parse_xml(bpmn_path: str) -> etree._ElementTree:
    return etree.parse(bpmn_path)


def _find_tasks(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.findall(".//bpmn:task", NSMAP)


def _find_lanes(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.findall(".//bpmn:lane", NSMAP)


def _remove_node_refs(tree: etree._ElementTree, node_id: str):
    for sf in tree.findall(".//bpmn:sequenceFlow", NSMAP):
        if sf.get("sourceRef") == node_id or sf.get("targetRef") == node_id:
            sf.getparent().remove(sf)
    for ref in tree.findall(".//bpmn:flowNodeRef", NSMAP):
        if ref.text and ref.text.strip() == node_id:
            ref.getparent().remove(ref)
    for shape in tree.findall(".//{http://www.omg.org/spec/BPMN/20100524/DI}BPMNShape"):
        if shape.get("bpmnElement") == node_id:
            shape.getparent().remove(shape)
    for edge in tree.findall(".//{http://www.omg.org/spec/BPMN/20100524/DI}BPMNEdge"):
        ref = edge.get("bpmnElement", "")
        flow = tree.find(f".//bpmn:sequenceFlow[@id='{ref}']", NSMAP)
        if flow is None and ref.startswith("SequenceFlow"):
            try:
                edge.getparent().remove(edge)
            except Exception:
                pass


def mutate_m1_delete_task(bpmn_path: str, task_index: int = 0) -> tuple[str, str, str]:
    """M1: Delete a task from the BPMN model."""
    tree = _parse_xml(bpmn_path)
    tasks = _find_tasks(tree)
    if not tasks or task_index >= len(tasks):
        return ("M1: no task to delete", "", etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())

    target = tasks[task_index]
    task_id = target.get("id")
    task_name = target.get("name", "").replace("\n", " ").replace("&#10;", " ")

    incoming = tree.findall(f".//bpmn:sequenceFlow[@targetRef='{task_id}']", NSMAP)
    outgoing = tree.findall(f".//bpmn:sequenceFlow[@sourceRef='{task_id}']", NSMAP)
    if incoming and outgoing:
        out_target = outgoing[0].get("targetRef")
        incoming[0].set("targetRef", out_target)

    target.getparent().remove(target)
    _remove_node_refs(tree, task_id)

    desc = f"M1: Deleted task '{task_name}' (id: {task_id})"
    return (desc, task_name, etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())


def mutate_m2_add_task(bpmn_path: str) -> tuple[str, str, str]:
    """M2: Add a spurious task not described in the text."""
    tree = _parse_xml(bpmn_path)
    process = tree.find(".//bpmn:process", NSMAP)
    spurious_name = "Verify compliance documents"
    spurious_id = "Task_MUTATED_M2"

    task_elem = etree.SubElement(process, f"{{{NSMAP['bpmn']}}}task", id=spurious_id, name=spurious_name)
    tasks = _find_tasks(tree)
    real_tasks = [t for t in tasks if t.get("id") != spurious_id]

    if real_tasks:
        anchor_id = real_tasks[0].get("id")
        out_flow = tree.find(f".//bpmn:sequenceFlow[@sourceRef='{anchor_id}']", NSMAP)
        if out_flow is not None:
            original_target = out_flow.get("targetRef")
            out_flow.set("targetRef", spurious_id)
            etree.SubElement(task_elem, f"{{{NSMAP['bpmn']}}}incoming").text = out_flow.get("id")
            new_flow_id = "SequenceFlow_MUTATED_M2"
            etree.SubElement(process, f"{{{NSMAP['bpmn']}}}sequenceFlow", id=new_flow_id, sourceRef=spurious_id, targetRef=original_target)
            etree.SubElement(task_elem, f"{{{NSMAP['bpmn']}}}outgoing").text = new_flow_id

    desc = f"M2: Added spurious task '{spurious_name}' (id: {spurious_id})"
    return (desc, spurious_name, etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())


def mutate_m3_swap_tasks(bpmn_path: str, idx_a: int = 0, idx_b: int = 1) -> tuple[str, str, str]:
    """M3: Swap the names of two tasks."""
    tree = _parse_xml(bpmn_path)
    tasks = _find_tasks(tree)
    if len(tasks) < 2 or idx_a >= len(tasks) or idx_b >= len(tasks):
        return ("M3: not enough tasks", "", etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())

    a, b = tasks[idx_a], tasks[idx_b]
    name_a, name_b = a.get("name", ""), b.get("name", "")
    a.set("name", name_b)
    b.set("name", name_a)

    swap_info = f"'{name_a}' <-> '{name_b}'"
    desc = f"M3: Swapped task names {swap_info}"
    return (desc, swap_info, etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())


def mutate_m4_remove_branch(bpmn_path: str) -> tuple[str, str, str]:
    """M4: Remove one outgoing branch from a gateway."""
    tree = _parse_xml(bpmn_path)
    for gw in tree.findall(".//bpmn:exclusiveGateway", NSMAP):
        gw_id = gw.get("id")
        outgoing = tree.findall(f".//bpmn:sequenceFlow[@sourceRef='{gw_id}']", NSMAP)
        if len(outgoing) >= 2:
            removed = outgoing[-1]
            removed_name = removed.get("name", "unnamed")
            removed_target = removed.get("targetRef", "?")
            removed.getparent().remove(removed)
            for out_ref in gw.findall(f"{{{NSMAP['bpmn']}}}outgoing"):
                if out_ref.text and out_ref.text.strip() == removed.get("id"):
                    gw.remove(out_ref)

            desc = f"M4: Removed branch '{removed_name}' from gateway {gw_id} (target: {removed_target})"
            return (desc, f"branch '{removed_name}' -> {removed_target}", etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())

    return ("M4: no gateway found", "", etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())


def mutate_m5_change_lane(bpmn_path: str) -> tuple[str, str, str]:
    """M5: Move a task to a different lane."""
    tree = _parse_xml(bpmn_path)
    lanes = _find_lanes(tree)
    if len(lanes) < 2:
        return ("M5: not enough lanes (skipped)", "", etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())

    lane_0_refs = lanes[0].findall(f"{{{NSMAP['bpmn']}}}flowNodeRef")
    tasks = _find_tasks(tree)
    task_ids = {t.get("id") for t in tasks}

    for ref in lane_0_refs:
        node_id = ref.text.strip() if ref.text else ""
        if node_id in task_ids:
            lanes[0].remove(ref)
            new_ref = etree.SubElement(lanes[1], f"{{{NSMAP['bpmn']}}}flowNodeRef")
            new_ref.text = node_id
            task_name = ""
            for t in tasks:
                if t.get("id") == node_id:
                    task_name = t.get("name", "").replace("\n", " ")
                    break
            lane_0_name = lanes[0].get("name", "Lane 0")
            lane_1_name = lanes[1].get("name", "Lane 1")
            change_info = f"'{task_name}': {lane_0_name} -> {lane_1_name}"
            desc = f"M5: Moved task '{task_name}' from {lane_0_name} to {lane_1_name}"
            return (desc, change_info, etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())

    return ("M5: no movable task found", "", etree.tostring(tree, xml_declaration=True, encoding="UTF-8").decode())


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

MUTATORS = [
    ("M1_delete_task", mutate_m1_delete_task),
    ("M2_add_task", mutate_m2_add_task),
    ("M3_swap_tasks", mutate_m3_swap_tasks),
    ("M4_remove_branch", mutate_m4_remove_branch),
    ("M5_change_lane", mutate_m5_change_lane),
]


async def run_hybrid(bpmn_path: str, text_path: str) -> dict:
    """Full hybrid pipeline: parse, map, extract, verify, explain (re-extraction arm)."""
    from src.orchestrator import build_pipeline, initial_state
    pipeline = build_pipeline()
    result = await pipeline.ainvoke(initial_state(bpmn_path, text_path))
    if result.get("report"):
        return result["report"].model_dump()
    return {"error": result.get("errors", ["Unknown error"])}


async def extract_constraints_once(bpmn_path: str, text_path: str) -> dict:
    """Extract the fixed constraint set (T1.9): parse + map + formalize, ONE time.

    Deliberately does not run Module C or D here — verification is done per
    model afterwards by :func:`verify_frozen`, and explanation is not needed
    to measure violation counts.

    Returns:
        Dict with ``graph``, ``fragments``, ``mappings``, ``constraints`` and
        ``meta`` (generation params, timestamp, source paths).
    """
    from src.module_a.text_loader import load_text
    from src.module_b.formalizer_agent import extract_constraints
    from src.module_b.mapper_agent import map_text_to_bpmn

    graph = parse_bpmn(bpmn_path)
    fragments = load_text(text_path)
    mappings = await map_text_to_bpmn(graph, fragments)
    constraints = await extract_constraints(graph, fragments, mappings)

    meta = generation_params_meta()
    meta["source_bpmn"] = bpmn_path
    meta["source_text"] = text_path
    meta["n_fragments"] = len(fragments)
    meta["n_mappings"] = len(mappings)
    meta["n_constraints"] = len(constraints)

    return {
        "graph": graph,
        "fragments": fragments,
        "mappings": mappings,
        "constraints": constraints,
        "meta": meta,
    }


def verify_frozen(bpmn_path: str, constraints: list[DeclareConstraint]) -> dict:
    """Verify a fixed constraint set against one BPMN model. No LLM call.

    This is the operation that varies only the model, per mutant, in the
    frozen arm — it calls Module C (``verify_constraints``) directly and
    nothing else.
    """
    graph = parse_bpmn(bpmn_path)
    verification = verify_constraints(graph, constraints, bpmn_file_path=bpmn_path)
    return {
        "process_name": graph.process_name,
        "mode": "frozen",
        "constraints": [c.model_dump() for c in constraints],
        "verification": verification.model_dump(),
    }


async def run_baseline(bpmn_path: str, text_path: str) -> dict:
    from scripts.baseline_pure_llm import run_pure_llm_baseline
    return await run_pure_llm_baseline(bpmn_path, text_path)


def _save_frozen_set(out: Path, extraction: dict) -> Path:
    frozen_path = out / "frozen_constraints.json"
    frozen_path.write_text(
        json.dumps(
            {
                "meta": extraction["meta"],
                "mappings": [m.model_dump() for m in extraction["mappings"]],
                "constraints": [c.model_dump() for c in extraction["constraints"]],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return frozen_path


async def run_mutations(
    bpmn_path: str,
    text_path: str,
    output_dir: str,
    run_baselines: bool = True,
    freeze: bool = True,
) -> list[dict]:
    """Generate mutations and run both methods on each.

    Args:
        freeze: If True (default, T1.9), extract constraints once from the
            unmutated model and reuse that fixed set across all five
            mutants — only the BPMN model varies. If False, re-invoke the
            full hybrid pipeline (including the Formalizer) on every
            mutant, as before T1.9; kept for comparison only.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    extraction: dict | None = None
    if freeze:
        print("\n  [freeze] Extracting constraints ONCE from the unmutated model...")
        extraction = await extract_constraints_once(bpmn_path, text_path)
        frozen_path = _save_frozen_set(out, extraction)
        print(f"    {len(extraction['constraints'])} constraint(s) frozen — saved to {frozen_path}")

    # ── Run on original ──
    print("\n  [0/5] Running on ORIGINAL (unmutated) model...")
    if freeze:
        hybrid_report = verify_frozen(bpmn_path, extraction["constraints"])
        hybrid_violated = hybrid_report["verification"]["violated"]
        (out / "ORIGINAL_hybrid.json").write_text(
            json.dumps(hybrid_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        try:
            hybrid_report = await run_hybrid(bpmn_path, text_path)
            hybrid_violated = hybrid_report.get("verification", {}).get("violated", 0)
            (out / "ORIGINAL_hybrid.json").write_text(
                json.dumps(hybrid_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.error("Hybrid failed on original: %s", e)
            hybrid_report = {"error": str(e)}
            hybrid_violated = -1

    baseline_violated = -1
    if run_baselines:
        await _rate_limit_pause("baseline on original")
        try:
            baseline_report = await run_baseline(bpmn_path, text_path)
            baseline_violated = len(baseline_report.get("llm_raw_output", {}).get("inconsistencies", []))
            (out / "ORIGINAL_baseline.json").write_text(
                json.dumps(baseline_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.error("Baseline failed on original: %s", e)

    results.append({
        "mutation": "ORIGINAL",
        "description": "No mutation (original model)",
        "detail": "",
        "mode": "frozen" if freeze else "reextract",
        "hybrid_violated": hybrid_violated,
        "baseline_violated": baseline_violated,
        "expected_new_violations": 0,
    })

    # ── Run each mutation ──
    for i, (name, mutator) in enumerate(MUTATORS):
        print(f"\n  [{i+1}/5] Generating mutation {name}...")
        desc, detail, xml_str = mutator(bpmn_path)
        print(f"    {desc}")

        # Skip if mutation failed (e.g., no lanes for M5)
        if "skipped" in desc.lower() or "not enough" in desc.lower():
            print(f"    ⚠ Mutation not applicable — skipping")
            results.append({
                "mutation": name,
                "description": desc,
                "detail": detail,
                "mode": "frozen" if freeze else "reextract",
                "hybrid_violated": -1,
                "baseline_violated": -1,
                "expected_new_violations": 0,
                "skipped": True,
            })
            continue

        mut_path = out / f"{name}.bpmn"
        mut_path.write_text(xml_str, encoding="utf-8")

        # ── Hybrid ──
        if freeze:
            print(f"    Verifying frozen constraint set against mutated model (no LLM call)...")
            try:
                h_report = verify_frozen(str(mut_path), extraction["constraints"])
                h_violated = h_report["verification"]["violated"]
                (out / f"{name}_hybrid.json").write_text(
                    json.dumps(h_report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                logger.error("Frozen verification failed on %s: %s", name, e)
                h_report = {"error": str(e)}
                h_violated = -1
        else:
            print(f"    Running hybrid pipeline (re-extracting constraints)...")
            try:
                h_report = await run_hybrid(str(mut_path), text_path)
                h_violated = h_report.get("verification", {}).get("violated", 0)
                (out / f"{name}_hybrid.json").write_text(
                    json.dumps(h_report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                logger.error("Hybrid failed on %s: %s", name, e)
                h_report = {"error": str(e)}
                h_violated = -1

        # ── Baseline ──
        b_violated = -1
        if run_baselines:
            await _rate_limit_pause("baseline on mutation")
            print(f"    Running pure-LLM baseline...")
            try:
                b_report = await run_baseline(str(mut_path), text_path)
                b_violated = len(b_report.get("llm_raw_output", {}).get("inconsistencies", []))
                (out / f"{name}_baseline.json").write_text(
                    json.dumps(b_report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                logger.error("Baseline failed on %s: %s", name, e)

        results.append({
            "mutation": name,
            "description": desc,
            "detail": detail,
            "mode": "frozen" if freeze else "reextract",
            "hybrid_violated": h_violated,
            "baseline_violated": b_violated,
            "expected_new_violations": 1,
        })

    return results


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    # Filter out skipped mutations
    active = [r for r in results if not r.get("skipped")]
    mode = active[0]["mode"] if active and "mode" in active[0] else "unknown"

    print(f"\n{'='*80}")
    print(f"  MUTATION TESTING RESULTS  (arm: {mode})")
    print(f"{'='*80}")
    print(f"  {'Mutation':<22} {'Description':<30} {'Hybrid':>8} {'PureLLM':>8}")
    print(f"  {'-'*22} {'-'*30} {'-'*8} {'-'*8}")

    for r in active:
        mut = r["mutation"][:22]
        desc = (r["detail"][:30] if r["detail"] else r["description"][:30])
        h = str(r["hybrid_violated"]) if r["hybrid_violated"] >= 0 else "ERR"
        b = str(r["baseline_violated"]) if r["baseline_violated"] >= 0 else "N/A"
        print(f"  {mut:<22} {desc:<30} {h:>8} {b:>8}")

    print(f"{'='*80}")

    mutations = [r for r in active if r["mutation"] != "ORIGINAL"]
    original = next((r for r in active if r["mutation"] == "ORIGINAL"), None)
    original_h = original["hybrid_violated"] if original else 0
    original_b = original["baseline_violated"] if original and original["baseline_violated"] >= 0 else 0

    hybrid_increased = sum(
        1 for r in mutations
        if r["hybrid_violated"] > original_h and r["hybrid_violated"] >= 0
    )
    hybrid_any_change = sum(
        1 for r in mutations
        if r["hybrid_violated"] != original_h and r["hybrid_violated"] >= 0
    )
    baseline_increased = sum(
        1 for r in mutations
        if r["baseline_violated"] > original_b and r["baseline_violated"] >= 0
    )
    baseline_any_change = sum(
        1 for r in mutations
        if r["baseline_violated"] != original_b and r["baseline_violated"] >= 0
    )

    total = len(mutations)
    print(f"\n  Detection rate, arm={mode}, criterion 'increase only' (violations > original):")
    print(f"    Hybrid:   {hybrid_increased}/{total} mutations detected")
    print(f"    Pure LLM: {baseline_increased}/{total} mutations detected")
    print(f"\n  Detection rate, arm={mode}, criterion 'any change' (violations != original):")
    print(f"    Hybrid:   {hybrid_any_change}/{total} mutations detected")
    print(f"    Pure LLM: {baseline_any_change}/{total} mutations detected")
    print()


def main():
    parser = argparse.ArgumentParser(description="Mutation testing for BPMN-text conformance.")
    parser.add_argument("bpmn_file", help="Path to original BPMN 2.0 XML file.")
    parser.add_argument("text_file", help="Path to text description file.")
    parser.add_argument("--output", "-o", default="results/mutations/", help="Directory for results.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip pure-LLM baseline.")
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Use the pre-T1.9 behaviour: re-run the full hybrid pipeline "
             "(including a fresh Formalizer call) on every mutant instead of "
             "reusing one frozen constraint set. Kept for comparison only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock the mapper/formalizer LLM calls — zero Gemini API calls. "
             "Use this to verify the frozen-arm wiring before a live run. "
             "Forces --no-baseline (the pure-LLM baseline has no offline stub).",
    )
    args = parser.parse_args()

    run_baselines = not args.no_baseline and not args.dry_run
    if args.dry_run and not args.no_baseline:
        print("  [dry-run] Forcing --no-baseline: the pure-LLM baseline path has no offline stub.")

    run_kwargs = dict(
        run_baselines=run_baselines,
        freeze=not args.reextract,
    )
    if args.dry_run:
        with _mock_llm_agents():
            results = asyncio.run(run_mutations(args.bpmn_file, args.text_file, args.output, **run_kwargs))
    else:
        results = asyncio.run(run_mutations(args.bpmn_file, args.text_file, args.output, **run_kwargs))
    print_comparison_table(results)

    summary = {
        "meta": generation_params_meta(),
        "arm": "reextract" if args.reextract else "frozen",
        "results": results,
    }
    summary_path = Path(args.output) / "mutation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
