"""Run full evaluation suite and produce paper-ready results.

Includes rate limiting for Gemini API (15 RPM).

Usage::

    python scripts/run_paper_evaluation.py

    # Run only one process (faster):
    python scripts/run_paper_evaluation.py --process dispatch

    # Skip baseline (much faster, hybrid only):
    python scripts/run_paper_evaluation.py --no-baseline
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import evaluate_mappings, evaluate_constraints, Scores
from scripts.mutate import run_mutations, print_comparison_table

# ── Rate limiting ─────────────────────────────────────────────────────────────
PAUSE_BETWEEN_PHASES = 25  # seconds between major phases


async def _phase_pause(label: str):
    print(f"\n  ⏳ Waiting {PAUSE_BETWEEN_PHASES}s before {label} (rate limit)...")
    await asyncio.sleep(PAUSE_BETWEEN_PHASES)


# ── Configuration ─────────────────────────────────────────────────────────────

ALL_PROCESSES = [
    {
        "key": "dispatch",
        "name": "Dispatch of Goods",
        "bpmn": "data/dispatch/Dispatch-of-goods.bpmn",
        "text": "data/dispatch/DispatchDescription.txt",
        "gt": "data/ground_truth/dispatch_mapping.json",
    },
    {
        "key": "recourse",
        "name": "Recourse",
        "bpmn": "data/recourse/Recourse.bpmn",
        "text": "data/recourse/RecourseDescription.txt",
        "gt": "data/ground_truth/recourse_mapping.json",
    },
]

RESULTS_DIR = Path("results")


# ── Step 1: Run hybrid pipeline ──────────────────────────────────────────────


async def run_hybrid_pipeline(proc: dict) -> dict:
    from src.orchestrator import build_pipeline, initial_state

    print(f"\n  Running hybrid pipeline on {proc['name']}...")
    pipeline = build_pipeline()
    result = await pipeline.ainvoke(initial_state(proc["bpmn"], proc["text"]))

    if result.get("errors"):
        print(f"    ERRORS: {result['errors']}")

    report = result.get("report")
    if report:
        report_dict = report.model_dump()
        out_path = RESULTS_DIR / "hybrid" / f"{proc['key']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False))
        print(f"    ✓ Saved to {out_path}")
        print(f"    Mappings: {len(report_dict.get('mappings', []))}")
        print(f"    Constraints: {len(report_dict.get('constraints', []))}")
        print(f"    Violated: {report_dict.get('verification', {}).get('violated', 0)}")
        return report_dict

    return {}


# ── Step 2: Evaluate accuracy ────────────────────────────────────────────────


def evaluate_process(report: dict, gt_path: str, proc_name: str) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    mapping_scores = evaluate_mappings(report.get("mappings", []), gt.get("mappings", []))
    constraint_scores = evaluate_constraints(report.get("constraints", []), gt.get("constraints", []))

    return {
        "process": proc_name,
        "mapping": {
            "total_predicted": len(report.get("mappings", [])),
            "total_gt": len(gt.get("mappings", [])),
            "precision": mapping_scores.precision,
            "recall": mapping_scores.recall,
            "f1": mapping_scores.f1,
        },
        "constraints": {
            "total_predicted": len(report.get("constraints", [])),
            "total_gt": len(gt.get("constraints", [])),
            "precision": constraint_scores.precision,
            "recall": constraint_scores.recall,
            "f1": constraint_scores.f1,
        },
        "violations": {
            "total": report.get("verification", {}).get("violated", 0),
            "conformant": report.get("verification", {}).get("is_conformant", True),
        },
    }


# ── Pretty tables ────────────────────────────────────────────────────────────


def print_table_iii(evaluations: list[dict]):
    print(f"\n{'='*70}")
    print(f"  TABLE III: MAPPING ACCURACY (Mapper Agent)")
    print(f"{'='*70}")
    print(f"  {'Process':<25} {'Predicted':>10} {'GT':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for e in evaluations:
        m = e["mapping"]
        print(
            f"  {e['process']:<25} "
            f"{m['total_predicted']:>10} "
            f"{m['total_gt']:>10} "
            f"{m['precision']:>9.0%} "
            f"{m['recall']:>9.0%} "
            f"{m['f1']:>9.0%}"
        )
    print(f"{'='*70}")


def print_table_iv(evaluations: list[dict]):
    print(f"\n{'='*70}")
    print(f"  TABLE IV: CONSTRAINT EXTRACTION QUALITY (Formalizer Agent)")
    print(f"{'='*70}")
    print(f"  {'Process':<25} {'Predicted':>10} {'GT':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for e in evaluations:
        c = e["constraints"]
        print(
            f"  {e['process']:<25} "
            f"{c['total_predicted']:>10} "
            f"{c['total_gt']:>10} "
            f"{c['precision']:>9.0%} "
            f"{c['recall']:>9.0%} "
            f"{c['f1']:>9.0%}"
        )
    print(f"{'='*70}")


def print_table_ii(mutation_results: dict):
    print(f"\n{'='*70}")
    print(f"  TABLE II: END-TO-END INCONSISTENCY DETECTION")
    print(f"{'='*70}")

    for proc_name, results in mutation_results.items():
        active = [r for r in results if not r.get("skipped")]
        mutations_only = [r for r in active if r["mutation"] != "ORIGINAL"]
        original = next((r for r in active if r["mutation"] == "ORIGINAL"), None)
        orig_h = original["hybrid_violated"] if original else 0
        orig_b = original["baseline_violated"] if original and original["baseline_violated"] >= 0 else 0

        hybrid_detected = sum(
            1 for r in mutations_only
            if r["hybrid_violated"] > orig_h and r["hybrid_violated"] >= 0
        )
        baseline_has_data = any(r["baseline_violated"] >= 0 for r in mutations_only)
        baseline_detected = sum(
            1 for r in mutations_only
            if r["baseline_violated"] > orig_b and r["baseline_violated"] >= 0
        )
        total = len(mutations_only)
        h_rate = hybrid_detected / total if total > 0 else 0

        print(f"\n  {proc_name} ({total} mutations):")
        print(f"    {'Method':<30} {'Detected':>10} {'Total':>10} {'Rate':>10}")
        print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        print(f"    {'Hybrid (ours)':<30} {hybrid_detected:>10} {total:>10} {h_rate:>9.0%}")

        if baseline_has_data:
            b_rate = baseline_detected / total if total > 0 else 0
            print(f"    {'Pure LLM (baseline)':<30} {baseline_detected:>10} {total:>10} {b_rate:>9.0%}")
        else:
            print(f"    {'Pure LLM (baseline)':<30} {'N/A':>10} {total:>10} {'N/A':>10}")

    print(f"\n{'='*70}")


# ── Estimated time ───────────────────────────────────────────────────────────


def estimate_time(n_processes: int, run_baselines: bool) -> str:
    # Per process: 1 hybrid original + 5 mutations hybrid + (optional) 6 baseline
    # Each run: ~3 LLM calls for hybrid, ~1 for baseline
    # With 20s pauses between runs
    runs_per_process = 6  # original + 5 mutations
    if run_baselines:
        runs_per_process *= 2  # double for baseline

    total_runs = runs_per_process * n_processes
    # ~20s pause + ~10s for actual LLM call per run
    total_seconds = total_runs * 30
    minutes = total_seconds // 60
    return f"~{minutes} minutes"


# ── Main ──────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="Full evaluation suite for the paper.")
    parser.add_argument(
        "--process", "-p", choices=["dispatch", "recourse", "all"], default="all",
        help="Which process to evaluate (default: all).",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip pure-LLM baseline (2x faster).",
    )
    parser.add_argument(
        "--no-mutations", action="store_true",
        help="Skip mutation testing (much faster, only Tables III/IV).",
    )
    args = parser.parse_args()

    processes = ALL_PROCESSES
    if args.process != "all":
        processes = [p for p in ALL_PROCESSES if p["key"] == args.process]

    run_baselines = not args.no_baseline
    run_muts = not args.no_mutations

    est = estimate_time(len(processes), run_baselines) if run_muts else "~2 minutes"

    print(f"\n{'#'*70}")
    print(f"  KhPIWeek 2026 — FULL EVALUATION SUITE")
    print(f"  Processes: {', '.join(p['name'] for p in processes)}")
    print(f"  Baseline: {'yes' if run_baselines else 'no'}")
    print(f"  Mutations: {'yes' if run_muts else 'no'}")
    print(f"  Estimated time: {est}")
    print(f"{'#'*70}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ══ Phase 1: Hybrid pipeline ══
    print(f"\n{'─'*70}")
    print(f"  PHASE 1: Running hybrid pipeline")
    print(f"{'─'*70}")

    reports = {}
    for i, proc in enumerate(processes):
        if i > 0:
            await _phase_pause("next process")
        try:
            reports[proc["name"]] = await run_hybrid_pipeline(proc)
        except Exception as e:
            print(f"    FAILED: {e}")
            reports[proc["name"]] = {}

    # ══ Phase 2: Evaluate accuracy ══
    print(f"\n{'─'*70}")
    print(f"  PHASE 2: Evaluating against ground truth")
    print(f"{'─'*70}")

    evaluations = []
    for proc in processes:
        if reports.get(proc["name"]):
            ev = evaluate_process(reports[proc["name"]], proc["gt"], proc["name"])
            evaluations.append(ev)
            print(f"  {proc['name']}:")
            print(f"    Mapping:     P={ev['mapping']['precision']:.0%}  R={ev['mapping']['recall']:.0%}  F1={ev['mapping']['f1']:.0%}")
            print(f"    Constraints: P={ev['constraints']['precision']:.0%}  R={ev['constraints']['recall']:.0%}  F1={ev['constraints']['f1']:.0%}")

    # ══ Phase 3: Mutation testing ══
    mutation_results = {}
    if run_muts:
        print(f"\n{'─'*70}")
        print(f"  PHASE 3: Mutation testing")
        print(f"{'─'*70}")

        for i, proc in enumerate(processes):
            if i > 0 or reports:
                await _phase_pause("mutation testing")
            print(f"\n  Processing {proc['name']}...")
            out_dir = RESULTS_DIR / "mutations" / proc["key"]
            try:
                results = await run_mutations(
                    proc["bpmn"], proc["text"], str(out_dir), run_baselines=run_baselines
                )
                mutation_results[proc["name"]] = results
            except Exception as e:
                print(f"    FAILED: {e}")
                mutation_results[proc["name"]] = []

    # ══ Print tables ══
    print(f"\n\n{'#'*70}")
    print(f"  PAPER TABLES — READY TO COPY")
    print(f"{'#'*70}")

    if evaluations:
        print_table_iii(evaluations)
        print_table_iv(evaluations)

    if mutation_results:
        print_table_ii(mutation_results)

    # ══ Save ══
    summary = {
        "evaluations": evaluations,
        "mutation_results": {k: v for k, v in mutation_results.items()},
    }
    summary_path = RESULTS_DIR / "paper_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
