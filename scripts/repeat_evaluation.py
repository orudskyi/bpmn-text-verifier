"""Repeated-run evaluation harness (docs/ROADMAP.md T1.10 — "one draw is not a result").

Runs the full hybrid pipeline N times (default 5) on a process, evaluates
each run independently against ground truth, and reports median / mean /
standard deviation for:

    - mapping precision / recall / F1
    - constraint extraction precision / recall / F1
    - violation count on the unmutated model

Every individual run is persisted in full (report + per-run metrics), not
just the aggregate — reviewer R#3 asked for variance, so the raw draws must
stay inspectable, not be collapsed into a single summary line.

Also records, for every run, the exact model string, temperature and any
other generation parameters actually passed to the Gemini API — read from
``src/config.py`` and the LLM call sites, not asserted from memory.

Rate limiting: no manual pause is added here. ``src/rate_limiter.py``
already enforces a 14 RPM floor inside every mapper/formalizer/explainer
call (the singleton is shared process-wide), so N sequential pipeline
invocations are paced correctly without a fourth pacing mechanism on top
of it (see docs/AUDIT.md §6.5 on the three that already coexist).

Usage::

    # Dry run first — no API calls, verifies the wiring and the stats code:
    python scripts/repeat_evaluation.py --process dispatch --runs 2 --dry-run

    # Live:
    python scripts/repeat_evaluation.py --process dispatch --runs 5
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import evaluate_constraints, evaluate_mappings  # noqa: E402
from scripts.mutate import generation_params_meta  # noqa: E402

RESULTS_DIR = Path("results/repeated_runs")

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


# ══════════════════════════════════════════════════════════════════════════════
# --dry-run stubs: replace the three LLM call sites with deterministic,
# network-free functions matching the real signatures exactly. Module C
# (the verifier) is never mocked — it has no LLM calls per CLAUDE.md, so
# leaving it real exercises the same code path a live run would use.
# ══════════════════════════════════════════════════════════════════════════════


async def _dry_run_map_text_to_bpmn(graph, fragments):
    from src.models import BPMNNodeType, Mapping

    tasks = [n for n in graph.nodes if n.type == BPMNNodeType.TASK]
    if not tasks:
        return []
    return [
        Mapping(
            fragment_id=frag.id,
            fragment_text=frag.text,
            node_id=tasks[i % len(tasks)].id,
            node_name=tasks[i % len(tasks)].name,
            confidence=0.42,
        )
        for i, frag in enumerate(fragments)
    ]


async def _dry_run_extract_constraints(graph, fragments, mappings):
    from src.models import BPMNNodeType, DeclareConstraint, DeclareTemplate

    tasks = [n for n in graph.nodes if n.type == BPMNNodeType.TASK]
    if not tasks:
        return []
    constraints = [
        DeclareConstraint(
            id="dryrun_01",
            template=DeclareTemplate.EXISTENCE,
            activity_a=tasks[0].name,
            source_text="[dry-run stub — no LLM call made]",
        )
    ]
    if len(tasks) >= 2:
        constraints.append(
            DeclareConstraint(
                id="dryrun_02",
                template=DeclareTemplate.RESPONSE,
                activity_a=tasks[0].name,
                activity_b=tasks[1].name,
                source_text="[dry-run stub — no LLM call made]",
            )
        )
    return constraints


async def _dry_run_explain_violations(verification, constraints, graph):
    return verification  # explanations left empty; shape matches the real return value


@contextmanager
def _mock_llm_agents():
    with patch("src.orchestrator.map_text_to_bpmn", _dry_run_map_text_to_bpmn), \
         patch("src.orchestrator.extract_constraints", _dry_run_extract_constraints), \
         patch("src.orchestrator.explain_violations", _dry_run_explain_violations):
        yield


# ══════════════════════════════════════════════════════════════════════════════
# One pipeline invocation
# ══════════════════════════════════════════════════════════════════════════════


async def _invoke_pipeline(proc: dict) -> dict:
    from src.orchestrator import build_pipeline, initial_state

    pipeline = build_pipeline()
    result = await pipeline.ainvoke(initial_state(proc["bpmn"], proc["text"]))
    report = result.get("report")
    if report:
        return report.model_dump()
    return {"error": "no report produced", "errors": result.get("errors", [])}


def compute_run_metrics(report: dict, gt: dict) -> dict:
    mapping_scores = evaluate_mappings(report.get("mappings", []), gt.get("mappings", []))
    constraint_scores = evaluate_constraints(report.get("constraints", []), gt.get("constraints", []))
    return {
        "mapping": {
            "precision": mapping_scores.precision,
            "recall": mapping_scores.recall,
            "f1": mapping_scores.f1,
            "tp": mapping_scores.tp,
            "fp": mapping_scores.fp,
            "fn": mapping_scores.fn,
        },
        "constraints": {
            "precision": constraint_scores.precision,
            "recall": constraint_scores.recall,
            "f1": constraint_scores.f1,
            "tp": constraint_scores.tp,
            "fp": constraint_scores.fp,
            "fn": constraint_scores.fn,
        },
        "violations_unmutated": report.get("verification", {}).get("violated", 0),
    }


def _stat_block(values: list[float]) -> dict:
    n = len(values)
    return {
        "n": n,
        "values": values,
        "mean": statistics.mean(values) if n else None,
        "median": statistics.median(values) if n else None,
        # sample stdev needs n>=2; a single draw has undefined variance, not zero.
        "stdev": statistics.stdev(values) if n >= 2 else None,
        "min": min(values) if n else None,
        "max": max(values) if n else None,
    }


def aggregate_metrics(run_metrics: list[dict]) -> dict:
    fields = {
        "mapping_precision": [m["mapping"]["precision"] for m in run_metrics],
        "mapping_recall": [m["mapping"]["recall"] for m in run_metrics],
        "mapping_f1": [m["mapping"]["f1"] for m in run_metrics],
        "constraint_precision": [m["constraints"]["precision"] for m in run_metrics],
        "constraint_recall": [m["constraints"]["recall"] for m in run_metrics],
        "constraint_f1": [m["constraints"]["f1"] for m in run_metrics],
        "violations_unmutated": [m["violations_unmutated"] for m in run_metrics],
    }
    return {name: _stat_block(values) for name, values in fields.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════


async def run_repeated(proc: dict, n_runs: int, output_dir: Path, dry_run: bool) -> dict:
    with open(proc["gt"], "r", encoding="utf-8") as f:
        gt = json.load(f)

    proc_dir = output_dir / proc["key"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict] = []
    run_metrics: list[dict] = []

    for i in range(n_runs):
        label = "DRY-RUN, no API calls" if dry_run else "LIVE"
        print(f"\n  Run {i + 1}/{n_runs} [{label}] on {proc['name']}...")

        t0 = time.perf_counter()
        if dry_run:
            with _mock_llm_agents():
                report_dict = await _invoke_pipeline(proc)
        else:
            report_dict = await _invoke_pipeline(proc)
        duration_s = time.perf_counter() - t0

        run_dir = proc_dir / f"run_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        metrics = compute_run_metrics(report_dict, gt)
        metrics["duration_s"] = duration_s
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        metrics["pipeline_errors"] = report_dict.get("errors", [])
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        run_metrics.append(metrics)
        run_records.append({
            "run_index": i,
            "duration_s": duration_s,
            "report_path": str(run_dir / "report.json"),
            "metrics_path": str(run_dir / "metrics.json"),
        })

        m, c = metrics["mapping"], metrics["constraints"]
        print(
            f"    mapping F1={m['f1']:.2%}  constraints F1={c['f1']:.2%}  "
            f"violated={metrics['violations_unmutated']}  ({duration_s:.1f}s)"
        )

    summary = {
        "process": proc["name"],
        "process_key": proc["key"],
        "n_runs": n_runs,
        "dry_run": dry_run,
        "generation_params": generation_params_meta(),
        "runs": run_records,
        "aggregate": aggregate_metrics(run_metrics),
    }
    summary_path = proc_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Summary (median/mean/sd over {n_runs} run(s)) saved to {summary_path}")
    return summary


def print_summary_table(summary: dict):
    agg = summary["aggregate"]
    print(f"\n{'=' * 78}")
    print(f"  REPEATED RUN SUMMARY — {summary['process']}  (N={summary['n_runs']}, "
          f"{'DRY-RUN' if summary['dry_run'] else 'LIVE'})")
    print(f"{'=' * 78}")
    print(f"  {'Metric':<24} {'median':>9} {'mean':>9} {'sd':>9} {'min':>9} {'max':>9}")
    print(f"  {'-' * 24} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")
    for name, block in agg.items():
        sd = f"{block['stdev']:.4f}" if block["stdev"] is not None else "n/a"
        print(
            f"  {name:<24} {block['median']:>9.4f} {block['mean']:>9.4f} {sd:>9} "
            f"{block['min']:>9.4f} {block['max']:>9.4f}"
        )
    print(f"{'=' * 78}")


# ── Estimated cost/time for a LIVE run (no API access from this script) ───────


def estimate_live_run(n_runs: int, n_processes: int) -> dict:
    """Estimate a floor on wall-clock time for a live repeated run.

    Deliberately conservative and grounded in code, not memory:

    - Each pipeline invocation makes 2 LLM calls unconditionally (mapper,
      formalizer) and a 3rd (explainer) only if that run finds >=1 violation
      (src/orchestrator.py `_should_explain`).
    - src/rate_limiter.py enforces a 60/14 ≈ 4.3s minimum interval between
      calls (`RateLimiter(rpm=14)`), shared process-wide.
    - Actual Gemini response latency and any src/llm_retry.py backoff on
      429s are NOT included — neither is measured anywhere in this
      codebase (docs/ROADMAP.md T2.7), so real elapsed time will be higher
      than this floor, not lower.
    """
    interval_s = 60 / 14
    calls_min = n_runs * n_processes * 2
    calls_max = n_runs * n_processes * 3
    return {
        "n_runs": n_runs,
        "n_processes": n_processes,
        "llm_calls_total_range": [calls_min, calls_max],
        "wall_clock_floor_seconds_range": [
            round(calls_min * interval_s, 1),
            round(calls_max * interval_s, 1),
        ],
        "note": (
            "This is a floor from src/rate_limiter.py pacing alone "
            "(60/14s per call); it excludes actual Gemini response latency "
            "and any 429 backoff in src/llm_retry.py — real elapsed time "
            "will be higher."
        ),
        "cost": (
            "Not computed. No token counts are recorded anywhere in src/ "
            "or scripts/ (confirmed by inspection; no token-accounting "
            "code exists). A dollar figure needs either per-call token "
            "instrumentation — out of scope for this task, it would touch "
            "src/ — or the current Gemini API price sheet applied manually "
            "to the call counts above."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="Run the full evaluation N times and report variance (T1.10)."
    )
    parser.add_argument(
        "--process", "-p", choices=["dispatch", "recourse", "all"], default="dispatch",
        help="Which process to evaluate (default: dispatch — the only one currently "
             "evaluated per CLAUDE.md).",
    )
    parser.add_argument(
        "--runs", "-n", type=int, default=5,
        help="Number of repeated pipeline invocations (default: 5).",
    )
    parser.add_argument(
        "--output", "-o", default=str(RESULTS_DIR),
        help="Directory for per-run and aggregate results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mock all three LLM call sites (mapper, formalizer, explainer) — "
             "makes zero Gemini API calls. Use this to verify the runner and "
             "the statistics code before spending a live run.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    processes = ALL_PROCESSES if args.process == "all" else [
        p for p in ALL_PROCESSES if p["key"] == args.process
    ]

    est = estimate_live_run(args.runs, len(processes))

    print(f"\n{'#' * 78}")
    print(f"  REPEATED-RUN EVALUATION (T1.10)")
    print(f"  Process(es): {', '.join(p['name'] for p in processes)}")
    print(f"  Runs: {args.runs}    Mode: {'DRY-RUN (no API calls)' if args.dry_run else 'LIVE'}")
    if not args.dry_run:
        lo, hi = est["wall_clock_floor_seconds_range"]
        print(f"  Estimated time floor: {lo:.0f}s–{hi:.0f}s ({lo/60:.1f}–{hi/60:.1f} min)")
        print(f"  Estimated LLM calls: {est['llm_calls_total_range'][0]}–{est['llm_calls_total_range'][1]}")
        print(f"  Cost: {est['cost']}")
    print(f"{'#' * 78}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for proc in processes:
        summary = await run_repeated(proc, args.runs, output_dir, dry_run=args.dry_run)
        print_summary_table(summary)
        all_summaries[proc["key"]] = summary

    (output_dir / "estimate.json").write_text(
        json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
