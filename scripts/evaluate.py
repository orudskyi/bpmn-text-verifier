"""Evaluate pipeline results against ground truth.

Calculates Precision / Recall / F1 for:
  - Mapping accuracy  (Table III in the paper)
  - Constraint extraction quality  (Table IV)
  - End-to-end violation detection  (feeds into Table II)

Usage::

    # Run pipeline first, save report:
    python scripts/run_pipeline.py data/dispatch/Dispatch-of-goods.bpmn \
        data/dispatch/DispatchDescription.txt -o results/dispatch_report.json

    # Evaluate against ground truth:
    python scripts/evaluate.py results/dispatch_report.json \
        data/ground_truth/dispatch_mapping.json
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass

# ── Add project root to path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ConformanceReport


# ── Metric helpers ────────────────────────────────────────────────────────────


@dataclass
class Scores:
    """Precision / Recall / F1 container."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"  TP={self.tp}  FP={self.fp}  FN={self.fn}\n"
            f"  Precision = {self.precision:.2%}\n"
            f"  Recall    = {self.recall:.2%}\n"
            f"  F1        = {self.f1:.2%}"
        )


# ── Mapping evaluation ───────────────────────────────────────────────────────


def evaluate_mappings(
    predicted: list[dict],
    ground_truth: list[dict],
    confidence_threshold: float = 0.0,
) -> Scores:
    """Compare predicted mappings with ground truth.

    A mapping is correct (TP) if fragment_id → node_id matches.
    Predicted mappings not in GT are FP.
    GT mappings not in predicted are FN.

    Args:
        predicted: List of Mapping dicts from ConformanceReport.
        ground_truth: List of mapping dicts from ground truth JSON.
        confidence_threshold: Ignore predicted mappings below this confidence.

    Returns:
        Scores with TP/FP/FN counts.
    """
    # Build GT lookup: fragment_id → node_id
    # Handle multi-node mappings (like frag_06 in Recourse which maps to two nodes)
    gt_map: dict[str, set[str]] = {}
    for m in ground_truth:
        fid = m["fragment_id"]
        if fid not in gt_map:
            gt_map[fid] = set()
        # Support both single node_id and primary/secondary pattern
        if "node_id" in m:
            gt_map[fid].add(m["node_id"])
        if "node_id_primary" in m:
            gt_map[fid].add(m["node_id_primary"])
        if "node_id_secondary" in m:
            gt_map[fid].add(m["node_id_secondary"])

    # Build predicted lookup: fragment_id → node_id
    pred_map: dict[str, str] = {}
    for m in predicted:
        if m.get("confidence", 1.0) >= confidence_threshold:
            # Skip NONE mappings
            if m.get("node_id", "NONE") != "NONE":
                pred_map[m["fragment_id"]] = m["node_id"]

    scores = Scores()

    # Check predictions against GT
    for fid, pred_nid in pred_map.items():
        if fid in gt_map and pred_nid in gt_map[fid]:
            scores.tp += 1
        else:
            scores.fp += 1

    # Count GT entries not found in predictions
    for fid, gt_nids in gt_map.items():
        if fid not in pred_map:
            scores.fn += len(gt_nids)
        elif pred_map[fid] not in gt_nids:
            # Predicted wrong node — already counted as FP above
            scores.fn += 1

    return scores


# ── Constraint evaluation ────────────────────────────────────────────────────


def _constraint_key(c: dict) -> tuple:
    """Create a comparable key from a constraint dict."""
    return (
        c.get("template", "").lower(),
        c.get("activity_a", "").lower().strip(),
        c.get("activity_b", "").lower().strip(),
    )


def evaluate_constraints(
    predicted: list[dict],
    ground_truth: list[dict],
) -> Scores:
    """Compare predicted DECLARE constraints with ground truth.

    A constraint is correct (TP) if (template, activity_a, activity_b) matches
    (case-insensitive, ignoring condition field).

    Args:
        predicted: List of DeclareConstraint dicts from ConformanceReport.
        ground_truth: List of constraint dicts from ground truth JSON.

    Returns:
        Scores with TP/FP/FN counts.
    """
    gt_keys = {_constraint_key(c) for c in ground_truth}
    pred_keys = {_constraint_key(c) for c in predicted}

    tp = len(gt_keys & pred_keys)
    fp = len(pred_keys - gt_keys)
    fn = len(gt_keys - pred_keys)

    return Scores(tp=tp, fp=fp, fn=fn)


# ── Violation detection evaluation ───────────────────────────────────────────


def evaluate_violations(
    report: dict,
    expected_violations: list[dict] | None = None,
) -> dict:
    """Summarize violation detection results.

    Args:
        report: Full ConformanceReport dict.
        expected_violations: Optional list of expected violation descriptions.

    Returns:
        Summary dict with counts.
    """
    verification = report.get("verification", {})
    return {
        "is_conformant": verification.get("is_conformant", True),
        "total_constraints": verification.get("total_constraints", 0),
        "satisfied": verification.get("satisfied", 0),
        "violated": verification.get("violated", 0),
        "violations": [
            {
                "constraint": v.get("constraint_description", ""),
                "trace_length": len(v.get("trace", [])),
                "has_explanation": bool(v.get("explanation", "")),
            }
            for v in verification.get("violations", [])
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline results against ground truth."
    )
    parser.add_argument(
        "report_json",
        help="Path to the pipeline ConformanceReport JSON file.",
    )
    parser.add_argument(
        "ground_truth_json",
        help="Path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence to consider a mapping (default: 0.0).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional: save evaluation results to JSON file.",
    )
    args = parser.parse_args()

    # Load data
    with open(args.report_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    with open(args.ground_truth_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    process_name = report.get("process_name", "Unknown")

    print(f"\n{'='*60}")
    print(f"  EVALUATION: {process_name}")
    print(f"{'='*60}")

    # ── Mapping evaluation ──
    print(f"\n--- Mapping Accuracy ---")
    mapping_scores = evaluate_mappings(
        report.get("mappings", []),
        gt.get("mappings", []),
        confidence_threshold=args.confidence_threshold,
    )
    print(mapping_scores)
    print(f"  Total predicted: {len(report.get('mappings', []))}")
    print(f"  Total ground truth: {len(gt.get('mappings', []))}")

    # ── Constraint evaluation ──
    print(f"\n--- Constraint Extraction ---")
    constraint_scores = evaluate_constraints(
        report.get("constraints", []),
        gt.get("constraints", []),
    )
    print(constraint_scores)
    print(f"  Total predicted: {len(report.get('constraints', []))}")
    print(f"  Total ground truth: {len(gt.get('constraints', []))}")

    # ── Violation summary ──
    print(f"\n--- Violation Detection ---")
    violation_summary = evaluate_violations(report)
    print(f"  Conformant: {violation_summary['is_conformant']}")
    print(f"  Constraints checked: {violation_summary['total_constraints']}")
    print(f"  Satisfied: {violation_summary['satisfied']}")
    print(f"  Violated: {violation_summary['violated']}")
    for v in violation_summary["violations"]:
        print(f"    • {v['constraint']} (trace len: {v['trace_length']})")

    print(f"\n{'='*60}\n")

    # ── Save results ──
    if args.output:
        results = {
            "process_name": process_name,
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
            "violations": violation_summary,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
