"""CLI entrypoint for the BPMN-Text Conformance Checker pipeline.

Usage::

    # Print report to stdout:
    python scripts/run_pipeline.py data/dispatch/Dispatch-of-goods.bpmn \
        data/dispatch/DispatchDescription.txt

    # Save report to JSON file:
    python scripts/run_pipeline.py data/dispatch/Dispatch-of-goods.bpmn \
        data/dispatch/DispatchDescription.txt \
        -o results/dispatch_report.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator import build_pipeline, initial_state


async def main():
    parser = argparse.ArgumentParser(
        description="Run the BPMN-Text Conformance Checker pipeline."
    )
    parser.add_argument("bpmn_file", help="Path to BPMN 2.0 XML file.")
    parser.add_argument("text_file", help="Path to text description file.")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save report JSON to this file path.",
    )
    args = parser.parse_args()

    pipeline = build_pipeline()
    result = await pipeline.ainvoke(initial_state(args.bpmn_file, args.text_file))

    if result.get("errors"):
        print(f"ERRORS: {result['errors']}", file=sys.stderr)

    report = result.get("report")
    if report:
        report_json = report.model_dump_json(indent=2)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(report_json, encoding="utf-8")
            print(f"Report saved to {args.output}")
        else:
            print(report_json)
    else:
        print("No report generated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
