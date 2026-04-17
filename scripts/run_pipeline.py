#!/usr/bin/env python3
"""CLI entrypoint for the BPMN-Text Conformance Checker pipeline.

Usage:
    python -m scripts.run_pipeline <bpmn_file> <text_file>

Example:
    python -m scripts.run_pipeline \\
        data/dispatch/Dispatch-of-goods.bpmn \\
        data/dispatch/DispatchDescription.txt
"""

import asyncio
import json
import sys

from src.orchestrator import build_pipeline, initial_state


async def main() -> None:
    """Entry point: parse CLI args and run the full pipeline."""
    if len(sys.argv) != 3:
        print(
            "Usage: python -m scripts.run_pipeline <bpmn_file> <text_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    bpmn_file = sys.argv[1]
    text_file = sys.argv[2]

    print(f"Running pipeline for:\n  BPMN: {bpmn_file}\n  Text: {text_file}\n",
          file=sys.stderr)

    pipeline = build_pipeline()
    result = await pipeline.ainvoke(initial_state(bpmn_file, text_file))

    errors: list[str] = result.get("errors", [])
    if errors:
        print("PIPELINE ERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)

    report = result.get("report")
    if report:
        print(report.model_dump_json(indent=2))
    else:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
