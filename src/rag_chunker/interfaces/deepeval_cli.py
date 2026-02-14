from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..use_cases.deepeval_gates import DeepEvalGateConfig, run_deepeval_gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic DeepEval quality gates for chunking artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--eval-report", type=Path, default=Path("artifacts/eval_report.json"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/deepeval_gate_report.json"))
    parser.add_argument("--tiny-chunk-tokens", type=int, default=20)
    parser.add_argument("--max-tiny-chunk-pct", type=float, default=1.0)
    parser.add_argument("--max-duplicate-instance-pct", type=float, default=0.0)
    parser.add_argument("--max-overlap-p95-chars", type=int, default=30)
    parser.add_argument("--max-missing-metadata-pct", type=float, default=0.0)
    parser.add_argument("--overlap-scan-chars", type=int, default=240)
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DeepEvalGateConfig(
        artifacts_dir=args.artifacts_dir,
        eval_report_path=args.eval_report,
        output_json=args.output_json,
        tiny_chunk_tokens=args.tiny_chunk_tokens,
        max_tiny_chunk_pct=args.max_tiny_chunk_pct,
        max_duplicate_instance_pct=args.max_duplicate_instance_pct,
        max_overlap_p95_chars=args.max_overlap_p95_chars,
        max_missing_metadata_pct=args.max_missing_metadata_pct,
        overlap_scan_chars=args.overlap_scan_chars,
    )
    try:
        report = run_deepeval_gates(config)
    except AssertionError as exc:
        print(f"DeepEval gates failed: {exc}")
        if args.fail_on_threshold:
            sys.exit(2)
        return

    print(f"DeepEval gates passed: {report['summary']['gates_passed']}")
    print(f"Tiny chunk pct: {report['metrics']['tiny_chunk_pct']}")
    print(f"Duplicate instance pct: {report['metrics']['duplicate_instance_pct']}")
    print(f"Overlap p95 chars: {report['metrics']['overlap_p95_chars']}")
    print(f"Missing required metadata pct: {report['metrics']['missing_required_metadata_pct']}")
    print(f"JSON report: {config.output_json}")


if __name__ == "__main__":
    main()
