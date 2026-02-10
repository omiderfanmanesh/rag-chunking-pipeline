from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluator import EvalConfig, run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate chunking and metadata quality from artifacts.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing chunks.jsonl, documents.jsonl, and run_manifest.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/eval_report.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/eval_report.md"),
        help="Output Markdown report path",
    )
    parser.add_argument("--target-tokens", type=int, default=450)
    parser.add_argument("--max-tokens", type=int, default=520)
    parser.add_argument("--small-chunk-threshold", type=int, default=20)
    parser.add_argument("--moderate-chunk-threshold", type=int, default=50)
    parser.add_argument("--max-small-chunk-pct", type=float, default=12.0)
    parser.add_argument("--max-article-mixed-pct", type=float, default=5.0)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with code 2 when quality gates fail",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EvalConfig(
        artifacts_dir=args.artifacts_dir,
        output_json=args.output_json,
        output_md=args.output_md,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        small_chunk_threshold=args.small_chunk_threshold,
        moderate_chunk_threshold=args.moderate_chunk_threshold,
        max_small_chunk_pct=args.max_small_chunk_pct,
        max_article_mixed_pct=args.max_article_mixed_pct,
        sample_size=args.sample_size,
    )
    report = run_evaluation(config)
    summary = report["summary"]
    gates = report["quality_gates"]
    print(f"Overall score: {summary['overall_score']} ({summary['overall_status']})")
    print(f"Documents: {summary['documents']}")
    print(f"Chunks: {summary['chunks']}")
    print(f"Quality gates passed: {gates['passed']}")
    print(f"JSON report: {config.output_json}")
    print(f"Markdown report: {config.output_md}")
    if args.fail_on_threshold and not gates["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
