from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MinerU-aware deterministic RAG chunking pipeline")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to MinerU output root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output artifacts directory")
    parser.add_argument(
        "--source-priority",
        type=str,
        default="block_first",
        choices=["block_first"],
        help="Source selection policy",
    )
    parser.add_argument("--target-tokens", type=int, default=450)
    parser.add_argument("--max-tokens", type=int, default=520)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--min-chars", type=int, default=220)
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if a document fails processing")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source_priority=args.source_priority,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chars=args.min_chars,
        fail_fast=args.fail_fast,
    )
    manifest = run_pipeline(config)
    print(f"Processed documents: {manifest['documents']}")
    print(f"Produced chunks: {manifest['chunks']}")
    if manifest["errors"]:
        print(f"Errors: {len(manifest['errors'])}")


if __name__ == "__main__":
    main()
