from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeepEvalGateConfig:
    """Configuration for deterministic DeepEval quality gates."""

    artifacts_dir: Path
    eval_report_path: Path
    output_json: Path
    tiny_chunk_tokens: int = 20
    max_tiny_chunk_pct: float = 0.5
    max_duplicate_instance_pct: float = 0.0
    max_overlap_p95_chars: int = 30
    max_missing_metadata_pct: float = 0.0
    overlap_scan_chars: int = 240
    max_mixed_article_pct: float = 2.0
    min_median_tokens: int = 100
    min_coverage_ratio: float = 95.0
