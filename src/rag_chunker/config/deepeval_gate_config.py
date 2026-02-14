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
    max_tiny_chunk_pct: float = 1.0
    max_duplicate_instance_pct: float = 0.0
    max_overlap_p95_chars: int = 30
    max_missing_metadata_pct: float = 0.0
    overlap_scan_chars: int = 240
