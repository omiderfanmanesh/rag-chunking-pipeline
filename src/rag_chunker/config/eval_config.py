from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalConfig:
    """Configuration for artifact evaluation report generation."""

    artifacts_dir: Path
    output_json: Path
    output_md: Path
    target_tokens: int = 450
    max_tokens: int = 520
    small_chunk_threshold: int = 20
    moderate_chunk_threshold: int = 50
    sample_size: int = 8
    max_small_chunk_pct: float = 12.0
    max_article_mixed_pct: float = 5.0
