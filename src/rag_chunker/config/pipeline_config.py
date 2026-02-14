from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Runtime configuration for the chunking pipeline.

    This model intentionally keeps only primitive and path fields so that it
    can be serialized into incremental cache signatures without custom logic.
    """

    input_dir: Path
    output_dir: Path
    source_priority: str = "block_first"
    target_tokens: int = 420
    max_tokens: int = 480
    overlap_tokens: int = 30
    max_overlap_chars: int = 200
    min_chars: int = 220
    min_chunk_tokens: int = 24
    min_viable_chunk_tokens: int = 50
    drop_toc: bool = True
    dedupe_chunks: bool = True
    incremental: bool = True
    fail_fast: bool = False
    tokenizer_name: str = "Cohere/Cohere-embed-multilingual-v3.0"
