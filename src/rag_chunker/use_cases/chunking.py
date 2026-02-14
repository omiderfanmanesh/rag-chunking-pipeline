from __future__ import annotations

from ..domain.models import CanonicalBlock, Segment
from .services.chunking_service import ChunkingService

# Backward-compatible module facade. Consumers can keep importing these symbols,
# while logic is owned by the ChunkingService class.
_DEFAULT_SERVICE = ChunkingService()


def count_tokens(text: str) -> int:
    return _DEFAULT_SERVICE.count_tokens(text)


def build_segments(blocks: list[CanonicalBlock]) -> list[Segment]:
    return _DEFAULT_SERVICE.build_segments(blocks)


def split_text_by_tokens(
    text: str,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    return _DEFAULT_SERVICE.split_text_by_tokens(
        text=text,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
