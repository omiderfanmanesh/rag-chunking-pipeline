from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PageRef:
    page_idx: int
    block_id: str | None = None
    block_position: str | None = None


@dataclass
class CanonicalBlock:
    text: str
    block_type: str
    page_refs: list[PageRef] = field(default_factory=list)
    heading_level: int | None = None
    source_hint: str | None = None


@dataclass
class Segment:
    text: str
    page_refs: list[PageRef]
    section: str | None
    article: str | None
    subarticle: str | None
    heading_path: list[str]


@dataclass
class SourceChoice:
    mode: str
    folder: Path
    block_path: Path | None = None
    content_path: Path | None = None
    md_path: Path | None = None
    fallback_reason: str | None = None

