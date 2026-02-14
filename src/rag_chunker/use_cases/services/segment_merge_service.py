from __future__ import annotations

import re
from typing import Any

from ...domain.models import PageRef, Segment

def _dedupe_page_refs(page_refs: list[PageRef]) -> list[PageRef]:
    seen: set[tuple[int, str | None, str | None]] = set()
    out: list[PageRef] = []
    for ref in page_refs:
        key = (ref.page_idx, ref.block_id, ref.block_position)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _same_structure(a, b) -> bool:
    return a.section == b.section and a.article == b.article and a.subarticle == b.subarticle


def _merge_toc_segments(segments: list[Segment], *, max_tokens: int, drop_toc: bool) -> list[Segment]:
    if drop_toc:
        return [segment for segment in segments if not _is_toc_segment(segment)]
    if not segments:
        return []

    merged: list[Segment] = []
    buffer: list[Segment] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        text_parts = [segment.text.strip() for segment in buffer if segment.text.strip()]
        if not text_parts:
            return
        combined_text = "\n\n".join(text_parts)
        combined_refs = _dedupe_page_refs([ref for segment in buffer for ref in segment.page_refs])
        candidate = Segment(
            text=combined_text,
            page_refs=combined_refs,
            section="TABLE OF CONTENTS",
            article=None,
            subarticle=None,
            heading_path=["TABLE OF CONTENTS"],
        )
        if count_tokens(candidate.text) > max_tokens:
            # Keep a single marker segment for very large indexes; avoid polluting retrieval.
            summary_text = "\n".join(text_parts[:20]).strip()
            candidate = Segment(
                text=f"TABLE OF CONTENTS\n\n{summary_text}",
                page_refs=combined_refs,
                section="TABLE OF CONTENTS",
                article=None,
                subarticle=None,
                heading_path=["TABLE OF CONTENTS"],
            )
        merged.append(candidate)

    for segment in segments:
        if _is_toc_segment(segment) or (buffer and _is_toc_continuation(segment)):
            buffer.append(segment)
            continue
        flush_buffer()
        buffer = []
        merged.append(segment)
    flush_buffer()
    return merged


def _same_structure(a, b) -> bool:
    return a.section == b.section and a.article == b.article and a.subarticle == b.subarticle


def _compatible_for_small_merge(a, b) -> bool:
    if _same_structure(a, b):
        return True
    if a.article and b.article and a.article == b.article:
        return True
    if a.section and b.section and a.section == b.section and not a.article and not b.article:
        return True
    # Never merge segments whose heading_path differs at level ≥ 2
    if len(a.heading_path) >= 2 and len(b.heading_path) >= 2 and a.heading_path[1] != b.heading_path[1]:
        return False
    # Never merge across article boundaries even when both are None but heading text differs
    if a.article != b.article:
        return False
    if a.article is None and b.article is None and a.heading_path and b.heading_path and a.heading_path[0] != b.heading_path[0]:
        return False
    return False


def _merge_two_segments(a, b):
    heading_path = a.heading_path if a.heading_path else b.heading_path
    return type(a)(
        text=f"{a.text.rstrip()}\n\n{b.text.lstrip()}",
        page_refs=_dedupe_page_refs(a.page_refs + b.page_refs),
        section=a.section,
        article=a.article,
        subarticle=a.subarticle,
        heading_path=heading_path,
    )


def _prepend_segment_to_next(prefix, nxt):
    heading_path = nxt.heading_path if nxt.heading_path else prefix.heading_path
    return type(prefix)(
        text=f"{prefix.text.rstrip()}\n\n{nxt.text.lstrip()}",
        page_refs=_dedupe_page_refs(prefix.page_refs + nxt.page_refs),
        section=nxt.section,
        article=nxt.article,
        subarticle=nxt.subarticle,
        heading_path=heading_path,
    )


def _looks_structural_stub(text: str, *, token_count: int, threshold: int) -> bool:
    if token_count > threshold:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    if all((TOC_ENTRY_RE.match(line) or TOC_NUMBERED_ENTRY_RE.match(line)) for line in lines):
        if any(PAGE_TAIL_RE.search(line) for line in lines):
            return True
    if all(PAGE_TAIL_RE.search(line) for line in lines):
        if any(
            TOC_NUMBERED_ENTRY_RE.match(line)
            or re.match(r"^\s*(?:#\s*)?(?:art\.?|article|articolo|section|sezione)\b", line, re.IGNORECASE)
            for line in lines
        ):
            return True
    if any("|" in line for line in lines):
        return False
    if all(line.startswith("#") for line in lines):
        return True
    joined = " ".join(lines)
    if STRUCTURAL_STUB_RE.match(joined):
        return True
    if token_count <= 12 and joined.upper() == joined and any(ch.isalpha() for ch in joined):
        return True
    if token_count <= 12:
        words = re.findall(r"[A-Za-zÀ-ÿ']+", joined)
        if 2 <= len(words) <= 10 and all(word[0].isupper() for word in words if word and word[0].isalpha()):
            return True
    return False


def _merge_small_segments(segments, *, min_tokens: int, max_tokens: int):
    if not segments:
        return []

    working = list(segments)
    out = []
    idx = 0
    while idx < len(working):
        segment = working[idx]
        seg_tokens = count_tokens(segment.text)

        while seg_tokens < min_tokens and idx + 1 < len(working):
            nxt = working[idx + 1]
            if not _compatible_for_small_merge(segment, nxt):
                break
            merged_candidate = _merge_two_segments(segment, nxt)
            if count_tokens(merged_candidate.text) > max_tokens:
                break
            segment = merged_candidate
            idx += 1
            seg_tokens = count_tokens(segment.text)

        if seg_tokens < min_tokens and idx + 1 < len(working):
            nxt = working[idx + 1]
            line_count = len([line for line in segment.text.splitlines() if line.strip()])
            looks_like_label = line_count <= 2 and len(segment.text) <= 120
            if looks_like_label and _compatible_for_small_merge(segment, nxt):
                merged_candidate = _prepend_segment_to_next(segment, nxt)
                if count_tokens(merged_candidate.text) <= max_tokens:
                    segment = merged_candidate
                    idx += 1
                    seg_tokens = count_tokens(segment.text)

        if out and seg_tokens < min_tokens and _compatible_for_small_merge(out[-1], segment):
            merged_candidate = _merge_two_segments(out[-1], segment)
            if count_tokens(merged_candidate.text) <= max_tokens:
                out[-1] = merged_candidate
                idx += 1
                continue

        out.append(segment)
        idx += 1
    return out


def _dedup_chunk_boundaries(chunk_texts: list[str], *, overlap_tokens: int) -> list[str]:
    if len(chunk_texts) < 2:
        return chunk_texts
    out = [chunk_texts[0]]
    for curr in chunk_texts[1:]:
        prev = out[-1]
        overlap_chars = ChunkingService._max_suffix_prefix_overlap(prev, curr)
        min_len = min(len(prev), len(curr))
        if overlap_chars > 0.2 * min_len:
            # Trim the overlapping prefix from curr, preserving overlap_tokens worth of text
            # Approximate overlap_tokens tokens as overlap_tokens * 4 chars (rough estimate)
            preserve_chars = overlap_tokens * 4
            if overlap_chars > preserve_chars:
                trim_chars = overlap_chars - preserve_chars
                if trim_chars < len(curr):
                    curr = curr[trim_chars:].lstrip()
        out.append(curr)
    return out