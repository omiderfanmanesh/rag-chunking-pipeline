from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .use_cases.augment import build_augmented_text
from .use_cases.chunking import build_segments, count_tokens, split_text_by_tokens
from .use_cases.cleaning import IMAGE_LINE_RE, clean_text
from .config import PipelineConfig
from .infrastructure.io import choose_source, discover_document_folders, read_json, read_text, write_json, write_jsonl
from .use_cases.metadata import detect_language_hint, extract_brief_description, extract_document_name, extract_year
from .domain.models import CanonicalBlock, PageRef, Segment, SourceChoice
from .use_cases.services.chunking_service import ChunkingService
from .use_cases.services.incremental_cache_service import IncrementalCacheService
from .use_cases.services.global_chunk_dedupe_service import GlobalChunkDedupeService
from .use_cases.services.tiny_chunk_sweep_service import TinyChunkSweepService
from .use_cases.services.block_loader_service import load_canonical_blocks
from .use_cases.services.segment_merge_service import _merge_toc_segments, _merge_small_segments, _dedup_chunk_boundaries
from .use_cases.services.chunk_assembly_service import _chunk_segment_texts, _merge_tiny_chunk_texts, _split_table_rows, _is_table_chunk_text
from .use_cases.services.structure_resolver_service import _resolve_structure, _resolve_chunk_article

UUID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
TOC_KEYWORD_RE = re.compile(r"(?i)\b(summary|sommario|indice|table of contents)\b")
TOC_ENTRY_RE = re.compile(r"(?i)^(?:#\s*)?(?:art\.?|article|articolo)\s*\d+(?:\.\d+)?\b.*\b\d{1,3}\s*$")
TOC_NUMBERED_ENTRY_RE = re.compile(r"(?i)^\s*(?:\d+(?:\.\d+){0,4})(?:\.?\s+|[.:])\S.*\s+\d{1,3}\s*$")
PAGE_TAIL_RE = re.compile(r"\b\d{1,3}\s*$")
STRUCTURAL_STUB_RE = re.compile(r"(?i)^\s*(?:#\s*)?(?:art\.?|article|articolo|section|sezione)\b")
SECTION_HEADING_RE = re.compile(r"(?i)\b(?:section|sezione)\s+[IVXLC0-9]+\b")
ARTICLE_HEADING_RE = re.compile(r"(?i)\b(?:art\.?|article|articolo)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b")
ARTICLE_LINE_RE = re.compile(r"(?i)^\s*(?:#\s*)?(?:art\.?|article|articolo)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b")
LEADING_NUMERIC_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _normalized_folder_title(folder_name: str) -> str:
    no_uuid = UUID_SUFFIX_RE.sub("", folder_name)
    no_pdf = no_uuid.removesuffix(".pdf")
    text = no_pdf.replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)





def _clean_blocks(blocks: list[CanonicalBlock]) -> list[CanonicalBlock]:
    cleaned: list[CanonicalBlock] = []
    for block in blocks:
        text = clean_text(block.text)
        if not text:
            continue
        cleaned.append(
            CanonicalBlock(
                text=text,
                block_type=block.block_type,
                page_refs=block.page_refs,
                heading_level=block.heading_level,
                source_hint=block.source_hint,
            )
        )
    return cleaned


def _page_meta(page_refs: list[PageRef]) -> tuple[int | None, int | None, list[dict[str, Any]]]:
    cleaned_refs = [ref for ref in page_refs if ref.page_idx >= 0]
    if not cleaned_refs:
        return None, None, []

    refs = sorted(
        {(ref.page_idx, ref.block_id, ref.block_position) for ref in cleaned_refs},
        key=lambda item: (item[0], item[2] or "", item[1] or ""),
    )
    # Convert to human-facing 1-based page numbers.
    page_start = min(item[0] for item in refs) + 1
    page_end = max(item[0] for item in refs) + 1
    refs_payload = [
        {"page_idx": page_idx + 1, "block_id": block_id, "block_position": block_position}
        for page_idx, block_id, block_position in refs
    ]
    return page_start, page_end, refs_payload


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


def _same_chunk_structure(chunk_row: dict[str, Any], *, section: str | None, article: str | None, subarticle: str | None) -> bool:
    metadata = chunk_row.get("metadata", {})
    return (
        metadata.get("section") == section
        and metadata.get("article") == article
        and metadata.get("subarticle") == subarticle
    )


def _compatible_chunk_structure(
    chunk_row: dict[str, Any],
    *,
    section: str | None,
    article: str | None,
    subarticle: str | None,
) -> bool:
    if _same_chunk_structure(chunk_row, section=section, article=article, subarticle=subarticle):
        return True
    metadata = chunk_row.get("metadata", {})
    existing_article = metadata.get("article")
    existing_section = metadata.get("section")
    if existing_article and article and existing_article == article:
        return True
    if (
        not existing_article
        and not article
        and existing_section
        and section
        and existing_section == section
    ):
        return True
    return False


def _merge_page_ref_payload(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, str | None, str | None]] = set()
    merged: list[dict[str, Any]] = []
    for ref in [*existing, *incoming]:
        page_idx = ref.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        key = (page_idx, ref.get("block_id"), ref.get("block_position"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "page_idx": page_idx,
                "block_id": ref.get("block_id"),
                "block_position": ref.get("block_position"),
            }
        )
    return sorted(
        merged,
        key=lambda item: (item["page_idx"], item.get("block_position") or "", item.get("block_id") or ""),
    )


def _resolve_structure(segment: Segment) -> tuple[str | None, str | None, str | None]:
    section = segment.section
    article = segment.article
    subarticle = segment.subarticle
    if section and article:
        return section, article, subarticle
    for heading in segment.heading_path:
        heading_text = heading.strip()
        if not section and SECTION_HEADING_RE.search(heading_text):
            section = heading_text
        article_match = ARTICLE_HEADING_RE.search(heading_text)
        if not article and article_match:
            full = article_match.group(1)
            article = full.split(".")[0]
            if not subarticle and "." in full:
                subarticle = full
    if not section and segment.heading_path:
        for heading in segment.heading_path:
            heading_text = heading.strip()
            if not heading_text:
                continue
            if ARTICLE_HEADING_RE.search(heading_text):
                continue
            section = heading_text
            break
        if not section:
            section = segment.heading_path[0].strip() or None
    return section, article, subarticle


def _line_article_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:10]:
        match = ARTICLE_LINE_RE.match(line)
        if not match:
            continue
        mentions.append(match.group(1))
    return mentions


def _resolve_chunk_article(
    chunk_text: str,
    *,
    fallback_article: str | None,
    fallback_subarticle: str | None,
) -> tuple[str | None, str | None]:
    mentions = _line_article_mentions(chunk_text)
    if not mentions:
        return fallback_article, fallback_subarticle

    roots = [value.split(".")[0] for value in mentions]
    unique_roots = sorted(set(roots))
    if fallback_article is None and len(unique_roots) == 1:
        chosen = mentions[0]
        chosen_root = chosen.split(".")[0]
        chosen_sub = chosen if "." in chosen else None
        return chosen_root, chosen_sub

    if fallback_article is not None and fallback_article not in unique_roots and len(unique_roots) == 1:
        chosen = mentions[0]
        chosen_root = chosen.split(".")[0]
        chosen_sub = chosen if "." in chosen else fallback_subarticle
        return chosen_root, chosen_sub

    return fallback_article, fallback_subarticle


def _article_from_section_label(section: str | None) -> tuple[str | None, str | None]:
    if not section:
        return None, None
    match = LEADING_NUMERIC_SECTION_RE.match(section.strip())
    if not match:
        return None, None
    value = match.group(1)
    root = value.split(".")[0]
    sub = value if "." in value else None
    return root, sub




    if not chunk_texts:
        return []
    working = [chunk.strip() for chunk in chunk_texts if chunk and chunk.strip()]
    if len(working) <= 1:
        return working

    out: list[str] = []
    idx = 0
    while idx < len(working):
        chunk = working[idx]
        tokens = count_tokens(chunk)
        if tokens >= min_tokens:
            out.append(chunk)
            idx += 1
            continue

        merged = False
        if idx + 1 < len(working):
            nxt = working[idx + 1]
            if _is_table_chunk_text(chunk) == _is_table_chunk_text(nxt):
                forward_candidate = chunk.rstrip() + "\n\n" + nxt.lstrip()
                if count_tokens(forward_candidate) <= max_tokens:
                    working[idx + 1] = forward_candidate
                    merged = True
        if merged:
            idx += 1
            continue

        if out and _is_table_chunk_text(out[-1]) == _is_table_chunk_text(chunk):
            backward_candidate = out[-1].rstrip() + "\n\n" + chunk.lstrip()
            if count_tokens(backward_candidate) <= max_tokens:
                out[-1] = backward_candidate
                idx += 1
                continue

        out.append(chunk)
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


def _final_tiny_chunk_sweep(chunk_rows: list[dict[str, Any]], *, max_tokens: int, sweep_tokens: int, min_viable_chunk_tokens: int) -> list[dict[str, Any]]:
    service = TinyChunkSweepService(
        max_tokens=max_tokens,
        sweep_tokens=sweep_tokens,
        count_tokens=count_tokens,
        looks_structural_stub=lambda text, token_count: _looks_structural_stub(text, token_count=token_count, threshold=min_viable_chunk_tokens),
        compatible_chunk_structure=lambda row, section, article, subarticle: _compatible_chunk_structure(
            row,
            section=section,
            article=article,
            subarticle=subarticle,
        ),
        merge_page_ref_payload=_merge_page_ref_payload,
        build_augmented_text=lambda text, name, year, brief_description, section, article, subarticle: build_augmented_text(
            text,
            name=name,
            year=year,
            brief_description=brief_description,
            section=section,
            article=article,
            subarticle=subarticle,
        ),
        sha1_func=_sha1,
        is_table_chunk_text=_is_table_chunk_text,
    )
    return service.sweep(chunk_rows)


def _is_toc_segment(segment: Segment) -> bool:
    heading_blob = " ".join(segment.heading_path).strip()
    text = segment.text.strip()
    if not text:
        return False
    if TOC_KEYWORD_RE.search(heading_blob) or TOC_KEYWORD_RE.search(text[:240]):
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    toc_entries = sum(1 for line in lines if TOC_ENTRY_RE.match(line))
    numbered_entries = sum(1 for line in lines if TOC_NUMBERED_ENTRY_RE.match(line))
    page_tail_lines = sum(1 for line in lines if PAGE_TAIL_RE.search(line))
    art_refs = sum(1 for line in lines if re.search(r'(?i)art\.?\s*\d+', line))
    return (
        toc_entries >= 3
        or numbered_entries >= 3
        or (toc_entries >= 2 and page_tail_lines >= 3)
        or (numbered_entries >= 2 and page_tail_lines >= 3)
        or art_refs >= 5
    )


def _is_toc_continuation(segment: Segment) -> bool:
    text = segment.text.strip()
    if not text:
        return False
    if segment.heading_path:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    toc_entries = sum(1 for line in lines if TOC_ENTRY_RE.match(line))
    numbered_entries = sum(1 for line in lines if TOC_NUMBERED_ENTRY_RE.match(line))
    page_tail_lines = sum(1 for line in lines if PAGE_TAIL_RE.search(line))
    return toc_entries >= 2 or numbered_entries >= 2 or page_tail_lines >= 2


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



    normalized = text.strip()
    if not normalized:
        return []

    def split_table_payload(payload: str) -> tuple[str | None, str]:
        payload_lines = payload.splitlines()
        table_lines: list[str] = []
        saw_pipe = False
        idx = 0
        while idx < len(payload_lines):
            line = payload_lines[idx].strip()
            if not line:
                if table_lines:
                    idx += 1
                    while idx < len(payload_lines) and not payload_lines[idx].strip():
                        idx += 1
                    break
                idx += 1
                continue

            if table_lines and saw_pipe and "|" not in line:
                break

            table_lines.append(line)
            if "|" in line:
                saw_pipe = True
            idx += 1

        trailing = "\n".join(payload_lines[idx:]).strip()
        if not table_lines:
            return None, payload.strip()
        if not saw_pipe:
            prose_like = any(len(line.split()) > 12 or line.endswith((".", ";")) for line in table_lines)
            if prose_like:
                prose = "\n".join(table_lines).strip()
                combined = "\n\n".join(part for part in [prose, trailing] if part).strip()
                return None, combined
        return "Table:\n" + "\n".join(table_lines), trailing

    fragments: list[tuple[str, str]] = []
    if "Table:\n" not in normalized:
        fragments.append(("text", normalized))
    else:
        parts = normalized.split("Table:\n")
        prefix = parts[0].strip()
        if prefix:
            fragments.append(("text", prefix))
        for payload in parts[1:]:
            table_fragment, trailing_fragment = split_table_payload(payload)
            if table_fragment:
                fragments.append(("table", table_fragment))
            if trailing_fragment:
                fragments.append(("text", trailing_fragment))

    chunk_pairs: list[tuple[str, str]] = []
    for fragment_kind, fragment_text in fragments:
        if not fragment_text.strip():
            continue
        if fragment_kind == "table":
            for table_chunk in _split_table_rows(fragment_text, max_tokens=max_tokens):
                if table_chunk.strip():
                    chunk_pairs.append(("table", table_chunk))
            continue
        text_chunks = split_text_by_tokens(
            fragment_text,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for text_chunk in text_chunks:
            if text_chunk.strip():
                chunk_pairs.append(("text", text_chunk))

    def normalize_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for chunk_kind, chunk_text in pairs:
            normalized_text = _normalize_table_chunk_text(chunk_text)
            if not normalized_text:
                continue
            normalized_kind = "table" if _is_table_chunk_text(normalized_text) else chunk_kind
            normalized.append((normalized_kind, normalized_text))
        return normalized

    chunk_pairs = normalize_pairs(chunk_pairs)
    chunks = [chunk_text for _, chunk_text in chunk_pairs]

    if len(chunks) <= 1:
        return chunks

    # Merge tiny tail chunk if it fits in the max token budget.
    if len(chunks[-1]) < min_chars and chunk_pairs[-1][0] == "text" and chunk_pairs[-2][0] == "text":
        merged_tail = chunks[-2].rstrip() + "\n\n" + chunks[-1].lstrip()
        if count_tokens(merged_tail) <= max_tokens:
            chunks = chunks[:-2] + [merged_tail]
            chunk_pairs = chunk_pairs[:-2] + [("text", merged_tail)]

    # Hard guard: never emit a chunk over max_tokens.
    fixed_pairs: list[tuple[str, str]] = []
    safe_overlap = max(1, min(overlap_tokens, target_tokens - 1, max_tokens - 1))
    for chunk_kind, chunk in chunk_pairs:
        if count_tokens(chunk) <= max_tokens:
            fixed_pairs.append((chunk_kind, chunk))
            continue
        split_chunks = split_text_by_tokens(
            chunk,
            target_tokens=max_tokens,
            max_tokens=max_tokens,
            overlap_tokens=safe_overlap,
        )
        fixed_pairs.extend((chunk_kind, split_chunk) for split_chunk in split_chunks if split_chunk.strip())
    fixed_pairs = normalize_pairs(fixed_pairs)
    return [chunk_text for _, chunk_text in fixed_pairs]


def _process_document_folder(folder: Path, config: PipelineConfig) -> tuple[dict, list[dict], dict]:
    choice = choose_source(folder, source_priority=config.source_priority)
    raw_blocks, source_mode_used = load_canonical_blocks(choice)
    cleaned_blocks = _clean_blocks(raw_blocks)
    if not cleaned_blocks:
        raise ValueError("No usable text blocks extracted")

    source_folder = str(folder.resolve())
    md_path = str(choice.md_path.resolve()) if choice.md_path is not None else None
    source_file = choice.md_path.name if choice.md_path is not None else folder.name
    fallback_name = _normalized_folder_title(folder.name)
    doc_id = _sha1(source_folder)[:16]

    preview_text = "\n".join(block.text for block in cleaned_blocks[:30])
    name = extract_document_name(cleaned_blocks, fallback_name=fallback_name)
    year = extract_year(preview_text)
    brief_description = extract_brief_description(cleaned_blocks, title=name)
    language_hint = detect_language_hint(preview_text)

    segments = build_segments(cleaned_blocks)
    segments = _merge_toc_segments(segments, max_tokens=config.max_tokens, drop_toc=config.drop_toc)
    segments = _merge_small_segments(
        segments,
        min_tokens=config.min_viable_chunk_tokens,
        max_tokens=config.max_tokens,
    )
    chunk_rows: list[dict] = []
    seen_chunk_texts: set[str] = set()
    chunk_index = 0
    for segment in segments:
        resolved_section, resolved_article, resolved_subarticle = _resolve_structure(segment)
        chunk_texts = _chunk_segment_texts(
            segment.text,
            target_tokens=config.target_tokens,
            max_tokens=config.max_tokens,
            overlap_tokens=config.overlap_tokens,
            min_chars=config.min_chars,
        )
        chunk_texts = _merge_tiny_chunk_texts(
            chunk_texts,
            min_tokens=config.min_chunk_tokens,
            max_tokens=config.max_tokens,
        )
        chunk_texts = _dedup_chunk_boundaries(
            chunk_texts,
            overlap_tokens=config.overlap_tokens,
        )
        for chunk_text in chunk_texts:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunk_article, chunk_subarticle = _resolve_chunk_article(
                chunk_text,
                fallback_article=resolved_article,
                fallback_subarticle=resolved_subarticle,
            )
            if chunk_article is None:
                section_article, section_subarticle = _article_from_section_label(resolved_section)
                if section_article is not None:
                    chunk_article = section_article
                    if chunk_subarticle is None:
                        chunk_subarticle = section_subarticle
            if chunk_article is None:
                chunk_article = "front_matter"
            token_count = count_tokens(chunk_text)
            if token_count < config.min_chunk_tokens:
                if _looks_structural_stub(chunk_text, token_count=token_count, threshold=config.min_viable_chunk_tokens):
                    continue
                if chunk_rows and _compatible_chunk_structure(
                    chunk_rows[-1],
                    section=resolved_section,
                    article=chunk_article,
                    subarticle=chunk_subarticle,
                ):
                    merged_candidate = chunk_rows[-1]["text"].rstrip() + "\n\n" + chunk_text
                    if count_tokens(merged_candidate) <= config.max_tokens:
                        old_text = chunk_rows[-1]["text"]
                        chunk_rows[-1]["text"] = merged_candidate
                        chunk_rows[-1]["token_count"] = count_tokens(merged_candidate)
                        chunk_rows[-1]["char_count"] = len(merged_candidate)
                        merged_refs = _merge_page_ref_payload(chunk_rows[-1].get("page_refs", []), _page_meta(segment.page_refs)[2])
                        chunk_rows[-1]["page_refs"] = merged_refs
                        if merged_refs:
                            chunk_rows[-1]["page_start"] = min(ref["page_idx"] for ref in merged_refs)
                            chunk_rows[-1]["page_end"] = max(ref["page_idx"] for ref in merged_refs)
                        chunk_rows[-1]["augmented_text"] = build_augmented_text(
                            merged_candidate,
                            name=name,
                            year=year,
                            brief_description=brief_description,
                            section=resolved_section,
                            article=chunk_article,
                            subarticle=chunk_subarticle,
                        )
                        if config.dedupe_chunks:
                            seen_chunk_texts.discard(old_text)
                            seen_chunk_texts.add(merged_candidate)
                        continue

            if config.dedupe_chunks and chunk_text in seen_chunk_texts:
                continue

            page_start, page_end, page_refs = _page_meta(segment.page_refs)
            chunk_id = _sha1(f"{doc_id}:{chunk_index}:{chunk_text[:80]}")[:20]
            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "augmented_text": build_augmented_text(
                        chunk_text,
                        name=name,
                        year=year,
                        brief_description=brief_description,
                        section=resolved_section,
                        article=chunk_article,
                        subarticle=chunk_subarticle,
                    ),
                    "token_count": token_count,
                    "char_count": len(chunk_text),
                    "source_path": source_folder,
                    "source_file": source_file,
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_refs": page_refs,
                    "metadata": {k: v for k, v in {
                        "year": year,
                        "name": name,
                        "brief_description": brief_description,
                        "section": resolved_section,
                        "article": chunk_article,
                        "subarticle": chunk_subarticle,
                        "heading_path": segment.heading_path,
                        "language_hint": language_hint,
                    }.items() if v is not None},
                }
            )
            if config.dedupe_chunks:
                seen_chunk_texts.add(chunk_text)
            chunk_index += 1

    chunk_rows = _final_tiny_chunk_sweep(
        chunk_rows,
        max_tokens=config.max_tokens,
        sweep_tokens=config.min_viable_chunk_tokens,
        min_viable_chunk_tokens=config.min_viable_chunk_tokens,
    )

    pages = sorted(
        {
            page_ref["page_idx"]
            for row in chunk_rows
            for page_ref in row.get("page_refs", [])
            if isinstance(page_ref.get("page_idx"), int)
        }
    )
    total_tokens = sum(row["token_count"] for row in chunk_rows)
    document_row = {
        "doc_id": doc_id,
        "source_folder": source_folder,
        "source_md_path": md_path,
        "source_mode_used": source_mode_used,
        "name": name,
        "year": year,
        "brief_description": brief_description,
        "language_hint": language_hint,
        "stats": {
            "blocks": len(cleaned_blocks),
            "chunks": len(chunk_rows),
            "pages": len(pages),
            "tokens": total_tokens,
        },
    }
    result_manifest = {
        "doc_id": doc_id,
        "source_folder": source_folder,
        "source_mode_used": source_mode_used,
        "fallback_reason": choice.fallback_reason,
        "warnings": [],
    }
    if source_mode_used != "block_list":
        result_manifest["warnings"].append("block_list not used")
    if year is None:
        result_manifest["warnings"].append("year not detected")
    return document_row, chunk_rows, result_manifest


def run_pipeline(config: PipelineConfig) -> dict:
    folders = discover_document_folders(config.input_dir)
    cache_service = IncrementalCacheService(
        output_dir=config.output_dir,
        sha1_func=_sha1,
        read_json=read_json,
        version=2,
    )
    snapshot = cache_service.load_snapshot()
    current_signature = cache_service.processing_signature(config)
    dedupe_service = GlobalChunkDedupeService(sha1_func=_sha1)
    documents: list[dict] = []
    chunks: list[dict] = []
    errors: list[dict] = []
    doc_results: list[dict] = []
    reusable_hashes: dict[str, dict[str, str]] = {}
    reused_documents = 0
    processed_documents = 0

    for folder in folders:
        source_folder = str(folder.resolve())
        doc_id = _sha1(source_folder)[:16]
        folder_hash = cache_service.compute_folder_hash(folder)
        if config.incremental and cache_service.can_reuse(
            doc_id=doc_id,
            folder_hash=folder_hash,
            processing_signature=current_signature,
            snapshot=snapshot,
        ):
            document_row = snapshot.docs_by_id[doc_id]
            chunk_rows = snapshot.chunks_by_doc.get(doc_id, [])
            documents.append(document_row)
            chunks.extend(chunk_rows)
            doc_results.append(
                {
                    "doc_id": doc_id,
                    "source_folder": source_folder,
                    "source_mode_used": document_row.get("source_mode_used", "unknown"),
                    "fallback_reason": "incremental reuse",
                    "warnings": [],
                    "reused": True,
                }
            )
            reusable_hashes[doc_id] = cache_service.build_entry(
                source_folder=source_folder,
                folder_hash=folder_hash,
                processing_signature=current_signature,
            )
            reused_documents += 1
            continue
        try:
            document_row, chunk_rows, doc_manifest = _process_document_folder(folder, config)
            documents.append(document_row)
            chunks.extend(chunk_rows)
            doc_results.append(doc_manifest)
            reusable_hashes[str(document_row.get("doc_id", doc_id))] = cache_service.build_entry(
                source_folder=source_folder,
                folder_hash=folder_hash,
                processing_signature=current_signature,
            )
            processed_documents += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_row = {"source_folder": str(folder.resolve()), "error": str(exc)}
            errors.append(error_row)
            print(f"[rag-chunker] Failed to process {folder.name}: {exc}", file=sys.stderr)
            if config.fail_fast:
                raise

    if config.dedupe_chunks:
        chunks = dedupe_service.apply(chunks, documents)

    output_dir = config.output_dir
    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "chunks.jsonl", chunks)

    source_mode_counts: dict[str, int] = {}
    for row in documents:
        mode = row.get("source_mode_used", "unknown")
        source_mode_counts[mode] = source_mode_counts.get(mode, 0) + 1

    manifest = {
        "input_dir": str(config.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "documents": len(documents),
        "chunks": len(chunks),
        "source_modes": source_mode_counts,
        "incremental": {
            "enabled": config.incremental,
            "processed_documents": processed_documents,
            "reused_documents": reused_documents,
        },
        "document_results": doc_results,
        "errors": errors,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    cache_service.write_cache(
        generated_at_utc=manifest["processed_at_utc"],
        entries=reusable_hashes,
        write_json=write_json,
    )
    return manifest
