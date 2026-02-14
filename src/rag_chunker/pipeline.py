from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .augment import build_augmented_text
from .chunking import build_segments, count_tokens, split_text_by_tokens
from .cleaning import IMAGE_LINE_RE, clean_text
from .io import choose_source, discover_document_folders, read_json, read_text, write_json, write_jsonl
from .metadata import detect_language_hint, extract_brief_description, extract_document_name, extract_year
from .models import CanonicalBlock, PageRef, Segment, SourceChoice

UUID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*)$")

BLOCK_ALLOWED_TYPES = {"title", "text", "list", "table_body", "equation", "table_caption", "table_footnote"}
CONTENT_ALLOWED_TYPES = {"text", "list", "table", "equation"}
SMALL_SEGMENT_TOKEN_THRESHOLD = 30
TOC_KEYWORD_RE = re.compile(r"(?i)\b(summary|sommario|indice|table of contents)\b")
TOC_ENTRY_RE = re.compile(r"(?i)^(?:#\s*)?(?:art\.?|article|articolo)\s*\d+(?:\.\d+)?\b.*\b\d{1,3}\s*$")
TOC_NUMBERED_ENTRY_RE = re.compile(r"(?i)^\s*(?:\d+(?:\.\d+){0,4})(?:\.?\s+|[.:])\S.*\s+\d{1,3}\s*$")
PAGE_TAIL_RE = re.compile(r"\b\d{1,3}\s*$")
STRUCTURAL_STUB_RE = re.compile(r"(?i)^\s*(?:#\s*)?(?:art\.?|article|articolo|section|sezione)\b")
SECTION_HEADING_RE = re.compile(r"(?i)\b(?:section|sezione)\s+[IVXLC0-9]+\b")
ARTICLE_HEADING_RE = re.compile(r"(?i)\b(?:art\.?|article|articolo)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b")
ARTICLE_LINE_RE = re.compile(r"(?i)^\s*(?:#\s*)?(?:art\.?|article|articolo)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b")
LEADING_NUMERIC_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    source_priority: str = "block_first"
    target_tokens: int = 450
    max_tokens: int = 520
    overlap_tokens: int = 30
    min_chars: int = 220
    min_chunk_tokens: int = 24
    drop_toc: bool = True
    dedupe_chunks: bool = True
    fail_fast: bool = False


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _normalized_folder_title(folder_name: str) -> str:
    no_uuid = UUID_SUFFIX_RE.sub("", folder_name)
    no_pdf = no_uuid.removesuffix(".pdf")
    text = no_pdf.replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)


def _load_blocks_from_block_list(path: Path) -> list[CanonicalBlock]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return []
    pages = payload.get("pdfData", [])
    if not isinstance(pages, list):
        return []

    raw_blocks: list[CanonicalBlock] = []
    block_position_to_idx: dict[str, int] = {}
    block_positions: list[str | None] = []
    for page in pages:
        if not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, dict):
                continue
            if item.get("is_discarded") is True:
                continue
            block_type = str(item.get("type", "")).strip()
            if block_type not in BLOCK_ALLOWED_TYPES:
                continue

            if block_type == "table_body":
                text = str(item.get("table_body") or item.get("text") or "")
            else:
                text = str(item.get("text") or item.get("content") or "")
            page_idx = item.get("page_idx")
            page_ref = []
            if isinstance(page_idx, int):
                page_ref.append(PageRef(page_idx=page_idx, block_id=item.get("id"), block_position=item.get("block_position")))
            heading_level = item.get("level") if block_type == "title" else None
            if not isinstance(heading_level, int):
                heading_level = None
            raw_blocks.append(
                CanonicalBlock(
                    text=text,
                    block_type=block_type,
                    page_refs=page_ref,
                    heading_level=heading_level,
                    source_hint=item.get("block_position"),
                )
            )
            block_position = item.get("block_position")
            block_positions.append(block_position if isinstance(block_position, str) else None)
            if isinstance(block_position, str):
                block_position_to_idx[block_position] = len(raw_blocks) - 1

    merge_connections = payload.get("mergeConnections") or []
    replacements: dict[int, CanonicalBlock] = {}
    skip: set[int] = set()
    if isinstance(merge_connections, list):
        for connection in merge_connections:
            if not isinstance(connection, dict):
                continue
            if connection.get("type") != "merge":
                continue
            positions = connection.get("blocks")
            if not isinstance(positions, list):
                continue
            indices = [block_position_to_idx[pos] for pos in positions if isinstance(pos, str) and pos in block_position_to_idx]
            if len(indices) < 2:
                continue
            indices = sorted(set(indices))
            base_idx = min(indices)
            base_block = raw_blocks[base_idx]
            merged_texts: list[str] = []
            merged_refs: list[PageRef] = []
            for idx in indices:
                text = raw_blocks[idx].text.strip()
                if text and (not merged_texts or merged_texts[-1] != text):
                    merged_texts.append(text)
                merged_refs.extend(raw_blocks[idx].page_refs)
                if idx != base_idx:
                    skip.add(idx)
            if not merged_texts:
                continue
            replacements[base_idx] = CanonicalBlock(
                text="\n".join(merged_texts),
                block_type=base_block.block_type,
                page_refs=merged_refs,
                heading_level=base_block.heading_level,
                source_hint=base_block.source_hint,
            )

    merged_blocks: list[CanonicalBlock] = []
    for idx, block in enumerate(raw_blocks):
        if idx in skip:
            continue
        merged_blocks.append(replacements.get(idx, block))
    return merged_blocks


def _load_blocks_from_content_list(path: Path) -> list[CanonicalBlock]:
    payload = read_json(path)
    if not isinstance(payload, list):
        return []
    blocks: list[CanonicalBlock] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type", "")).strip()
        if block_type not in CONTENT_ALLOWED_TYPES:
            continue

        page_idx = item.get("page_idx")
        page_refs = [PageRef(page_idx=page_idx)] if isinstance(page_idx, int) else []
        if block_type == "table":
            table_text = str(item.get("table_body") or "")
            if item.get("table_caption"):
                table_text += "\n" + " ".join(str(v) for v in item.get("table_caption"))
            if item.get("table_footnote"):
                table_text += "\n" + " ".join(str(v) for v in item.get("table_footnote"))
            text = table_text
            normalized_type = "table_body"
            heading_level = None
        else:
            text = str(item.get("text") or "")
            text_level = item.get("text_level")
            heading_level = text_level if isinstance(text_level, int) and text_level > 0 else None
            normalized_type = "title" if heading_level is not None else block_type

        blocks.append(
            CanonicalBlock(
                text=text,
                block_type=normalized_type,
                page_refs=page_refs,
                heading_level=heading_level,
            )
        )
    return blocks


def _load_blocks_from_md(path: Path) -> list[CanonicalBlock]:
    text = read_text(path)
    blocks: list[CanonicalBlock] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = "\n".join(paragraph_lines).strip()
        if paragraph:
            blocks.append(CanonicalBlock(text=paragraph, block_type="text"))
        paragraph_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if IMAGE_LINE_RE.match(line):
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            blocks.append(CanonicalBlock(text=heading_text, block_type="title", heading_level=level))
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph_lines.append(line)
    flush_paragraph()
    return blocks


def _load_canonical_blocks(choice: SourceChoice) -> tuple[list[CanonicalBlock], str]:
    if choice.mode == "block_list" and choice.block_path is not None:
        blocks = _load_blocks_from_block_list(choice.block_path)
        if blocks:
            return blocks, "block_list"
    if choice.content_path is not None:
        blocks = _load_blocks_from_content_list(choice.content_path)
        if blocks:
            return blocks, "content_list"
    if choice.md_path is not None:
        blocks = _load_blocks_from_md(choice.md_path)
        if blocks:
            return blocks, "md"
    return [], "none"


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


def _looks_structural_stub(text: str, *, token_count: int) -> bool:
    if token_count > SMALL_SEGMENT_TOKEN_THRESHOLD:
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


def _is_table_chunk_text(text: str) -> bool:
    return text.lstrip().startswith("Table:")


def _merge_tiny_chunk_texts(chunk_texts: list[str], *, min_tokens: int, max_tokens: int) -> list[str]:
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
    return (
        toc_entries >= 3
        or numbered_entries >= 3
        or (toc_entries >= 2 and page_tail_lines >= 3)
        or (numbered_entries >= 2 and page_tail_lines >= 3)
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


def _merge_toc_segments(segments: list[Segment], *, max_tokens: int) -> list[Segment]:
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


def _split_table_rows(text: str, *, max_tokens: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if not lines[0].startswith("Table:"):
        return [text.strip()]

    data_lines = lines[1:]

    def maybe_strip_orphan_prefix(chunk_text: str) -> str:
        chunk_lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        if not chunk_lines:
            return ""
        if not chunk_lines[0].startswith("Table:"):
            return chunk_text.strip()
        body = chunk_lines[1:]
        if any("|" in line for line in body):
            return chunk_text.strip()
        # Keep single-column tables, but drop "Table:" for prose-like tails.
        prose_like = any(len(line.split()) > 12 or line.endswith((".", ";")) for line in body)
        if prose_like:
            return "\n".join(body).strip()
        return chunk_text.strip()

    if not any("|" in line for line in data_lines):
        stripped = maybe_strip_orphan_prefix(text)
        return [stripped] if stripped else []

    header = lines[0]
    rows = data_lines
    chunks: list[str] = []
    current_rows: list[str] = []

    def flush_rows() -> None:
        if current_rows:
            chunk_text = maybe_strip_orphan_prefix(header + "\n" + "\n".join(current_rows))
            if chunk_text:
                chunks.append(chunk_text)

    for row in rows:
        row_candidate = header + "\n" + row
        if count_tokens(row_candidate) > max_tokens:
            flush_rows()
            current_rows.clear()
            split_rows = split_text_by_tokens(
                row,
                target_tokens=max_tokens,
                max_tokens=max_tokens,
                overlap_tokens=1,
            )
            for split_row in split_rows:
                split_row = split_row.strip()
                if not split_row:
                    continue
                chunk_text = maybe_strip_orphan_prefix(header + "\n" + split_row)
                if chunk_text:
                    chunks.append(chunk_text)
            continue

        expanded = header + "\n" + "\n".join(current_rows + [row])
        if current_rows and count_tokens(expanded) > max_tokens:
            flush_rows()
            current_rows.clear()
        current_rows.append(row)
    flush_rows()
    return [chunk for chunk in chunks if chunk.strip()] or [text.strip()]


def _chunk_segment_texts(text: str, *, target_tokens: int, max_tokens: int, overlap_tokens: int, min_chars: int) -> list[str]:
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
    return [chunk_text for _, chunk_text in fixed_pairs]


def _process_document_folder(folder: Path, config: PipelineConfig) -> tuple[dict, list[dict], dict]:
    choice = choose_source(folder, source_priority=config.source_priority)
    raw_blocks, source_mode_used = _load_canonical_blocks(choice)
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
    segments = _merge_toc_segments(segments, max_tokens=config.max_tokens)
    if config.drop_toc:
        segments = [segment for segment in segments if segment.section != "TABLE OF CONTENTS" and not _is_toc_segment(segment)]
    segments = _merge_small_segments(
        segments,
        min_tokens=SMALL_SEGMENT_TOKEN_THRESHOLD,
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
                if _looks_structural_stub(chunk_text, token_count=token_count):
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
                    "metadata": {
                        "year": year,
                        "name": name,
                        "brief_description": brief_description,
                        "section": resolved_section,
                        "article": chunk_article,
                        "subarticle": chunk_subarticle,
                        "heading_path": segment.heading_path,
                        "language_hint": language_hint,
                    },
                }
            )
            if config.dedupe_chunks:
                seen_chunk_texts.add(chunk_text)
            chunk_index += 1

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
    documents: list[dict] = []
    chunks: list[dict] = []
    errors: list[dict] = []
    doc_results: list[dict] = []

    for folder in folders:
        try:
            document_row, chunk_rows, doc_manifest = _process_document_folder(folder, config)
            documents.append(document_row)
            chunks.extend(chunk_rows)
            doc_results.append(doc_manifest)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_row = {"source_folder": str(folder.resolve()), "error": str(exc)}
            errors.append(error_row)
            print(f"[rag-chunker] Failed to process {folder.name}: {exc}", file=sys.stderr)
            if config.fail_fast:
                raise

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
        "document_results": doc_results,
        "errors": errors,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
