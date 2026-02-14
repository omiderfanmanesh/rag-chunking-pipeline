from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ...infrastructure.io import read_json, read_text
from ...domain.models import CanonicalBlock, PageRef

UUID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*)$")

BLOCK_ALLOWED_TYPES = {"title", "text", "list", "table_body", "equation", "table_caption", "table_footnote"}
CONTENT_ALLOWED_TYPES = {"text", "list", "table", "equation"}


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
    from ...use_cases.cleaning import IMAGE_LINE_RE  # Import here to avoid circular import

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


def load_canonical_blocks(choice: SourceChoice) -> tuple[list[CanonicalBlock], str]:
    from ...domain.models import SourceChoice  # Import here to avoid circular import

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