from __future__ import annotations

import re
from typing import Any

from ...domain.models import Segment

ARTICLE_HEADING_RE = re.compile(r"(?i)\b(?:art\.?|article|articolo)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b")
ARTICLE_LINE_RE = re.compile(r"(?i)^\s*(?:#\s*)?(?:art\.?|article|articolo)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b")
LEADING_NUMERIC_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")
SECTION_HEADING_RE = re.compile(r"(?i)\b(?:section|sezione)\s+[IVXLC0-9]+\b")


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