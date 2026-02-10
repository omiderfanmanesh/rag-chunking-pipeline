from __future__ import annotations

import re

from .models import CanonicalBlock

YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*[/_\-]\s*(\d{2,4})\b")
AY_RE = re.compile(r"(?i)\bA\.?\s*Y\.?\s*(20\d{2})\s*[/_\-]\s*(\d{2,4})\b")
SINGLE_YEAR_RE = re.compile(r"\b(20\d{2})\b")

SECTION_RE = re.compile(
    r"(?i)^\s*(?:#\s*)?(SECTION\s+[IVXLC0-9]+.*|SEZIONE\s+[IVXLC0-9]+.*|PART\s+[IVXLC0-9]+.*|CHAPTER\s+[IVXLC0-9]+.*|CAPO\s+[IVXLC0-9]+.*|TITOLO\s+[IVXLC0-9]+.*|(?:\d+\.)+\s+.*|GENERAL SECTION.*|SCHOLARSHIP\b.*|BORSA\b.*|ACCOMODATION\b.*|ACCOMMODATION\b.*)$"
)
ARTICLE_RE = re.compile(r"(?i)^\s*(?:#\s*)?(?:ART\.?|ARTICLE|ARTICOLO)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b(.*)$")
SUBARTICLE_INLINE_RE = re.compile(r"\((\d+(?:\.\d+)?)\)")
SUBARTICLE_DOTTED_RE = re.compile(r"^\s*(\d+\.\d+)\b")
ARTICLE_TOKEN_RE = re.compile(r"(?i)\b(?:ART\.?|ARTICLE|ARTICOLO)\b")

ITALIAN_HINTS = {"articolo", "sezione", "borsa", "studenti", "domanda", "graduatoria", "requisiti"}
ENGLISH_HINTS = {"article", "section", "scholarship", "students", "application", "ranking", "requirements"}
GENERIC_TITLE_WORDS = {
    "call",
    "bando",
    "notice",
    "avviso",
    "announcement",
    "regulation",
    "regolamento",
    "policy",
    "guidelines",
}


def normalize_year_range(start: int, end_raw: str) -> str:
    if len(end_raw) == 2:
        end = (start // 100) * 100 + int(end_raw)
        if end < start:
            end += 100
    else:
        end = int(end_raw)
    return f"{start}-{end}"


def extract_year(text: str) -> str | None:
    match = AY_RE.search(text)
    if match:
        return normalize_year_range(int(match.group(1)), match.group(2))

    match = YEAR_RANGE_RE.search(text)
    if match:
        return normalize_year_range(int(match.group(1)), match.group(2))

    match = SINGLE_YEAR_RE.search(text)
    if match:
        year = int(match.group(1))
        return str(year)
    return None


def extract_document_name(blocks: list[CanonicalBlock], fallback_name: str) -> str:
    title_candidates: list[str] = []
    for block in blocks[:60]:
        if block.block_type != "title":
            continue
        text = block.text.strip().lstrip("#").strip()
        if len(text) >= 4:
            title_candidates.append(text)

    def score_title(value: str) -> int:
        words = re.findall(r"[A-Za-zÀ-ÿ0-9']+", value.lower())
        if not words:
            return -1
        generic_hits = sum(1 for word in words if word in GENERIC_TITLE_WORDS)
        length_score = min(len(value), 120)
        word_score = min(len(words), 16) * 5
        specificity_penalty = generic_hits * 20
        return length_score + word_score - specificity_penalty

    if title_candidates:
        best = max(title_candidates, key=score_title)
        if score_title(best) >= 20:
            return best

    for block in blocks:
        text = block.text.strip().lstrip("#").strip()
        if len(text) >= 8:
            return text
    return fallback_name


def extract_brief_description(blocks: list[CanonicalBlock], title: str) -> str:
    candidates: list[str] = []
    for block in blocks:
        if block.block_type == "title":
            continue
        text = block.text.strip()
        if len(text) < 60:
            continue
        if text.lower().startswith("table:"):
            continue
        candidates.append(text)
        if len(candidates) >= 3:
            break
    if not candidates:
        return title
    merged = " ".join(candidates)
    merged = re.sub(r"\s+", " ", merged).strip()
    if len(merged) <= 320:
        return merged
    return merged[:317].rstrip() + "..."


def detect_language_hint(text: str) -> str | None:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return None
    it_score = sum(1 for token in words if token in ITALIAN_HINTS)
    en_score = sum(1 for token in words if token in ENGLISH_HINTS)
    if it_score >= 3 and it_score > en_score:
        return "it"
    if en_score >= 3 and en_score >= it_score:
        return "en"
    return None


def update_structure_state(
    text: str,
    section: str | None,
    article: str | None,
    subarticle: str | None,
    *,
    is_heading: bool = False,
) -> tuple[str | None, str | None, str | None]:
    line = text.strip()
    if not line:
        return section, article, subarticle

    section_match = SECTION_RE.match(line)
    if section_match:
        section = section_match.group(1).strip()

    article_match = ARTICLE_RE.match(line)
    if article_match:
        # Avoid treating body sentences that mention "Article X" as structural boundaries.
        article_mentions = len(ARTICLE_TOKEN_RE.findall(line))
        looks_like_title = is_heading or len(line) <= 140 or line.isupper()
        if article_mentions > 1 and not is_heading:
            looks_like_title = False
        if not looks_like_title:
            return section, article, subarticle

        article_full = article_match.group(1)
        article = article_full.split(".")[0]
        subarticle = article_full if "." in article_full else None
        remainder = article_match.group(2)
        inline_sub = SUBARTICLE_INLINE_RE.search(remainder) if is_heading else None
        if inline_sub:
            subarticle = inline_sub.group(1)
        return section, article, subarticle

    dotted = SUBARTICLE_DOTTED_RE.match(line)
    if dotted and article is not None:
        value = dotted.group(1)
        if value.startswith(f"{article}."):
            subarticle = value

    return section, article, subarticle
