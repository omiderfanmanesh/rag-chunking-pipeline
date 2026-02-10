from __future__ import annotations

import re
import warnings

from .metadata import update_structure_state
from .models import CanonicalBlock, PageRef, Segment

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?;:][\"')\]]*$")
SENTENCE_BREAK_RE = re.compile(r"[.!?;:]\s+")

try:  # pragma: no cover - import fallback covered by behavior tests
    import tiktoken
except Exception:  # pragma: no cover - package may be absent in lightweight environments
    tiktoken = None

_TOKENIZER = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
FALLBACK_TOKEN_SAFETY_FACTOR = 1.2
if _TOKENIZER is None:  # pragma: no cover - behavior is validated indirectly in tests
    warnings.warn(
        "tiktoken is not installed; using conservative regex token fallback. Install dependencies for exact token budgets.",
        RuntimeWarning,
    )


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text, disallowed_special=()))
    return int(len(TOKEN_RE.findall(text)) * FALLBACK_TOKEN_SAFETY_FACTOR + 0.5)


def _fallback_budget(value: int) -> int:
    return max(1, int(value / FALLBACK_TOKEN_SAFETY_FACTOR))


def _ends_with_sentence_boundary(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith("\n\n"):
        return True
    tail = stripped[-6:]
    return bool(SENTENCE_END_RE.search(tail))


def _strip_fragment_prefix(text: str, *, is_first_chunk: bool) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if is_first_chunk:
        return cleaned
    if cleaned.startswith(("Table:", "#", "- ", "* ")):
        return cleaned
    if cleaned[0].isupper() or cleaned[0].isdigit():
        return cleaned
    match = SENTENCE_BREAK_RE.search(cleaned[:220])
    if not match:
        return cleaned
    candidate = cleaned[match.end() :].lstrip()
    if len(candidate) < 24:
        return cleaned
    return candidate


def _rebalance_tiny_tail(chunks: list[str], *, target_tokens: int, max_tokens: int) -> list[str]:
    if len(chunks) < 2:
        return chunks
    tail_threshold = max(20, target_tokens // 6)
    if count_tokens(chunks[-1]) >= tail_threshold:
        return chunks
    merged = chunks[-2].rstrip() + "\n\n" + chunks[-1].lstrip()
    if count_tokens(merged) > max_tokens:
        return chunks
    return chunks[:-2] + [merged]


def _split_text_by_regex_tokens(
    text: str,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    matches = list(TOKEN_RE.finditer(text))
    token_count = len(matches)
    if token_count == 0:
        return []
    if token_count <= max_tokens:
        return [text.strip()]

    out: list[str] = []
    start = 0
    while start < token_count:
        remaining = token_count - start
        if remaining <= max_tokens:
            end = token_count
        else:
            end = min(start + target_tokens, token_count)
        if end <= start:
            break
        start_char = matches[start].span()[0]
        end_char = matches[end - 1].span()[1]
        chunk_text = text[start_char:end_char]
        chunk_text = _strip_fragment_prefix(chunk_text, is_first_chunk=not out)
        if chunk_text:
            out.append(chunk_text)
        if end >= token_count:
            break
        next_start = end - overlap_tokens
        if next_start <= start:
            next_start = end
        start = next_start
    return _rebalance_tiny_tail(out, target_tokens=target_tokens, max_tokens=max_tokens)


def _choose_end_index_bpe(
    token_ids: list[int],
    *,
    start: int,
    target_tokens: int,
    max_tokens: int,
) -> int:
    candidate_end = min(start + target_tokens, len(token_ids))
    max_end = min(start + max_tokens, len(token_ids))
    min_end = min(start + max(32, target_tokens // 3), max_end)

    probe_forward_limit = min(max_end, candidate_end + 24)
    for end in range(candidate_end, probe_forward_limit + 1):
        if _ends_with_sentence_boundary(_TOKENIZER.decode(token_ids[start:end])):
            return end
    for end in range(candidate_end, min_end - 1, -1):
        if _ends_with_sentence_boundary(_TOKENIZER.decode(token_ids[start:end])):
            return end
    return candidate_end


def _dedupe_page_refs(page_refs: list[PageRef]) -> list[PageRef]:
    seen: set[tuple[int, str | None, str | None]] = set()
    deduped: list[PageRef] = []
    for ref in page_refs:
        key = (ref.page_idx, ref.block_id, ref.block_position)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def build_segments(blocks: list[CanonicalBlock]) -> list[Segment]:
    segments: list[Segment] = []
    heading_path: list[str] = []
    section: str | None = None
    article: str | None = None
    subarticle: str | None = None

    current_text_parts: list[str] = []
    current_refs: list[PageRef] = []
    current_heading_path: list[str] = []
    current_section: str | None = None
    current_article: str | None = None
    current_subarticle: str | None = None

    def flush() -> None:
        if not current_text_parts:
            return
        text = "\n\n".join(part for part in current_text_parts if part).strip()
        if not text:
            return
        segments.append(
            Segment(
                text=text,
                page_refs=_dedupe_page_refs(current_refs),
                section=current_section,
                article=current_article,
                subarticle=current_subarticle,
                heading_path=list(current_heading_path),
            )
        )

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue

        if block.heading_level is not None:
            level = max(1, block.heading_level)
            while len(heading_path) >= level:
                heading_path.pop()
            heading_path.append(text.lstrip("#").strip())

        new_section, new_article, new_subarticle = update_structure_state(
            text,
            section,
            article,
            subarticle,
            is_heading=block.heading_level is not None,
        )
        structure_changed = (new_section != section) or (new_article != article) or (new_subarticle != subarticle)
        is_boundary = block.heading_level is not None or structure_changed

        if current_text_parts and is_boundary:
            flush()
            current_text_parts = []
            current_refs = []

        section, article, subarticle = new_section, new_article, new_subarticle
        if not current_text_parts:
            current_heading_path = list(heading_path)
            current_section = section
            current_article = article
            current_subarticle = subarticle

        current_text_parts.append(text)
        current_refs.extend(block.page_refs)

    flush()
    return segments


def split_text_by_tokens(
    text: str,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if max_tokens < target_tokens:
        raise ValueError("max_tokens must be >= target_tokens")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")

    if _TOKENIZER is None:
        safe_target = _fallback_budget(target_tokens)
        safe_max = max(safe_target, _fallback_budget(max_tokens))
        safe_overlap = min(max(1, _fallback_budget(overlap_tokens)), safe_target - 1)
        return _split_text_by_regex_tokens(text, safe_target, safe_max, safe_overlap)

    token_ids = _TOKENIZER.encode(text, disallowed_special=())
    token_count = len(token_ids)
    if token_count == 0:
        return []
    if token_count <= max_tokens:
        return [text.strip()]

    out: list[str] = []
    start = 0
    while start < token_count:
        remaining = token_count - start
        if remaining <= max_tokens:
            end = token_count
        else:
            end = _choose_end_index_bpe(
                token_ids,
                start=start,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
            )

        if end <= start:
            break

        chunk_text = _TOKENIZER.decode(token_ids[start:end])
        chunk_text = _strip_fragment_prefix(chunk_text, is_first_chunk=not out)
        if chunk_text:
            out.append(chunk_text)
        if end >= token_count:
            break

        next_start = end - overlap_tokens
        if next_start <= start:
            next_start = end
        start = next_start

    return _rebalance_tiny_tail(out, target_tokens=target_tokens, max_tokens=max_tokens)
