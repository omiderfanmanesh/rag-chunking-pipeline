from __future__ import annotations

import re
import warnings

from ...domain.models import CanonicalBlock, PageRef, Segment
from ..metadata import update_structure_state

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?;:][\"')\]]*$")
SENTENCE_BREAK_RE = re.compile(r"[.!?;:]\s+")
OVERLAP_SCAN_CHARS = 320
MAX_CONSECUTIVE_OVERLAP_CHARS = 30


class ChunkingService:
    """Token-aware segmentation and chunk splitting service."""

    def __init__(self) -> None:
        self._fallback_token_safety_factor = 1.2
        self._tokenizer = self._load_tokenizer()
        if self._tokenizer is None:  # pragma: no cover
            warnings.warn(
                "tiktoken is not installed; using conservative regex token fallback. Install dependencies for exact token budgets.",
                RuntimeWarning,
            )

    @staticmethod
    def _load_tokenizer():
        try:  # pragma: no cover
            from tokenizers import Tokenizer

            return Tokenizer.from_pretrained("Cohere/Cohere-embed-multilingual-v3.0")
        except Exception:  # pragma: no cover
            try:  # pragma: no cover
                import tiktoken

                return tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover
                return None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            encoding = self._tokenizer.encode(text)
            return len(encoding.ids)
        return int(len(TOKEN_RE.findall(text)) * self._fallback_token_safety_factor + 0.5)

    def _fallback_budget(self, value: int) -> int:
        return max(1, int(value / self._fallback_token_safety_factor))

    @staticmethod
    def _ends_with_sentence_boundary(text: str) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return False
        if stripped.endswith("\n\n"):
            return True
        tail = stripped[-6:]
        return bool(SENTENCE_END_RE.search(tail))

    @staticmethod
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

    def _rebalance_tiny_tail(self, chunks: list[str], *, target_tokens: int, max_tokens: int) -> list[str]:
        if len(chunks) < 2:
            return chunks
        tail_threshold = max(20, target_tokens // 6)
        if self.count_tokens(chunks[-1]) >= tail_threshold:
            return chunks
        merged = chunks[-2].rstrip() + "\n\n" + chunks[-1].lstrip()
        if self.count_tokens(merged) > max_tokens:
            return chunks
        return chunks[:-2] + [merged]

    @staticmethod
    def _max_suffix_prefix_overlap(left: str, right: str, *, scan_chars: int = OVERLAP_SCAN_CHARS) -> int:
        left_tail = left[-scan_chars:]
        right_head = right[:scan_chars]
        max_len = min(len(left_tail), len(right_head))
        for overlap in range(max_len, 0, -1):
            if left_tail[-overlap:] == right_head[:overlap]:
                return overlap
        return 0

    @staticmethod
    def _looks_sentence_start(text: str) -> bool:
        cleaned = text.lstrip()
        if not cleaned:
            return False
        if cleaned.startswith(("Table:", "#", "- ", "* ")):
            return True
        return cleaned[0].isupper() or cleaned[0].isdigit()

    def _next_start_index_regex(
        self,
        text: str,
        matches: list[re.Match[str]],
        *,
        start: int,
        end: int,
        overlap_tokens: int,
    ) -> int:
        min_start = max(start + 1, end - overlap_tokens)
        end_char = matches[end - 1].span()[1]
        idx = min_start
        while idx < end:
            overlap_chars = end_char - matches[idx].span()[0]
            if overlap_chars <= MAX_CONSECUTIVE_OVERLAP_CHARS:
                break
            idx += 1
        for probe in range(idx, end):
            probe_start = matches[probe].span()[0]
            if end_char - probe_start > MAX_CONSECUTIVE_OVERLAP_CHARS:
                continue
            if self._looks_sentence_start(text[probe_start : min(len(text), probe_start + 64)]):
                return probe
        return min(idx, end)

    def _next_start_index_bpe(
        self,
        token_ids: list[int],
        *,
        start: int,
        end: int,
        overlap_tokens: int,
    ) -> int:
        min_start = max(start + 1, end - overlap_tokens)
        idx = min_start
        while idx < end:
            overlap_text = self._tokenizer.decode(token_ids[idx:end])
            if len(overlap_text) <= MAX_CONSECUTIVE_OVERLAP_CHARS:
                break
            idx += 1
        for probe in range(idx, end):
            overlap_text = self._tokenizer.decode(token_ids[probe:end])
            if len(overlap_text) > MAX_CONSECUTIVE_OVERLAP_CHARS:
                continue
            preview = self._tokenizer.decode(token_ids[probe : min(end, probe + 16)])
            if self._looks_sentence_start(preview):
                return probe
        return min(idx, end)

    def _split_text_by_regex_tokens(
        self,
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
            chunk_text = self._strip_fragment_prefix(chunk_text, is_first_chunk=not out)
            if chunk_text:
                out.append(chunk_text)
            if end >= token_count:
                break
            next_start = self._next_start_index_regex(
                text,
                matches,
                start=start,
                end=end,
                overlap_tokens=overlap_tokens,
            )
            if next_start <= start:
                next_start = end
            start = next_start
        return self._rebalance_tiny_tail(out, target_tokens=target_tokens, max_tokens=max_tokens)

    def _choose_end_index_bpe(
        self,
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
            if self._ends_with_sentence_boundary(self._tokenizer.decode(token_ids[start:end])):
                return end
        for end in range(candidate_end, min_end - 1, -1):
            if self._ends_with_sentence_boundary(self._tokenizer.decode(token_ids[start:end])):
                return end
        return candidate_end

    @staticmethod
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

    def build_segments(self, blocks: list[CanonicalBlock]) -> list[Segment]:
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
                    page_refs=self._dedupe_page_refs(current_refs),
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
        self,
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

        if self._tokenizer is None:
            safe_target = self._fallback_budget(target_tokens)
            safe_max = max(safe_target, self._fallback_budget(max_tokens))
            safe_overlap = min(max(1, self._fallback_budget(overlap_tokens)), safe_target - 1)
            return self._split_text_by_regex_tokens(text, safe_target, safe_max, safe_overlap)

        token_ids = self._tokenizer.encode(text).ids
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
                end = self._choose_end_index_bpe(
                    token_ids,
                    start=start,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                )

            if end <= start:
                break

            chunk_text = self._tokenizer.decode(token_ids[start:end])
            chunk_text = self._strip_fragment_prefix(chunk_text, is_first_chunk=not out)
            if chunk_text:
                out.append(chunk_text)
            if end >= token_count:
                break

            next_start = self._next_start_index_bpe(
                token_ids,
                start=start,
                end=end,
                overlap_tokens=overlap_tokens,
            )
            if next_start <= start:
                next_start = end
            start = next_start

        return self._rebalance_tiny_tail(out, target_tokens=target_tokens, max_tokens=max_tokens)
