from __future__ import annotations

from typing import Any, Callable


class TinyChunkSweepService:
    """Final pass that merges or drops residual tiny chunks safely.

    The service relies on injected callbacks to keep policy logic (structure
    compatibility and table/prose checks) in the caller module.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        sweep_tokens: int,
        count_tokens: Callable[[str], int],
        looks_structural_stub: Callable[[str, int], bool],
        compatible_chunk_structure: Callable[[dict[str, Any], str | None, str | None, str | None], bool],
        merge_page_ref_payload: Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[dict[str, Any]]],
        build_augmented_text: Callable[[str, str | None, str | None, str | None, str | None, str | None, str | None], str],
        sha1_func: Callable[[str], str],
        is_table_chunk_text: Callable[[str], bool],
    ) -> None:
        self.max_tokens = max_tokens
        self.sweep_tokens = sweep_tokens
        self._count_tokens = count_tokens
        self._looks_structural_stub = looks_structural_stub
        self._compatible_chunk_structure = compatible_chunk_structure
        self._merge_page_ref_payload = merge_page_ref_payload
        self._build_augmented_text = build_augmented_text
        self._sha1 = sha1_func
        self._is_table_chunk_text = is_table_chunk_text

    def sweep(self, chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(chunk_rows) <= 1:
            return chunk_rows

        working = [row for row in chunk_rows if str(row.get("text", "")).strip()]
        if len(working) <= 1:
            return working

        idx = 0
        while idx < len(working):
            row = working[idx]
            text = str(row.get("text", "")).strip()
            token_count = int(row.get("token_count", self._count_tokens(text)))
            if token_count >= self.sweep_tokens:
                idx += 1
                continue

            if self._looks_structural_stub(text, token_count):
                del working[idx]
                continue

            metadata = row.get("metadata", {})
            section = metadata.get("section")
            article = metadata.get("article")
            subarticle = metadata.get("subarticle")
            is_table = self._is_table_chunk_text(text)

            merged = False
            if idx > 0:
                prev = working[idx - 1]
                if self._is_compatible_neighbor(prev, section, article, subarticle, is_table):
                    merged_text = str(prev.get("text", "")).rstrip() + "\n\n" + text
                    if self._count_tokens(merged_text) <= self.max_tokens:
                        merged_refs = self._merge_page_ref_payload(prev.get("page_refs", []), row.get("page_refs", []))
                        self._refresh_chunk_row_text(prev, merged_text)
                        self._set_chunk_row_page_meta(prev, merged_refs)
                        del working[idx]
                        merged = True
            if merged:
                continue

            if idx + 1 < len(working):
                nxt = working[idx + 1]
                if self._is_compatible_neighbor(nxt, section, article, subarticle, is_table):
                    merged_text = text + "\n\n" + str(nxt.get("text", "")).lstrip()
                    if self._count_tokens(merged_text) <= self.max_tokens:
                        merged_refs = self._merge_page_ref_payload(row.get("page_refs", []), nxt.get("page_refs", []))
                        self._refresh_chunk_row_text(nxt, merged_text)
                        self._set_chunk_row_page_meta(nxt, merged_refs)
                        del working[idx]
                        continue

            idx += 1

        for new_idx, row in enumerate(working):
            row["chunk_index"] = new_idx
            row["chunk_id"] = self._sha1(f"{row.get('doc_id')}:{new_idx}:{str(row.get('text', ''))[:80]}")[:20]
        return working

    def _is_compatible_neighbor(
        self,
        neighbor: dict[str, Any],
        section: str | None,
        article: str | None,
        subarticle: str | None,
        is_table: bool,
    ) -> bool:
        return self._compatible_chunk_structure(neighbor, section, article, subarticle) and (
            self._is_table_chunk_text(str(neighbor.get("text", ""))) == is_table
        )

    def _refresh_chunk_row_text(self, chunk_row: dict[str, Any], text: str) -> None:
        metadata = chunk_row.get("metadata", {})
        chunk_row["text"] = text
        chunk_row["token_count"] = self._count_tokens(text)
        chunk_row["char_count"] = len(text)
        chunk_row["augmented_text"] = self._build_augmented_text(
            text,
            metadata.get("name"),
            metadata.get("year"),
            metadata.get("brief_description"),
            metadata.get("section"),
            metadata.get("article"),
            metadata.get("subarticle"),
        )

    @staticmethod
    def _set_chunk_row_page_meta(chunk_row: dict[str, Any], refs: list[dict[str, Any]]) -> None:
        chunk_row["page_refs"] = refs
        if refs:
            chunk_row["page_start"] = min(ref["page_idx"] for ref in refs)
            chunk_row["page_end"] = max(ref["page_idx"] for ref in refs)
        else:
            chunk_row["page_start"] = None
            chunk_row["page_end"] = None
