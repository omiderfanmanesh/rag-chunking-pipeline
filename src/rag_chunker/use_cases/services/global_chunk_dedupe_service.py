from __future__ import annotations

import re
from typing import Any, Callable


class GlobalChunkDedupeService:
    """Applies global exact-text dedupe with normalized whitespace/case keys."""

    def __init__(self, *, sha1_func: Callable[[str], str]) -> None:
        self._sha1 = sha1_func
        self._ws_re = re.compile(r"\s+")

    def apply(self, chunks: list[dict[str, Any]], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_keys: set[str] = set()
        deduped_chunks: list[dict[str, Any]] = []
        for row in chunks:
            key = self._dedupe_key(str(row.get("text", "")))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_chunks.append(row)

        self._refresh_document_stats(deduped_chunks, documents)
        return deduped_chunks

    def _dedupe_key(self, text: str) -> str:
        normalized = self._ws_re.sub(" ", text).strip().casefold()
        return self._sha1(normalized)

    @staticmethod
    def _refresh_document_stats(chunks: list[dict[str, Any]], documents: list[dict[str, Any]]) -> None:
        chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
        for row in chunks:
            doc_id = str(row.get("doc_id", ""))
            chunks_by_doc.setdefault(doc_id, []).append(row)

        for document in documents:
            doc_chunks = chunks_by_doc.get(str(document.get("doc_id", "")), [])
            doc_pages = {
                page_ref["page_idx"]
                for row in doc_chunks
                for page_ref in row.get("page_refs", [])
                if isinstance(page_ref.get("page_idx"), int)
            }
            doc_stats = document.get("stats")
            if not isinstance(doc_stats, dict):
                doc_stats = {}
                document["stats"] = doc_stats
            doc_stats["chunks"] = len(doc_chunks)
            doc_stats["tokens"] = sum(int(row.get("token_count", 0)) for row in doc_chunks)
            doc_stats["pages"] = len(doc_pages)
