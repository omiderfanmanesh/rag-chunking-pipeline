from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ...config.pipeline_config import PipelineConfig


@dataclass
class IncrementalCacheSnapshot:
    docs_by_id: dict[str, dict[str, Any]]
    chunks_by_doc: dict[str, list[dict[str, Any]]]
    cache_by_doc: dict[str, dict[str, str]]


class IncrementalCacheService:
    """Encapsulates cache read/write and reuse decisions for incremental runs."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sha1_func: Callable[[str], str],
        read_json: Callable[[Path], dict[str, Any] | list[Any]],
        version: int = 2,
    ) -> None:
        self.output_dir = output_dir
        self._sha1 = sha1_func
        self._read_json = read_json
        self.version = version

    def processing_signature(self, config: PipelineConfig) -> str:
        payload = {
            "cache_version": self.version,
            "source_priority": config.source_priority,
            "target_tokens": config.target_tokens,
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "min_chars": config.min_chars,
            "min_chunk_tokens": config.min_chunk_tokens,
            "drop_toc": config.drop_toc,
            "dedupe_chunks": config.dedupe_chunks,
        }
        return self._sha1(json.dumps(payload, sort_keys=True))

    def compute_folder_hash(self, folder: Path) -> str:
        hasher = hashlib.sha1()
        files = sorted([path for path in folder.rglob("*") if path.is_file()], key=lambda p: str(p.relative_to(folder)))
        for path in files:
            rel = str(path.relative_to(folder)).replace("\\", "/")
            hasher.update(rel.encode("utf-8"))
            hasher.update(b"\0")
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
        return hasher.hexdigest()

    def load_snapshot(self) -> IncrementalCacheSnapshot:
        docs_by_id: dict[str, dict[str, Any]] = {}
        chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
        cache_by_doc: dict[str, dict[str, str]] = {}

        for row in self._load_jsonl(self.output_dir / "documents.jsonl"):
            doc_id = str(row.get("doc_id", ""))
            if doc_id:
                docs_by_id[doc_id] = row

        for row in self._load_jsonl(self.output_dir / "chunks.jsonl"):
            doc_id = str(row.get("doc_id", ""))
            if not doc_id:
                continue
            chunks_by_doc.setdefault(doc_id, []).append(row)

        cache_path = self.output_dir / "doc_hashes.json"
        if cache_path.exists():
            payload = self._read_json(cache_path)
            if isinstance(payload, dict):
                documents_map = payload.get("documents", {})
                if isinstance(documents_map, dict):
                    for doc_id, item in documents_map.items():
                        if not isinstance(item, dict):
                            continue
                        content_hash = item.get("content_hash")
                        processing_signature = item.get("processing_signature")
                        if isinstance(content_hash, str):
                            cache_by_doc[str(doc_id)] = {
                                "content_hash": content_hash,
                                "processing_signature": processing_signature if isinstance(processing_signature, str) else "",
                            }

        return IncrementalCacheSnapshot(
            docs_by_id=docs_by_id,
            chunks_by_doc=chunks_by_doc,
            cache_by_doc=cache_by_doc,
        )

    def can_reuse(
        self,
        *,
        doc_id: str,
        folder_hash: str,
        processing_signature: str,
        snapshot: IncrementalCacheSnapshot,
    ) -> bool:
        prior = snapshot.cache_by_doc.get(doc_id, {})
        if prior.get("content_hash") != folder_hash:
            return False
        if prior.get("processing_signature") != processing_signature:
            return False
        if doc_id not in snapshot.docs_by_id:
            return False
        if doc_id in snapshot.chunks_by_doc:
            return True
        return int(snapshot.docs_by_id[doc_id].get("stats", {}).get("chunks", 0)) == 0

    @staticmethod
    def build_entry(*, source_folder: str, folder_hash: str, processing_signature: str) -> dict[str, str]:
        return {
            "source_folder": source_folder,
            "content_hash": folder_hash,
            "processing_signature": processing_signature,
        }

    def write_cache(self, *, generated_at_utc: str, entries: dict[str, dict[str, str]], write_json: Callable[[Path, dict[str, Any]], None]) -> None:
        write_json(
            self.output_dir / "doc_hashes.json",
            {
                "version": self.version,
                "generated_at_utc": generated_at_utc,
                "documents": entries,
            },
        )

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
