from __future__ import annotations

import json
from pathlib import Path

from .models import SourceChoice


def discover_document_folders(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted([path for path in input_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower())


def choose_source(folder: Path, source_priority: str = "block_first") -> SourceChoice:
    block_path = folder / "block_list.json"
    content_paths = sorted(folder.glob("*_content_list.json"))
    md_paths = sorted(folder.glob("*.md"))
    content_path = content_paths[0] if content_paths else None
    md_path = md_paths[0] if md_paths else None

    if source_priority != "block_first":
        raise ValueError(f"Unsupported source priority: {source_priority}")

    if block_path.exists():
        return SourceChoice(mode="block_list", folder=folder, block_path=block_path, content_path=content_path, md_path=md_path)
    if content_path is not None:
        return SourceChoice(
            mode="content_list",
            folder=folder,
            content_path=content_path,
            md_path=md_path,
            fallback_reason="block_list.json missing",
        )
    if md_path is not None:
        return SourceChoice(
            mode="md",
            folder=folder,
            md_path=md_path,
            fallback_reason="block_list.json and *_content_list.json missing",
        )
    return SourceChoice(
        mode="none",
        folder=folder,
        fallback_reason="No block_list.json, *_content_list.json, or .md file found",
    )


def read_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

