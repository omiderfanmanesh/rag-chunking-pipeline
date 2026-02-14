from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


@dataclass
class DeepEvalGateConfig:
    artifacts_dir: Path
    eval_report_path: Path
    output_json: Path
    tiny_chunk_tokens: int = 20
    max_tiny_chunk_pct: float = 12.0
    max_duplicate_instance_pct: float = 0.0
    max_overlap_p95_chars: int = 50
    max_missing_metadata_pct: float = 0.0
    overlap_scan_chars: int = 240


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100.0, 2)


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, int(0.95 * len(ordered)) - 1)
    return float(ordered[idx])


def _max_suffix_prefix_overlap(left: str, right: str, scan_chars: int) -> int:
    left_tail = left[-scan_chars:]
    right_head = right[:scan_chars]
    max_len = min(len(left_tail), len(right_head))
    for overlap in range(max_len, 0, -1):
        if left_tail[-overlap:] == right_head[:overlap]:
            return overlap
    return 0


def _consecutive_overlaps(chunks: list[dict[str, Any]], scan_chars: int) -> list[int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        grouped.setdefault(doc_id, []).append(chunk)

    overlaps: list[int] = []
    for doc_chunks in grouped.values():
        ordered = sorted(doc_chunks, key=lambda row: int(row.get("chunk_index", 0)))
        for idx in range(1, len(ordered)):
            prev_text = str(ordered[idx - 1].get("text", ""))
            curr_text = str(ordered[idx].get("text", ""))
            overlaps.append(_max_suffix_prefix_overlap(prev_text, curr_text, scan_chars=scan_chars))
    return overlaps


class ThresholdMetric(BaseMetric):
    def __init__(self, name: str, actual: float, max_allowed: float) -> None:
        self._metric_name = name
        self.actual = float(actual)
        self.max_allowed = float(max_allowed)
        self.threshold = 1.0
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str | None = None
        self.error = None
        self.async_mode = False
        self.evaluation_model = "deterministic"
        self.verbose_mode = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # noqa: ARG002
        self.success = self.actual <= self.max_allowed
        self.score = 1.0 if self.success else 0.0
        self.reason = f"actual={self.actual} max_allowed={self.max_allowed}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:  # noqa: ARG002
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return self._metric_name


def run_deepeval_gates(config: DeepEvalGateConfig) -> dict[str, Any]:
    eval_report = _load_json(config.eval_report_path)
    chunks = _load_jsonl(config.artifacts_dir / "chunks.jsonl")

    total_chunks = len(chunks)
    tiny_chunks = [row for row in chunks if int(row.get("token_count", 0)) < config.tiny_chunk_tokens]
    tiny_chunk_pct = _pct(len(tiny_chunks), total_chunks)

    duplicate_instance_pct = float(eval_report["chunk_metrics"]["duplicates"]["duplicate_instance_pct"])

    required_fields = ["year", "name", "brief_description", "section", "article", "language_hint"]
    missing_metadata_count = 0
    for row in chunks:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict) or any(not metadata.get(field) for field in required_fields):
            missing_metadata_count += 1
    missing_metadata_pct = _pct(missing_metadata_count, total_chunks)

    overlaps = _consecutive_overlaps(chunks, scan_chars=config.overlap_scan_chars)
    overlap_p95_chars = _p95(overlaps)

    checks = [
        {
            "name": "tiny_chunk_pct",
            "actual": tiny_chunk_pct,
            "expected_max": config.max_tiny_chunk_pct,
            "passed": tiny_chunk_pct <= config.max_tiny_chunk_pct,
        },
        {
            "name": "duplicate_instance_pct",
            "actual": duplicate_instance_pct,
            "expected_max": config.max_duplicate_instance_pct,
            "passed": duplicate_instance_pct <= config.max_duplicate_instance_pct,
        },
        {
            "name": "overlap_p95_chars",
            "actual": overlap_p95_chars,
            "expected_max": float(config.max_overlap_p95_chars),
            "passed": overlap_p95_chars <= config.max_overlap_p95_chars,
        },
        {
            "name": "missing_required_metadata_pct",
            "actual": missing_metadata_pct,
            "expected_max": config.max_missing_metadata_pct,
            "passed": missing_metadata_pct <= config.max_missing_metadata_pct,
        },
    ]

    report = {
        "summary": {
            "chunks": total_chunks,
            "eval_report_overall_status": eval_report.get("summary", {}).get("overall_status"),
            "gates_passed": all(check["passed"] for check in checks),
        },
        "metrics": {
            "tiny_chunk_pct": tiny_chunk_pct,
            "duplicate_instance_pct": duplicate_instance_pct,
            "overlap_p95_chars": overlap_p95_chars,
            "missing_required_metadata_pct": missing_metadata_pct,
        },
        "checks": checks,
    }

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    test_case = LLMTestCase(
        input="rag chunk quality gates",
        actual_output=json.dumps({"checks": checks}),
        expected_output="all checks must pass",
    )
    metrics = [ThresholdMetric(check["name"], check["actual"], check["expected_max"]) for check in checks]
    assert_test(test_case, metrics, run_async=False)
    return report
