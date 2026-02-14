import json

import pytest

from rag_chunker import DeepEvalGateConfig, run_deepeval_gates


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _base_chunk(chunk_id, chunk_index, text, token_count=30, metadata=None):
    return {
        "chunk_id": chunk_id,
        "doc_id": "d1",
        "chunk_index": chunk_index,
        "text": text,
        "token_count": token_count,
        "char_count": len(text),
        "page_start": 1,
        "page_end": 1,
        "page_refs": [{"page_idx": 1}],
        "metadata": metadata
        or {
            "year": "2025-2026",
            "name": "Doc One",
            "brief_description": "Desc",
            "section": "SECTION I",
            "article": "1",
            "subarticle": None,
            "language_hint": "en",
        },
    }


def _write_eval_report(path, duplicate_instance_pct=0.0):
    payload = {
        "summary": {"overall_status": "excellent", "coverage_ratio": 95.0},
        "chunk_metrics": {
            "duplicates": {"duplicate_instance_pct": duplicate_instance_pct},
            "token_stats": {"median": 150.0}
        },
        "metadata_metrics": {
            "consistency": {
                "article_mixed_chunks": {"pct": 0.0}
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_deepeval_gates_pass(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    chunks = [
        _base_chunk("c1", 0, "A" * 120 + " unique one"),
        _base_chunk("c2", 1, "B" * 120 + " unique two"),
    ]
    _write_jsonl(artifacts / "chunks.jsonl", chunks)
    _write_eval_report(artifacts / "eval_report.json", duplicate_instance_pct=0.0)

    report = run_deepeval_gates(
        DeepEvalGateConfig(
            artifacts_dir=artifacts,
            eval_report_path=artifacts / "eval_report.json",
            output_json=artifacts / "deepeval_gate_report.json",
            max_tiny_chunk_pct=1.0,
            max_duplicate_instance_pct=0.0,
            max_overlap_p95_chars=30,
            max_missing_metadata_pct=0.0,
        )
    )

    assert report["summary"]["gates_passed"] is True
    assert (artifacts / "deepeval_gate_report.json").exists()


def test_deepeval_gates_fail_on_overlap(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    overlap = "X" * 80
    chunks = [
        _base_chunk("c1", 0, "prefix " + overlap),
        _base_chunk("c2", 1, overlap + " suffix"),
    ]
    _write_jsonl(artifacts / "chunks.jsonl", chunks)
    _write_eval_report(artifacts / "eval_report.json", duplicate_instance_pct=0.0)

    with pytest.raises(AssertionError):
        run_deepeval_gates(
            DeepEvalGateConfig(
                artifacts_dir=artifacts,
                eval_report_path=artifacts / "eval_report.json",
                output_json=artifacts / "deepeval_gate_report.json",
                max_overlap_p95_chars=30,
            )
        )


def test_deepeval_gates_fail_on_tiny_chunks(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    chunks = [
        _base_chunk("c1", 0, "tiny one", token_count=3),
        _base_chunk("c2", 1, "tiny two", token_count=3),
    ]
    _write_jsonl(artifacts / "chunks.jsonl", chunks)
    _write_eval_report(artifacts / "eval_report.json", duplicate_instance_pct=0.0)

    with pytest.raises(AssertionError):
        run_deepeval_gates(
            DeepEvalGateConfig(
                artifacts_dir=artifacts,
                eval_report_path=artifacts / "eval_report.json",
                output_json=artifacts / "deepeval_gate_report.json",
                max_tiny_chunk_pct=1.0,
            )
        )


def test_deepeval_gates_fail_on_duplicate_pct(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    chunks = [
        _base_chunk("c1", 0, "unique one"),
        _base_chunk("c2", 1, "unique two"),
    ]
    _write_jsonl(artifacts / "chunks.jsonl", chunks)
    _write_eval_report(artifacts / "eval_report.json", duplicate_instance_pct=0.5)

    with pytest.raises(AssertionError):
        run_deepeval_gates(
            DeepEvalGateConfig(
                artifacts_dir=artifacts,
                eval_report_path=artifacts / "eval_report.json",
                output_json=artifacts / "deepeval_gate_report.json",
                max_duplicate_instance_pct=0.0,
            )
        )


def test_deepeval_gates_fail_on_missing_metadata(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    chunks = [
        _base_chunk("c1", 0, "ok metadata"),
        _base_chunk(
            "c2",
            1,
            "missing metadata",
            metadata={
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "1",
                "subarticle": None,
                "language_hint": "",
            },
        ),
    ]
    _write_jsonl(artifacts / "chunks.jsonl", chunks)
    _write_eval_report(artifacts / "eval_report.json", duplicate_instance_pct=0.0)

    with pytest.raises(AssertionError):
        run_deepeval_gates(
            DeepEvalGateConfig(
                artifacts_dir=artifacts,
                eval_report_path=artifacts / "eval_report.json",
                output_json=artifacts / "deepeval_gate_report.json",
                max_missing_metadata_pct=0.0,
            )
        )
