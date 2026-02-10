import json

from rag_chunker.evaluator import EvalConfig, run_evaluation


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluator_generates_reports(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    docs = [
        {
            "doc_id": "d1",
            "source_mode_used": "block_list",
            "name": "Doc One",
            "year": "2025-2026",
            "brief_description": "Desc",
            "language_hint": "en",
        }
    ]
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "text": "ART. 1 Intro text",
            "token_count": 4,
            "char_count": 18,
            "page_start": 0,
            "page_end": 0,
            "page_refs": [{"page_idx": 0}],
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "1",
                "subarticle": None,
                "heading_path": ["SECTION I"],
                "language_hint": "en",
            },
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "chunk_index": 1,
            "text": "Article 1 body " + ("x " * 600),
            "token_count": 603,
            "char_count": 1200,
            "page_start": 1,
            "page_end": 1,
            "page_refs": [{"page_idx": 1}],
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "1",
                "subarticle": None,
                "heading_path": ["SECTION I"],
                "language_hint": "en",
            },
        },
    ]
    manifest = {"documents": 1, "chunks": 2, "source_modes": {"block_list": 1}, "errors": []}

    _write_jsonl(artifacts / "documents.jsonl", docs)
    _write_jsonl(artifacts / "chunks.jsonl", chunks)
    (artifacts / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out_json = artifacts / "eval_report.json"
    out_md = artifacts / "eval_report.md"
    report = run_evaluation(
        EvalConfig(
            artifacts_dir=artifacts,
            output_json=out_json,
            output_md=out_md,
            max_tokens=520,
            small_chunk_threshold=20,
        )
    )

    assert out_json.exists()
    assert out_md.exists()
    assert report["summary"]["documents"] == 1
    assert report["summary"]["chunks"] == 2
    assert report["chunk_metrics"]["oversized_chunks"]["count"] == 1
    assert "Overall score" in out_md.read_text(encoding="utf-8")

