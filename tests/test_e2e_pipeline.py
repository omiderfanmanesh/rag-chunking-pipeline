import json

from rag_chunker.pipeline import PipelineConfig, run_pipeline
import rag_chunker.pipeline as pipeline_module


def test_e2e_pipeline_with_fallbacks(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    doc_block = data_dir / "DocA.pdf-11111111-1111-1111-1111-111111111111"
    doc_block.mkdir()
    block_payload = {
        "pdfData": [
            [
                {
                    "type": "header",
                    "text": "HEADER",
                    "is_discarded": True,
                    "page_idx": 0,
                    "block_position": "0-0",
                },
                {
                    "type": "title",
                    "text": "# ART. 1 INTRO",
                    "page_idx": 0,
                    "block_position": "0-1",
                    "id": "a",
                    "level": 1,
                },
                {
                    "type": "text",
                    "text": "A.Y. 2025/26 Students can apply.",
                    "page_idx": 0,
                    "block_position": "0-2",
                    "id": "b",
                },
                {
                    "type": "table_body",
                    "table_body": "<table><tr><td>K</td><td>V</td></tr></table>",
                    "page_idx": 0,
                    "block_position": "0-3",
                    "id": "merge-id",
                },
            ],
            [
                {
                    "type": "table_body",
                    "table_body": "<table><tr><td>K2</td><td>V2</td></tr></table>",
                    "page_idx": 1,
                    "block_position": "1-0",
                    "id": "merge-id",
                }
            ],
        ],
        "mergeConnections": [{"id": "merge-id", "blocks": ["0-3", "1-0"], "type": "merge"}],
    }
    (doc_block / "block_list.json").write_text(json.dumps(block_payload), encoding="utf-8")
    (doc_block / "DocA.md").write_text("# fallback md", encoding="utf-8")

    doc_content = data_dir / "DocB.pdf-22222222-2222-2222-2222-222222222222"
    doc_content.mkdir()
    content_payload = [
        {"type": "text", "text": "A.Y. 2025/2026", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Article 2 - Eligibility requirements", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "The application must be submitted online.", "page_idx": 0},
    ]
    (doc_content / "x_content_list.json").write_text(json.dumps(content_payload), encoding="utf-8")

    output_dir = tmp_path / "artifacts"
    config = PipelineConfig(input_dir=data_dir, output_dir=output_dir)
    manifest = run_pipeline(config)

    assert manifest["documents"] == 2
    assert manifest["chunks"] > 0

    chunks_path = output_dir / "chunks.jsonl"
    docs_path = output_dir / "documents.jsonl"
    assert chunks_path.exists()
    assert docs_path.exists()

    docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    modes = {doc["source_mode_used"] for doc in docs}
    assert "block_list" in modes
    assert "content_list" in modes


def test_pipeline_drops_toc_and_dedupes_exact_text(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    doc = data_dir / "DocC.pdf-33333333-3333-3333-3333-333333333333"
    doc.mkdir()
    (doc / "DocC.md").write_text(
        "# Summary\n"
        "Article 1 Intro 5\n"
        "Article 2 Rules 7\n\n"
        "# ART. 1 Intro\n"
        "A sufficiently long body paragraph that survives filtering and is chunked.\n",
        encoding="utf-8",
    )

    # Force duplicate chunk texts from one segment to exercise dedupe logic.
    monkeypatch.setattr(
        "rag_chunker.pipeline._chunk_segment_texts",
        lambda *_args, **_kwargs: [
            "Duplicate chunk text with enough tokens to remain in output after filtering.",
            "Duplicate chunk text with enough tokens to remain in output after filtering.",
        ],
    )

    output_dir = tmp_path / "artifacts"
    manifest = run_pipeline(PipelineConfig(input_dir=data_dir, output_dir=output_dir, min_chunk_tokens=1))
    assert manifest["documents"] == 1

    chunks = [json.loads(line) for line in (output_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert chunks
    assert all("TABLE OF CONTENTS" not in row.get("text", "") for row in chunks)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Duplicate chunk text with enough tokens to remain in output after filtering."


def test_pipeline_dedupes_across_documents(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    doc_one = data_dir / "DocD.pdf-44444444-4444-4444-4444-444444444444"
    doc_one.mkdir()
    (doc_one / "DocD.md").write_text(
        "# ART. 1 Intro\n"
        "doc-one-marker\n"
        "Body paragraph that will be replaced by a deterministic test chunk.\n",
        encoding="utf-8",
    )

    doc_two = data_dir / "DocE.pdf-55555555-5555-5555-5555-555555555555"
    doc_two.mkdir()
    (doc_two / "DocE.md").write_text(
        "# ART. 1 Intro\n"
        "doc-two-marker\n"
        "Another body paragraph that will be replaced by the same normalized chunk.\n",
        encoding="utf-8",
    )

    base_text = (
        "Shared policy paragraph with enough tokens to survive filtering and remain in output "
        "after chunking across both documents."
    )

    def fake_chunk_segment_texts(text, **_kwargs):
        if "doc-two-marker" in text:
            return [f"  Shared policy paragraph with enough tokens   to survive filtering and remain in output after chunking across both documents.  "]
        return [base_text]

    monkeypatch.setattr("rag_chunker.pipeline._chunk_segment_texts", fake_chunk_segment_texts)

    output_dir = tmp_path / "artifacts"
    manifest = run_pipeline(PipelineConfig(input_dir=data_dir, output_dir=output_dir, min_chunk_tokens=1))
    assert manifest["documents"] == 2
    assert manifest["chunks"] == 1

    chunks = [json.loads(line) for line in (output_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(chunks) == 1

    documents = [json.loads(line) for line in (output_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines() if line]
    chunk_counts = sorted(doc["stats"]["chunks"] for doc in documents)
    assert chunk_counts == [0, 1]


def test_pipeline_incremental_reuses_unchanged_documents(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc = data_dir / "DocF.pdf-66666666-6666-6666-6666-666666666666"
    doc.mkdir()
    (doc / "DocF.md").write_text(
        "# ART. 1 Intro\n"
        "This document should be reused without reprocessing on the second run.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    first = run_pipeline(PipelineConfig(input_dir=data_dir, output_dir=output_dir))
    assert first["incremental"]["processed_documents"] == 1
    assert first["incremental"]["reused_documents"] == 0

    monkeypatch.setattr(
        "rag_chunker.pipeline._process_document_folder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not reprocess unchanged docs")),
    )
    second = run_pipeline(PipelineConfig(input_dir=data_dir, output_dir=output_dir))
    assert second["documents"] == 1
    assert second["errors"] == []
    assert second["incremental"]["processed_documents"] == 0
    assert second["incremental"]["reused_documents"] == 1


def test_pipeline_incremental_reprocesses_changed_document(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc = data_dir / "DocG.pdf-77777777-7777-7777-7777-777777777777"
    doc.mkdir()
    md_path = doc / "DocG.md"
    md_path.write_text(
        "# ART. 1 Intro\n"
        "First version.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"
    run_pipeline(PipelineConfig(input_dir=data_dir, output_dir=output_dir))

    md_path.write_text(
        "# ART. 1 Intro\n"
        "Second version changed.\n",
        encoding="utf-8",
    )

    calls = {"count": 0}
    original = pipeline_module._process_document_folder

    def wrapped(folder, config):
        calls["count"] += 1
        return original(folder, config)

    monkeypatch.setattr("rag_chunker.pipeline._process_document_folder", wrapped)
    second = run_pipeline(PipelineConfig(input_dir=data_dir, output_dir=output_dir))
    assert second["errors"] == []
    assert calls["count"] == 1
    assert second["incremental"]["processed_documents"] == 1
    assert second["incremental"]["reused_documents"] == 0
