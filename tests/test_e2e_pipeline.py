import json

from rag_chunker.pipeline import PipelineConfig, run_pipeline


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
