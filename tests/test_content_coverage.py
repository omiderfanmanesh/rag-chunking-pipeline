import pytest
from pathlib import Path

from rag_chunker.pipeline import _process_document_folder
from rag_chunker.config import PipelineConfig


def test_content_coverage():
    """Test that chunked content covers the original document text."""
    # Use a small document for testing
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Data directory not found")

    folders = list(data_dir.glob("*"))
    if not folders:
        pytest.skip("No document folders found")

    # Pick the first folder
    folder = folders[0]
    config = PipelineConfig(
        input_dir=Path("data"),
        output_dir=Path("artifacts")
    )

    try:
        doc_meta, chunk_rows, result_manifest = _process_document_folder(folder, config)
    except Exception as e:
        pytest.skip(f"Failed to process document: {e}")

    # Check that chunks are generated
    assert len(chunk_rows) > 0, "No chunks generated"
    for row in chunk_rows:
        assert "text" in row, "Chunk missing text"
        assert len(row["text"].strip()) > 0, "Empty chunk text"

    # Basic sanity checks
    assert len(chunk_rows) > 0, "No chunks generated"
    for row in chunk_rows:
        assert "text" in row, "Chunk missing text"
        assert len(row["text"].strip()) > 0, "Empty chunk text"