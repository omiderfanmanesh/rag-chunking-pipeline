from rag_chunker import PageRef, Segment
from rag_chunker.pipeline import (
    _article_from_section_label,
    _chunk_segment_texts,
    _final_tiny_chunk_sweep,
    _looks_structural_stub,
    _is_toc_segment,
    _merge_tiny_chunk_texts,
    _merge_toc_segments,
    _page_meta,
    _resolve_chunk_article,
    _resolve_structure,
)


def test_merge_toc_segments_consolidates_index_blocks():
    segments = [
        Segment(
            text="# Summary\nArticle 1 - Intro 5\nArticle 2 - Rules 8",
            page_refs=[PageRef(0)],
            section=None,
            article=None,
            subarticle=None,
            heading_path=["Summary"],
        ),
        Segment(
            text="Article 3 - Payments 11\nArticle 4 - Appeals 17",
            page_refs=[PageRef(1)],
            section=None,
            article=None,
            subarticle=None,
            heading_path=[],
        ),
        Segment(
            text="# ART. 1 Intro\nEligibility requirements apply.",
            page_refs=[PageRef(2)],
            section="SECTION I",
            article="1",
            subarticle=None,
            heading_path=["ART. 1 Intro"],
        ),
    ]
    merged = _merge_toc_segments(segments, max_tokens=520)
    assert len(merged) == 2
    assert merged[0].section == "TABLE OF CONTENTS"
    assert "Article 4 - Appeals" in merged[0].text
    assert merged[1].article == "1"


def test_chunk_segment_texts_splits_table_rows():
    text = "Table:\n" + "\n".join(f"Row {i} | value {i}" for i in range(200))
    chunks = _chunk_segment_texts(text, target_tokens=120, max_tokens=150, overlap_tokens=20, min_chars=30)
    assert len(chunks) > 1
    assert all(chunk.startswith("Table:") for chunk in chunks)


def test_chunk_segment_texts_splits_multiple_tables_and_keeps_prose():
    text = (
        "Intro paragraph.\n\n"
        "Table:\nA | 1\nB | 2\n\n"
        "Narrative after first table.\n\n"
        "Table:\nC | 3\nD | 4"
    )
    chunks = _chunk_segment_texts(text, target_tokens=60, max_tokens=80, overlap_tokens=10, min_chars=20)
    table_chunks = [chunk for chunk in chunks if chunk.startswith("Table:")]
    prose_chunks = [chunk for chunk in chunks if not chunk.startswith("Table:")]

    assert len(table_chunks) == 2
    assert "A | 1" in table_chunks[0]
    assert "C | 3" in table_chunks[1]
    assert any("Narrative after first table." in chunk for chunk in prose_chunks)


def test_chunk_segment_texts_strips_orphan_table_prefix_for_prose():
    text = "Table:\nFor grant recipients who are allocated an accommodation place and give it up early."
    chunks = _chunk_segment_texts(text, target_tokens=80, max_tokens=100, overlap_tokens=20, min_chars=30)
    assert len(chunks) == 1
    assert not chunks[0].startswith("Table:")
    assert chunks[0].startswith("For grant recipients")


def test_chunk_segment_texts_normalizes_pipe_table_without_prefix():
    text = "Benefit | Date\nAccommodation place | 29 July 2025\nScholarship publication | 8 August 2025"
    chunks = _chunk_segment_texts(text, target_tokens=80, max_tokens=120, overlap_tokens=20, min_chars=30)
    assert chunks
    assert chunks[0].startswith("Table:")
    assert "Benefit | Date" in chunks[0]


def test_chunk_segment_texts_normalizes_pipe_table_with_caption_context():
    text = "Contacts - Udine Office\nService | Phone\nScholarship | 0432 245772"
    chunks = _chunk_segment_texts(text, target_tokens=80, max_tokens=120, overlap_tokens=20, min_chars=30)
    assert chunks
    assert chunks[0].startswith("Table:")
    assert "Contacts - Udine Office" in chunks[0]
    assert "Service | Phone" in chunks[0]


def test_page_meta_uses_one_based_page_numbers():
    page_start, page_end, refs = _page_meta([PageRef(0), PageRef(2)])
    assert page_start == 1
    assert page_end == 3
    assert refs[0]["page_idx"] == 1


def test_resolve_structure_falls_back_to_heading_path():
    segment = Segment(
        text="Eligibility details",
        page_refs=[PageRef(1)],
        section=None,
        article=None,
        subarticle=None,
        heading_path=["SECTION II GENERAL RULES", "ART. 14.2 Request for recognition"],
    )
    section, article, subarticle = _resolve_structure(segment)
    assert section == "SECTION II GENERAL RULES"
    assert article == "14"
    assert subarticle == "14.2"


def test_resolve_chunk_article_uses_single_heading_line_match():
    article, subarticle = _resolve_chunk_article(
        "ART. 7.2 Accommodation benefits\nStudents can apply online.",
        fallback_article=None,
        fallback_subarticle=None,
    )
    assert article == "7"
    assert subarticle == "7.2"


def test_resolve_chunk_article_corrects_clear_mismatch():
    article, subarticle = _resolve_chunk_article(
        "Article 12 Appeals\nAppeal requests must be submitted in writing.",
        fallback_article="5",
        fallback_subarticle=None,
    )
    assert article == "12"
    assert subarticle is None


def test_merge_tiny_chunk_texts_merges_same_kind_neighbors():
    merged = _merge_tiny_chunk_texts(
        [
            "Brief intro",
            "This is a longer paragraph with enough context to absorb the intro and still remain within budget.",
        ],
        min_tokens=20,
        max_tokens=80,
    )
    assert len(merged) == 1
    assert "Brief intro" in merged[0]


def test_merge_tiny_chunk_texts_does_not_merge_table_into_plain_text():
    merged = _merge_tiny_chunk_texts(
        [
            "Table:\nA | 1",
            "Narrative text that should remain separate from table chunks.",
        ],
        min_tokens=20,
        max_tokens=120,
    )
    assert len(merged) == 2


def test_final_tiny_chunk_sweep_merges_small_row_into_previous_when_compatible():
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "text": "This is a substantial chunk with enough context and content to remain as the main paragraph.",
            "token_count": 30,
            "char_count": 95,
            "page_refs": [{"page_idx": 1}],
            "page_start": 1,
            "page_end": 1,
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "1",
                "subarticle": None,
                "language_hint": "en",
            },
            "augmented_text": "",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "chunk_index": 1,
            "text": "tiny tail",
            "token_count": 3,
            "char_count": 9,
            "page_refs": [{"page_idx": 2}],
            "page_start": 2,
            "page_end": 2,
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "1",
                "subarticle": None,
                "language_hint": "en",
            },
            "augmented_text": "",
        },
    ]
    swept = _final_tiny_chunk_sweep(rows, max_tokens=120, sweep_tokens=20)
    assert len(swept) == 1
    assert "tiny tail" in swept[0]["text"]
    assert swept[0]["page_start"] == 1
    assert swept[0]["page_end"] == 2


def test_final_tiny_chunk_sweep_drops_structural_stub():
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "text": "Article 3 Appeals 9",
            "token_count": 4,
            "char_count": 19,
            "page_refs": [{"page_idx": 1}],
            "page_start": 1,
            "page_end": 1,
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "3",
                "subarticle": None,
                "language_hint": "en",
            },
            "augmented_text": "",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "chunk_index": 1,
            "text": "This is the substantive article body with enough information to remain in the final output.",
            "token_count": 28,
            "char_count": 95,
            "page_refs": [{"page_idx": 2}],
            "page_start": 2,
            "page_end": 2,
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "3",
                "subarticle": None,
                "language_hint": "en",
            },
            "augmented_text": "",
        },
    ]
    swept = _final_tiny_chunk_sweep(rows, max_tokens=120, sweep_tokens=20)
    assert len(swept) == 1
    assert "substantive article body" in swept[0]["text"]


def test_final_tiny_chunk_sweep_does_not_merge_tiny_table_into_prose():
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "text": "Narrative chunk with enough context and details to remain separate from tables.",
            "token_count": 24,
            "char_count": 77,
            "page_refs": [{"page_idx": 1}],
            "page_start": 1,
            "page_end": 1,
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "1",
                "subarticle": None,
                "language_hint": "en",
            },
            "augmented_text": "",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "chunk_index": 1,
            "text": "Table:\nA | 1",
            "token_count": 6,
            "char_count": 12,
            "page_refs": [{"page_idx": 1}],
            "page_start": 1,
            "page_end": 1,
            "metadata": {
                "year": "2025-2026",
                "name": "Doc One",
                "brief_description": "Desc",
                "section": "SECTION I",
                "article": "1",
                "subarticle": None,
                "language_hint": "en",
            },
            "augmented_text": "",
        },
    ]
    swept = _final_tiny_chunk_sweep(rows, max_tokens=120, sweep_tokens=20)
    assert len(swept) == 2
    assert swept[0]["text"].startswith("Narrative chunk")
    assert swept[1]["text"].startswith("Table:")


def test_is_toc_segment_detects_numbered_outline_entries():
    segment = Segment(
        text=(
            "3.1 General requirements 9\n"
            "3.1.1 Grounds for exclusion 9\n"
            "3.1.2 Academic regularity 10\n"
            "3.1.3 Foreign qualifications 11"
        ),
        page_refs=[PageRef(0)],
        section=None,
        article=None,
        subarticle=None,
        heading_path=[],
    )
    assert _is_toc_segment(segment)


def test_is_toc_segment_detects_compact_numbered_entries_without_space_after_dot():
    segment = Segment(
        text="11.1.Total forfeiture 22\n11.2.Partial forfeiture 22\n11.3.Consequences 23",
        page_refs=[PageRef(0)],
        section=None,
        article=None,
        subarticle=None,
        heading_path=[],
    )
    assert _is_toc_segment(segment)


def test_looks_structural_stub_detects_index_entry_lines():
    text = "3.1 General requirements 9"
    assert _looks_structural_stub(text, token_count=10)


def test_looks_structural_stub_detects_title_like_short_labels():
    text = "Accommodation Service Degree Awards UPDATE"
    assert _looks_structural_stub(text, token_count=9)


def test_article_from_section_label_extracts_root_and_subarticle():
    article, subarticle = _article_from_section_label("3.2.1 Access requirement")
    assert article == "3"
    assert subarticle == "3.2.1"


def test_merge_small_segments_compatible_with_same_article_different_subarticle():
    s1 = Segment(
        text="11.1 Total forfeiture",
        page_refs=[PageRef(1)],
        section="SECTION IV",
        article="11",
        subarticle="11.1",
        heading_path=["SECTION IV", "11.1"],
    )
    s2 = Segment(
        text="11.2 Partial forfeiture details",
        page_refs=[PageRef(1)],
        section="SECTION IV",
        article="11",
        subarticle="11.2",
        heading_path=["SECTION IV", "11.2"],
    )
    from rag_chunker.pipeline import _merge_small_segments

    merged = _merge_small_segments([s1, s2], min_tokens=30, max_tokens=120)
    assert len(merged) == 1
    assert "11.1 Total forfeiture" in merged[0].text
    assert "11.2 Partial forfeiture details" in merged[0].text
