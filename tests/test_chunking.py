from rag_chunker.chunking import build_segments, count_tokens, split_text_by_tokens
from rag_chunker.models import CanonicalBlock, PageRef


def test_build_segments_respects_article_boundaries():
    blocks = [
        CanonicalBlock(text="ART. 1 FIRST", block_type="title", heading_level=1, page_refs=[PageRef(0)]),
        CanonicalBlock(text="text in article one", block_type="text", page_refs=[PageRef(0)]),
        CanonicalBlock(text="ART. 2 SECOND", block_type="title", heading_level=1, page_refs=[PageRef(1)]),
        CanonicalBlock(text="text in article two", block_type="text", page_refs=[PageRef(1)]),
    ]
    segments = build_segments(blocks)
    assert len(segments) >= 2
    assert segments[0].article == "1"
    assert segments[1].article == "2"


def test_split_text_by_tokens_with_overlap():
    text = " ".join(f"token{i}" for i in range(1200))
    chunks = split_text_by_tokens(text, target_tokens=450, max_tokens=520, overlap_tokens=40)
    assert len(chunks) >= 3
    assert all(chunk.strip() for chunk in chunks)
    assert all(count_tokens(chunk) <= 520 for chunk in chunks)


def test_split_text_by_tokens_prefers_sentence_boundaries():
    sentence = "Students who satisfy all eligibility requirements can submit the application before the deadline."
    text = " ".join(sentence for _ in range(120))
    chunks = split_text_by_tokens(text, target_tokens=120, max_tokens=140, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    # Most chunk starts should not be obvious sentence fragments.
    fragment_starts = sum(1 for chunk in chunks[1:] if chunk[0].islower())
    assert fragment_starts <= 1
