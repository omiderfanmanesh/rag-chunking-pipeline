from rag_chunker import build_segments, count_tokens, split_text_by_tokens, CanonicalBlock, PageRef


def _max_suffix_prefix_overlap(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    for overlap in range(max_len, 0, -1):
        if left[-overlap:] == right[:overlap]:
            return overlap
    return 0


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


def test_split_text_by_tokens_caps_consecutive_overlap_chars():
    text = " ".join(
        f"Sentence {idx} has a unique marker token U{idx} and enough words to exercise overlap limits."
        for idx in range(360)
    )
    chunks = split_text_by_tokens(text, target_tokens=80, max_tokens=100, overlap_tokens=30)
    assert len(chunks) > 2
    overlaps = [_max_suffix_prefix_overlap(chunks[idx - 1], chunks[idx]) for idx in range(1, len(chunks))]
    assert max(overlaps) <= 30


def test_split_text_by_tokens_overlap_trim_prefers_sentence_break():
    sentence_a = "Students must submit complete documentation before the stated deadline."
    sentence_b = "Late applications are rejected automatically by the admissions office."
    text = " ".join(([sentence_a, sentence_b] * 120))
    chunks = split_text_by_tokens(text, target_tokens=90, max_tokens=110, overlap_tokens=35)
    assert len(chunks) > 2
    starts = [chunk.lstrip()[:1] for chunk in chunks[1:]]
    assert sum(1 for ch in starts if ch and ch.islower()) <= 1
