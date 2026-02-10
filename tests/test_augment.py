from rag_chunker.augment import build_augmented_text


def test_build_augmented_text_omits_empty_metadata_fields():
    body = "This body text contains enough words to trigger metadata augmentation and verify field omission behavior."
    text = build_augmented_text(
        body,
        name="Doc Name",
        year=None,
        brief_description="",
        section=None,
        article="5",
        subarticle=None,
    )
    assert "N/A" not in text
    assert "meta: doc=Doc Name" in text
    assert "article=5" in text
    assert "year=" not in text
    assert "section=" not in text
    assert "subarticle=" not in text
    assert "Context:" not in text


def test_build_augmented_text_skips_metadata_for_tiny_chunks():
    text = build_augmented_text(
        "Short chunk",
        name="Doc Name",
        year="2025",
        brief_description="desc",
        section="SECTION I",
        article="1",
        subarticle=None,
    )
    assert text == "Short chunk"
