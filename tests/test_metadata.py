from rag_chunker import extract_year, update_structure_state, CanonicalBlock, extract_brief_description, extract_document_name


def test_extract_year_variants():
    assert extract_year("A.Y. 2025/2026") == "2025-2026"
    assert extract_year("a.y. 2025/26") == "2025-2026"
    assert extract_year("Academic year 2025-2026") == "2025-2026"


def test_extract_name_and_description():
    blocks = [
        CanonicalBlock(text="UNIFIED CALL FOR REGIONAL BENEFITS", block_type="title", heading_level=1),
        CanonicalBlock(
            text="Students wishing to apply for benefits must complete the online application by deadlines.",
            block_type="text",
        ),
    ]
    name = extract_document_name(blocks, fallback_name="Fallback")
    brief = extract_brief_description(blocks, title=name)
    assert name == "UNIFIED CALL FOR REGIONAL BENEFITS"
    assert "Students wishing to apply" in brief


def test_extract_document_name_prefers_specific_title_over_generic():
    blocks = [
        CanonicalBlock(text="CALL", block_type="title", heading_level=1),
        CanonicalBlock(text="ARDIS FRIULI VENEZIA GIULIA ACADEMIC YEAR 2025-2026", block_type="title", heading_level=2),
    ]
    name = extract_document_name(blocks, fallback_name="Fallback")
    assert name == "ARDIS FRIULI VENEZIA GIULIA ACADEMIC YEAR 2025-2026"


def test_update_structure_state():
    section, article, subarticle = update_structure_state("# SECTION II. GENERAL PROVISIONS", None, None, None)
    assert section and "SECTION II" in section.upper()
    section, article, subarticle = update_structure_state("ART. 14.2 Request for recognition", section, article, subarticle)
    assert article == "14"
    assert subarticle == "14.2"


def test_update_structure_state_ignores_body_mentions():
    section, article, subarticle = update_structure_state("# ART. 5 Students with disabilities", None, None, None, is_heading=True)
    assert article == "5"
    section, article2, subarticle2 = update_structure_state(
        "Students with disabilities (pursuant to Article 3(1) of Law no. 104) may apply.",
        section,
        article,
        subarticle,
        is_heading=False,
    )
    assert article2 == "5"
    assert subarticle2 is None
