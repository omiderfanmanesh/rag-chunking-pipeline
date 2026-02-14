from rag_chunker import clean_text, flatten_html_table, normalize_inline_math


def test_flatten_html_table():
    html = "<table><tr><td>Col A</td><td>Col B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    flattened = flatten_html_table(html)
    assert "Table:" in flattened
    assert "Col A | Col B" in flattened
    assert "1 | 2" in flattened


def test_clean_text_normalizes_math_entities_and_noise():
    raw = "Line 1\n\nArticle with $66\\%$ threshold.\nErrore. Il segnalibro non è definito.\nTom &amp; Jerry\n"
    cleaned = clean_text(raw)
    assert "66%" in cleaned
    assert "segnalibro" not in cleaned.lower()
    assert "Tom & Jerry" in cleaned


def test_clean_text_removes_inline_bookmark_and_paren_math_escapes():
    raw = "Value formula: \\( BDS_{MAX} - \\frac{A}{B} \\). Errore. Il segnalibro non è definito."
    cleaned = clean_text(raw)
    assert "\\(" not in cleaned and "\\)" not in cleaned
    assert "\\frac" not in cleaned
    assert "segnalibro" not in cleaned.lower()


def test_normalize_inline_math_fraction_and_operators():
    raw = r"Formula: \left(\frac{X}{Y}\right) \times 100 and \frac{\frac{A}{B}}{C}"
    normalized = normalize_inline_math(raw)
    assert r"\left" not in normalized
    assert r"\right" not in normalized
    assert r"\frac" not in normalized
    assert "X / Y" in normalized
    assert "A / B / C" in normalized
    assert "× 100" in normalized


def test_normalize_inline_math_fraction_with_nested_braces():
    raw = r"\frac{( ISEE_S - 2 / 3 soglia ) \times (BDS_{MAX} - BDS_{MIN})}{( soglia - 2 / 3 soglia )}"
    normalized = normalize_inline_math(raw)
    assert r"\frac" not in normalized
    assert "frac{" not in normalized
    assert "BDS_{MAX}" in normalized


def test_clean_text_dedupes_consecutive_duplicate_sentences():
    raw = (
        "Students must submit the online application before the deadline. "
        "Students must submit the online application before the deadline. "
        "Late submissions are not accepted."
    )
    cleaned = clean_text(raw)
    assert cleaned.count("Students must submit the online application before the deadline.") == 1
    assert "Late submissions are not accepted." in cleaned


def test_clean_text_dedupes_non_consecutive_long_lines():
    raw = (
        "The issue of documentation is free of charge and available through the designated office.\n"
        "Different informational line.\n"
        "The issue of documentation is free of charge and available through the designated office.\n"
    )
    cleaned = clean_text(raw)
    assert cleaned.count("The issue of documentation is free of charge and available through the designated office.") == 1
