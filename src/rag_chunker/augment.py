from __future__ import annotations


def _meta_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def build_augmented_text(
    chunk_text: str,
    *,
    name: str,
    year: str | None,
    brief_description: str,
    section: str | None,
    article: str | None,
    subarticle: str | None,
) -> str:
    prefix = [f"doc={name.strip()}"]
    year_value = _meta_value(year)
    section_value = _meta_value(section)
    article_value = _meta_value(article)
    subarticle_value = _meta_value(subarticle)
    _ = brief_description

    if year_value:
        prefix.append(f"year={year_value}")
    if section_value:
        prefix.append(f"section={section_value}")
    if article_value:
        prefix.append(f"article={article_value}")
    if subarticle_value:
        prefix.append(f"subarticle={subarticle_value}")
    meta_line = "meta: " + " | ".join(prefix)
    body = chunk_text.strip()
    meta_words = len(meta_line.split())
    body_words = len(body.split())
    if body_words == 0:
        return meta_line
    if meta_words / (meta_words + body_words) > 0.7:
        return body
    return meta_line + "\n\n" + body
