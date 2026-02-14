from __future__ import annotations

import html
import re

TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
TR_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
TD_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>\n]+>")
INLINE_MATH_RE = re.compile(r"\$([^$\n]+)\$")
PAREN_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
BRACKET_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*]\([^)]*\)\s*$")
NOISE_PATTERNS = [
    re.compile(r"(?i)^\s*errore\.\s*il segnalibro non.*$"),
    re.compile(r"(?i)^\s*error\.\s*bookmark not defined.*$"),
    re.compile(r"^\s*[#=\-*_]{4,}\s*$"),
]
BOOKMARK_INLINE_RE = re.compile(
    r"(?i)\s*errore\.\s*il\s+(?:segnalibro|segnalbro)\s+non\s+.*?definit[oa]\.?\s*"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_cell(cell: str) -> str:
    cell = html.unescape(cell)
    cell = TAG_RE.sub(" ", cell)
    cell = re.sub(r"\s+", " ", cell)
    return cell.strip()


def flatten_html_table(table_html: str) -> str:
    rows = []
    for tr in TR_RE.findall(table_html):
        cells = TD_RE.findall(tr)
        if not cells:
            continue
        cleaned = [_clean_cell(cell) for cell in cells]
        cleaned = [cell for cell in cleaned if cell]
        if cleaned:
            rows.append(" | ".join(cleaned))
    if not rows:
        raw = _clean_cell(table_html)
        return f"Table: {raw}" if raw else ""
    return "Table:\n" + "\n".join(rows)


def _extract_braced(text: str, start: int) -> tuple[str | None, int]:
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    idx = start
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
        idx += 1
    return None, start


def _replace_frac_commands(text: str) -> str:
    frac_cmd_re = re.compile(r"\\(?:d|t)?frac\s*")
    out: list[str] = []
    idx = 0
    while idx < len(text):
        match = frac_cmd_re.search(text, idx)
        if not match:
            out.append(text[idx:])
            break
        out.append(text[idx : match.start()])
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        numerator, cursor_after_num = _extract_braced(text, cursor)
        if numerator is None:
            out.append(text[match.start() : match.end()])
            idx = match.end()
            continue
        cursor = cursor_after_num
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        denominator, cursor_after_den = _extract_braced(text, cursor)
        if denominator is None:
            out.append(text[match.start():cursor])
            idx = cursor
            continue
        out.append(f"{numerator} / {denominator}")
        idx = cursor_after_den
    return "".join(out)


def normalize_inline_math(text: str) -> str:
    while True:
        updated = _replace_frac_commands(text)
        if updated == text:
            break
        text = updated
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\times", "×")
    text = text.replace("\\cdot", "·")
    # Add LaTeX command mappings
    latex_mappings = {
        "\\succ": ">",
        "\\prec": "<",
        "\\geq": "≥",
        "\\leq": "≤",
        "\\bullet": "•",
        "\\neq": "≠",
        "\\approx": "≈",
    }
    for cmd, sym in latex_mappings.items():
        text = text.replace(cmd, sym)

    def replacer(match: re.Match[str]) -> str:
        inner = match.group(1)
        inner = inner.replace("\\%", "%")
        inner = inner.replace("\\", "")
        return inner.strip()

    text = INLINE_MATH_RE.sub(replacer, text)
    text = PAREN_MATH_RE.sub(lambda m: m.group(1), text)
    text = BRACKET_MATH_RE.sub(lambda m: m.group(1), text)
    text = text.replace("\\%", "%")
    text = text.replace(r"\(", "(").replace(r"\)", ")")
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    # Catch-all for unknown LaTeX commands
    text = re.sub(r"\\[a-z]+\b", "", text)
    return text


def _is_noise_line(line: str) -> bool:
    if not line:
        return False
    if IMAGE_LINE_RE.match(line):
        return True
    if all(not ch.isalnum() for ch in line) and len(line) >= 4:
        return True
    return any(pattern.match(line) for pattern in NOISE_PATTERNS)


def _dedupe_consecutive_sentences(text: str) -> str:
    parts = SENTENCE_SPLIT_RE.split(text)
    if len(parts) <= 1:
        return text
    deduped: list[str] = []
    seen_norms: set[str] = set()
    for part in parts:
        candidate = part.strip()
        if not candidate:
            continue
        norm = re.sub(r"\s+", " ", candidate).strip().lower()
        # Remove repeated full sentences, but keep short labels/captions.
        if norm in seen_norms and len(norm) >= 24:
            continue
        deduped.append(candidate)
        if len(norm) >= 24:
            seen_norms.add(norm)
    return " ".join(deduped) if deduped else text


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = BOOKMARK_INLINE_RE.sub(" ", text)
    text = TABLE_RE.sub(lambda m: "\n" + flatten_html_table(m.group(0)) + "\n", text)
    text = normalize_inline_math(text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "")
    text = re.sub(r"([0-9]{1,2}(?:st|nd|rd|th))(20[0-9]{2})", r"\1 \2", text)

    cleaned_lines: list[str] = []
    seen_long_non_table_lines: set[str] = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if _is_noise_line(line):
            continue
        if cleaned_lines and cleaned_lines[-1] == line:
            continue
        if "|" not in line and not line.startswith("#") and len(line) >= 48:
            normalized_line = re.sub(r"\s+", " ", line).strip().lower()
            if normalized_line in seen_long_non_table_lines:
                continue
            seen_long_non_table_lines.add(normalized_line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = _dedupe_consecutive_sentences(cleaned)
    return cleaned.strip()
