from __future__ import annotations

from ...use_cases.chunking import count_tokens, split_text_by_tokens


def _is_table_chunk_text(text: str) -> bool:
    return text.lstrip().startswith("Table:")


def _normalize_table_chunk_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    if lines[0].startswith("Table:"):
        lines[0] = "Table:"
        return "\n".join(lines).strip()

    pipe_lines = [line for line in lines if "|" in line]
    pipe_count = cleaned.count("|")
    looks_table_like = len(pipe_lines) >= 2 or pipe_count >= 6
    if not looks_table_like:
        return cleaned

    first_pipe_idx = next((idx for idx, line in enumerate(lines) if "|" in line), 0)
    prefix_lines = lines[:first_pipe_idx]
    table_lines = lines[first_pipe_idx:] if first_pipe_idx < len(lines) else lines
    out_lines = ["Table:"]
    if prefix_lines:
        # Keep pre-table context as a caption row so no information is lost.
        out_lines.append(" | ".join(prefix_lines))
    out_lines.extend(table_lines)
    return "\n".join(out_lines).strip()


def _split_table_rows(text: str, *, max_tokens: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if not lines[0].startswith("Table:"):
        return [text.strip()]

    data_lines = lines[1:]

    def maybe_strip_orphan_prefix(chunk_text: str) -> str:
        chunk_lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        if not chunk_lines:
            return ""
        if not chunk_lines[0].startswith("Table:"):
            return chunk_text.strip()
        body = chunk_lines[1:]
        if any("|" in line for line in body):
            return chunk_text.strip()
        # Keep single-column tables, but drop "Table:" for prose-like tails.
        prose_like = any(len(line.split()) > 12 or line.endswith((".", ";")) for line in body)
        if prose_like:
            return "\n".join(body).strip()
        return chunk_text.strip()

    if not any("|" in line for line in data_lines):
        stripped = maybe_strip_orphan_prefix(text)
        return [stripped] if stripped else []

    header = lines[0]
    rows = data_lines
    chunks: list[str] = []
    current_rows: list[str] = []

    def flush_rows() -> None:
        if current_rows:
            chunk_text = maybe_strip_orphan_prefix(header + "\n" + "\n".join(current_rows))
            if chunk_text:
                chunks.append(chunk_text)

    for row in rows:
        row_candidate = header + "\n" + row
        if count_tokens(row_candidate) > max_tokens:
            flush_rows()
            current_rows.clear()
            split_rows = split_text_by_tokens(
                row,
                target_tokens=max_tokens,
                max_tokens=max_tokens,
                overlap_tokens=1,
            )
            for split_row in split_rows:
                split_row = split_row.strip()
                if not split_row:
                    continue
                chunk_text = maybe_strip_orphan_prefix(header + "\n" + split_row)
                if chunk_text:
                    chunks.append(chunk_text)
            continue

        expanded = header + "\n" + "\n".join(current_rows + [row])
        if current_rows and count_tokens(expanded) > max_tokens:
            flush_rows()
            current_rows.clear()
        current_rows.append(row)
    flush_rows()
    return [chunk for chunk in chunks if chunk.strip()] or [text.strip()]


def _merge_tiny_chunk_texts(chunk_texts: list[str], *, min_tokens: int, max_tokens: int) -> list[str]:
    if not chunk_texts:
        return []
    working = [chunk.strip() for chunk in chunk_texts if chunk and chunk.strip()]
    if len(working) <= 1:
        return working

    out: list[str] = []
    idx = 0
    while idx < len(working):
        chunk = working[idx]
        tokens = count_tokens(chunk)
        if tokens >= min_tokens:
            out.append(chunk)
            idx += 1
            continue

        merged = False
        if idx + 1 < len(working):
            nxt = working[idx + 1]
            if _is_table_chunk_text(chunk) == _is_table_chunk_text(nxt):
                forward_candidate = chunk.rstrip() + "\n\n" + nxt.lstrip()
                if count_tokens(forward_candidate) <= max_tokens:
                    working[idx + 1] = forward_candidate
                    merged = True
        if merged:
            idx += 1
            continue

        if out and _is_table_chunk_text(out[-1]) == _is_table_chunk_text(chunk):
            backward_candidate = out[-1].rstrip() + "\n\n" + chunk.lstrip()
            if count_tokens(backward_candidate) <= max_tokens:
                out[-1] = backward_candidate
                idx += 1
                continue

        out.append(chunk)
        idx += 1
    return out


def _chunk_segment_texts(text: str, *, target_tokens: int, max_tokens: int, overlap_tokens: int, min_chars: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    def split_table_payload(payload: str) -> tuple[str | None, str]:
        payload_lines = payload.splitlines()
        table_lines: list[str] = []
        saw_pipe = False
        idx = 0
        while idx < len(payload_lines):
            line = payload_lines[idx].strip()
            if not line:
                if table_lines:
                    idx += 1
                    while idx < len(payload_lines) and not payload_lines[idx].strip():
                        idx += 1
                    break
                idx += 1
                continue

            if table_lines and saw_pipe and "|" not in line:
                break

            table_lines.append(line)
            if "|" in line:
                saw_pipe = True
            idx += 1

        trailing = "\n".join(payload_lines[idx:]).strip()
        if not table_lines:
            return None, payload.strip()
        if not saw_pipe:
            prose_like = any(len(line.split()) > 12 or line.endswith((".", ";")) for line in table_lines)
            if prose_like:
                prose = "\n".join(table_lines).strip()
                combined = "\n\n".join(part for part in [prose, trailing] if part).strip()
                return None, combined
        return "Table:\n" + "\n".join(table_lines), trailing

    fragments: list[tuple[str, str]] = []
    if "Table:\n" not in normalized:
        fragments.append(("text", normalized))
    else:
        parts = normalized.split("Table:\n")
        prefix = parts[0].strip()
        if prefix:
            fragments.append(("text", prefix))
        for payload in parts[1:]:
            table_fragment, trailing_fragment = split_table_payload(payload)
            if table_fragment:
                fragments.append(("table", table_fragment))
            if trailing_fragment:
                fragments.append(("text", trailing_fragment))

    chunk_pairs: list[tuple[str, str]] = []
    for fragment_kind, fragment_text in fragments:
        if not fragment_text.strip():
            continue
        if fragment_kind == "table":
            for table_chunk in _split_table_rows(fragment_text, max_tokens=max_tokens):
                if table_chunk.strip():
                    chunk_pairs.append(("table", table_chunk))
            continue
        text_chunks = split_text_by_tokens(
            fragment_text,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for text_chunk in text_chunks:
            if text_chunk.strip():
                chunk_pairs.append(("text", text_chunk))

    def normalize_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for chunk_kind, chunk_text in pairs:
            normalized_text = _normalize_table_chunk_text(chunk_text)
            if not normalized_text:
                continue
            normalized_kind = "table" if _is_table_chunk_text(normalized_text) else chunk_kind
            normalized.append((normalized_kind, normalized_text))
        return normalized

    chunk_pairs = normalize_pairs(chunk_pairs)
    chunks = [chunk_text for _, chunk_text in chunk_pairs]

    if len(chunks) <= 1:
        return chunks

    # Merge tiny tail chunk if it fits in the max token budget.
    if len(chunks[-1]) < min_chars and chunk_pairs[-1][0] == "text" and chunk_pairs[-2][0] == "text":
        merged_tail = chunks[-2].rstrip() + "\n\n" + chunks[-1].lstrip()
        if count_tokens(merged_tail) <= max_tokens:
            chunks = chunks[:-2] + [merged_tail]
            chunk_pairs = chunk_pairs[:-2] + [("text", merged_tail)]

    # Hard guard: never emit a chunk over max_tokens.
    fixed_pairs: list[tuple[str, str]] = []
    safe_overlap = max(1, min(overlap_tokens, target_tokens - 1, max_tokens - 1))
    for chunk_kind, chunk in chunk_pairs:
        if count_tokens(chunk) <= max_tokens:
            fixed_pairs.append((chunk_kind, chunk))
            continue
        split_chunks = split_text_by_tokens(
            chunk,
            target_tokens=max_tokens,
            max_tokens=max_tokens,
            overlap_tokens=safe_overlap,
        )
        fixed_pairs.extend((chunk_kind, split_chunk) for split_chunk in split_chunks if split_chunk.strip())
    fixed_pairs = normalize_pairs(fixed_pairs)
    return [chunk_text for _, chunk_text in fixed_pairs]