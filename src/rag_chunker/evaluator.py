from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTICLE_RE = re.compile(r"(?im)^\s*(?:#\s*)?(?:ART\.?|ARTICLE|ARTICOLO)\s*[-.:]?\s*(\d+(?:\.\d+)*)\b")
HTML_LEFTOVER_RE = re.compile(r"<\/?(?:table|tr|td|th)\b", re.IGNORECASE)
INLINE_MATH_RE = re.compile(r"\$[^$\n]+\$")
ESCAPED_LATEX_RE = re.compile(r"\\\(|\\\)|\\\[|\\\]|\\[A-Za-z]+")
OCR_BOOKMARK_RE = re.compile(
    r"(?i)errore\.\s*il\s+(?:segnalibro|segnalbro)\s+non\s+.*?definit[oa]"
)


@dataclass
class EvalConfig:
    artifacts_dir: Path
    output_json: Path
    output_md: Path
    target_tokens: int = 450
    max_tokens: int = 520
    small_chunk_threshold: int = 20
    moderate_chunk_threshold: int = 50
    sample_size: int = 8
    max_small_chunk_pct: float = 12.0
    max_article_mixed_pct: float = 5.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100.0, 2)


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, int(0.95 * len(ordered)) - 1)
    return float(ordered[idx])


def _safe_median(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _dimension_status(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "warning"
    return "critical"


def _article_roots(text: str) -> set[str]:
    return {match.split(".")[0] for match in ARTICLE_RE.findall(text)}


def _sample_issues(chunks: list[dict[str, Any]], predicate, sample_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        if predicate(chunk):
            out.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "token_count": chunk.get("token_count"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "text_preview": str(chunk.get("text", "")).replace("\n", " ")[:220],
                }
            )
            if len(out) >= sample_size:
                break
    return out


def evaluate_artifacts(config: EvalConfig) -> dict[str, Any]:
    chunks_path = config.artifacts_dir / "chunks.jsonl"
    documents_path = config.artifacts_dir / "documents.jsonl"
    manifest_path = config.artifacts_dir / "run_manifest.json"

    chunks = _load_jsonl(chunks_path)
    documents = _load_jsonl(documents_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    total_chunks = len(chunks)
    total_docs = len(documents)
    doc_by_id = {doc.get("doc_id"): doc for doc in documents}

    tokens = [int(chunk.get("token_count", 0)) for chunk in chunks]
    chars = [int(chunk.get("char_count", 0)) for chunk in chunks]
    small_chunks = [chunk for chunk in chunks if int(chunk.get("token_count", 0)) < config.small_chunk_threshold]
    moderate_chunks = [chunk for chunk in chunks if int(chunk.get("token_count", 0)) < config.moderate_chunk_threshold]
    oversized_chunks = [chunk for chunk in chunks if int(chunk.get("token_count", 0)) > config.max_tokens]
    target_range_chunks = [
        chunk for chunk in chunks if config.small_chunk_threshold <= int(chunk.get("token_count", 0)) <= config.max_tokens
    ]

    html_leftovers = [chunk for chunk in chunks if HTML_LEFTOVER_RE.search(str(chunk.get("text", "")))]
    inline_math_leftovers = [chunk for chunk in chunks if INLINE_MATH_RE.search(str(chunk.get("text", "")))]
    escaped_latex_leftovers = [chunk for chunk in chunks if ESCAPED_LATEX_RE.search(str(chunk.get("text", "")))]
    ocr_bookmark_leftovers = [chunk for chunk in chunks if OCR_BOOKMARK_RE.search(str(chunk.get("text", "")))]

    empty_page_refs = [chunk for chunk in chunks if not chunk.get("page_refs")]
    missing_page_range = [
        chunk for chunk in chunks if chunk.get("page_start") is None or chunk.get("page_end") is None
    ]
    invalid_page_range = [
        chunk
        for chunk in chunks
        if chunk.get("page_start") is not None
        and chunk.get("page_end") is not None
        and int(chunk.get("page_start")) > int(chunk.get("page_end"))
    ]

    article_mixed_chunks = []
    article_mismatch_chunks = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        roots = _article_roots(text)
        if len(roots) >= 2:
            article_mixed_chunks.append(chunk)

        chunk_article = str(chunk.get("metadata", {}).get("article") or "")
        if chunk_article and roots and chunk_article not in roots:
            article_mismatch_chunks.append(chunk)

    chunk_year_missing = [chunk for chunk in chunks if not chunk.get("metadata", {}).get("year")]
    chunk_name_missing = [chunk for chunk in chunks if not chunk.get("metadata", {}).get("name")]
    chunk_desc_missing = [chunk for chunk in chunks if not chunk.get("metadata", {}).get("brief_description")]
    chunk_section_missing = [chunk for chunk in chunks if not chunk.get("metadata", {}).get("section")]
    chunk_article_missing = [chunk for chunk in chunks if not chunk.get("metadata", {}).get("article")]
    chunk_subarticle_missing = [chunk for chunk in chunks if not chunk.get("metadata", {}).get("subarticle")]
    chunk_lang_missing = [chunk for chunk in chunks if not chunk.get("metadata", {}).get("language_hint")]

    doc_year_missing = [doc for doc in documents if not doc.get("year")]
    doc_name_missing = [doc for doc in documents if not doc.get("name")]
    doc_desc_missing = [doc for doc in documents if not doc.get("brief_description")]
    doc_lang_missing = [doc for doc in documents if not doc.get("language_hint")]

    chunk_doc_year_mismatch = []
    chunk_doc_name_mismatch = []
    for chunk in chunks:
        doc = doc_by_id.get(chunk.get("doc_id"))
        if not doc:
            continue
        cy = chunk.get("metadata", {}).get("year")
        dy = doc.get("year")
        if cy and dy and cy != dy:
            chunk_doc_year_mismatch.append(chunk)
        cn = chunk.get("metadata", {}).get("name")
        dn = doc.get("name")
        if cn and dn and cn != dn:
            chunk_doc_name_mismatch.append(chunk)

    text_counter = Counter(str(chunk.get("text", "")) for chunk in chunks)
    duplicate_instances = sum(count for count in text_counter.values() if count > 1)
    duplicate_unique_texts = sum(1 for count in text_counter.values() if count > 1)

    source_modes = Counter(str(doc.get("source_mode_used", "unknown")) for doc in documents)

    size_score = max(
        0.0,
        100.0
        - (_pct(len(oversized_chunks), total_chunks) * 2.0)
        - (_pct(len(small_chunks), total_chunks) * 0.6),
    )
    cleanliness_penalty = (
        _pct(len(html_leftovers), total_chunks) * 3.0
        + _pct(len(inline_math_leftovers), total_chunks) * 2.0
        + _pct(len(escaped_latex_leftovers), total_chunks) * 2.5
        + _pct(len(ocr_bookmark_leftovers), total_chunks) * 5.0
    )
    cleanliness_score = max(0.0, 100.0 - cleanliness_penalty)
    metadata_completeness = (
        100.0
        - _pct(len(chunk_year_missing), total_chunks) * 0.15
        - _pct(len(chunk_name_missing), total_chunks) * 0.15
        - _pct(len(chunk_desc_missing), total_chunks) * 0.15
        - _pct(len(chunk_section_missing), total_chunks) * 0.2
        - _pct(len(chunk_article_missing), total_chunks) * 0.2
        - _pct(len(chunk_lang_missing), total_chunks) * 0.15
    )
    metadata_consistency_penalty = _pct(len(article_mismatch_chunks), total_chunks) * 1.2 + _pct(
        len(chunk_doc_year_mismatch), total_chunks
    ) * 2.0 + _pct(len(chunk_doc_name_mismatch), total_chunks) * 2.0
    metadata_score = max(0.0, metadata_completeness - metadata_consistency_penalty)
    provenance_score = max(
        0.0,
        100.0
        - _pct(len(empty_page_refs), total_chunks) * 3.0
        - _pct(len(missing_page_range), total_chunks) * 3.0
        - _pct(len(invalid_page_range), total_chunks) * 8.0,
    )

    overall_score = round(
        size_score * 0.3 + cleanliness_score * 0.2 + metadata_score * 0.3 + provenance_score * 0.2,
        2,
    )

    report = {
        "summary": {
            "documents": total_docs,
            "chunks": total_chunks,
            "source_modes": dict(source_modes),
            "overall_score": overall_score,
            "overall_status": _dimension_status(overall_score),
        },
        "dimensions": {
            "size": {"score": round(size_score, 2), "status": _dimension_status(size_score)},
            "cleanliness": {"score": round(cleanliness_score, 2), "status": _dimension_status(cleanliness_score)},
            "metadata": {"score": round(metadata_score, 2), "status": _dimension_status(metadata_score)},
            "provenance": {"score": round(provenance_score, 2), "status": _dimension_status(provenance_score)},
        },
        "chunk_metrics": {
            "token_stats": {
                "min": min(tokens) if tokens else 0,
                "median": _safe_median(tokens),
                "p95": _p95(tokens),
                "max": max(tokens) if tokens else 0,
                "avg": round(sum(tokens) / len(tokens), 2) if tokens else 0.0,
            },
            "char_stats": {
                "min": min(chars) if chars else 0,
                "median": _safe_median(chars),
                "p95": _p95(chars),
                "max": max(chars) if chars else 0,
                "avg": round(sum(chars) / len(chars), 2) if chars else 0.0,
            },
            "small_chunks": {
                "threshold": config.small_chunk_threshold,
                "count": len(small_chunks),
                "pct": _pct(len(small_chunks), total_chunks),
            },
            "moderate_chunks": {
                "threshold": config.moderate_chunk_threshold,
                "count": len(moderate_chunks),
                "pct": _pct(len(moderate_chunks), total_chunks),
            },
            "oversized_chunks": {
                "max_tokens": config.max_tokens,
                "count": len(oversized_chunks),
                "pct": _pct(len(oversized_chunks), total_chunks),
            },
            "in_target_range": {
                "count": len(target_range_chunks),
                "pct": _pct(len(target_range_chunks), total_chunks),
            },
            "duplicates": {
                "duplicate_instances": duplicate_instances,
                "duplicate_unique_texts": duplicate_unique_texts,
                "duplicate_instance_pct": _pct(duplicate_instances, total_chunks),
            },
        },
        "metadata_metrics": {
            "chunk_completeness": {
                "year_missing": {"count": len(chunk_year_missing), "pct": _pct(len(chunk_year_missing), total_chunks)},
                "name_missing": {"count": len(chunk_name_missing), "pct": _pct(len(chunk_name_missing), total_chunks)},
                "brief_description_missing": {
                    "count": len(chunk_desc_missing),
                    "pct": _pct(len(chunk_desc_missing), total_chunks),
                },
                "section_missing": {
                    "count": len(chunk_section_missing),
                    "pct": _pct(len(chunk_section_missing), total_chunks),
                },
                "article_missing": {
                    "count": len(chunk_article_missing),
                    "pct": _pct(len(chunk_article_missing), total_chunks),
                },
                "subarticle_missing": {
                    "count": len(chunk_subarticle_missing),
                    "pct": _pct(len(chunk_subarticle_missing), total_chunks),
                },
                "language_missing": {"count": len(chunk_lang_missing), "pct": _pct(len(chunk_lang_missing), total_chunks)},
            },
            "document_completeness": {
                "year_missing": {"count": len(doc_year_missing), "pct": _pct(len(doc_year_missing), total_docs)},
                "name_missing": {"count": len(doc_name_missing), "pct": _pct(len(doc_name_missing), total_docs)},
                "brief_description_missing": {"count": len(doc_desc_missing), "pct": _pct(len(doc_desc_missing), total_docs)},
                "language_missing": {"count": len(doc_lang_missing), "pct": _pct(len(doc_lang_missing), total_docs)},
            },
            "consistency": {
                "article_mixed_chunks": {
                    "count": len(article_mixed_chunks),
                    "pct": _pct(len(article_mixed_chunks), total_chunks),
                },
                "article_metadata_mismatch": {
                    "count": len(article_mismatch_chunks),
                    "pct": _pct(len(article_mismatch_chunks), total_chunks),
                },
                "chunk_doc_year_mismatch": {
                    "count": len(chunk_doc_year_mismatch),
                    "pct": _pct(len(chunk_doc_year_mismatch), total_chunks),
                },
                "chunk_doc_name_mismatch": {
                    "count": len(chunk_doc_name_mismatch),
                    "pct": _pct(len(chunk_doc_name_mismatch), total_chunks),
                },
            },
        },
        "cleanliness_metrics": {
            "html_leftovers": {"count": len(html_leftovers), "pct": _pct(len(html_leftovers), total_chunks)},
            "inline_math_leftovers": {"count": len(inline_math_leftovers), "pct": _pct(len(inline_math_leftovers), total_chunks)},
            "escaped_latex_leftovers": {
                "count": len(escaped_latex_leftovers),
                "pct": _pct(len(escaped_latex_leftovers), total_chunks),
            },
            "ocr_bookmark_leftovers": {
                "count": len(ocr_bookmark_leftovers),
                "pct": _pct(len(ocr_bookmark_leftovers), total_chunks),
            },
        },
        "provenance_metrics": {
            "empty_page_refs": {"count": len(empty_page_refs), "pct": _pct(len(empty_page_refs), total_chunks)},
            "missing_page_range": {"count": len(missing_page_range), "pct": _pct(len(missing_page_range), total_chunks)},
            "invalid_page_range": {"count": len(invalid_page_range), "pct": _pct(len(invalid_page_range), total_chunks)},
        },
        "samples": {
            "oversized_chunks": _sample_issues(chunks, lambda c: int(c.get("token_count", 0)) > config.max_tokens, config.sample_size),
            "small_chunks": _sample_issues(
                chunks, lambda c: int(c.get("token_count", 0)) < config.small_chunk_threshold, config.sample_size
            ),
            "article_mixed_chunks": _sample_issues(
                chunks, lambda c: len(_article_roots(str(c.get("text", "")))) >= 2, config.sample_size
            ),
            "ocr_bookmark_leftovers": _sample_issues(
                chunks, lambda c: OCR_BOOKMARK_RE.search(str(c.get("text", ""))) is not None, config.sample_size
            ),
        },
        "manifest_echo": {
            "documents": manifest.get("documents"),
            "chunks": manifest.get("chunks"),
            "errors": manifest.get("errors", []),
            "source_modes": manifest.get("source_modes", {}),
        },
    }
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    dims = report["dimensions"]
    chunk_metrics = report["chunk_metrics"]
    meta_metrics = report["metadata_metrics"]
    clean_metrics = report["cleanliness_metrics"]
    prov_metrics = report["provenance_metrics"]
    gates = report["quality_gates"]

    lines: list[str] = []
    lines.append("# Chunking and Metadata Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Documents: {summary['documents']}")
    lines.append(f"- Chunks: {summary['chunks']}")
    lines.append(f"- Overall score: {summary['overall_score']} ({summary['overall_status']})")
    lines.append(f"- Source modes: {summary['source_modes']}")
    lines.append("")
    lines.append("## Dimension Scores")
    lines.append("| Dimension | Score | Status |")
    lines.append("|---|---:|---|")
    for key in ["size", "cleanliness", "metadata", "provenance"]:
        lines.append(f"| {key} | {dims[key]['score']} | {dims[key]['status']} |")
    lines.append("")
    lines.append("## Quality Gates")
    lines.append(f"- Overall pass: **{gates['passed']}**")
    lines.append("| Gate | Actual | Expected | Passed |")
    lines.append("|---|---:|---:|---|")
    for gate in gates["checks"]:
        lines.append(f"| {gate['name']} | {gate['actual']} | {gate['expected']} | {gate['passed']} |")
    lines.append("")
    lines.append("## Chunk Quality")
    lines.append(
        f"- Token stats: min={chunk_metrics['token_stats']['min']}, median={chunk_metrics['token_stats']['median']}, "
        f"p95={chunk_metrics['token_stats']['p95']}, max={chunk_metrics['token_stats']['max']}, avg={chunk_metrics['token_stats']['avg']}"
    )
    lines.append(
        f"- Small chunks (<{chunk_metrics['small_chunks']['threshold']}): {chunk_metrics['small_chunks']['count']} ({chunk_metrics['small_chunks']['pct']}%)"
    )
    lines.append(
        f"- Oversized chunks (>{chunk_metrics['oversized_chunks']['max_tokens']}): {chunk_metrics['oversized_chunks']['count']} ({chunk_metrics['oversized_chunks']['pct']}%)"
    )
    lines.append(f"- In target range: {chunk_metrics['in_target_range']['count']} ({chunk_metrics['in_target_range']['pct']}%)")
    lines.append(
        f"- Duplicate instances: {chunk_metrics['duplicates']['duplicate_instances']} "
        f"({chunk_metrics['duplicates']['duplicate_instance_pct']}%)"
    )
    lines.append("")
    lines.append("## Metadata Quality")
    lines.append(
        f"- Chunk missing section/article/subarticle: "
        f"{meta_metrics['chunk_completeness']['section_missing']['pct']}% / "
        f"{meta_metrics['chunk_completeness']['article_missing']['pct']}% / "
        f"{meta_metrics['chunk_completeness']['subarticle_missing']['pct']}%"
    )
    lines.append(
        f"- Chunk-doc year/name mismatch: "
        f"{meta_metrics['consistency']['chunk_doc_year_mismatch']['count']} / "
        f"{meta_metrics['consistency']['chunk_doc_name_mismatch']['count']}"
    )
    lines.append(
        f"- Mixed-article chunks: {meta_metrics['consistency']['article_mixed_chunks']['count']} "
        f"({meta_metrics['consistency']['article_mixed_chunks']['pct']}%)"
    )
    lines.append("")
    lines.append("## Cleanliness and Provenance")
    lines.append(
        f"- Residual HTML/math/latex/OCR: "
        f"{clean_metrics['html_leftovers']['count']} / "
        f"{clean_metrics['inline_math_leftovers']['count']} / "
        f"{clean_metrics['escaped_latex_leftovers']['count']} / "
        f"{clean_metrics['ocr_bookmark_leftovers']['count']}"
    )
    lines.append(
        f"- Missing or invalid page provenance: "
        f"{prov_metrics['empty_page_refs']['count']} empty refs, "
        f"{prov_metrics['missing_page_range']['count']} missing ranges, "
        f"{prov_metrics['invalid_page_range']['count']} invalid ranges"
    )
    lines.append("")
    lines.append("## Actionable Focus")
    lines.append("- Reduce small chunk ratio by merging ToC-like fragments with adjacent semantic blocks.")
    lines.append("- Lower mixed-article chunk count by stronger article boundary segmentation for summary sections.")
    lines.append("- Keep page provenance strict (already healthy if zeros above).")
    return "\n".join(lines) + "\n"


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    report = evaluate_artifacts(config)
    small_pct = report["chunk_metrics"]["small_chunks"]["pct"]
    oversized = report["chunk_metrics"]["oversized_chunks"]["count"]
    mixed_article_pct = report["metadata_metrics"]["consistency"]["article_mixed_chunks"]["pct"]
    clean_total = (
        report["cleanliness_metrics"]["html_leftovers"]["count"]
        + report["cleanliness_metrics"]["inline_math_leftovers"]["count"]
        + report["cleanliness_metrics"]["escaped_latex_leftovers"]["count"]
        + report["cleanliness_metrics"]["ocr_bookmark_leftovers"]["count"]
    )
    provenance_total = (
        report["provenance_metrics"]["empty_page_refs"]["count"]
        + report["provenance_metrics"]["missing_page_range"]["count"]
        + report["provenance_metrics"]["invalid_page_range"]["count"]
    )
    checks = [
        {
            "name": "oversized_chunks",
            "actual": oversized,
            "expected": "<= 0",
            "passed": oversized <= 0,
        },
        {
            "name": "small_chunks_pct",
            "actual": small_pct,
            "expected": f"<= {config.max_small_chunk_pct}",
            "passed": small_pct <= config.max_small_chunk_pct,
        },
        {
            "name": "mixed_article_chunks_pct",
            "actual": mixed_article_pct,
            "expected": f"<= {config.max_article_mixed_pct}",
            "passed": mixed_article_pct <= config.max_article_mixed_pct,
        },
        {
            "name": "residual_noise_count",
            "actual": clean_total,
            "expected": "<= 0",
            "passed": clean_total <= 0,
        },
        {
            "name": "provenance_issues_count",
            "actual": provenance_total,
            "expected": "<= 0",
            "passed": provenance_total <= 0,
        },
    ]
    report["quality_gates"] = {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_md.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    config.output_md.write_text(render_markdown_report(report), encoding="utf-8")
    return report
