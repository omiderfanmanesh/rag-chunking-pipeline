# Chunking and Metadata Evaluation Report

## Summary
- Documents: 8
- Chunks: 1610
- Overall score: 99.38 (excellent)
- Source modes: {'block_list': 7, 'content_list': 1}

## Dimension Scores
| Dimension | Score | Status |
|---|---:|---|
| size | 97.95 | excellent |
| cleanliness | 100.0 | excellent |
| metadata | 100.0 | excellent |
| provenance | 100.0 | excellent |

## Quality Gates
- Overall pass: **True**
| Gate | Actual | Expected | Passed |
|---|---:|---:|---|
| oversized_chunks | 0 | <= 0 | True |
| small_chunks_pct | 3.42 | <= 12.0 | True |
| mixed_article_chunks_pct | 0.31 | <= 5.0 | True |
| residual_noise_count | 0 | <= 0 | True |
| provenance_issues_count | 0 | <= 0 | True |

## Chunk Quality
- Token stats: min=17, median=144.0, p95=429.0, max=478, avg=183.27
- Small chunks (<50): 55 (3.42%)
- Oversized chunks (>480): 0 (0.0%)
- In target range: 1555 (96.58%)
- Duplicate instances: 0 (0.0%)

## Metadata Quality
- Chunk missing section/article/subarticle: 0.0% / 0.0% / 59.25%
- Chunk-doc year/name mismatch: 0 / 0
- Mixed-article chunks: 5 (0.31%)

## Cleanliness and Provenance
- Residual HTML/math/latex/OCR: 0 / 0 / 0 / 0
- Missing or invalid page provenance: 0 empty refs, 0 missing ranges, 0 invalid ranges

## Actionable Focus
- Reduce small chunk ratio by merging ToC-like fragments with adjacent semantic blocks.
- Lower mixed-article chunk count by stronger article boundary segmentation for summary sections.
- Keep page provenance strict (already healthy if zeros above).
