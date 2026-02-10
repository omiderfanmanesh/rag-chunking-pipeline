# Chunking and Metadata Evaluation Report

## Summary
- Documents: 8
- Chunks: 1867
- Overall score: 99.74 (excellent)
- Source modes: {'block_list': 7, 'content_list': 1}

## Dimension Scores
| Dimension | Score | Status |
|---|---:|---|
| size | 99.42 | excellent |
| cleanliness | 100.0 | excellent |
| metadata | 99.7 | excellent |
| provenance | 100.0 | excellent |

## Quality Gates
- Overall pass: **True**
| Gate | Actual | Expected | Passed |
|---|---:|---:|---|
| oversized_chunks | 0 | <= 0 | True |
| small_chunks_pct | 0.96 | <= 12.0 | True |
| mixed_article_chunks_pct | 0.21 | <= 5.0 | True |
| residual_noise_count | 0 | <= 0 | True |
| provenance_issues_count | 0 | <= 0 | True |

## Chunk Quality
- Token stats: min=6, median=117.0, p95=451.0, max=520, avg=162.39
- Small chunks (<20): 18 (0.96%)
- Oversized chunks (>520): 0 (0.0%)
- In target range: 1849 (99.04%)
- Duplicate instances: 0 (0.0%)

## Metadata Quality
- Chunk missing section/article/subarticle: 0.0% / 1.5% / 56.61%
- Chunk-doc year/name mismatch: 0 / 0
- Mixed-article chunks: 4 (0.21%)

## Cleanliness and Provenance
- Residual HTML/math/latex/OCR: 0 / 0 / 0 / 0
- Missing or invalid page provenance: 0 empty refs, 0 missing ranges, 0 invalid ranges

## Actionable Focus
- Reduce small chunk ratio by merging ToC-like fragments with adjacent semantic blocks.
- Lower mixed-article chunk count by stronger article boundary segmentation for summary sections.
- Keep page provenance strict (already healthy if zeros above).
