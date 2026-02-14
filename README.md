# MinerU-Aware RAG Chunker

Deterministic chunking pipeline for MinerU PDF extraction outputs.

## Run

```bash
python -m rag_chunker.cli \
  --input-dir /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/data \
  --output-dir /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/artifacts \
  --source-priority block_first \
  --target-tokens 450 \
  --max-tokens 520 \
  --overlap-tokens 80
```

## Outputs

- `artifacts/chunks.jsonl`
- `artifacts/documents.jsonl`
- `artifacts/run_manifest.json`

## Evaluate Chunking and Metadata Quality

```bash
python -m rag_chunker.eval_cli \
  --artifacts-dir /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/artifacts \
  --output-json /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/artifacts/eval_report.json \
  --output-md /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/artifacts/eval_report.md \
  --target-tokens 450 \
  --max-tokens 520 \
  --fail-on-threshold
```

Generated reports:
- `artifacts/eval_report.json` (machine-readable metrics and samples)
- `artifacts/eval_report.md` (human-readable scorecard)

## Run DeepEval Quality Gates

```bash
python -m rag_chunker.deepeval_cli \
  --artifacts-dir /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/artifacts \
  --eval-report /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/artifacts/eval_report.json \
  --output-json /Users/omiderfanmanesh/Projects/rag-chunking-pipeline/artifacts/deepeval_gate_report.json \
  --max-overlap-p95-chars 50 \
  --max-duplicate-instance-pct 0 \
  --max-missing-metadata-pct 0 \
  --fail-on-threshold
```

Generated report:
- `artifacts/deepeval_gate_report.json` (deterministic DeepEval gate checks for CI)
