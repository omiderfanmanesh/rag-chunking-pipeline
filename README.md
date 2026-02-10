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
