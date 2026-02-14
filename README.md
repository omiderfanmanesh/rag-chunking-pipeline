# MinerU-Aware RAG Chunker

Deterministic chunking pipeline for MinerU PDF extraction outputs.

## Architecture

The codebase follows clean architecture principles with clear separation of concerns:

```text
src/rag_chunker/
  config/
    pipeline_config.py              # PipelineConfig
    eval_config.py                  # EvalConfig
    deepeval_gate_config.py         # DeepEvalGateConfig
  domain/
    models.py                       # Domain models (CanonicalBlock, etc.)
  infrastructure/
    io.py                           # IO utilities (read_json, write_jsonl, etc.)
  interfaces/
    cli.py                          # Command-line interface
  use_cases/
    pipeline.py                     # Main orchestration
    evaluator.py                    # Evaluation use case
    deepeval_gates.py               # Quality gates use case
    services/
      block_loader_service.py       # Document loading and block processing
      segment_merge_service.py      # TOC and segment merging
      chunk_assembly_service.py     # Chunking and assembly logic
      structure_resolver_service.py # Structure resolution
      artifact_evaluation_service.py # Artifact evaluation
      deepeval_gate_service.py      # DeepEval gate checking
      incremental_cache_service.py  # Caching
      global_chunk_dedupe_service.py # Deduplication
      tiny_chunk_sweep_service.py   # Tiny chunk handling
```

## Run Pipeline

```bash
PYTHONPATH=src python -m rag_chunker.interfaces.cli \
  --input-dir data \
  --output-dir artifacts \
  --max-tokens 480 \
  --max-overlap-chars 200 \
  --drop-toc \
  --min-viable-chunk-tokens 50
```

Incremental mode is enabled by default: unchanged document folders are reused from cache (`artifacts/doc_hashes.json`).
To force full reprocessing, add `--no-incremental`.

## Evaluate Quality

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from rag_chunker import EvalConfig, run_evaluation
config = EvalConfig(
    artifacts_dir=Path('artifacts'),
    output_json=Path('artifacts/eval_report.json'),
    output_md=Path('artifacts/eval_report.md'),
    max_tokens=480,
    small_chunk_threshold=50
)
run_evaluation(config)
"
```

## Check Quality Gates

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from rag_chunker import DeepEvalGateConfig, run_deepeval_gates
config = DeepEvalGateConfig(
    artifacts_dir=Path('artifacts'),
    eval_report_path=Path('artifacts/eval_report.json'),
    output_json=Path('artifacts/deepeval_gate_report.json'),
    max_tiny_chunk_pct=1.0,
    max_duplicate_instance_pct=0.0,
    max_overlap_p95_chars=200,
    max_missing_metadata_pct=0.0
)
run_deepeval_gates(config)
"
```

## Outputs

- `artifacts/chunks.jsonl` - Chunked text with metadata
- `artifacts/documents.jsonl` - Document metadata
- `artifacts/run_manifest.json` - Processing summary
- `artifacts/eval_report.json` - Quality metrics
- `artifacts/eval_report.md` - Human-readable evaluation
- `artifacts/deepeval_gate_report.json` - Gate check results
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
  --max-overlap-p95-chars 30 \
  --max-tiny-chunk-pct 1.0 \
  --max-duplicate-instance-pct 0 \
  --max-missing-metadata-pct 0 \
  --fail-on-threshold
```

Generated report:
- `artifacts/deepeval_gate_report.json` (deterministic DeepEval gate checks for CI)
