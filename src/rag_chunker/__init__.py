"""MinerU-aware deterministic RAG chunking pipeline."""

from .pipeline import run_pipeline
from .use_cases.evaluator import run_evaluation
from .use_cases.augment import build_augmented_text
from .use_cases.chunking import build_segments, count_tokens, split_text_by_tokens
from .domain.models import CanonicalBlock, PageRef, Segment
from .use_cases.metadata import extract_year, update_structure_state, extract_brief_description, extract_document_name
from .use_cases.cleaning import clean_text, flatten_html_table, normalize_inline_math
from .use_cases.deepeval_gates import DeepEvalGateConfig, run_deepeval_gates
from .config.eval_config import EvalConfig

__all__ = [
    "run_pipeline",
    "run_evaluation",
    "build_augmented_text",
    "build_segments",
    "count_tokens",
    "split_text_by_tokens",
    "CanonicalBlock",
    "PageRef",
    "Segment",
    "extract_year",
    "update_structure_state",
    "extract_brief_description",
    "extract_document_name",
    "clean_text",
    "flatten_html_table",
    "normalize_inline_math",
    "DeepEvalGateConfig",
    "run_deepeval_gates",
    "EvalConfig",
]
