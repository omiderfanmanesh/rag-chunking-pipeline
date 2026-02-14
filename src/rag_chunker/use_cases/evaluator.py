from __future__ import annotations

from ..config.eval_config import EvalConfig
from .services.artifact_evaluation_service import ArtifactEvaluationService

# Backward-compatible facade over the service-based implementation.
_DEFAULT_SERVICE = ArtifactEvaluationService()


def evaluate_artifacts(config: EvalConfig) -> dict:
    return _DEFAULT_SERVICE.evaluate_artifacts(config)


def render_markdown_report(report: dict) -> str:
    return _DEFAULT_SERVICE.render_markdown_report(report)


def run_evaluation(config: EvalConfig) -> dict:
    return _DEFAULT_SERVICE.run(config)
