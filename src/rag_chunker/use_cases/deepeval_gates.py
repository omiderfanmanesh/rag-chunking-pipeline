from __future__ import annotations

from ..config.deepeval_gate_config import DeepEvalGateConfig
from .services.deepeval_gate_service import DeepEvalGateService

# Backward-compatible facade.
_DEFAULT_SERVICE = DeepEvalGateService()


def run_deepeval_gates(config: DeepEvalGateConfig) -> dict:
    return _DEFAULT_SERVICE.run(config)
