"""Concrete evaluator implementations.

Each evaluator is registered via ``@registry.architecture_validator(...)``.
"""

import mlcombine.evaluators.holdout  # noqa: F401 — triggers @registry.architecture_validator decorator
import mlcombine.evaluators.cv  # noqa: F401 — triggers @registry.architecture_validator decorator

from mlcombine.evaluators.cv import CVEvaluator
from mlcombine.evaluators.holdout import HoldoutArchitectureValidator

__all__ = [
    "CVEvaluator",
    "HoldoutArchitectureValidator",
]
