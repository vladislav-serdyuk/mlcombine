"""BaseArchitectureValidator — abstract base class for architecture validators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mlcombine.core.types import MLCombineConfig, PipelineContext


class BaseArchitectureValidator(ABC):
    """Base class for architecture/hyperparameter validators (CV, holdout).

    Builds fresh models internally from a blueprint and evaluates the
    architecture/hyperparameters via resampling.
    """

    def __init__(self, cfg: MLCombineConfig | None = None, **params: Any) -> None:
        self.cfg = cfg
        self.params = params

    @abstractmethod
    def validate(
        self,
        blueprint: Any,  # ModelBlueprint - imported locally to avoid circular deps
        context: PipelineContext,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate architecture/hyperparams via resampling.

        Args:
            blueprint: Lazy model specification (ModelBlueprint) to build fresh models per fold/split.
            context: Pipeline context containing train data.
            **kwargs: Additional parameters (e.g., fold-specific params via with_params()).

        Returns:
            Dict of metric_name -> aggregated value (mean over folds for CV, single value for holdout).
        """
        ...

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        """Return ``False`` to skip this validator (e.g. based on config fields)."""
        return True
