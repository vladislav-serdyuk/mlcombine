"""Structural protocols for core registry extension points.

Each protocol describes the minimal interface a registered extension must
satisfy.  Because these are ``Protocol`` classes (PEP 544), providers written
outside the framework satisfy them structurally — no explicit inheritance
required.

All types used here are either builtins or from ``core.enums`` so that this
module imports nothing from higher layers, guaranteeing zero risk of circular
imports.
"""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    from typing import Self
    from mlcombine.core.tensor import UnifiedTensor
    from mlcombine.core.types import PipelineContext


class LayerBuilderProtocol(Protocol):
    """Protocol matching layer-builder callables.

    Implementations receive ``(cfg, prev_dim, **kwargs)`` and return an
    ``nn.Module`` (the exact type is opaque to the protocol).
    """

    def __call__(self, cfg: dict[str, Any], prev_dim: int, **kwargs: Any) -> Any: ...


class ActivationProtocol(Protocol):
    """Protocol matching activation-class constructors.

    Implementations are expected to return an ``nn.Module`` instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class MLModelProtocol[T: np.generic](Protocol):
    """Protocol defining the minimal interface for ML models (fit, predict, predict_proba).

    Type parameter ``T`` is the dtype of ``predict()`` output (e.g. ``np.int_`` for
    class labels, ``np.float64`` for regression values).
    """

    def fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        **kwargs: Any,
    ) -> Self:
        """Fit the model to training data and return self."""
        ...

    def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[T]:
        """Predict target values for given features."""
        ...

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        """Predict class probabilities for given features."""
        ...


@runtime_checkable
class ProbabilityModelProtocol(MLModelProtocol[np.float64], Protocol):
    """Protocol for models outputting probability (0-1) per sample."""


class EvaluatorProtocol(Protocol):
    """Protocol for model evaluators.

    Evaluator is configured with data/context at init, then evaluates fitted models.
    Returns a dict of metric_name -> value.
    """

    def evaluate(self, model: MLModelProtocol[Any], context: PipelineContext | None = None) -> dict[str, float]:
        """Evaluate the model and return dict of metric_name -> value."""
        ...


class ArchitectureValidatorProtocol(Protocol):
    """Protocol for architecture/hyperparameter validators (CV, holdout, bootstrap).

    Unlike EvaluatorProtocol which assesses a *fitted* model instance,
    ArchitectureValidator builds fresh models internally from a blueprint
    and evaluates the architecture/hyperparameters via resampling.
    """

    def validate(
        self,
        blueprint: Any,  # ModelBlueprint - avoids circular import
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


type SupportedModel = MLModelProtocol[Any]
"""A model of any output dtype — use when the specific dtype doesn't matter."""


__all__ = [
    "ActivationProtocol",
    "ArchitectureValidatorProtocol",
    "EvaluatorProtocol",
    "LayerBuilderProtocol",
    "MLModelProtocol",
    "ProbabilityModelProtocol",
    "SupportedModel",
]
