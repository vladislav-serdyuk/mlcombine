"""Ensemble meta-provider — weighted vote/average from multiple models.

Usage in YAML::

    model:
      - id: "catboost"
        provider: "catboost"
        params: { iterations: 3000, depth: 8 }
      - id: "lgbm"
        provider: "lightgbm"
        params: { iterations: 3000, num_leaves: 64 }
      - provider: "ensemble"
        models:
          - "catboost"
          - "lgbm"
        params:
          weights: [0.6, 0.4]
          vote: "hard"  # "hard" | "soft"
"""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mlcombine.core.protocols import SupportedModel
from mlcombine.core.registry import registry
from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.tensor import UnifiedTensor

logger = logging.getLogger(__name__)


class EnsembleWrapper:
    """Weighted ensemble of multiple fitted models.

    Args:
        models: List of ``ModelBlueprint`` objects to build and fit lazily.
        weights: Optional list of weights (same length as *models*).
            If ``None``, uniform averaging is used.
        vote: Voting strategy — ``"hard"`` (weighted majority, default)
            or ``"soft"`` (average probabilities → argmax).
    """

    def __init__(
        self,
        models: list[ModelBlueprint],
        weights: list[float] | None = None,
        vote: str = "hard",
    ) -> None:
        if not models:
            raise ValueError("Ensemble requires at least one model")
        self._blueprints = list(models)
        self._models: list[SupportedModel] = []
        self._weights = weights
        self._vote = vote
        self._model_classes: list[np.ndarray | None] = []
        if weights is not None and len(weights) != len(models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")

    @property
    def is_fitted(self) -> bool:
        return len(self._models) > 0

    def fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        **kwargs: Any,
    ) -> Self:
        """Build and fit any unfitted sub-models, then return self."""
        self._models = []
        self._model_classes = []
        for i, bp in enumerate(self._blueprints):
            logger.info("Building and fitting sub-model %d/%d inside ensemble", i + 1, len(self._blueprints))
            model = bp.build()
            model.fit(x, y, **kwargs)
            self._models.append(model)
            inner = getattr(model, "_model", model)
            cls = getattr(inner, "classes_", None)
            self._model_classes.append(cls)
            logger.info("Sub-model %d/%d fitted (%s)", i + 1, len(self._blueprints), type(model).__name__)
        return self

    def _common_classes(self) -> np.ndarray | None:
        all_c = [c for c in self._model_classes if c is not None]
        if not all_c:
            return None
        return np.unique(np.concatenate([np.asarray(c).ravel() for c in all_c]))

    def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[Any]:
        if not self._models:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Soft voting: average probabilities → argmax
        if self._vote == "soft":
            common = self._common_classes()
            probas = self.predict_proba(x)
            if common is not None:
                return common[probas.argmax(axis=1)]  # type: ignore[no-any-return]
            return probas.argmax(axis=1).astype(np.int64)  # type: ignore[no-any-return]

        # Hard voting: weighted majority
        all_preds = [m.predict(x) for m in self._models]
        classes = np.unique(np.concatenate([np.asarray(p).ravel() for p in all_preds]))
        votes = np.zeros((len(x), len(classes)), dtype=np.float64)
        for i, pred in enumerate(all_preds):
            w = self._weights[i] if self._weights else 1.0
            for j, c in enumerate(classes):
                votes[:, j] += w * (np.asarray(pred).ravel() == c)
        return classes[votes.argmax(axis=1)]  # type: ignore[no-any-return]

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        if not self._models:
            raise RuntimeError("Ensemble must be fitted before prediction")
        common = self._common_classes()
        probas = []
        total_weight = 0.0

        for i, m in enumerate(self._models):
            p = np.asarray(m.predict_proba(x), dtype=np.float64)
            w = self._weights[i] if self._weights else 1.0

            if common is not None and self._model_classes[i] is not None:
                mc = np.asarray(self._model_classes[i])
                if len(mc) != len(common) or not np.array_equal(mc, common):
                    col_map = {c: j for j, c in enumerate(common)}
                    aligned = np.zeros((p.shape[0], len(common)), dtype=np.float64)
                    for j, c in enumerate(mc):
                        if c in col_map:
                            aligned[:, col_map[c]] = p[:, j]
                    p = aligned

            probas.append(w * p)
            total_weight += w

        return np.sum(probas, axis=0) / total_weight  # type: ignore[no-any-return]


@registry.model_provider("ensemble")
def ensemble_provider(models: list[ModelBlueprint], **params: Any) -> SupportedModel:
    """Create an ``EnsembleWrapper`` averaging multiple models.

    Required ``**params`` (optional):
        weights (list[float], optional) — one weight per model.
        vote (str, optional) — ``"hard"`` or ``"soft"`` (default ``"hard"``).
    """
    weights: list[float] | None = params.pop("weights", None)
    vote: str = params.pop("vote", "hard")
    return EnsembleWrapper(models, weights=weights, vote=vote)
