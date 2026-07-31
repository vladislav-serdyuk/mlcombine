"""Stacking meta-provider — trains a meta-model on base model predictions.

Supports optional OOF (out-of-fold) stacking for unbiased meta-training.

Usage in YAML::

    model:
      - id: "cb"
        provider: "catboost"
      - id: "rf"
        provider: "sklearn"
        params: { backbone: "random_forest", n_estimators: 500 }
      - provider: "stacking"
        models: ["cb", "rf"]
        params:
          meta_model: "logistic"
          n_folds: 3
          stratified: true
          oof_params:
            rf: { n_estimators: 300 }   # fold override
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Self

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mlcombine.core.protocols import SupportedModel
from mlcombine.core.registry import registry
from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.tensor import UnifiedTensor

logger = logging.getLogger(__name__)

_META_MODELS: dict[str, type] = {}


def _register_meta(name: str) -> Callable[[type], type]:
    def wrapper(cls: type) -> type:
        _META_MODELS[name] = cls
        return cls

    return wrapper


@_register_meta("logistic")
class _LogisticMeta:
    def __init__(self, **params: Any) -> None:
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(max_iter=1000, **params)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)  # type: ignore[no-any-return]

    def predict_proba(self, x: np.ndarray) -> NDArray[np.float64]:
        return self._model.predict_proba(x)  # type: ignore[no-any-return]


@_register_meta("ridge")
class _RidgeMeta:
    def __init__(self, **params: Any) -> None:
        from sklearn.linear_model import RidgeClassifier

        self._model = RidgeClassifier(**params)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)  # type: ignore[no-any-return]

    def predict_proba(self, x: np.ndarray) -> NDArray[np.float64]:
        raise RuntimeError("RidgeClassifier does not support predict_proba")


class StackingWrapper:
    """Stacking ensemble: base models → stacked probas → meta-model.

    When ``n_folds > 0``, OOF (out-of-fold) stacking is used to avoid
    meta-model overfitting.  Base models are first trained per fold with
    optional ``oof_params`` overrides, then refitted on the full dataset.

    Args:
        blueprints: List of ``ModelBlueprint`` for base models.
        meta_model: Meta-classifier name (``"logistic"`` or ``"ridge"``).
        n_folds: Number of CV folds (``0`` = direct, ``>0`` = OOF).
        stratified: Use ``StratifiedKFold`` when ``True``.
        oof_params: Per-provider param overrides for OOF fitting.
        **meta_params: Additional kwargs for the meta-classifier.
    """

    def __init__(
        self,
        blueprints: list[ModelBlueprint],
        meta_model: str = "logistic",
        n_folds: int = 0,
        stratified: bool = True,
        oof_params: dict[str, dict[str, Any]] | None = None,
        **meta_params: Any,
    ) -> None:
        if not blueprints:
            raise ValueError("Stacking requires at least one base model")
        self._blueprints = list(blueprints)
        self._base_models: list[SupportedModel] = []
        self._model_classes: list[np.ndarray | None] = []
        self._meta_type = meta_model
        self._meta_params = meta_params
        self._meta_model: _LogisticMeta | _RidgeMeta | None = None
        self._n_folds = n_folds
        self._stratified = stratified
        self._oof_params = oof_params or {}

        if meta_model not in _META_MODELS:
            raise ValueError(f"Unknown meta_model '{meta_model}'. Available: {list(_META_MODELS)}")

    @property
    def is_fitted(self) -> bool:
        return self._meta_model is not None

    @staticmethod
    def _classes_from(model: SupportedModel) -> np.ndarray | None:
        inner = getattr(model, "_model", model)
        return getattr(inner, "classes_", None)

    @staticmethod
    def _align_proba(
        p: np.ndarray,
        model_classes: np.ndarray | None,
        target_classes: np.ndarray,
    ) -> np.ndarray:
        if model_classes is None or np.array_equal(model_classes, target_classes):
            return p
        col_map = {c: j for j, c in enumerate(target_classes)}
        aligned = np.zeros((p.shape[0], len(target_classes)), dtype=np.float64)
        for j, c in enumerate(np.asarray(model_classes)):
            if c in col_map:
                aligned[:, col_map[c]] = p[:, j]
        return aligned

    def fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        **kwargs: Any,
    ) -> Self:
        """Fit base models, then meta-model on stacked probas."""
        y_arr = np.asarray(y).ravel()

        if self._n_folds > 0:
            # ── OOF stacking ──
            oof_stacked = self._oof_fit(x, y_arr, **kwargs)
            logger.info("OOF proba shape: %s — fitting meta-model (%s)", oof_stacked.shape, self._meta_type)
            meta_cls = _META_MODELS[self._meta_type]
            self._meta_model = meta_cls(**self._meta_params)
            self._meta_model.fit(oof_stacked, y_arr)
            logger.info("Meta-model fitted on OOF: %s", type(self._meta_model._model).__name__)
            # Refit base models on full data
            self._refit_full(x, y_arr, **kwargs)
        else:
            # ── Direct stacking (train on train_preds) ──
            self._base_models = []
            self._model_classes = []
            for i, bp in enumerate(self._blueprints):
                logger.info(
                    "Building and fitting base model %d/%d inside stacking",
                    i + 1,
                    len(self._blueprints),
                )
                model = bp.build()
                model.fit(x, y, **kwargs)
                self._base_models.append(model)
                self._model_classes.append(self._classes_from(model))
                logger.info(
                    "Base model %d/%d fitted (%s)",
                    i + 1,
                    len(self._blueprints),
                    type(model).__name__,
                )

            stacked = self._stack_probas(x)
            logger.info("Stacked proba shape: %s — fitting meta-model (%s)", stacked.shape, self._meta_type)
            meta_cls = _META_MODELS[self._meta_type]
            self._meta_model = meta_cls(**self._meta_params)
            self._meta_model.fit(stacked, y_arr)
            logger.info("Meta-model fitted: %s", type(self._meta_model._model).__name__)

        return self

    def _oof_fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y_arr: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """K-fold OOF: train models on folds, collect OOF stacked probas."""
        from sklearn.model_selection import KFold, StratifiedKFold

        cv: KFold | StratifiedKFold
        if self._stratified:
            cv = StratifiedKFold(n_splits=self._n_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self._n_folds, shuffle=True, random_state=42)

        common = np.unique(y_arr)
        oof_parts: list[np.ndarray] = []
        fold_indices: list[np.ndarray] = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y_arr)), y_arr)):
            logger.info("OOF fold %d/%d — train=%d val=%d", fold + 1, self._n_folds, len(train_idx), len(val_idx))

            if isinstance(x, pd.DataFrame):
                x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            else:
                x_train, x_val = x[train_idx], x[val_idx]
            y_train = y_arr[train_idx]

            fold_probas: list[np.ndarray] = []
            for bp in self._blueprints:
                overrides = self._oof_params.get(bp.provider, {})
                model = bp.with_params(**overrides).build()
                model.fit(x_train, y_train, **kwargs)
                p = np.asarray(model.predict_proba(x_val), dtype=np.float64)
                mc = self._classes_from(model)
                p = self._align_proba(p, mc, common)
                fold_probas.append(p)

            oof_parts.append(np.concatenate(fold_probas, axis=1))
            fold_indices.append(val_idx)

        # Restore original row order
        order = np.concatenate(fold_indices)
        return np.concatenate(oof_parts, axis=0)[np.argsort(order)]

    def _refit_full(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y_arr: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Refit base models on full data with original (non-OOF) params."""
        self._base_models = []
        self._model_classes = []
        for i, bp in enumerate(self._blueprints):
            logger.info("Refitting base model %d/%d on full data", i + 1, len(self._blueprints))
            model = bp.build()
            model.fit(x, y_arr, **kwargs)
            self._base_models.append(model)
            self._model_classes.append(self._classes_from(model))
            logger.info("Base model refitted (%s)", type(model).__name__)

    def _common_classes(self) -> np.ndarray | None:
        all_c = [c for c in self._model_classes if c is not None]
        if not all_c:
            return None
        return np.unique(np.concatenate([np.asarray(c).ravel() for c in all_c]))

    def _stack_probas(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> np.ndarray:
        common = self._common_classes()
        probas: list[np.ndarray] = []
        for i, m in enumerate(self._base_models):
            p = np.asarray(m.predict_proba(x), dtype=np.float64)
            if common is not None:
                p = self._align_proba(p, self._model_classes[i], common)
            probas.append(p)
        return np.concatenate(probas, axis=1)

    def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.int_]:
        if not self.is_fitted:
            raise RuntimeError("StackingWrapper must be fitted before prediction")
        assert self._meta_model is not None
        stacked = self._stack_probas(x)
        return self._meta_model.predict(stacked)

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("StackingWrapper must be fitted before prediction")
        assert self._meta_model is not None
        stacked = self._stack_probas(x)
        return self._meta_model.predict_proba(stacked)


@registry.model_provider("stacking")
def stacking_provider(models: list[ModelBlueprint], **params: Any) -> SupportedModel:
    """Create a ``StackingWrapper``.

    ``params`` keys consumed by this provider:
        meta_model (str) — ``"logistic"`` (default) or ``"ridge"``.
        n_folds (int) — number of CV folds (``0`` = direct, default 0).
        stratified (bool) — use ``StratifiedKFold``.
        oof_params (dict) — per-provider param overrides for OOF.
    """
    meta_model: str = params.pop("meta_model", "logistic")
    n_folds: int = int(params.pop("n_folds", 0))
    stratified: bool = bool(params.pop("stratified", True))
    oof_params: dict[str, dict[str, Any]] = dict(params.pop("oof_params", {}))
    # Discard ModelBuilder-injected params
    for key in ("task_type", "objective", "num_classes", "input_size", "backbone"):
        params.pop(key, None)
    return StackingWrapper(
        models,
        meta_model=meta_model,
        n_folds=n_folds,
        stratified=stratified,
        oof_params=oof_params,
        **params,
    )
