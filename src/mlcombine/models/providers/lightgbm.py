"""LightGBM provider — uses explicit TaskType instead of guessing from metric name."""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mlcombine.core.tensor import UnifiedTensor
from mlcombine.core.enums import ModelObjective, TaskType
from mlcombine.core.registry import registry
from mlcombine.core.protocols import SupportedModel

try:
    import lightgbm as lgb

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


if _LGBM_AVAILABLE:

    class LightGBMWrapper:
        """Wrapper making LightGBM models conform to MLModelProtocol."""

        def __init__(self, model: lgb.LGBMClassifier | lgb.LGBMRegressor) -> None:
            self._model = model

        def fit(
            self,
            x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
            y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
            **kwargs: Any,
        ) -> Self:
            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            if isinstance(y, UnifiedTensor):
                y = y.numpy()
            logger.info("Fitting LightGBM on %d samples", len(x))
            self._model.fit(x, y, **kwargs)
            logger.info("LightGBM fitted")
            return self

        def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[Any]:
            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            return self._model.predict(x)  # type: ignore[no-any-return]

        def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            if not hasattr(self._model, "predict_proba"):
                raise RuntimeError(f"{type(self._model).__name__} does not support predict_proba (use classification mode)")
            return self._model.predict_proba(x)  # type: ignore[no-any-return]


logger = logging.getLogger(__name__)


@registry.model_provider("lightgbm", package="lightgbm", module="mlcombine.models.providers.lightgbm")
def lightgbm_provider(
    backbone: str = "gradient_boosting",
    task_type: TaskType = TaskType.REGRESSION,
    objective: ModelObjective = ModelObjective.RMSE,
    num_classes: int | None = None,
    input_size: int | None = None,
    **params: Any,
) -> SupportedModel:
    """Create a LightGBM classifier or regressor based on task type."""
    if not _LGBM_AVAILABLE:
        logger.error("LightGBM is not installed. Install with: uv add lightgbm")
        raise ImportError("LightGBM is required for lightgbm provider")
    if task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK):
        n_classes = num_classes or 2
        p: dict[str, Any] = {
            "objective": "multiclass" if n_classes > 2 else "binary",
            "metric": "multi_logloss" if n_classes > 2 else ("f1" if objective == ModelObjective.F1 else "accuracy"),
            "num_class": n_classes if n_classes > 2 else None,
            "verbose": -1,
        }
        p.update(params)
        model = lgb.LGBMClassifier(**{k: v for k, v in p.items() if v is not None})
        logger.info("Created LightGBM: objective=%s, num_class=%s", p.get("objective"), p.get("num_class"))
        return LightGBMWrapper(model)
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mape" if objective == ModelObjective.MAPE else "rmse",
        verbose=-1,
        **params,
    )
    logger.info("Created LightGBM: objective=regression, metric=%s", model.metric)
    return LightGBMWrapper(model)
