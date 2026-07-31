"""scikit-learn provider — RandomForest backed, uses explicit TaskType."""

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
from mlcombine.core.types import ConfigurationError

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR


class SklearnWrapper:
    """Wrapper making scikit-learn models conform to MLModelProtocol."""

    def __init__(self, model: RandomForestClassifier | RandomForestRegressor) -> None:
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
        logger.info("Fitting sklearn %s on %d samples", type(self._model).__name__, len(x))
        self._model.fit(x, y, **kwargs)
        logger.info("sklearn fitted")
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


@registry.model_provider("sklearn")
def sklearn_provider(
    backbone: str = "random_forest",
    task_type: TaskType = TaskType.REGRESSION,
    objective: ModelObjective = ModelObjective.RMSE,
    num_classes: int | None = None,
    input_size: int | None = None,
    **params: Any,
) -> SupportedModel:
    """Create a scikit-learn model dispatching on *backbone*.

    Checks ``registry`` for custom backbones first, then falls back to
    the built-in ``match/case`` dispatch.
    """
    try:
        is_classif = task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK)

        # Custom backbones from registry — callable receives (is_classif, **kwargs)
        custom_model_fn = registry.get_backbone("sklearn", backbone)
        if custom_model_fn is not None:
            model = custom_model_fn(is_classif, **params)
            logger.info("Created sklearn: backbone=%s (custom)", backbone)
            return SklearnWrapper(model)

        match backbone:
            case "gradient_boosting":
                if is_classif:
                    model = GradientBoostingClassifier(random_state=42, **params)
                else:
                    model = GradientBoostingRegressor(random_state=42, **params)
            case "svm":
                if is_classif:
                    model = SVC(random_state=42, **params)
                else:
                    model = SVR(**params)
            case "logistic_regression":
                if is_classif:
                    model = LogisticRegression(random_state=42, **params)
                else:
                    raise ConfigurationError(
                        "LogisticRegression requires classification task_type. "
                        "Use task_type=classification or choose a different backbone "
                        "(e.g. random_forest, gradient_boosting)."
                    )
            case "mlp":
                if is_classif:
                    model = MLPClassifier(random_state=42, **params)
                else:
                    model = MLPRegressor(random_state=42, **params)
            case _:
                if is_classif:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, **params)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, **params)
        logger.info("Created sklearn: backbone=%s, %s", backbone, "classifier" if is_classif else "regressor")
        return SklearnWrapper(model)
    except ImportError:
        logger.error("scikit-learn is not installed. Install with: uv add sklearn")
        raise


# Register built-in sklearn backbones so users can see/override them via registry
@registry.backbone("sklearn", "random_forest")
def _rf(is_classif: bool, **kwargs: Any) -> Any:
    rf_kwargs: dict[str, Any] = {"n_estimators": 100, "random_state": 42}
    rf_kwargs.update(kwargs)
    if is_classif:
        return RandomForestClassifier(**rf_kwargs)
    return RandomForestRegressor(**rf_kwargs)
