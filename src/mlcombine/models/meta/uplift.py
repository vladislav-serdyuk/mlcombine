"""Uplift modeling — TLearner and SLearner using ModelBlueprint for safe duplication.

Each arm builds a fresh model instance from the blueprint to avoid sharing state.
"""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mlcombine.core.protocols import ProbabilityModelProtocol, SupportedModel
from mlcombine.core.registry import registry
from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.tensor import UnifiedTensor

logger = logging.getLogger(__name__)


def _to_array(x: pd.DataFrame | pd.Series | np.ndarray | UnifiedTensor[Any]) -> np.ndarray:
    """Coerce DataFrame/Series to NumPy array for sklearn internals."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return np.asarray(x)


class TLearner:
    """Two-model uplift strategy: one model for treatment, one for control.

    Each arm builds a fresh model from the blueprint.
    """

    def __init__(self, base_model: ModelBlueprint) -> None:
        """Initialize with a blueprint to clone for treatment and control."""
        self._blueprint = base_model
        self.model_treatment: ProbabilityModelProtocol | None = None
        self.model_control: ProbabilityModelProtocol | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        *,
        treatment: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit separate models on treatment and control splits.

        Side Effects:
            - Sets self.model_treatment, self.model_control, self.is_fitted.
        """
        if treatment is None:
            raise ValueError("treatment is required for TLearner")
        X_arr = _to_array(x)
        y_arr = _to_array(y).ravel()
        t_arr = np.asarray(treatment).astype(int)

        treatment_mask = t_arr == 1
        control_mask = t_arr == 0

        X_t, y_t = X_arr[treatment_mask], y_arr[treatment_mask]
        X_c, y_c = X_arr[control_mask], y_arr[control_mask]

        logger.info("Fitting TLearner: %d treatment, %d control samples", len(X_t), len(X_c))

        # Build fresh model instances from the blueprint for each arm
        self.model_treatment = self._blueprint.build()
        self.model_control = self._blueprint.build()

        if len(X_t) > 0:
            self.model_treatment.fit(X_t, y_t, **kwargs)
        if len(X_c) > 0:
            self.model_control.fit(X_c, y_c, **kwargs)

        self.is_fitted = True
        return self

    def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        """Difference P(Y|T=1) - P(Y|T=0) for uplift estimation."""
        X_arr = _to_array(x)
        if not self.is_fitted or self.model_treatment is None or self.model_control is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model_treatment.predict(X_arr) - self.model_control.predict(X_arr)

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        """Difference in class probabilities with/without treatment."""
        X_arr = _to_array(x)
        if not self.is_fitted or self.model_treatment is None or self.model_control is None:
            raise RuntimeError("Model must be fitted before prediction")
        proba_t = self.model_treatment.predict_proba(X_arr)
        proba_c = self.model_control.predict_proba(X_arr)
        return proba_t - proba_c


class SLearner:
    """Single-model uplift strategy: treatment indicator as a feature.

    Builds a fresh model from the blueprint.
    """

    def __init__(self, base_model: ModelBlueprint) -> None:
        """Initialize with a blueprint to build the model from."""
        self._blueprint = base_model
        self.model: ProbabilityModelProtocol | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        *,
        treatment: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit single model with treatment indicator as extra feature.

        Side Effects:
            - Sets self.model, self.is_fitted.
        """
        if treatment is None:
            raise ValueError("treatment is required for SLearner")
        X_arr = _to_array(x)
        y_arr = _to_array(y).ravel()
        t_arr = np.asarray(treatment).astype(int)
        if t_arr.ndim == 1:
            t_arr = t_arr.reshape(-1, 1)

        self.model = self._blueprint.build()
        self.model.fit(np.hstack([X_arr, t_arr]), y_arr, **kwargs)
        self.is_fitted = True
        logger.info("SLearner fitted: %s, %d samples", type(self.model).__name__, len(X_arr))
        return self

    def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        """ITE estimate: predict with treatment=1 minus treatment=0."""
        X_arr = _to_array(x)
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        pred_t = self.model.predict(np.hstack([X_arr, np.ones((len(X_arr), 1))]))
        pred_c = self.model.predict(np.hstack([X_arr, np.zeros((len(X_arr), 1))]))
        return pred_t - pred_c

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        """Difference in class probabilities with/without treatment."""
        X_arr = _to_array(x)
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        proba_t = self.model.predict_proba(np.hstack([X_arr, np.ones((len(X_arr), 1))]))
        proba_c = self.model.predict_proba(np.hstack([X_arr, np.zeros((len(X_arr), 1))]))
        return proba_t - proba_c


# ── Provider functions ─────────────────────────────────────────────────


@registry.model_provider("t_learner")
def t_learner_provider(model: ModelBlueprint, **params: Any) -> SupportedModel:
    """Wrap a base model blueprint in TLearner for uplift."""
    return TLearner(model)


@registry.model_provider("s_learner")
def s_learner_provider(model: ModelBlueprint, **params: Any) -> SupportedModel:
    """Wrap a base model blueprint in SLearner for uplift."""
    return SLearner(model)
