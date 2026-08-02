"""CatBoost provider — uses explicit TaskType instead of guessing from metric name."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mlcombine.core.tensor import UnifiedTensor
from mlcombine.core.enums import ModelObjective, TaskType
from mlcombine.core.registry import registry
from mlcombine.core.protocols import SupportedModel

if TYPE_CHECKING:
    from catboost import CatBoostClassifier, CatBoostRegressor

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


class CatBoostLoggingCallback:
    """Routes CatBoost per-iteration metrics to Python logging."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self.logger = logger
        self.level = level

    def after_iteration(self, info: Any) -> bool:
        iteration = getattr(info, "iteration", None)
        learn = getattr(info, "learn", {})
        test = getattr(info, "test", {})
        elapsed = getattr(info, "elapsed_time", 0)

        parts = [f"iter {iteration}"]
        if learn:
            parts.append(f"learn: {learn}")
        if test:
            parts.append(f"val: {test}")
        if elapsed:
            parts.append(f"{elapsed:.1f}s")

        self.logger.log(self.level, " | ".join(parts))
        return True


class CatBoostWrapper:
    """Wrapper making CatBoost models conform to MLModelProtocol."""

    def __init__(self, model: CatBoostClassifier | CatBoostRegressor) -> None:
        self._model = model
        self._text_features: list[str] = []

    @staticmethod
    def _is_text_column(series: pd.Series) -> bool:
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        lengths = sample.astype(str).str.len()
        return lengths.mean() > 50

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
        if isinstance(x, pd.DataFrame) and "cat_features" not in kwargs:
            self._text_features = list(self._model.get_params().get("text_features") or [])
            if not self._text_features:
                self._text_features = [col for col in x.select_dtypes(include=["object", "string"]).columns if self._is_text_column(x[col])]
            cat_cols = x.select_dtypes(include=["object", "string"]).columns.tolist()
            cat_cols = [c for c in cat_cols if c not in self._text_features]
            if self._text_features:
                kwargs["text_features"] = self._text_features
            if cat_cols:
                kwargs["cat_features"] = cat_cols
        is_gpu = self._model.get_params().get("task_type") == "GPU"
        if not is_gpu and "callbacks" not in kwargs:
            _cb_logger = logging.getLogger("catboost")
            kwargs["callbacks"] = [CatBoostLoggingCallback(_cb_logger)]
        elif is_gpu:
            kwargs.pop("callbacks", None)
        n_rows = len(x)
        if is_gpu and isinstance(x, pd.DataFrame) and self._text_features:
            text_cols = [c for c in self._text_features if c in x.columns]
            if text_cols:
                x = x.copy()
                x[text_cols] = x[text_cols].fillna("").astype(str)
        logger.info("Fitting CatBoost on %d samples", n_rows)
        self._model.fit(x, y, **kwargs)
        logger.info("CatBoost fitted")
        return self

    def _cast_text_features_as_str(self, x: pd.DataFrame) -> pd.DataFrame:
        """Cast CatBoost text-feature columns + remaining object/string to str."""
        cols_to_cast: set[str] = set()
        try:
            feature_names = self._model.feature_names_
            text_indices = self._model.get_text_feature_indices()
            if text_indices and feature_names:
                cols_to_cast.update(feature_names[i] for i in text_indices if i < len(feature_names) and feature_names[i] in x.columns)
            elif self._text_features:
                cols_to_cast.update(col for col in self._text_features if col in x.columns)
        except Exception:
            pass
        for col in x.select_dtypes(include=["object", "string"]).columns:
            if col not in cols_to_cast:
                cols_to_cast.add(col)
        if not cols_to_cast:
            return x
        x = x.copy()
        for col in cols_to_cast:
            x[col] = x[col].fillna("").astype(str)
        return x

    def _ensure_text_as_str(self, x: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_names = self._model.feature_names_
            text_indices = self._model.get_text_feature_indices()
        except Exception:
            return self._cast_text_features_as_str(x)
        if not feature_names:
            return x
        if not text_indices:
            if self._text_features:
                cols = [c for c in self._text_features if c in x.columns]
                if cols:
                    x = x.copy()
                    x[cols] = x[cols].fillna("").astype(str)
            return x
        cols = [feature_names[i] for i in text_indices if i < len(feature_names) and feature_names[i] in x.columns]
        if not cols:
            return x
        x = x.copy()
        x[cols] = x[cols].fillna("").astype(str)
        return x

    def _align_columns(self, x: pd.DataFrame) -> pd.DataFrame:
        try:
            fn = self._model.feature_names_
            if fn is not None:
                matched = [c for c in fn if c in x.columns]
                if matched and len(matched) == len(fn):
                    x = x[matched]
                elif matched:
                    missing = set(fn) - set(x.columns)
                    logger.warning(
                        "CatBoost predict: %d features missing (%s) — using %d available",
                        len(missing),
                        ",".join(sorted(missing))[:200],
                        len(matched),
                    )
                    x = x[matched]
        except Exception:
            pass
        return x

    def _predict_with_text_fallback(self, x: pd.DataFrame, method: str = "predict") -> np.ndarray:
        """Call predict/predict_proba; fall back to blanket str cast on error."""
        predict_fn = getattr(self._model, method)
        try:
            return predict_fn(x)  # type: ignore[no-any-return]
        except Exception as e:
            err_str = str(e)
            if "text_features must have string type" not in err_str and "Bad value for" not in err_str:
                raise
            logger.warning(
                "CatBoost dtype error in %s: %s — casting text feature cols to str and retrying",
                method,
                err_str,
            )
            x = self._cast_text_features_as_str(x)
            return predict_fn(x)  # type: ignore[no-any-return]

    def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[Any]:
        if isinstance(x, UnifiedTensor):
            x = x.numpy()
        if isinstance(x, pd.DataFrame):
            x = self._align_columns(x)
            x = self._ensure_text_as_str(x)
            return self._predict_with_text_fallback(x, "predict")
        return self._model.predict(x)  # type: ignore[no-any-return]

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        if isinstance(x, UnifiedTensor):
            x = x.numpy()
        if not hasattr(self._model, "predict_proba"):
            raise RuntimeError(f"{type(self._model).__name__} does not support predict_proba (use classification mode)")
        if isinstance(x, pd.DataFrame):
            x = self._align_columns(x)
            x = self._ensure_text_as_str(x)
            return self._predict_with_text_fallback(x, "predict_proba")
        return self._model.predict_proba(x)  # type: ignore[no-any-return]


logger = logging.getLogger(__name__)


_REGRESSION_LOSSES = frozenset({"RMSE", "MAE", "MAPE", "HUBER", "QUANTILE"})
_CLASSIFICATION_LOSSES = frozenset({"LOGLOSS", "CROSSENTROPY", "MULTICLASS"})
_LOSS_ALIASES = {
    "RMSE": "RMSE",
    "MAE": "MAE",
    "MAPE": "MAPE",
    "HUBER": "Huber",
    "QUANTILE": "Quantile",
    "LOGLOSS": "Logloss",
    "CROSSENTROPY": "CrossEntropy",
    "MULTICLASS": "MultiClass",
}


@registry.model_provider("catboost", package="catboost", module="mlcombine.models.providers.catboost")
def catboost_provider(
    backbone: str = "gradient_boosting",
    task_type: TaskType = TaskType.REGRESSION,
    objective: ModelObjective | str = ModelObjective.RMSE,
    num_classes: int | None = None,
    input_size: int | None = None,
    **params: Any,
) -> SupportedModel:
    """Create a CatBoost classifier or regressor based on task type.

    Set ``params["gpu"] = True`` to enable GPU training
    (adds ``task_type="GPU"`` to the CatBoost constructor).

    By default ``verbose=False`` and a logging callback is added to route
    per-iteration metrics through Python logging. Override with
    ``params={"verbose": True, "callbacks": [...]}`` for native output.

    ``objective`` may also be a raw string (e.g. ``"MAE"``) — it is then
    used as ``loss_function`` and (unless set explicitly in params)
    ``eval_metric``. Supported: RMSE, MAE, MAPE, HUBER, QUANTILE for
    regression; Logloss, CrossEntropy, MultiClass for classification.
    """
    if not _CATBOOST_AVAILABLE:
        logger.error("CatBoost is not installed. Install with: uv add catboost")
        raise ImportError("CatBoost is required for catboost provider")
    try:
        use_gpu: bool = bool(params.pop("gpu", False))
        if use_gpu:
            params["task_type"] = "GPU"

        # Disable native verbose output; our callback handles logging
        params.setdefault("verbose", False)

        objective_str: str | None = None
        if type(objective) is str:  # raw string (not StrEnum) — e.g. "MAE"
            objective_str = objective.upper()
        if objective_str is not None and objective_str not in _REGRESSION_LOSSES | _CLASSIFICATION_LOSSES:
            logger.warning("Unsupported objective %r — ignoring", objective_str)
            objective_str = None

        if task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK):
            n_classes = num_classes or 2
            is_multi = n_classes > 2
            if objective_str == "MULTICLASS" and is_multi:
                loss_fn = "MultiClass"
            elif objective_str in {"LOGLOSS", "CROSSENTROPY"} and not is_multi:
                loss_fn = _LOSS_ALIASES[objective_str]
            else:
                loss_fn = "MultiClass" if is_multi else "Logloss"
            if "eval_metric" not in params:
                params["eval_metric"] = "F1" if objective == ModelObjective.F1 else "Accuracy"
            model = CatBoostClassifier(
                loss_function=loss_fn,
                thread_count=-1,
                **params,
            )
            logger.info("Created CatBoost: task_type=%s, loss=%s, gpu=%s", task_type, loss_fn, use_gpu)
            return CatBoostWrapper(model)
        loss_fn = _LOSS_ALIASES[objective_str] if objective_str in _REGRESSION_LOSSES else "RMSE"
        if "eval_metric" not in params:
            params["eval_metric"] = loss_fn if objective_str in _REGRESSION_LOSSES else ("MAPE" if objective == ModelObjective.MAPE else "RMSE")
        model = CatBoostRegressor(
            loss_function=loss_fn,
            thread_count=-1,
            **params,
        )
        logger.info("Created CatBoost: task_type=%s, loss=%s, gpu=%s", task_type, loss_fn, use_gpu)
        return CatBoostWrapper(model)
    except ImportError:
        logger.error("CatBoost is not installed. Install with: uv add catboost")
        raise
