"""Optuna-based hyperparameter tuner meta-provider.

Usage in YAML::

    model:
      - provider: "tuner"
        params:
          n_trials: 50
          target_provider: "catboost"
          target_params:
            iterations: 3000
            depth: 8
          search_space:
            learning_rate: { type: "float", low: 0.01, high: 0.3, log: true }
          evaluator: "cv"
          tune_metric: "f1"
          evaluator_params:
            n_folds: 3
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Self

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mlcombine.core.enums import TaskType
from mlcombine.core.protocols import SupportedModel
from mlcombine.core.registry import registry
from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.tensor import UnifiedTensor
from mlcombine.core.types import PipelineContext, PipelineData

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

# Metrics where a lower value is better (optuna study direction "minimize").
# Everything else is maximized (accuracy, f1, auc, ...).
_MINIMIZE_METRICS = frozenset({"rmse", "mse", "mae", "mape", "logloss"})

# Module-level cache: maps config hash → best params discovered.
# Shared across all TunerWrapper instances so that OOF folds 2+ skip optuna.
_TUNER_BEST_PARAMS: dict[str, dict[str, Any]] = {}


class TunerWrapper:
    """Optuna-based tuner wrapping a target model provider.

    Uses a registered architecture validator (``"cv"``, ``"holdout"``, etc.)
    for trial evaluation via blueprint → validate().
    """

    def __init__(
        self,
        target_provider: str,
        target_params: dict[str, Any],
        search_space: dict[str, Any],
        *,
        n_trials: int = 50,
        task_type: str = "regression",
        num_classes: int | None = None,
        input_size: int | None = None,
        evaluator: str = "cv",
        tune_metric: str | None = None,
        evaluator_params: dict[str, Any] | None = None,
        cache_best_params: bool = True,
    ) -> None:
        self._target_provider = target_provider
        self._target_params = dict(target_params)
        self._search_space = dict(search_space)
        self._n_trials = n_trials
        self._task_type = task_type
        self._num_classes = num_classes
        self._input_size = input_size
        self._validator_name = evaluator
        self._tune_metric = tune_metric
        self._validator_params = dict(evaluator_params or {})
        self._cache_enabled = cache_best_params

        self._best_params: dict[str, Any] | None = None
        self._model: SupportedModel | None = None
        self.is_fitted: bool = False

    def _compute_cache_key(self) -> str:
        """Hash of (target_provider, target_params, search_space) for cross-instance cache."""
        data = {
            "target_provider": self._target_provider,
            "target_params": dict(sorted(self._target_params.items())),
            "search_space": dict(sorted(self._search_space.items())),
        }
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _build_and_fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        **kwargs: Any,
    ) -> None:
        assert self._best_params is not None
        self._model = self._build_model(self._best_params)
        self._model.fit(x, y, **kwargs)
        self.is_fitted = True

    def _build_model(self, extra_params: dict[str, Any] | None = None) -> SupportedModel:
        """Instantiate the target model with merged params."""
        params = dict(self._target_params)
        if extra_params:
            params.update(extra_params)
        provider_fn = registry.get_model_provider(self._target_provider)
        if provider_fn is None:
            raise ValueError(f"Target provider {self._target_provider!r} not found. Available: {list(registry.model_providers)}")
        kwargs: dict[str, Any] = dict(params)
        kwargs["task_type"] = self._task_type
        if self._num_classes is not None:
            kwargs["num_classes"] = self._num_classes
        if self._input_size is not None:
            kwargs["input_size"] = self._input_size
        return provider_fn(**kwargs)

    def _default_tune_metric(self) -> str:
        if self._task_type in ("classification", "multitask"):
            return "f1"
        return "rmse"

    def _study_direction(self) -> str:
        metric = self._tune_metric or self._default_tune_metric()
        return "minimize" if metric in _MINIMIZE_METRICS else "maximize"

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, object]:
        """Convert search_space config to optuna trial suggestions."""
        suggested: dict[str, Any] = {}
        for name, cfg in self._search_space.items():
            typ = cfg.get("type", "float")
            if typ == "int":
                suggested[name] = trial.suggest_int(
                    name,
                    cfg["low"],
                    cfg["high"],
                    step=cfg.get("step", 1),
                    log=cfg.get("log", False),
                )
            elif typ == "float":
                suggested[name] = trial.suggest_float(
                    name,
                    cfg["low"],
                    cfg["high"],
                    log=cfg.get("log", False),
                )
            elif typ == "categorical":
                suggested[name] = trial.suggest_categorical(name, cfg["choices"])
        return suggested

    def _trial_callback(self, study: optuna.Study, trial: Any) -> None:
        n = trial.number
        if n % 10 == 0 or n == self._n_trials - 1:
            logger.info(
                "Trial %d/%d: %s=%.4f  params=%s",
                n + 1,
                self._n_trials,
                self._tune_metric or self._default_tune_metric(),
                trial.value if trial.value is not None else float("nan"),
                trial.params,
            )

    def _objective(
        self,
        trial: optuna.Trial,
        context: PipelineContext,
        **kwargs: object,
    ) -> float:
        params = self._suggest_params(trial)

        blueprint = ModelBlueprint(
            self._target_provider,
            {**self._target_params, **params},
            task_type=self._task_type,
            num_classes=self._num_classes,
            input_size=self._input_size,
        )

        validator_cls = registry.get_architecture_validator(self._validator_name)
        if validator_cls is None:
            raise ValueError(f"Architecture validator {self._validator_name!r} not found. Available: {list(registry.architecture_validator_names)}")
        validator = validator_cls(cfg=None, **self._validator_params)

        results = validator.validate(blueprint, context)

        # cleanup GPU memory for next trial
        del validator, blueprint
        gc.collect()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        metric = self._tune_metric or self._default_tune_metric()
        score = results.get(metric)
        if score is None:
            raise ValueError(f"Validator {self._validator_name!r} did not return metric {metric!r}. Available: {list(results)}")
        return float(score)

    def fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        **kwargs: Any,
    ) -> Self:
        cache_key = self._compute_cache_key() if self._cache_enabled else ""

        # ── Check global cache (avoids re-running optuna per OOF fold) ──
        if self._cache_enabled and cache_key in _TUNER_BEST_PARAMS:
            self._best_params = dict(_TUNER_BEST_PARAMS[cache_key])
            self._build_and_fit(x, y, **kwargs)
            logger.info(
                "Tuner: reused cached best params — optuna skipped: %s",
                self._best_params,
            )
            return self

        import optuna

        # Build a combined DataFrame for the evaluator
        if isinstance(x, pd.DataFrame):
            x_df = x
        else:
            x_df = pd.DataFrame(np.asarray(x))
        y_arr = np.asarray(y).ravel() if not isinstance(y, np.ndarray) else np.asarray(y).ravel()
        target_col = "__tuner_target__"
        train_df = x_df.copy()
        train_df[target_col] = y_arr

        context = PipelineContext(
            data=PipelineData(
                train_df=train_df,
                target_col=target_col,
                task_type=TaskType(self._task_type),
            )
        )

        study = optuna.create_study(direction=self._study_direction())
        interrupted = False

        try:
            study.optimize(
                lambda trial: self._objective(trial, context, **kwargs),
                n_trials=self._n_trials,
                callbacks=[self._trial_callback],
            )
        except KeyboardInterrupt:
            study.stop()
            interrupted = True
            logger.warning(
                "Optuna interrupted by user after %d trial(s). Continuing with best params found so far.",
                len(study.trials),
            )

        self._finalize(study, x, y, interrupted, **kwargs)

        # Store in cache for sibling instances (e.g. other OOF folds)
        if self._cache_enabled and self._best_params is not None:
            _TUNER_BEST_PARAMS[cache_key] = dict(self._best_params)
            logger.info("Tuner: cached best params for future reuse")
        return self

    def _finalize(
        self,
        study: optuna.Study,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        interrupted: bool,
        **kwargs: Any,
    ) -> None:
        """Build best model and fit on full data."""
        if study.best_params:
            self._best_params = dict(study.best_params)
            logger.info(
                "Optuna %s: best score=%.4f  params=%s",
                "interrupted" if interrupted else "finished",
                study.best_value,
                self._best_params,
            )
            self._build_and_fit(x, y, **kwargs)
        else:
            raise RuntimeError("Optuna was interrupted before any trial completed. No model to use.")

    def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[Any]:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Tuner model must be fitted before prediction")
        return self._model.predict(x)

    def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Tuner model must be fitted before prediction")
        return self._model.predict_proba(x)


@registry.model_provider("tuner", package="optuna", module="mlcombine.models.meta.tuner")
def tuner_provider(**params: Any) -> SupportedModel:
    """Create a TunerWrapper wrapping a target provider with optuna search.

    Required ``**params`` keys:
        target_provider — name of the model provider to tune.
        target_params — fixed hyperparameters for the target model.
        search_space — optuna search space definition.
        n_trials — number of trials (default 50).

    Validation params:
        evaluator — architecture validator name (``"cv"``, ``"holdout"``, etc.).
        tune_metric — metric to optimize (e.g. ``"f1"``, ``"rmse"``).
        evaluator_params — dict passed to validator constructor.

    Also accepts ``task_type``, ``num_classes``, ``input_size``.
    """
    target_provider: str = params.pop("target_provider")
    target_params: dict[str, Any] = params.pop("target_params")
    search_space: dict[str, Any] = params.pop("search_space")
    n_trials: int = int(params.pop("n_trials", 50))
    task_type: str = params.pop("task_type", "regression")
    num_classes: int | None = params.pop("num_classes", None)
    input_size: int | None = params.pop("input_size", None)
    evaluator: str = params.pop("evaluator", "cv")
    tune_metric: str | None = params.pop("tune_metric", None)
    evaluator_params: dict[str, Any] | None = params.pop("evaluator_params", None)
    cache_best_params: bool = bool(params.pop("cache_best_params", True))

    return TunerWrapper(
        target_provider,
        target_params,
        search_space,
        n_trials=n_trials,
        task_type=task_type,
        num_classes=num_classes,
        input_size=input_size,
        evaluator=evaluator,
        tune_metric=tune_metric,
        evaluator_params=evaluator_params,
        cache_best_params=cache_best_params,
    )
