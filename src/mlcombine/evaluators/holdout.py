"""HoldoutArchitectureValidator — train/val split for architecture validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from mlcombine.core.evaluator import BaseArchitectureValidator
from mlcombine.core.metric import DEFAULT_METRICS
from mlcombine.core.registry import registry
from mlcombine.core.types import MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.architecture_validator("holdout")
class HoldoutArchitectureValidator(BaseArchitectureValidator):
    """Validator that splits ``train_df`` into train/val, builds a fresh model
    from blueprint, fits on train, and computes metrics on val.

    Parameters (via ``params``):
        val_fraction (float): Fraction of data to hold out (default 0.2).
        stratified (bool): Use stratified split for classification (default True).
        random_state (int): Random seed for reproducibility (default 42).
    """

    def __init__(self, cfg: MLCombineConfig | None = None, **params: Any) -> None:
        super().__init__(cfg, **params)
        self._val_fraction = float(params.get("val_fraction", 0.2))
        self._stratified = bool(params.get("stratified", True))
        self._random_state = int(params.get("random_state", 42))

    def validate(
        self,
        blueprint: Any,
        context: PipelineContext,
        **kwargs: Any,
    ) -> dict[str, float]:
        if context is None or context.data.train_df is None:
            logger.warning("HoldoutArchitectureValidator: no train_df — skipping")
            return {}

        df = context.data.train_df
        target_col = context.data.target_col
        if not isinstance(target_col, str) or target_col not in df.columns:
            logger.warning("HoldoutArchitectureValidator: target_col not found — skipping")
            return {}

        x = df.drop(columns=[target_col])
        y = df[target_col]

        if self._val_fraction <= 0 or self._val_fraction >= 1:
            logger.warning("HoldoutArchitectureValidator: val_fraction must be in (0, 1) — got %s", self._val_fraction)
            return {}

        task_type = context.data.task_type
        task_str = str(task_type) if task_type else "regression"
        is_classification = "classification" in task_str.lower()

        # — Split
        if is_classification and self._stratified:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self._val_fraction,
                random_state=self._random_state,
            )
            train_idx, val_idx = next(splitter.split(x, y))
            x_train = x.iloc[train_idx]
            x_val = x.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=self._val_fraction,
                random_state=self._random_state,
            )

        # — Build model from blueprint
        try:
            model = blueprint.build()
        except Exception as e:
            logger.error("HoldoutArchitectureValidator: failed to build model — %s", e)
            return {}

        # — Fit on train subset
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            logger.error("HoldoutArchitectureValidator: model fit failed — %s", e)
            return {}

        # — Predict
        try:
            y_pred = model.predict(x_val)
        except Exception as e:
            logger.error("HoldoutArchitectureValidator: predict failed — %s", e)
            return {}

        # — Metrics
        metrics_list: list[str] | None = None
        if not metrics_list:
            task_key = "regression"
            if task_type is not None:
                raw = str(task_type).lower()
                if "classification" in raw or "multitask" in raw:
                    task_key = raw
            metrics_list = DEFAULT_METRICS.get(task_key, ["rmse"])

        y_true_arr = np.asarray(y_val).ravel()
        y_pred_arr = y_pred.ravel() if y_pred.ndim > 1 else y_pred

        results: dict[str, float] = {}
        for metric_name in metrics_list:
            entry = registry.metric.get(metric_name)
            if entry is None:
                logger.warning("Unknown metric: %s — skipping", metric_name)
                continue
            fn, kwargs_metric = entry

            try:
                results[metric_name] = float(fn(y_true_arr, y_pred_arr, **kwargs_metric))
            except Exception as e:
                logger.warning("Metric %s failed: %s", metric_name, e)

        return results


__all__ = ["HoldoutArchitectureValidator"]
