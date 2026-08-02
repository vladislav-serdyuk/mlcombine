"""EvaluateStep — computes metrics on holdout (unbiased) or train_df (biased)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from mlcombine.core.metric import DEFAULT_METRICS
from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("EvaluateStep")
class EvaluateStep(BaseStep[PipelineContext]):
    """Compute evaluation metrics on holdout_df or train_df.

    Resolution order:
    1. **holdout_df** — unbiased evaluation on data held out by ``SplitStep``.
    2. **train_df** — biased in-sample fallback (logs a warning).
    3. No data — silently skipped.

    Metrics are configured via ``cfg.step_config.evaluate.metrics``,
    or default to ``DEFAULT_METRICS`` per task type.
    Custom metrics can be added via ``@registry.metric(name, **kwargs)``.
    """

    train = True
    predict = False

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        ecfg = getattr(cfg.step_config, "evaluate", None) or {}
        self._metrics: list[str] | None = ecfg.get("metrics", None)
        self._target_col = cfg.data.target_col
        self._treatment_col = cfg.data.treatment_col
        self._id_col = cfg.data.id_col
        self._drop_columns: list[str] = cfg.data.drop_columns

    def run(self, context: PipelineContext) -> PipelineContext:
        """Compute metrics and store them on the context."""
        model = context.artifacts.model

        # — Primary: holdout_df from SplitStep (unbiased)
        if context.data.holdout_df is not None and model is not None:
            logger.info("EvaluateStep: using holdout_df evaluation")
            return self._run_split(context, context.data.holdout_df)

        # — Fallback: train_df (biased — model saw these targets)
        if context.data.train_df is not None and model is not None:
            logger.warning("EvaluateStep: no holdout_df — using train_df (biased metrics)")
            return self._run_split(context, context.data.train_df)

        logger.warning("EvaluateStep: no data available — skipping")
        return context

    def _run_split(self, context: PipelineContext, df: pd.DataFrame) -> PipelineContext:
        target = self._target_col
        if not isinstance(target, str) or target not in df.columns:
            return context

        drop_cols = [target]
        if self._treatment_col and self._treatment_col in df.columns:
            drop_cols.append(self._treatment_col)
        if self._drop_columns:
            drop_cols.extend(c for c in self._drop_columns if c in df.columns)
        if self._id_col and self._id_col in df.columns:
            drop_cols.append(self._id_col)
        y_true = df[target]
        x = df.drop(columns=drop_cols)
        model = context.artifacts.model
        if model is None:
            return context

        y_pred = model.predict(x)
        results = self._compute_metrics(y_true, y_pred, context)
        context.artifacts.evaluation_results = results
        self._log_results(results)
        return context

    def _compute_metrics(
        self,
        y_true: pd.Series | pd.DataFrame,
        y_pred: np.ndarray,
        context: PipelineContext,
    ) -> dict[str, float]:
        task_type = context.data.task_type
        task_key = "regression"
        if task_type is not None:
            raw = str(task_type).lower()
            if "classification" in raw or "multitask" in raw:
                task_key = raw

        metrics = self._metrics or DEFAULT_METRICS.get(task_key, ["rmse"])

        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = y_pred.ravel() if y_pred.ndim > 1 else y_pred

        # Align string y_true → int when model returns integer class indices
        if y_true_arr.dtype.kind in ("U", "S", "O") and y_pred_arr.dtype.kind in ("i", "u"):
            le = LabelEncoder()
            y_true_arr = le.fit_transform(y_true_arr.astype(str))

        x: pd.DataFrame | None = None
        model = context.artifacts.model
        if model is not None and context.data.train_df is not None:
            target = self._target_col
            if isinstance(target, str) and target in context.data.train_df.columns:
                drop_cols = [target]
                if self._treatment_col and self._treatment_col in context.data.train_df.columns:
                    drop_cols.append(self._treatment_col)
                if self._id_col and self._id_col in context.data.train_df.columns:
                    drop_cols.append(self._id_col)
                x = context.data.train_df.drop(columns=drop_cols)

        results: dict[str, float] = {}
        for metric_name in metrics:
            entry = registry.metric.get(metric_name)
            if entry is None:
                logger.warning("Unknown metric: %s — skipping", metric_name)
                continue
            fn, kwargs = entry

            # logloss / auc need predict_proba
            if metric_name in ("logloss", "auc") and x is not None and model is not None and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(x)
                    if metric_name == "logloss":
                        results[metric_name] = float(fn(y_true_arr, proba, **kwargs))
                    else:
                        results[metric_name] = float(roc_auc_score(y_true_arr, proba, multi_class="ovo", average="weighted"))
                except Exception as e:
                    logger.warning("Metric %s failed: %s", metric_name, e)
                continue

            try:
                results[metric_name] = float(fn(y_true_arr, y_pred_arr, **kwargs))
            except Exception as e:
                logger.warning("Metric %s failed: %s", metric_name, e)

        return results

    @staticmethod
    def _log_results(results: dict[str, float]) -> None:
        for name, value in results.items():
            logger.info("  %s: %.4f", name, value)


__all__ = [
    "EvaluateStep",
]
