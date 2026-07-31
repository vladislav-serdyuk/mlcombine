"""CVEvaluator — K-fold cross-validation evaluation.

Creates K folds, trains a fresh model on each fold, collects OOF predictions,
and computes metrics. Optionally supports OOF-safe target encoding and GroupKFold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from mlcombine.core.evaluator import BaseArchitectureValidator
from mlcombine.core.metric import DEFAULT_METRICS
from mlcombine.core.protocols import SupportedModel

from mlcombine.core.registry import registry
from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.types import MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


def _target_encode(
    X: pd.DataFrame,
    y_arr: np.ndarray,
    cols: list[str],
    *,
    train_idx: npt.NDArray[Any],
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """OOF-safe target encoding: compute target mean on *train_idx*, encode all."""
    X = X.copy()
    global_mean = float(np.mean(y_arr[train_idx]))
    for col in cols:
        train_df = X.iloc[train_idx]
        train_target = y_arr[train_idx]
        means = train_df[[col]].copy()
        means["target"] = train_target
        stats = means.groupby(col)["target"].agg(["count", "mean"]).to_dict("index")

        def _encode_value(row: pd.Series) -> float:
            cat = row[col]
            if cat in stats:
                cnt: float = stats[cat]["count"]
                mean_val: float = stats[cat]["mean"]
                return (mean_val * cnt + global_mean * smoothing) / (cnt + smoothing)
            return global_mean

        X[col] = X.apply(_encode_value, axis=1)
    return X


@registry.architecture_validator("cv")
class CVEvaluator(BaseArchitectureValidator):
    """K-fold cross-validation evaluator.

    Builds a fresh model, creates K folds, trains one model per fold,
    collects OOF predictions, and computes metrics.

    Parameters (via ``params``):
        n_folds (int): Number of folds (default 5).
        stratified (bool): Stratified folds for classification (default True).
        shuffle (bool): Shuffle before splitting (default True).
        random_state (int): Random seed (default 42).
        target_encode_cols (list[str]): Columns for OOF-safe target encoding.
        target_encode_smoothing (float): Smoothing factor (default 10.0).
        group_col (str): Column name for GroupKFold (default None).
    """

    def __init__(self, cfg: MLCombineConfig | None = None, **params: Any) -> None:
        super().__init__(cfg, **params)
        self._n_folds = int(params.get("n_folds", 5))
        self._stratified = bool(params.get("stratified", True))
        self._shuffle = bool(params.get("shuffle", True))
        self._random_state = int(params.get("random_state", 42))
        self._target_encode_cols: list[str] = list(params.get("target_encode_cols", []))
        self._target_encode_smoothing = float(params.get("target_encode_smoothing", 10.0))
        self._group_col: str | None = params.get("group_col", None)
        self._metrics: list[str] | None = params.get("metrics", None)

    def validate(self, blueprint: ModelBlueprint, context: PipelineContext, **kwargs: Any) -> dict[str, float]:
        if context is None or context.data.train_df is None:
            logger.warning("CVEvaluator: no train_df — skipping")
            return {}

        df = context.data.train_df
        target_col = context.data.target_col
        if not isinstance(target_col, str) or target_col not in df.columns:
            logger.warning("CVEvaluator: target_col must be a column in train_df — skipping")
            return {}

        x = df.drop(columns=[target_col])
        y = df[target_col]
        y_arr = np.asarray(y).ravel()

        task_type = context.data.task_type
        task_str = str(task_type) if task_type else "regression"
        is_classification = "classification" in task_str.lower()

        # — Splitter (use kwargs for fold-specific params)
        groups = None
        if self._group_col and self._group_col in x.columns:
            groups = x[self._group_col].to_numpy()
            x = x.drop(columns=[self._group_col])
            splitter: KFold | StratifiedKFold | GroupKFold = GroupKFold(n_splits=self._n_folds)
            split_kwargs: dict[str, Any] = {"groups": groups}
        elif is_classification and self._stratified:
            splitter = StratifiedKFold(
                n_splits=self._n_folds,
                shuffle=self._shuffle,
                random_state=self._random_state,
            )
            split_kwargs = {}
        else:
            splitter = KFold(
                n_splits=self._n_folds,
                shuffle=self._shuffle,
                random_state=self._random_state,
            )
            split_kwargs = {}

        # — Label encoding for string targets
        label_encoder: LabelEncoder | None = None
        if y_arr.dtype.kind in ("U", "S", "O"):
            le = LabelEncoder()
            y_arr = le.fit_transform(y_arr.astype(str))
            label_encoder = le

        # — K-fold training
        oof_preds = np.full(len(x), np.nan)
        fold_models: list[SupportedModel] = []
        fold_val_indices: list[npt.NDArray[Any]] = []

        for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y_arr, **split_kwargs)):
            X_fold = x.iloc[train_idx].copy()
            X_val = x.iloc[val_idx].copy()
            y_train = y_arr[train_idx]

            if self._target_encode_cols:
                X_all = pd.concat([X_fold, X_val], axis=0)
                X_enc = _target_encode(
                    X_all,
                    y_arr,
                    self._target_encode_cols,
                    train_idx=np.arange(len(X_fold)),
                    smoothing=self._target_encode_smoothing,
                )
                X_fold = X_enc.iloc[: len(X_fold)]
                X_val = X_enc.iloc[len(X_fold) :]

            # Build fresh model for this fold
            fold_model = blueprint.build()
            try:
                fold_model.fit(X_fold, y_train)
            except Exception as e:
                logger.warning("CVEvaluator: fold %d fit failed — %s", fold + 1, e)
                continue

            try:
                preds = np.asarray(fold_model.predict(X_val)).ravel()
                oof_preds[val_idx] = preds
            except Exception as e:
                logger.warning("CVEvaluator: fold %d predict failed — %s", fold + 1, e)

            fold_models.append(fold_model)
            fold_val_indices.append(val_idx)

            logger.info("Fold %d/%d done — %d train / %d val", fold + 1, self._n_folds, len(X_fold), len(X_val))

        # — Store OOF predictions on context
        valid_mask = ~np.isnan(oof_preds)
        n_valid = valid_mask.sum()
        if n_valid == 0:
            raise ValueError(
                f"CVEvaluator: all {self._n_folds} folds failed. Check the model/provider for compatibility issues (e.g., GPU callbacks not supported)."
            )
        if n_valid > 0:
            context.artifacts.oof_preds = pd.Series(oof_preds, name="oof_preds")

        # — Metrics
        metrics_list = self._metrics
        if not metrics_list:
            task_key = "regression"
            if task_type is not None:
                raw = str(task_type).lower()
                if "classification" in raw or "multitask" in raw:
                    task_key = raw
            metrics_list = DEFAULT_METRICS.get(task_key, ["rmse"])

        y_true_arr = y_arr.astype(float) if label_encoder is None else y_arr.astype(float)
        y_pred_arr = oof_preds[valid_mask]
        y_true_valid = y_true_arr[valid_mask]

        results: dict[str, float] = {}
        for metric_name in metrics_list:
            entry = registry.metric.get(metric_name)
            if entry is None:
                logger.warning("Unknown metric: %s — skipping", metric_name)
                continue
            fn, kwargs = entry

            if metric_name in ("logloss", "auc"):
                try:
                    proba = self._compute_oof_probas(x, y_arr, fold_models, fold_val_indices)
                    if proba is not None:
                        if metric_name == "logloss":
                            results[metric_name] = float(fn(y_true_arr, proba, **kwargs))
                        else:
                            results[metric_name] = float(roc_auc_score(y_true_arr, proba, multi_class="ovo", average="weighted"))
                except Exception as e:
                    logger.warning("CVEvaluator: %s failed — %s", metric_name, e)
                continue

            try:
                results[metric_name] = float(fn(y_true_valid, y_pred_arr, **kwargs))
            except Exception as e:
                logger.warning("CVEvaluator: metric %s failed — %s", metric_name, e)

        return results

    def _compute_oof_probas(
        self,
        x: pd.DataFrame,
        y_arr: np.ndarray,
        fold_models: list[SupportedModel],
        fold_val_indices: list[npt.NDArray[Any]],
    ) -> np.ndarray | None:
        """Compute OOF probabilities using already-trained fold models."""
        n_samples = len(x)
        n_classes = len(np.unique(y_arr))
        oof_probas = np.full((n_samples, n_classes), np.nan)

        for fold_model, val_idx in zip(fold_models, fold_val_indices, strict=False):
            X_val = x.iloc[val_idx]
            try:
                proba = np.asarray(fold_model.predict_proba(X_val))
                if proba.ndim == 1:
                    proba = proba.reshape(-1, 1)
                oof_probas[val_idx] = proba
            except Exception as e:
                logger.warning("CVEvaluator: predict_proba fold failed — %s", e)
                continue

        valid_mask = ~np.isnan(oof_probas[:, 0])
        if valid_mask.sum() == 0:
            return None
        return oof_probas


__all__ = ["CVEvaluator"]
