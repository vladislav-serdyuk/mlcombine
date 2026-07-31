from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("SplitStep", before="CreateModelStep")
class SplitStep(BaseStep[PipelineContext]):
    """SplitStep — splits train_df into train_df + holdout_df.

    Runs after feature engineering steps (Impute, EncodeScale, CrossEncoder, etc.)
    but before CreateModelStep, so the model never sees holdout data during training.

    Configure via ``step_config.split``::

        split:
          val_fraction: 0.2
          stratified: true
          group_cols:
            - authorId_left
            - authorId_right

    When ``group_cols`` is set, all rows sharing a value in any of those columns
    are kept together (either fully in train or fully in holdout) to prevent
    data leakage via entity overlap between splits.
    """

    train = True
    predict = False

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        sc = getattr(cfg.step_config, "split", None) or {}
        self._val_fraction = sc.get("val_fraction", 0.2)
        self._stratified = sc.get("stratified", True)
        self._random_state = sc.get("random_state", 42)
        self._target_col = cfg.data.target_col
        self._group_cols = sc.get("group_cols", None)

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        return getattr(cfg.step_config, "split", None) is not None

    def run(self, context: PipelineContext) -> PipelineContext:
        df = context.data.train_df
        if df is None:
            logger.warning("SplitStep: no train_df — skipping")
            return context

        target = self._target_col
        if not isinstance(target, str) or target not in df.columns:
            logger.warning("SplitStep: target_col %r not in train_df — skipping", target)
            return context

        y = df[target]
        task_type = context.data.task_type
        is_classification = task_type is not None and "classification" in str(task_type).lower()

        if self._group_cols:
            train_df, holdout_df = self._group_split(df, y, is_classification=is_classification)
        else:
            train_df, holdout_df = self._random_split(df, y, is_classification=is_classification)

        context.data.train_df = train_df
        context.data.holdout_df = holdout_df

        logger.info(
            "Split: train=%d holdout=%d (val_fraction=%.2f)",
            len(train_df),
            len(holdout_df),
            self._val_fraction,
        )
        return context

    def _random_split(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        is_classification: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = df.drop(columns=[y.name])

        if is_classification and self._stratified:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self._val_fraction,
                random_state=self._random_state,
            )
            train_idx, val_idx = next(splitter.split(x, y))
            train_df = df.iloc[train_idx].copy()
            holdout_df = df.iloc[val_idx].copy()
        else:
            train_df, holdout_df = train_test_split(
                df,
                test_size=self._val_fraction,
                random_state=self._random_state,
            )
        return train_df, holdout_df

    def _group_split(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        is_classification: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Group-aware split: values in ``group_cols`` never cross train/holdout."""
        assert self._group_cols is not None
        present_cols = [c for c in self._group_cols if c in df.columns]
        if not present_cols:
            logger.warning("None of group_cols %r found in DataFrame — falling back to random split", self._group_cols)
            return self._random_split(df, y, is_classification=is_classification)

        all_series = [df[c] for c in present_cols]
        stacked = pd.concat(all_series, ignore_index=True)
        unique_values = stacked.dropna().unique()
        unique_values_set = set(unique_values)

        # Map each unique value to its majority target class
        value_target: dict[Any, Any] = {}
        if len(present_cols) == 1:
            # vectorised path — single group column
            col = present_cols[0]
            y_name: str = str(y.name)
            value_target = (
                df.groupby(col, observed=True)[y_name]
                .agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else y.mode().iloc[0],
                )
                .to_dict()
            )
        else:
            for val in unique_values:
                mask = pd.Series(False, index=df.index)
                for col in present_cols:
                    mask |= df[col] == val
                targets = y[mask]
                if len(targets) > 0:
                    value_target[val] = targets.mode().iloc[0]
                else:
                    value_target[val] = y.mode().iloc[0]

        values_arr = np.array(list(unique_values_set))
        targets_arr = np.array([value_target[v] for v in values_arr])

        if is_classification and self._stratified:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self._val_fraction,
                random_state=self._random_state,
            )
            train_val_idx, holdout_val_idx = next(sss.split(values_arr, targets_arr))
        else:
            train_val_idx, holdout_val_idx = next(
                StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=self._val_fraction,
                    random_state=self._random_state,
                ).split(values_arr, np.zeros(len(values_arr)))
            )

        train_values = set(values_arr[train_val_idx])
        holdout_values = set(values_arr[holdout_val_idx])

        # Row belongs to holdout if ANY of its group values is assigned to holdout
        holdout_mask = pd.Series(False, index=df.index)
        for col in present_cols:
            holdout_mask |= df[col].isin(holdout_values)

        train_df = df[~holdout_mask].copy()
        holdout_df = df[holdout_mask].copy()

        actual_frac = len(holdout_df) / len(df) if len(df) > 0 else 0
        logger.info(
            "Group split: %d unique values — train values=%d holdout values=%d (actual holdout frac=%.3f)",
            len(unique_values_set),
            len(train_values),
            len(holdout_values),
            actual_frac,
        )

        return train_df, holdout_df


__all__ = ["SplitStep"]
