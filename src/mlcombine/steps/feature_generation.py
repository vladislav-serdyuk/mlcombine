"""FeatureGenerationStep — count encoding, frequency encoding for categorical columns.

Adds aggregated features for each categorical column:
  - ``count`` — how many times each value appears in the training set.
  - ``freq`` — count normalised by total rows.

Configure via ``step_config.feature_generation``::

    step_config:
      feature_generation:
        count_encode: true
        freq_encode: true
        max_unique: 1000  # skip high-cardinality columns
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import (
    BaseStep,
    FeatureMap,
    FeatureType,
    MLCombineConfig,
    PipelineContext,
)

logger = logging.getLogger(__name__)


@registry.step("FeatureGenerationStep")
class FeatureGenerationStep(BaseStep[PipelineContext]):
    """Add count / frequency encoding features for categorical columns.

    Side Effects:
        - Adds new numeric columns (``{col}_count``, ``{col}_freq``) to
          ``train_df`` and ``test_df``.
        - Updates ``detected_types`` to mark new columns as ``FeatureType.NUMBER``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        fg = getattr(cfg.step_config, "feature_generation", None) or {}
        self._count_encode = fg.get("count_encode", True)
        self._freq_encode = fg.get("freq_encode", True)
        self._max_unique = fg.get("max_unique", 1000)
        self._exclude_cols: set[str] = set()
        target = cfg.data.target_col
        if isinstance(target, str):
            self._exclude_cols.add(target)
        elif isinstance(target, list):
            self._exclude_cols.update(target)
        elif isinstance(target, dict):
            self._exclude_cols.update(target.values())

        self._mappings: dict[str, dict[str, dict[str, int | float]]] = {}

    def _fit(self, df: pd.DataFrame, detected: FeatureMap) -> None:
        cat_cols = [col for col, ft in detected.items() if ft == FeatureType.CATEGORY and col in df.columns and col not in self._exclude_cols]
        nrows = len(df)

        for col in cat_cols:
            n_unique = int(df[col].nunique())
            if n_unique > self._max_unique or n_unique <= 1:
                continue

            mapping: dict[str, Any] = {}
            counts = df[col].value_counts()
            if self._count_encode:
                mapping["count"] = counts.to_dict()
            if self._freq_encode:
                mapping["freq"] = (counts / nrows).to_dict()

            self._mappings[col] = mapping

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._mappings:
            return df

        df = df.copy()
        for col, mapping in self._mappings.items():
            if col not in df.columns:
                continue
            if "count" in mapping:
                df[f"{col}_count"] = df[col].map(mapping["count"]).fillna(0).astype(int)
            if "freq" in mapping:
                df[f"{col}_freq"] = df[col].map(mapping["freq"]).fillna(0.0).astype(float)
        return df

    def _update_detected(self, detected: FeatureMap) -> None:
        for col in self._mappings:
            if "count" in self._mappings[col]:
                detected[f"{col}_count"] = FeatureType.NUMBER
            if "freq" in self._mappings[col]:
                detected[f"{col}_freq"] = FeatureType.NUMBER

    def run(self, context: PipelineContext) -> PipelineContext:
        detected = context.data.detected_types
        if detected is None:
            return context

        if context.data.train_df is not None:
            self._fit(context.data.train_df, detected)
            context.data.train_df = self._transform(context.data.train_df)

        if context.data.test_df is not None:
            context.data.test_df = self._transform(context.data.test_df)

        self._update_detected(detected)
        return context


__all__ = [
    "FeatureGenerationStep",
]
