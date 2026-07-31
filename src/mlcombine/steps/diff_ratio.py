"""DiffRatioStep — add abs-diff and clipped ratio for column pairs.

Configure via ``step_config.diff_ratio``::

    diff_ratio:
      pairs:
        - ["a_left", "a_right"]
        - ["b_left", "b_right"]
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("DiffRatioStep", before="SplitStep")
class DiffRatioStep(BaseStep[PipelineContext]):
    """Add ``{col1}_diff`` and ``{col1}_ratio`` for each column pair.

    Side Effects:
        - Adds new numeric columns to ``train_df`` and ``test_df``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        dr = getattr(cfg.step_config, "diff_ratio", None) or {}
        self._pairs = dr.get("pairs", [])

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        dr = getattr(cfg.step_config, "diff_ratio", None) or {}
        return bool(dr.get("pairs", []))

    @staticmethod
    def _transform_features(df: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
        for pair in pairs:
            if len(pair) != 2:
                continue
            a, b = pair[0], pair[1]
            if a not in df.columns or b not in df.columns:
                continue
            df = df.copy()
            col_a = df[a].astype(float)
            col_b = df[b].astype(float).replace(0, np.nan)
            df[f"{a}_diff"] = (col_a - col_b).abs()
            df[f"{a}_ratio"] = (col_a / col_b).clip(0, 10).fillna(0)
        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.data.train_df is not None:
            context.data.train_df = self._transform_features(context.data.train_df, self._pairs)
        if context.data.test_df is not None:
            context.data.test_df = self._transform_features(context.data.test_df, self._pairs)
        if self._pairs:
            logger.info("Added diff/ratio features for %d column pairs", len(self._pairs))
        return context
