"""SameValueStep — add binary equal/not-equal features for column pairs.

Configure via ``step_config.same_value``::

    same_value:
      pairs:
        - ["a_left", "a_right"]
        - ["b_left", "b_right"]
      drop_source: false   # delete source columns after adding features
"""

from __future__ import annotations

import logging

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("SameValueStep", before="SplitStep")
class SameValueStep(BaseStep[PipelineContext]):
    """Add ``{col1}_same`` (int 0/1) for each column pair.

    Side Effects:
        - Adds new numeric columns to ``train_df`` and ``test_df``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        sv = getattr(cfg.step_config, "same_value", None) or {}
        self._pairs = sv.get("pairs", [])
        self._drop_source = sv.get("drop_source", False)

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        sv = getattr(cfg.step_config, "same_value", None) or {}
        return bool(sv.get("pairs", []))

    @staticmethod
    def _transform_features(df: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
        for pair in pairs:
            if len(pair) != 2:
                continue
            a, b = pair[0], pair[1]
            if a not in df.columns or b not in df.columns:
                continue
            df = df.copy()
            df[f"{a}_same"] = (df[a] == df[b]).astype(int)
        return df

    def _drop_source_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = []
        for pair in self._pairs:
            if len(pair) == 2:
                for c in pair:
                    if c in df:
                        cols_to_drop.append(c)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.data.train_df is not None:
            context.data.train_df = self._transform_features(context.data.train_df, self._pairs)
            if self._drop_source:
                context.data.train_df = self._drop_source_cols(context.data.train_df)
        if context.data.test_df is not None:
            context.data.test_df = self._transform_features(context.data.test_df, self._pairs)
            if self._drop_source:
                context.data.test_df = self._drop_source_cols(context.data.test_df)
        if self._pairs:
            logger.info("Added 'same' binary features for %d column pairs", len(self._pairs))
        return context
