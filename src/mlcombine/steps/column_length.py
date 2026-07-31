"""ColumnLengthStep — add character-length columns for string columns.

Configure via ``step_config.column_length``::

    column_length:
      columns: ["title", "content"]
"""

from __future__ import annotations

import logging

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("ColumnLengthStep", before="SplitStep")
class ColumnLengthStep(BaseStep[PipelineContext]):
    """Add ``{col}_len`` columns for each string column in config.

    Side Effects:
        - Adds new numeric columns to ``train_df`` and ``test_df``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        cl = getattr(cfg.step_config, "column_length", None) or {}
        self._columns = cl.get("columns", [])

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        cl = getattr(cfg.step_config, "column_length", None) or {}
        return bool(cl.get("columns", []))

    @staticmethod
    def _add_lengths(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[f"{col}_len"] = df[col].astype(str).str.len().astype(float)
        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.data.train_df is not None:
            context.data.train_df = self._add_lengths(context.data.train_df, self._columns)
        if context.data.test_df is not None:
            context.data.test_df = self._add_lengths(context.data.test_df, self._columns)
        if self._columns:
            logger.info("Added length features for %d columns", len(self._columns))
        return context
