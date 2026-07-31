"""ReferenceJoinStep — merge pair data against one or more reference tables.

Supports a list of join configs for sequential merges (e.g. left side
then right side).  Optionally filters training labels.

Configure via ``step_config.reference_join``::

    reference_join:
      joins:
        - reference_path: items.parquet
          left_on: leftItemId
          suffix: _left
        - reference_path: items.parquet
          left_on: rightItemId
          suffix: _right
      keep_labels:
        - "no_relevant"
        - "relevant_minus"
        - "relevant"
        - "relevant_plus"

Backward-compatible shorthand (single join, no keep_labels)::

    reference_join:
      reference_path: items.parquet
      left_on: leftItemId
      suffix: _left
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext, TargetColumn
from mlcombine.core.utils import safe_merge as _safe_merge

logger = logging.getLogger(__name__)


@registry.step("ReferenceJoinStep", before="TypeDetectStep")
class ReferenceJoinStep(BaseStep[PipelineContext]):
    """Merge pair data against reference tables (one or more sequential joins).

    Side Effects:
        - Replaces ``context.data.train_df`` and ``context.data.test_df``
          with merged copies.
        - Optionally filters ``train_df`` rows to keep only specified labels.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        rj = getattr(cfg.step_config, "reference_join", None) or {}
        if "reference_path" in rj and "joins" not in rj:
            joins_raw = [rj]
        else:
            joins_raw = rj.get("joins", [])
        self._joins = joins_raw
        self._keep_labels: list[str] | None = rj.get("keep_labels", None)
        target_col: TargetColumn | None = cfg.data.target_col
        self._target_col: str | None = target_col if isinstance(target_col, str) else None

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        rj = getattr(cfg.step_config, "reference_join", None) or {}
        if "reference_path" in rj and "joins" not in rj:
            return True
        return bool(rj.get("joins", []))

    @staticmethod
    def _load_reference(spec: dict[str, Any]) -> pd.DataFrame:
        fmt = spec.get("reference_format", "parquet")
        path = spec["reference_path"]
        if fmt == "parquet":
            try:
                df = pd.read_parquet(path, engine="fastparquet")
            except ImportError:
                df = pd.read_parquet(path, engine="pyarrow")  # type: ignore[call-overload]
        else:
            df = pd.read_csv(path)
        index_col = spec.get("reference_index_col", "itemId")
        if index_col in df.columns:
            df = df.set_index(index_col)
        return df

    @staticmethod
    def _exec_join(df: pd.DataFrame, reference: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
        left_on: str = spec.get("left_on", "leftItemId") or "leftItemId"
        suffix: str = spec.get("suffix", "")
        how: str = spec.get("how", "left")
        merged = _safe_merge(
            df,
            reference,
            how=how,  # type: ignore[arg-type]
            left_on=left_on,
            right_index=True,
            suffixes=("", suffix),
        )
        # Ensure all reference columns have the configured suffix
        # (pandas only appends when there's overlap; we always want it)
        rename = {}
        for col in reference.columns:
            renamed = f"{col}{suffix}"
            if col in merged.columns and renamed not in merged.columns:
                rename[col] = renamed
        if rename:
            merged = merged.rename(columns=rename)
        return merged

    def run(self, context: PipelineContext) -> PipelineContext:
        for spec in self._joins:
            reference = self._load_reference(spec)
            if context.data.train_df is not None:
                context.data.train_df = self._exec_join(context.data.train_df, reference, spec)
            if context.data.test_df is not None:
                context.data.test_df = self._exec_join(context.data.test_df, reference, spec)

        if self._keep_labels and context.data.train_df is not None and self._target_col:
            before = len(context.data.train_df)
            mask = context.data.train_df[self._target_col].isin(self._keep_labels)
            context.data.train_df = context.data.train_df[mask].copy()
            after = len(context.data.train_df)
            if after < before:
                logger.info("keep_labels filtered %d → %d rows", before, after)

        return context
