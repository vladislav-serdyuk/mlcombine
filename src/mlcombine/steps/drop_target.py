"""DropTargetColumnsStep — drop target columns from test_df for inference isolation."""

from __future__ import annotations

import logging

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext, TargetColumn

logger = logging.getLogger(__name__)


@registry.step("DropTargetColumnsStep")
class DropTargetColumnsStep(BaseStep[PipelineContext]):
    """Drop target columns from test_df for inference isolation.

    Side Effects:
        - Removes target columns from context.data.test_df in-place.
    """

    train = False
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        """Initialize with config — extracts target column."""
        self.target_col: TargetColumn | None = cfg.data.target_col

    def run(self, context: PipelineContext) -> PipelineContext:
        """Drop target columns from the test DataFrame."""
        if self.target_col is None or context.data.test_df is None:
            return context

        cols = (
            [self.target_col] if isinstance(self.target_col, str) else (self.target_col if isinstance(self.target_col, list) else list(self.target_col.keys()))
        )
        existing = [c for c in cols if c in context.data.test_df.columns]
        if existing:
            logger.info("Inference isolation: dropping %s", existing)
            context.data.test_df = context.data.test_df.drop(columns=existing)
        return context
