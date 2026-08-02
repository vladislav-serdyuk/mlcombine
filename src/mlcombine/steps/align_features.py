"""AlignFeaturesStep — aligns test features with training features."""

from __future__ import annotations

import logging

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("AlignFeaturesStep")
class AlignFeaturesStep(BaseStep[PipelineContext]):
    """Align test dataframe columns with the features used during training.

    Uses ``context.artifacts.feature_names`` (saved during training) to select
    only the columns the model expects. Missing features are filled with NaN,
    extra features are dropped.

    Runs only in predict mode, before SavePredictionsStep.
    """

    train = False
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self._id_col = cfg.data.id_col

    def run(self, context: PipelineContext) -> PipelineContext:
        test_df = context.data.test_df
        if test_df is None:
            logger.warning("No test data available — skipping feature alignment")
            return context

        feature_names = context.artifacts.feature_names
        if not feature_names:
            logger.warning("No feature_names in artifacts — skipping feature alignment")
            return context

        # preserve the id column before dropping non-feature columns
        if self._id_col and self._id_col in test_df.columns and context.data.prediction_ids is None:
            context.data.prediction_ids = test_df[self._id_col]

        missing = [f for f in feature_names if f not in test_df.columns]
        extra = [c for c in test_df.columns if c not in feature_names]

        if missing:
            logger.warning("Missing features in test data (will be filled with NaN): %s", missing)
        if extra:
            logger.info("Dropping extra features not seen during training: %s", extra)

        aligned = test_df.reindex(columns=feature_names)

        context.data.test_df = aligned
        logger.info("Aligned test features: %d columns", len(aligned.columns))
        return context
