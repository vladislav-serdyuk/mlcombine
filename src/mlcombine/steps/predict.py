"""ModelPredictStep — run model.predict() and store predictions in context."""

from __future__ import annotations

import logging

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("ModelPredictStep")
class ModelPredictStep(BaseStep[PipelineContext]):
    """Generate predictions using the trained model and store in context.

    Runs only in predict mode, before SavePredictionsStep.

    Side Effects:
        - Sets ``context.data.predictions`` and ``context.data.prediction_ids``.
    """

    train = False
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self._drop_columns = cfg.data.drop_columns
        _, self._label_map = cfg.make_label_maps()
        sp = getattr(cfg.step_config, "save_predictions", None) or {}
        self._id_col = sp.get("id_col", None)

    def run(self, context: PipelineContext) -> PipelineContext:
        test_df = context.data.test_df
        if test_df is None:
            logger.warning("No test data available — skipping predictions")
            return context

        model = context.artifacts.model
        if model is None:
            raise RuntimeError("No model in artifacts — LoadArtifactsStep must run before ModelPredictStep")

        if self._drop_columns:
            to_drop = [c for c in self._drop_columns if c in test_df.columns]
            if to_drop:
                test_df = test_df.drop(columns=to_drop)

        preds = model.predict(test_df)
        preds_series = pd.Series(preds.ravel())

        if self._label_map:
            preds_series = preds_series.astype(int).map(self._label_map)

        context.data.predictions = preds_series

        if self._id_col is not None and self._id_col != "" and self._id_col in test_df.columns:
            context.data.prediction_ids = test_df[self._id_col]
        elif context.data.prediction_ids is None:
            if self._id_col == "":
                context.data.prediction_ids = pd.Series(range(len(test_df)))
            else:
                context.data.prediction_ids = None

        logger.info("Generated %d predictions", len(preds_series))
        return context


__all__ = [
    "ModelPredictStep",
]
