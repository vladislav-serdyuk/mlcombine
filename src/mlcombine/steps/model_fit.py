"""ModelFitStep — fits the model on the training data from pipeline context."""

from __future__ import annotations

import logging

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext
from mlcombine.models.meta.uplift import TLearner, SLearner

logger = logging.getLogger(__name__)


@registry.step("ModelFitStep")
class ModelFitStep(BaseStep[PipelineContext]):
    """Fits the model on train_df using target_col, stores fitted model on context.

    Side Effects:
        - Mutates context.artifacts.model by calling .fit(x, y).
    """

    train = True
    predict = False

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self._drop_columns: list[str] = cfg.data.drop_columns

    def run(self, context: PipelineContext) -> PipelineContext:
        """Fit the model using train_df and target_col from context."""
        df = context.data.train_df
        if df is None:
            raise RuntimeError("Train data not loaded — DataLoaderStep must run before ModelFitStep")

        target = context.data.target_col
        if target is None:
            raise RuntimeError("target_col not set on context")

        if isinstance(target, str):
            cols_to_drop = [target]
        elif isinstance(target, list):
            cols_to_drop = target
        elif isinstance(target, dict):
            cols_to_drop = list(target.values())
        else:
            raise RuntimeError(f"Unsupported target_col type: {type(target).__name__}")

        x = df.drop(columns=cols_to_drop)
        if self._drop_columns:
            to_drop = [c for c in self._drop_columns if c in x.columns]
            if to_drop:
                x = x.drop(columns=to_drop)
        if len(cols_to_drop) == 1:
            y: pd.Series | pd.DataFrame = df[cols_to_drop[0]]
        else:
            y = df[cols_to_drop]

        treatment_col = context.data.treatment_col
        treatment = df[treatment_col] if treatment_col else None
        if treatment_col:
            x = x.drop(columns=[treatment_col])

        model = context.artifacts.model
        if model is None:
            raise RuntimeError("No model on context — CreateModelStep must run before ModelFitStep")

        # Materialise blueprint if needed (meta-providers defer build to fit time)
        if isinstance(model, ModelBlueprint):
            model = model.build()
            context.artifacts.model = model

        if treatment is not None and not isinstance(model, (TLearner, SLearner)):
            logger.warning(
                "treatment_col='%s' is set but model is not TLearner/SLearner "
                "(provider is not t_learner/s_learner). The treatment column will be dropped from "
                "features and the model may ignore the treatment argument.",
                treatment_col,
            )

        if isinstance(x, pd.DataFrame):
            n_rows, n_cols = x.shape
        else:
            n_rows, n_cols = len(x), 0
        logger.info("Fitting %s on %d samples, %d features", type(model).__name__, n_rows, n_cols)
        if treatment is not None:
            model.fit(x, y, treatment=treatment)
        else:
            model.fit(x, y)
        logger.info("Model fitted: %s", type(model).__name__)

        context.artifacts.feature_names = list(x.columns)
        return context
