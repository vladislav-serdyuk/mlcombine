"""CreateModelStep — creates a model via ModelBuilder and stores it on context."""

from __future__ import annotations

import logging

from mlcombine.core.registry import registry
from mlcombine.core.types import (
    BaseStep,
    MLCombineConfig,
    PipelineContext,
    TaskType,
)
from mlcombine.core.builder import ModelBuilder

logger = logging.getLogger(__name__)


@registry.step("CreateModelStep")
class CreateModelStep(BaseStep[PipelineContext]):
    """Creates a model by resolving the model DAG and stores it on context.

    If any node has ``input_size`` in its params AND the value is ``0`` or
    ``None``, the step auto-detects it from the training feature count.
    """

    train = True
    predict = False

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self._model_nodes = cfg.model

    def run(self, context: PipelineContext) -> PipelineContext:
        task_type: TaskType = TaskType.REGRESSION
        if context.data.task_type is not None:
            raw = context.data.task_type
            task_type = raw if isinstance(raw, TaskType) else TaskType.MULTITASK

        num_classes: int | None = None
        if task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK):
            num_classes = 2
            df = context.data.train_df
            target = context.data.target_col
            if df is not None and target is not None:
                if isinstance(target, str) and target in df.columns:
                    unique = df[target].nunique()
                    if unique > 0:
                        num_classes = unique
                elif isinstance(target, list):
                    unique = df[target[0]].nunique() if target else 2
                    if unique > 0:
                        num_classes = unique

        input_size: int | None = None
        if context.data.train_df is not None:
            exclude = {context.data.target_col} if isinstance(context.data.target_col, str) else set()
            if context.data.treatment_col:
                exclude.add(context.data.treatment_col)
            input_size = len([c for c in context.data.train_df.columns if c not in exclude])

        builder = ModelBuilder()
        model = builder.build_all(
            self._model_nodes,
            task_type=task_type,
            num_classes=num_classes,
            input_size=input_size,
        )
        context.artifacts.model = model  # type: ignore[assignment]
        context.artifacts.models = builder.built_dict  # type: ignore[assignment]
        logger.info("Built %d model nodes", len(self._model_nodes))
        return context
