"""Core type definitions — config schemas, enums, exceptions, and context models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlcombine.core.exceptions import (
    ConfigurationError,
    DatasetNotFoundError,
    EmptyDatasetError,
    MLCombineException,
    UnsupportedBackendError,
)
from mlcombine.core.schemas.config import (
    ArrayLike,
    DataFrame,
    DataConfig,
    EnvironmentConfig,
    HandlingCategoriesConfig,
    HandlingConfig,
    HandlingNumbersConfig,
    LayerConfig,
    MLCombineConfig,
    ModelNode,
    ModelTarget,
    PipelineConfig,
    Series,
    TrainerConfig,
)
from mlcombine.core.schemas.enums import (
    ColumnName,  # noqa: F401
    DatasetFormat,
    Device,
    EncodeStrategy,
    FeatureMap,
    FeatureType,
    ImputeStrategy,
    ModelObjective,
    ScaleStrategy,
    Separator,  # noqa: F401
    TargetColumn,  # noqa: F401
    TaskType,
    TensorBackendType,
    UpliftMethod,
)
from mlcombine.core.schemas.step_configs import StepConfigs

if TYPE_CHECKING:
    from mlcombine.core.protocols import MLModelProtocol

    _Model: TypeAlias = MLModelProtocol[Any] | None
else:
    _Model: TypeAlias = object | None


# ──────────────────────────────────────────────
#  Type aliases (kept for reference)
# ──────────────────────────────────────────────
# DataFrame / Series / ArrayLike / ModelTarget are re-exported from schemas.config


# ──────────────────────────────────────────────
#  Pipeline context types
# ──────────────────────────────────────────────
class PipelineData(BaseModel):
    """Data payload carried through the pipeline context."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    train_df: DataFrame | None = None
    holdout_df: DataFrame | None = None
    test_df: DataFrame | None = None
    train_df_path: Path | None = None
    test_df_path: Path | None = None
    detected_types: FeatureMap | None = None
    task_type: TaskType | dict[str, TaskType] | None = None
    target_col: str | list[str] | dict[str, str] | None = None
    treatment_col: str | None = None
    predictions: pd.Series | None = None
    prediction_ids: pd.Series | None = None


class PipelineArtifacts(BaseModel):
    """Artifacts (model, imputers, encoders, scaler) produced during pipeline execution."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model: _Model = None
    models: dict[str, object] = Field(default_factory=dict)
    imputers: dict[str, object] = Field(default_factory=dict)
    imputer_features: list[str] | None = None
    encoders: dict[str, object] = Field(default_factory=dict)
    scalers: dict[str, object] = Field(default_factory=dict)
    scaler_features: list[str] | None = None
    feature_names: list[str] | None = None
    target_mapping: dict[object, object] | None = None
    oof_preds: pd.Series | None = None
    evaluation_results: dict[str, float] | None = None
    test_evaluation: dict[str, float] | None = None


class PipelineContext(BaseModel):
    """Top-level context combining pipeline data and artifacts."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    data: PipelineData = Field(default_factory=PipelineData)
    artifacts: PipelineArtifacts = Field(default_factory=PipelineArtifacts)


# ──────────────────────────────────────────────
#  Step interface
# ──────────────────────────────────────────────
class BaseStep[StepContext: PipelineContext](ABC):
    """Abstract base class for all pipeline steps.

    Subclasses set *train* / *predict* flags and override ``is_required()``
    when the step should only appear conditionally.

    The constructor receives the full config plus runtime flags so that
    every step can be instantiated uniformly from the registry.
    """

    train: bool = True
    predict: bool = True

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        """Return ``False`` to skip this step (e.g. based on config fields)."""
        return True

    def __init__(
        self,
        cfg: MLCombineConfig,
        *,
        predict: bool = False,
        weights: str | None = None,
    ) -> None: ...

    @abstractmethod
    def run(self, context: StepContext) -> StepContext:
        """Execute the step, mutating and returning the pipeline context."""


SUPPORTED_DATASET_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".tsv", ".parquet", ".xlsx", ".xls"},
)


__all__ = [
    "ArrayLike",
    "BaseStep",
    "ConfigurationError",
    "DataConfig",
    "DataFrame",
    "DatasetFormat",
    "DatasetNotFoundError",
    "Device",
    "EmptyDatasetError",
    "EncodeStrategy",
    "EnvironmentConfig",
    "FeatureMap",
    "FeatureType",
    "HandlingCategoriesConfig",
    "HandlingConfig",
    "HandlingNumbersConfig",
    "ImputeStrategy",
    "LayerConfig",
    "MLCombineConfig",
    "MLCombineException",
    "ModelNode",
    "ModelObjective",
    "ModelTarget",
    "PipelineArtifacts",
    "PipelineConfig",
    "PipelineContext",
    "PipelineData",
    "ScaleStrategy",
    "Separator",
    "Series",
    "StepConfigs",
    "SUPPORTED_DATASET_SUFFIXES",
    "TargetColumn",
    "TaskType",
    "TensorBackendType",
    "TrainerConfig",
    "UnsupportedBackendError",
    "UpliftMethod",
]
