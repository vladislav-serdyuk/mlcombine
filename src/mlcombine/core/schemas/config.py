"""Pydantic config models for mlcombine pipeline configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlcombine.core.schemas.enums import (
    EncodeStrategy,
    FeatureMap,
    ImputeStrategy,
    ScaleStrategy,
    Separator,
    TargetColumn,
    TaskType,
)
from mlcombine.core.schemas.step_configs import StepConfigs

if TYPE_CHECKING:
    from mlcombine.core.protocols import MLModelProtocol

    _Model = MLModelProtocol[Any] | None
else:
    _Model = object | None


# ──────────────────────────────────────────────
#  Type aliases
# ──────────────────────────────────────────────
type DataFrame = pd.DataFrame
type Series = pd.Series
type ArrayLike = np.ndarray
type ModelTarget = pd.Series | pd.DataFrame | None


# ──────────────────────────────────────────────
#  Pydantic config models (extra="forbid" on all)
# ──────────────────────────────────────────────
class DataConfig(BaseModel):
    """Configuration for dataset paths and target column."""

    model_config = ConfigDict(extra="forbid")

    train_df: str
    test_df: str
    target_col: TargetColumn
    sep: Separator = ","
    treatment_col: str | None = None
    drop_columns: list[str] = []
    force_prepare_dataset: bool = False
    task_type: TaskType | None = None


class HandlingNumbersConfig(BaseModel):
    """Configuration for numerical feature handling (impute, scale)."""

    model_config = ConfigDict(extra="forbid")

    impute: ImputeStrategy = ImputeStrategy.MEDIAN
    scale: ScaleStrategy = ScaleStrategy.ROBUST


class HandlingCategoriesConfig(BaseModel):
    """Configuration for categorical feature encoding."""

    model_config = ConfigDict(extra="forbid")

    encode: EncodeStrategy = EncodeStrategy.ONEHOT
    smoothing: float = 10.0


class ColumnHandlingConfig(BaseModel):
    """Per-column overrides for feature handling (partial — inherit globals).

    Unset fields fall back to the global ``HandlingConfig`` strategies.
    ``none`` skips the column entirely (no imputation/encoding/scaling).
    """

    model_config = ConfigDict(extra="forbid")

    encode: EncodeStrategy | None = None
    impute: ImputeStrategy | None = None
    scale: ScaleStrategy | None = None
    fill_value: float | None = None


class HandlingConfig(BaseModel):
    """Aggregated configuration for all feature-type handling strategies."""

    model_config = ConfigDict(extra="forbid")

    numbers: HandlingNumbersConfig = Field(default_factory=HandlingNumbersConfig)
    categories: HandlingCategoriesConfig = Field(default_factory=HandlingCategoriesConfig)
    columns: dict[str, ColumnHandlingConfig] = Field(default_factory=dict)


class LayerConfig(BaseModel):
    """A single layer in a neural network architecture."""

    model_config = ConfigDict(extra="forbid")

    type: str
    in_features: int | None = None
    out_features: int | None = None
    bias: bool = True
    in_channels: int | None = None
    out_channels: int | None = None
    kernel_size: int | None = None
    stride: int = 1
    padding: int = 0
    num_features: int | None = None
    eps: float = 1e-5
    p: float = 0.5
    nhead: int | None = None
    dim_feedforward: int | None = None
    num_layers: int | None = None
    d_model: int | None = None


class ModelNode(BaseModel):
    """A single model node in the pipeline DAG.

    Can be a base model (created by a provider) or a meta-model (wrapping
    one or more other nodes via ``model`` / ``models``).

    The last node in ``MLCombineConfig.models`` is the final model used
    for fitting and prediction.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    provider: str
    model: str | None = None
    models: list[str] = Field(default_factory=list)
    params: dict[str, object] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    """Configuration for training output and error handling."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./outputs"
    output_file: str = "./outputs/result"


class EnvironmentConfig(BaseModel):
    """Configuration for runtime environment behavior."""

    model_config = ConfigDict(extra="forbid")

    auto_install: bool = True


class PipelineConfig(BaseModel):
    """Optional pipeline configuration.

    When *order* is set, it overrides any ``before``/``after`` hints from
    ``@registry.step`` decorators.  Steps not listed are skipped.
    *skip* unconditionally removes named steps from the resolved order.
    """

    model_config = ConfigDict(extra="forbid")

    order: list[str] | None = None
    skip: list[str] = []


class MLCombineConfig(BaseModel):
    """Top-level configuration for a mlcombine pipeline run."""

    model_config = ConfigDict(extra="forbid")

    plugins: list[str] = Field(default_factory=list)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    data: DataConfig
    force_types: FeatureMap | None = None
    target_label_map: dict[int, str] | None = None
    handling: HandlingConfig = Field(default_factory=HandlingConfig)
    model: list[ModelNode] = Field(min_length=1)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    step_config: StepConfigs = Field(default_factory=StepConfigs)

    def make_label_maps(
        self,
    ) -> tuple[dict[str, int], dict[int, str] | None]:
        """Return (str→int reverse map, int→str forward map) from target_label_map."""
        if self.target_label_map is None:
            return {}, None
        rev: dict[str, int] = {str(v): int(k) for k, v in self.target_label_map.items()}
        return rev, self.target_label_map


__all__ = [
    "ArrayLike",
    "DataConfig",
    "DataFrame",
    "EnvironmentConfig",
    "HandlingCategoriesConfig",
    "HandlingConfig",
    "HandlingNumbersConfig",
    "LayerConfig",
    "MLCombineConfig",
    "ModelNode",
    "ModelTarget",
    "PipelineConfig",
    "Series",
    "TrainerConfig",
]
