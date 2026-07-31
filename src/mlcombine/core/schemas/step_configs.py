from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = 0.2
    stratified: bool = True
    random_state: int = 42
    group_cols: list[str] | None = None


class EvaluateStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] | None = None


class ReferenceJoinJoinConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_path: str
    reference_format: str = "parquet"
    reference_index_col: str = "itemId"
    left_on: str = "leftItemId"
    suffix: str = ""
    how: str = "left"


class ReferenceJoinStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    joins: list[ReferenceJoinJoinConfig] = Field(default_factory=list)
    keep_labels: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _shorthand_to_full(cls, data: Any) -> Any:
        if isinstance(data, dict) and "reference_path" in data and "joins" not in data:
            return {"joins": [data]}
        return data


class CrossEncoderStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    pairs: list[list[str]] = Field(default_factory=list)
    batch_size: int = 64
    max_length: int | None = None
    predict_chunk: int = 512
    drop_source: bool = False
    device: str | None = None
    cache_dir: str | None = None


class PairwiseSimilarityStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "all-MiniLM-L6-v2"
    pairs: list[list[str]] = Field(default_factory=list)
    batch_size: int = 64
    max_length: int | None = None
    predict_chunk: int = 512
    drop_source: bool = False
    device: str | None = None


class TextOverlapStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[list[str]] = Field(default_factory=list)
    char_ngram: int = 0
    token_pattern: str = "[а-яёa-z]+"
    drop_source: bool = False


class ColumnLengthStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(default_factory=list)


class DiffRatioStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[list[str]] = Field(default_factory=list)


class SameValueStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[list[str]] = Field(default_factory=list)
    drop_source: bool = False


class FeatureGenerationStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count_encode: bool = True
    freq_encode: bool = True
    max_unique: int = 1000


class TextEmbeddingStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "intfloat/multilingual-e5-small"
    device: str = "cuda"
    batch_size: int = 64
    max_length: int = 512
    cache_dir: str | None = None


class SavePredictionsStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_col: str = "prediction"
    id_col: str | None = None


class StepConfigs(BaseModel):
    """Generic step config container — accepts any key for any step.

    Built-in steps and plugins alike access their own config via
    ``getattr(cfg.step_config, "step_name", {})`` and validate it internally.
    """

    model_config = ConfigDict(extra="allow")


__all__ = [
    "ColumnLengthStepConfig",
    "CrossEncoderStepConfig",
    "DiffRatioStepConfig",
    "EvaluateStepConfig",
    "FeatureGenerationStepConfig",
    "PairwiseSimilarityStepConfig",
    "ReferenceJoinJoinConfig",
    "ReferenceJoinStepConfig",
    "SameValueStepConfig",
    "SavePredictionsStepConfig",
    "SplitStepConfig",
    "StepConfigs",
    "TextEmbeddingStepConfig",
    "TextOverlapStepConfig",
]
