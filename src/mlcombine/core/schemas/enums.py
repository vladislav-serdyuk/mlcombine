"""Low-level enums and type aliases — no dependencies on the rest of mlcombine.

Houses all ``StrEnum`` subclasses and simple ``type`` aliases that are used
across multiple layers.  Separated from ``types.py`` to avoid circular imports
between ``types.py → base.py → tensor → registry → types``.
"""

from __future__ import annotations

from enum import StrEnum

# ── type aliases ─────────────────────────────────────────────────────

type ColumnName = str
type Separator = str
type TargetColumn = str | list[str] | dict[str, str]


# ── feature type system ──────────────────────────────────────────────


class FeatureType(StrEnum):
    """Supported feature/column types for automatic type detection."""

    NUMBER = "number"
    CATEGORY = "category"
    DATETIME = "datetime"
    IMAGE_PATH = "image_path"
    SEQUENCE_TOKEN = "sequence_token"
    TEXT = "text"
    UNKNOWN = "unknown"


type FeatureMap = dict[ColumnName, FeatureType | str]


# ── task type (multi-task aware) ──────────────────────────────────────


class TaskType(StrEnum):
    """Machine learning task types — supports multitask."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UPLIFT = "uplift"
    MULTITASK = "multitask"


# ── dataset formats ──────────────────────────────────────────────────


class DatasetFormat(StrEnum):
    """Supported dataset file formats."""

    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    XLSX = "xlsx"
    XLS = "xls"


# ── preprocessing strategies ─────────────────────────────────────────


class ImputeStrategy(StrEnum):
    """Missing-value imputation strategies."""

    NONE = "none"
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class ScaleStrategy(StrEnum):
    """Feature scaling strategies."""

    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class EncodeStrategy(StrEnum):
    """Categorical encoding strategies."""

    TARGET = "target"
    ORDINAL = "ordinal"
    ONEHOT = "onehot"
    NONE = "none"


# ── model objectives ─────────────────────────────────────────────────


class ModelObjective(StrEnum):
    """Optimization objectives / evaluation metrics."""

    MAPE = "MAPE"
    F1 = "F1"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "AUC"
    LOGLOSS = "logloss"
    RMSE = "RMSE"


# ── metric direction ────────────────────────────────────────────────


class MetricDirection(StrEnum):
    """Whether higher or lower metric values are better.

    Used by the metric registry (``@registry.metric(..., direction=...)``)
    so that consumers like the optuna tuner know how to optimize.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# ── uplift ───────────────────────────────────────────────────────────


class UpliftMethod(StrEnum):
    """Uplift modeling strategies."""

    T_LEARNER = "t_learner"
    S_LEARNER = "s_learner"


# ── device / backend ────────────────────────────────────────────────


class Device(StrEnum):
    """Target device for tensor execution."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class TensorBackendType(StrEnum):
    """Supported tensor backends."""

    NUMPY = "numpy"
    PYTORCH = "pytorch"


__all__ = [
    "DatasetFormat",
    "Device",
    "EncodeStrategy",
    "FeatureMap",
    "FeatureType",
    "ImputeStrategy",
    "ModelObjective",
    "ScaleStrategy",
    "Separator",
    "TargetColumn",
    "TaskType",
    "TensorBackendType",
    "UpliftMethod",
]
