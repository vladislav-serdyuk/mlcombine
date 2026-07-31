"""Core layer — pipeline orchestration, tensor abstraction, and type system.

Responsibility: Provide ``PipelineEngine``, ``BaseStep``, ``UnifiedTensor``,
and all Pydantic config schemas / enums / exceptions via the ``types``
submodule.

Domain constraints:
- ``BaseStep`` is the sole step interface; every pipeline entity must
  inherit it.
- All configuration models enforce ``ConfigDict(extra="forbid")``.
- ``PipelineEngine.run_all()`` requires an explicit ``PipelineContext``.
"""

from mlcombine.core.pipeline import PipelineContext, PipelineEngine
from mlcombine.core.protocols import EvaluatorProtocol
from mlcombine.core.tensor import TORCH_AVAILABLE, BaseAdapter, NumpyAdapter, TorchAdapter, UnifiedTensor
from mlcombine.core.types import (
    BaseStep,
    DataConfig,
    EncodeStrategy,
    FeatureMap,
    HandlingConfig,
    ImputeStrategy,
    MLCombineConfig,
    ModelNode,
    PipelineArtifacts,
    PipelineData,
    ScaleStrategy,
    TargetColumn,
    TaskType,
    TensorBackendType,
    TrainerConfig,
)

__all__ = [
    "TORCH_AVAILABLE",
    "BaseAdapter",
    "BaseStep",
    "DataConfig",
    "EncodeStrategy",
    "EvaluatorProtocol",
    "FeatureMap",
    "HandlingConfig",
    "ImputeStrategy",
    "MLCombineConfig",
    "ModelNode",
    "NumpyAdapter",
    "PipelineArtifacts",
    "PipelineContext",
    "PipelineData",
    "PipelineEngine",
    "ScaleStrategy",
    "TargetColumn",
    "TaskType",
    "TensorBackendType",
    "TorchAdapter",
    "TrainerConfig",
    "UnifiedTensor",
]
