"""mlcombine — Declarative Low-Code/No-Code framework for ML competitions.

Top-level package re-exports the public API from all layers:
``core`` (pipeline, tensor, types), ``data`` (download, COCO, joins),
``steps`` (type-detect, preprocess), and ``models`` (builder, uplift).

Exports are explicitly listed in ``__all__``; undocumented symbols are
considered private.

Importing this package does NOT trigger ``_registration`` — call
``mlcombine.load_builtins()`` before running a pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlcombine.core import (
        BaseAdapter,
        BaseStep,
        DataConfig,
        HandlingConfig,
        MLCombineConfig,
        ModelNode,
        NumpyAdapter,
        PipelineContext,
        PipelineEngine,
        TorchAdapter,
        TrainerConfig,
        UnifiedTensor,
    )


def load_builtins() -> None:
    """Load all built-in steps, providers, evaluators, and tensor adapters.

    Idempotent — safe to call multiple times.
    """
    import mlcombine._registration  # noqa: F401


# Lazy re-exports via __getattr__ (PEP 562) — no heavyweight imports at package load
_registered = False


def __getattr__(name: str) -> object:
    # Allow direct access to _registration submodule
    if name == "_registration":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r} — import mlcombine._registration instead")

    from mlcombine.core import __all__ as _core_all

    if name in _core_all:
        import mlcombine.core as _core

        return getattr(_core, name)

    if name in ("FeatureHandler", "ModelBuilder", "registry"):
        import mlcombine.core.registry as _reg

        return getattr(_reg, name)

    # Step classes
    _step_classes = {
        "EncodeScaleStep": "mlcombine.steps.preprocess",
        "ImputeStep": "mlcombine.steps.preprocess",
        "ReferenceJoinStep": "mlcombine.steps.reference_join",
        "TypeDetectStep": "mlcombine.steps.type_detect",
        "PrepareDatasetStep": "mlcombine.steps.prepare_dataset",
    }
    if name in _step_classes:
        import importlib

        mod = importlib.import_module(_step_classes[name])
        return getattr(mod, name)

    # Meta providers
    if name in ("SLearner", "TLearner"):
        import mlcombine.models.meta as _meta

        return getattr(_meta, name)

    # Data steps
    if name in ("GraphPairsJoinStep", "LogsJoinStep", "TabularJoinStep"):
        import mlcombine.data as _data

        return getattr(_data, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseAdapter",
    "BaseStep",
    "DataConfig",
    "EncodeScaleStep",
    "FeatureHandler",
    "GraphPairsJoinStep",
    "HandlingConfig",
    "ImputeStep",
    "LogsJoinStep",
    "MLCombineConfig",
    "ModelBuilder",
    "ModelNode",
    "NumpyAdapter",
    "PipelineContext",
    "PipelineEngine",
    "PrepareDatasetStep",
    "ReferenceJoinStep",
    "SLearner",
    "TLearner",
    "TabularJoinStep",
    "TorchAdapter",
    "TrainerConfig",
    "TypeDetectStep",
    "UnifiedTensor",
    "load_builtins",
    "registry",
]
