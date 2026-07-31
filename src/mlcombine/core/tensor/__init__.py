"""Tensor abstraction package — Bridge/Strategy pattern for array backends."""

from mlcombine.core.tensor.backend_numpy import NumpyAdapter
from mlcombine.core.tensor.backend_torch import TORCH_AVAILABLE
from mlcombine.core.tensor.base import BaseAdapter, TensorIndexKey
from mlcombine.core.tensor.unified import UnifiedTensor
from mlcombine.core.enums import Device

__all__ = [
    "TORCH_AVAILABLE",
    "BaseAdapter",
    "Device",
    "NumpyAdapter",
    "TensorIndexKey",
    "TorchAdapter",
    "UnifiedTensor",
]

if TORCH_AVAILABLE:
    from mlcombine.core.tensor.backend_torch import TorchAdapter  # noqa: F401
else:
    TorchAdapter = None  # type: ignore[assignment,misc]  # noqa: F401
