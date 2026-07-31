"""Concrete ``TorchAdapter`` — bridges ``torch.Tensor`` into ``BaseAdapter[torch.Tensor]``.

``ImportError`` is raised at construction time when PyTorch is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcombine.core.tensor.base import BaseAdapter, TensorIndexKey
from mlcombine.core.enums import Device
from mlcombine.core.registry import registry

try:
    import torch

    TORCH_AVAILABLE = True

    @registry.tensor_adapter("pytorch")
    class TorchAdapter(BaseAdapter[torch.Tensor]):
        """Adapter wrapping PyTorch tensors as the tensor backend."""

        # ── element-wise arithmetic ────────────────────────────

        def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Element-wise addition."""
            return a + b

        def sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Element-wise subtraction."""
            return a - b

        def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Element-wise multiplication."""
            return a * b

        def div(self, a: torch.Tensor, b: torch.Tensor | float | int) -> torch.Tensor:
            """Element-wise division."""
            return a / b

        def neg(self, a: torch.Tensor) -> torch.Tensor:
            """Element-wise negation."""
            return -a

        def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Matrix multiplication."""
            return a @ b

        def pow(self, a: torch.Tensor, exponent: float | int) -> torch.Tensor:
            """Element-wise power."""
            return a**exponent

        # ── unary math ─────────────────────────────────────────

        def abs(self, a: torch.Tensor) -> torch.Tensor:
            """Element-wise absolute value."""
            return torch.abs(a)

        def sqrt(self, a: torch.Tensor) -> torch.Tensor:
            """Element-wise square root."""
            return torch.sqrt(a)

        def exp(self, a: torch.Tensor) -> torch.Tensor:
            """Element-wise exponential."""
            return torch.exp(a)

        def log(self, a: torch.Tensor) -> torch.Tensor:
            """Element-wise natural logarithm."""
            return torch.log(a)

        def clip(self, a: torch.Tensor, min_val: float | int | None, max_val: float | int | None) -> torch.Tensor:
            """Clip values to a range."""
            return torch.clip(a, min_val, max_val)

        # ── reductions ─────────────────────────────────────────

        def sum(self, a: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
            """Sum along a dimension."""
            if dim is None:
                return a.sum()
            return a.sum(dim=dim, keepdim=keepdim)

        def mean(self, a: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
            """Mean along a dimension."""
            if dim is None:
                return a.mean()
            return a.mean(dim=dim, keepdim=keepdim)

        def min(self, a: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
            """Min along a dimension."""
            if dim is None:
                return a.min()
            return a.min(dim=dim, keepdim=keepdim).values

        def max(self, a: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
            """Max along a dimension."""
            if dim is None:
                return a.max()
            return a.max(dim=dim, keepdim=keepdim).values

        def argmin(self, a: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
            """Index of minimum along a dimension."""
            if dim is None:
                return a.argmin()
            return a.argmin(dim=dim, keepdim=keepdim)

        def argmax(self, a: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
            """Index of maximum along a dimension."""
            if dim is None:
                return a.argmax()
            return a.argmax(dim=dim, keepdim=keepdim)

        # ── shape operations ───────────────────────────────────

        def reshape(self, a: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
            """Reshape tensor."""
            return a.reshape(shape)

        def flatten(self, a: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
            """Flatten a contiguous range of dimensions."""
            return a.flatten(start_dim=start_dim, end_dim=end_dim)

        def transpose(self, a: torch.Tensor, axes: tuple[int, ...] | None = None) -> torch.Tensor:
            """Permute dimensions. None = reverse all."""
            if axes is None:
                return a.T
            return a.permute(*axes)

        def squeeze(self, a: torch.Tensor, dim: int | None = None) -> torch.Tensor:
            """Remove dimensions of size 1."""
            if dim is None:
                return a.squeeze()
            return a.squeeze(dim=dim)

        def unsqueeze(self, a: torch.Tensor, dim: int) -> torch.Tensor:
            """Add a dimension of size 1."""
            return a.unsqueeze(dim)

        # ── indexing / slicing ─────────────────────────────────

        def get_item(self, data: torch.Tensor, key: TensorIndexKey) -> torch.Tensor:
            """Index into tensor."""
            return data[key]

        # ── conversion ─────────────────────────────────────────

        def convert(self, data: object) -> torch.Tensor:
            """Convert arbitrary data to torch.Tensor."""
            if torch.is_tensor(data):
                return data
            return torch.tensor(np.array(data))

        def to_numpy(self, data: torch.Tensor) -> np.ndarray:
            """Convert tensor to NumPy array (detached, CPU)."""
            return data.detach().cpu().numpy()

        def tolist(self, data: torch.Tensor) -> list[Any]:
            """Convert to Python list."""
            return data.tolist()

        # ── utility ────────────────────────────────────────────

        def item(self, data: torch.Tensor) -> float | int:
            """Return scalar value."""
            return data.item()

        def copy(self, data: torch.Tensor) -> torch.Tensor:
            """Return a copy of the tensor."""
            return data.clone()

        def ndim(self, data: torch.Tensor) -> int:
            """Number of dimensions."""
            return data.ndim

        def shape_of(self, data: torch.Tensor) -> tuple[int, ...]:
            """Return tensor shape."""
            return tuple(data.shape)

        def dtype_of(self, data: torch.Tensor) -> torch.dtype:
            """Return element dtype."""
            return data.dtype

        def len_of(self, data: torch.Tensor) -> int:
            """Length of the first dimension."""
            return len(data)

        def device_of(self, data: torch.Tensor) -> Device:
            """Return the device the tensor lives on."""
            device_str = data.device.type
            if device_str == "cpu":
                return Device.CPU
            if device_str == "cuda":
                return Device.CUDA
            if device_str == "mps":
                return Device.MPS
            return Device.CPU

        # ── creation ───────────────────────────────────────────

        def concatenate(self, tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
            """Concatenate tensors along a dimension."""
            return torch.cat(tensors, dim=dim)

        def to_device(self, data: torch.Tensor, device: Device) -> torch.Tensor:
            """Move tensor to target device."""
            torch_device = device.value
            return data.to(torch_device)

        def zeros_like(self, data: torch.Tensor) -> torch.Tensor:
            """Create zero tensor with same shape."""
            return torch.zeros_like(data)

        def ones_like(self, data: torch.Tensor) -> torch.Tensor:
            """Create ones tensor with same shape."""
            return torch.ones_like(data)

except ImportError:
    TORCH_AVAILABLE = False
