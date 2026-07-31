"""Concrete ``NumpyAdapter`` — bridges ``np.ndarray`` into ``BaseAdapter[np.ndarray]``.

The only universal fallback adapter — no external dependencies required.
Conversion from torch tensors is handled by checking ``__class__``, not by
importing torch at the module level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcombine.core.registry import registry

from mlcombine.core.tensor.base import BaseAdapter, TensorIndexKey
from mlcombine.core.enums import Device


@registry.tensor_adapter("numpy")
class NumpyAdapter(BaseAdapter[np.ndarray]):
    """Adapter wrapping NumPy arrays as the tensor backend."""

    # ── element-wise arithmetic ────────────────────────────

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise addition."""
        return a + b  # type: ignore[no-any-return]

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise subtraction."""
        return a - b  # type: ignore[no-any-return]

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication."""
        return a * b  # type: ignore[no-any-return]

    def div(self, a: np.ndarray, b: np.ndarray | float | int) -> np.ndarray:
        """Element-wise division."""
        return a / b

    def neg(self, a: np.ndarray) -> np.ndarray:
        """Element-wise negation."""
        return -a

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication."""
        return a @ b  # type: ignore[no-any-return]

    def pow(self, a: np.ndarray, exponent: float | int) -> np.ndarray:
        """Element-wise power."""
        return a**exponent

    # ── unary math ─────────────────────────────────────────

    def abs(self, a: np.ndarray) -> np.ndarray:
        """Element-wise absolute value."""
        return np.abs(a)  # type: ignore[no-any-return]

    def sqrt(self, a: np.ndarray) -> np.ndarray:
        """Element-wise square root."""
        return np.sqrt(a)  # type: ignore[no-any-return]

    def exp(self, a: np.ndarray) -> np.ndarray:
        """Element-wise exponential."""
        return np.exp(a)  # type: ignore[no-any-return]

    def log(self, a: np.ndarray) -> np.ndarray:
        """Element-wise natural logarithm."""
        return np.log(a)  # type: ignore[no-any-return]

    def clip(self, a: np.ndarray, min_val: float | int | None, max_val: float | int | None) -> np.ndarray:
        """Clip values to a range."""
        return np.clip(a, min_val, max_val)  # type: ignore[no-any-return]

    # ── reductions ─────────────────────────────────────────

    def sum(self, a: np.ndarray, dim: int | None = None, keepdim: bool = False) -> np.ndarray:
        """Sum along a dimension."""
        return np.sum(a, axis=dim, keepdims=keepdim)  # type: ignore[no-any-return]

    def mean(self, a: np.ndarray, dim: int | None = None, keepdim: bool = False) -> np.ndarray:
        """Mean along a dimension."""
        return np.mean(a, axis=dim, keepdims=keepdim)  # type: ignore[no-any-return]

    def min(self, a: np.ndarray, dim: int | None = None, keepdim: bool = False) -> np.ndarray:
        """Min along a dimension."""
        return np.amin(a, axis=dim, keepdims=keepdim)  # type: ignore[no-any-return]

    def max(self, a: np.ndarray, dim: int | None = None, keepdim: bool = False) -> np.ndarray:
        """Max along a dimension."""
        return np.amax(a, axis=dim, keepdims=keepdim)  # type: ignore[no-any-return]

    def argmin(self, a: np.ndarray, dim: int | None = None, keepdim: bool = False) -> np.ndarray:
        """Index of minimum along a dimension."""
        if dim is None:
            return np.array(np.argmin(a))
        arr = np.argmin(a, axis=dim)
        if keepdim:
            arr = np.expand_dims(arr, axis=dim)
        return arr  # type: ignore[no-any-return]

    def argmax(self, a: np.ndarray, dim: int | None = None, keepdim: bool = False) -> np.ndarray:
        """Index of maximum along a dimension."""
        if dim is None:
            return np.array(np.argmax(a))
        arr = np.argmax(a, axis=dim)
        if keepdim:
            arr = np.expand_dims(arr, axis=dim)
        return arr  # type: ignore[no-any-return]

    # ── shape operations ───────────────────────────────────

    def reshape(self, a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Reshape tensor."""
        return a.reshape(shape)

    def flatten(self, a: np.ndarray, start_dim: int = 0, end_dim: int = -1) -> np.ndarray:
        """Flatten a contiguous range of dimensions."""
        if end_dim == -1:
            end_dim = a.ndim - 1
        shape = a.shape
        new_shape = shape[:start_dim] + (np.prod(shape[start_dim : end_dim + 1]),) + shape[end_dim + 1 :]
        return a.reshape(int(new_shape[0]) if len(new_shape) == 1 else new_shape)

    def transpose(self, a: np.ndarray, axes: tuple[int, ...] | None = None) -> np.ndarray:
        """Permute dimensions. None = reverse all."""
        return np.transpose(a, axes)

    def squeeze(self, a: np.ndarray, dim: int | None = None) -> np.ndarray:
        """Remove dimensions of size 1."""
        if dim is None:
            return np.squeeze(a)
        return np.squeeze(a, axis=dim)

    def unsqueeze(self, a: np.ndarray, dim: int) -> np.ndarray:
        """Add a dimension of size 1."""
        return np.expand_dims(a, axis=dim)

    # ── indexing / slicing ─────────────────────────────────

    def get_item(self, data: np.ndarray, key: TensorIndexKey) -> np.ndarray:
        """Index into array."""
        return data[key]

    # ── conversion ─────────────────────────────────────────

    def convert(self, data: object) -> np.ndarray:
        """Convert arbitrary data to np.ndarray.

        Handles:
        - Native np.ndarray (returned as-is).
        - Objects with a .detach() method (e.g. torch tensors).
        - Anything else via np.array().
        """
        if isinstance(data, np.ndarray):
            return data
        try:
            return data.detach().cpu().numpy()  # type: ignore[no-any-return,attr-defined]
        except AttributeError:
            pass
        try:
            return np.array(data.detach())  # type: ignore[attr-defined]
        except AttributeError:
            pass
        return np.array(data)

    def to_numpy(self, data: np.ndarray) -> np.ndarray:
        """Identity — *data* is already a NumPy array."""
        return data

    def tolist(self, data: np.ndarray) -> list[Any]:
        """Convert to Python list."""
        return data.tolist()  # type: ignore[no-any-return]

    # ── utility ────────────────────────────────────────────

    def item(self, data: np.ndarray) -> float | int:
        """Return scalar value."""
        return data.item()  # type: ignore[no-any-return]

    def copy(self, data: np.ndarray) -> np.ndarray:
        """Return a copy of the array."""
        return data.copy()

    def ndim(self, data: np.ndarray) -> int:
        """Number of dimensions."""
        return data.ndim

    def shape_of(self, data: np.ndarray) -> tuple[int, ...]:
        """Return array shape."""
        return tuple(data.shape)

    def dtype_of(self, data: np.ndarray) -> np.dtype[Any]:
        """Return element dtype."""
        return data.dtype

    def len_of(self, data: np.ndarray) -> int:
        """Length of the first dimension."""
        return len(data)

    def device_of(self, data: np.ndarray) -> Device:
        """NumPy arrays are CPU-only."""
        return Device.CPU

    # ── creation ───────────────────────────────────────────

    def concatenate(self, tensors: list[np.ndarray], dim: int = 0) -> np.ndarray:
        """Concatenate arrays along an axis."""
        return np.concatenate(tensors, axis=dim)

    def to_device(self, data: np.ndarray, device: Device) -> np.ndarray:
        """No-op — NumPy arrays are CPU-only."""
        if device != Device.CPU:
            raise RuntimeError(f"NumpyAdapter does not support device '{device}'. Use TorchAdapter for GPU execution.")
        return data

    def zeros_like(self, data: np.ndarray) -> np.ndarray:
        """Create zero array with same shape."""
        return np.zeros_like(data)

    def ones_like(self, data: np.ndarray) -> np.ndarray:
        """Create ones array with same shape."""
        return np.ones_like(data)
