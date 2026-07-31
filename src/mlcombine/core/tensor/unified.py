"""High-level ``UnifiedTensor`` — delegates all operations to a ``BaseAdapter`` strategy.

Fully generic on the underlying array type; depends only on ``BaseAdapter``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Self, cast

import numpy as np

from mlcombine.core.tensor.base import BaseAdapter, TensorIndexKey
from mlcombine.core.enums import Device, TensorBackendType
from mlcombine.core.registry import registry


@lru_cache(maxsize=None)
def _adapter_cls(backend: str) -> type | None:
    """Cached lookup of adapter class for *backend*."""
    return registry.get_tensor_adapter(backend)


def _resolve_adapter(backend: TensorBackendType | str = "numpy") -> BaseAdapter[Any]:
    adapter_cls = _adapter_cls(backend)
    if adapter_cls is None:
        known = ", ".join(sorted(registry.tensor_adapters))
        msg = f"Unknown tensor backend {backend!r}. Available backends: {known}"
        raise ValueError(msg)
    return cast(BaseAdapter[Any], adapter_cls())


class UnifiedTensor[T]:
    """Lightweight tensor wrapper delegating all operations to a ``BaseAdapter[T]``.

    Generic parameter ``T`` is the concrete array type
    (``np.ndarray`` or ``torch.Tensor``).
    """

    def __init__(
        self,
        data: object,
        adapter: BaseAdapter[T] | None = None,
        device: Device = Device.CPU,
        backend: TensorBackendType | str = "numpy",
    ) -> None:
        """Wrap *data* with a given or auto-resolved adapter.

        Args:
            data: Raw input to convert.
            adapter: Pre-configured adapter; auto-detected if ``None``.
            device: Target device (``Device.CPU``, ``Device.CUDA``, ...).
            backend: Preferred tensor backend when *adapter* is not given.

        Side Effects:
            - Assigns ``self._adapter``, ``self._device``, ``self._data``.

        """
        if adapter is None:
            adapter = _resolve_adapter(backend)
        self._adapter: BaseAdapter[T] = adapter
        self._device: Device = device
        self._data: T = self._adapter.to_device(self._adapter.convert(data), device)

    # ── helpers ──────────────────────────────────────────────

    def _wrap(self, data: T) -> Self:
        """Create a new UnifiedTensor with the same adapter and device."""
        return self.__class__(data, self._adapter, self._device)

    def _scalar(self, other: float | int) -> T:
        """Convert a Python scalar to the backend tensor type."""
        return self._adapter.convert(other)

    # ── arithmetic operators ──────────────────────────────

    def __add__(self, other: Self | float | int) -> Self:
        """Element-wise addition."""
        b: T = other._data if isinstance(other, UnifiedTensor) else self._scalar(other)
        return self._wrap(self._adapter.add(self._data, b))

    def __sub__(self, other: Self | float | int) -> Self:
        """Element-wise subtraction."""
        b: T = other._data if isinstance(other, UnifiedTensor) else self._scalar(other)
        return self._wrap(self._adapter.sub(self._data, b))

    def __mul__(self, other: Self | float | int) -> Self:
        """Element-wise multiplication."""
        b: T = other._data if isinstance(other, UnifiedTensor) else self._scalar(other)
        return self._wrap(self._adapter.mul(self._data, b))

    def __truediv__(self, other: Self | float | int) -> Self:
        """Element-wise division."""
        b: T = other._data if isinstance(other, UnifiedTensor) else self._scalar(other)
        return self._wrap(self._adapter.div(self._data, b))

    def __neg__(self) -> Self:
        """Element-wise negation."""
        return self._wrap(self._adapter.neg(self._data))

    def __matmul__(self, other: Self) -> Self:
        """Matrix multiplication."""
        return self._wrap(self._adapter.matmul(self._data, other._data))

    def __pow__(self, exponent: float | int) -> Self:
        """Element-wise power."""
        return self._wrap(self._adapter.pow(self._data, exponent))

    def __abs__(self) -> Self:
        """Element-wise absolute value."""
        return self._wrap(self._adapter.abs(self._data))

    # ── unary math ─────────────────────────────────────────

    def abs(self) -> Self:
        """Element-wise absolute value."""
        return self._wrap(self._adapter.abs(self._data))

    def sqrt(self) -> Self:
        """Element-wise square root."""
        return self._wrap(self._adapter.sqrt(self._data))

    def exp(self) -> Self:
        """Element-wise exponential."""
        return self._wrap(self._adapter.exp(self._data))

    def log(self) -> Self:
        """Element-wise natural logarithm."""
        return self._wrap(self._adapter.log(self._data))

    def clip(self, min_val: float | int | None = None, max_val: float | int | None = None) -> Self:
        """Clip values to a range."""
        return self._wrap(self._adapter.clip(self._data, min_val, max_val))

    # ── reductions ─────────────────────────────────────────

    def sum(self, dim: int | None = None, keepdim: bool = False) -> Self:
        """Sum along a dimension."""
        return self._wrap(self._adapter.sum(self._data, dim, keepdim))

    def mean(self, dim: int | None = None, keepdim: bool = False) -> Self:
        """Mean along a dimension."""
        return self._wrap(self._adapter.mean(self._data, dim, keepdim))

    def min(self, dim: int | None = None, keepdim: bool = False) -> Self:
        """Min along a dimension."""
        return self._wrap(self._adapter.min(self._data, dim, keepdim))

    def max(self, dim: int | None = None, keepdim: bool = False) -> Self:
        """Max along a dimension."""
        return self._wrap(self._adapter.max(self._data, dim, keepdim))

    def argmin(self, dim: int | None = None, keepdim: bool = False) -> Self:
        """Index of minimum along a dimension."""
        return self._wrap(self._adapter.argmin(self._data, dim, keepdim))

    def argmax(self, dim: int | None = None, keepdim: bool = False) -> Self:
        """Index of maximum along a dimension."""
        return self._wrap(self._adapter.argmax(self._data, dim, keepdim))

    # ── shape operations ───────────────────────────────────

    def reshape(self, *shape: int) -> Self:
        """Reshape tensor."""
        return self._wrap(self._adapter.reshape(self._data, shape))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Self:
        """Flatten a contiguous range of dimensions."""
        return self._wrap(self._adapter.flatten(self._data, start_dim, end_dim))

    def transpose(self, axes: tuple[int, ...] | None = None) -> Self:
        """Permute dimensions. None = reverse all."""
        return self._wrap(self._adapter.transpose(self._data, axes))

    def squeeze(self, dim: int | None = None) -> Self:
        """Remove dimensions of size 1."""
        return self._wrap(self._adapter.squeeze(self._data, dim))

    def unsqueeze(self, dim: int) -> Self:
        """Add a dimension of size 1."""
        return self._wrap(self._adapter.unsqueeze(self._data, dim))

    @property
    def T(self) -> Self:
        """Transpose (reverse all dimensions)."""
        return self._wrap(self._adapter.transpose(self._data))

    # ── indexing / slicing ─────────────────────────────────

    def __getitem__(self, key: TensorIndexKey) -> Self:
        """Index into tensor."""
        return self._wrap(self._adapter.get_item(self._data, key))

    # ── conversion ─────────────────────────────────────────

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """NumPy protocol — enables np.asarray(tensor) and np.array(tensor)."""
        arr = self._adapter.to_numpy(self._data)
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy:
            arr = arr.copy()
        return arr

    def numpy(self) -> np.ndarray:
        """Return tensor data as a NumPy array (detached, CPU)."""
        return self._adapter.to_numpy(self._data)

    def tolist(self) -> list[Any]:
        """Convert to Python list."""
        return self._adapter.tolist(self._data)

    # ── utility ────────────────────────────────────────────

    def item(self) -> float | int:
        """Return scalar value (tensor must be 0-dim or 1-element)."""
        return self._adapter.item(self._data)

    def copy(self) -> Self:
        """Return a deep copy with the same adapter and device."""
        return self.__class__(self._adapter.copy(self._data), self._adapter, self._device)

    def __len__(self) -> int:
        """Length of the first dimension."""
        return self._adapter.len_of(self._data)

    def __repr__(self) -> str:
        """String representation."""
        return f"UnifiedTensor({self._adapter.to_numpy(self._data)}, device={self._device}, backend={type(self._adapter).__name__})"

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._adapter.ndim(self._data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return tensor shape."""
        return self._adapter.shape_of(self._data)

    @property
    def dtype(self) -> Any:
        """Return element dtype."""
        return self._adapter.dtype_of(self._data)

    @property
    def device(self) -> Device:
        """Return the device the tensor lives on."""
        return self._adapter.device_of(self._data)
