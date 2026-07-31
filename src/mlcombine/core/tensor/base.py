"""Abstractions for device-agnostic tensor operations via Bridge/Strategy pattern.

Provides BaseAdapter generic over T (tensor type), type aliases for index keys
and device literals, and a factory detector for resolving the right adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from mlcombine.core.enums import Device

type TensorIndexKey = int | slice | list[int] | list[bool] | np.ndarray | tuple[int | slice | np.ndarray, ...]


class BaseAdapter[T](ABC):
    """Abstract strategy for tensor backends. All math delegates here."""

    # ── element-wise arithmetic ────────────────────────────

    @abstractmethod
    def add(self, a: T, b: T) -> T:
        """Element-wise addition."""

    @abstractmethod
    def sub(self, a: T, b: T) -> T:
        """Element-wise subtraction."""

    @abstractmethod
    def mul(self, a: T, b: T) -> T:
        """Element-wise multiplication."""

    @abstractmethod
    def div(self, a: T, b: T | float | int) -> T:
        """Element-wise division."""

    @abstractmethod
    def neg(self, a: T) -> T:
        """Element-wise negation."""

    @abstractmethod
    def matmul(self, a: T, b: T) -> T:
        """Matrix multiplication."""

    @abstractmethod
    def pow(self, a: T, exponent: float | int) -> T:
        """Element-wise power."""

    # ── unary math ─────────────────────────────────────────

    @abstractmethod
    def abs(self, a: T) -> T:
        """Element-wise absolute value."""

    @abstractmethod
    def sqrt(self, a: T) -> T:
        """Element-wise square root."""

    @abstractmethod
    def exp(self, a: T) -> T:
        """Element-wise exponential."""

    @abstractmethod
    def log(self, a: T) -> T:
        """Element-wise natural logarithm."""

    @abstractmethod
    def clip(self, a: T, min_val: float | int | None, max_val: float | int | None) -> T:
        """Clip values to a range."""

    # ── reductions ─────────────────────────────────────────

    @abstractmethod
    def sum(self, a: T, dim: int | None = None, keepdim: bool = False) -> T:
        """Sum along a dimension."""

    @abstractmethod
    def mean(self, a: T, dim: int | None = None, keepdim: bool = False) -> T:
        """Mean along a dimension."""

    @abstractmethod
    def min(self, a: T, dim: int | None = None, keepdim: bool = False) -> T:
        """Min along a dimension."""

    @abstractmethod
    def max(self, a: T, dim: int | None = None, keepdim: bool = False) -> T:
        """Max along a dimension."""

    @abstractmethod
    def argmin(self, a: T, dim: int | None = None, keepdim: bool = False) -> T:
        """Index of minimum along a dimension."""

    @abstractmethod
    def argmax(self, a: T, dim: int | None = None, keepdim: bool = False) -> T:
        """Index of maximum along a dimension."""

    # ── shape operations ───────────────────────────────────

    @abstractmethod
    def reshape(self, a: T, shape: tuple[int, ...]) -> T:
        """Reshape tensor."""

    @abstractmethod
    def flatten(self, a: T, start_dim: int = 0, end_dim: int = -1) -> T:
        """Flatten a contiguous range of dimensions."""

    @abstractmethod
    def transpose(self, a: T, axes: tuple[int, ...] | None = None) -> T:
        """Permute dimensions. None = reverse all."""

    @abstractmethod
    def squeeze(self, a: T, dim: int | None = None) -> T:
        """Remove dimensions of size 1."""

    @abstractmethod
    def unsqueeze(self, a: T, dim: int) -> T:
        """Add a dimension of size 1."""

    # ── indexing / slicing ─────────────────────────────────

    @abstractmethod
    def get_item(self, data: T, key: TensorIndexKey) -> T:
        """Index into tensor."""

    # ── conversion ─────────────────────────────────────────

    @abstractmethod
    def convert(self, data: object) -> T:
        """Coerce arbitrary data to this backend's tensor type."""

    @abstractmethod
    def to_numpy(self, data: T) -> np.ndarray:
        """Convert to NumPy array (always detached, CPU)."""

    @abstractmethod
    def tolist(self, data: T) -> list[Any]:
        """Convert to Python list."""

    # ── utility ────────────────────────────────────────────

    @abstractmethod
    def item(self, data: T) -> float | int:
        """Return scalar value (tensor must be 0-dim or 1-element)."""

    @abstractmethod
    def copy(self, data: T) -> T:
        """Return a deep copy of the tensor."""

    @abstractmethod
    def ndim(self, data: T) -> int:
        """Number of dimensions."""

    @abstractmethod
    def shape_of(self, data: T) -> tuple[int, ...]:
        """Return tensor shape."""

    @abstractmethod
    def dtype_of(self, data: T) -> Any:
        """Return element dtype."""

    @abstractmethod
    def len_of(self, data: T) -> int:
        """Length of the first dimension."""

    @abstractmethod
    def device_of(self, data: T) -> Device:
        """Return the device the tensor lives on."""

    # ── creation ───────────────────────────────────────────

    @abstractmethod
    def concatenate(self, tensors: list[T], dim: int = 0) -> T:
        """Join sequence of tensors along existing dimension."""

    @abstractmethod
    def to_device(self, data: T, device: Device) -> T:
        """Move tensor to target device."""

    @abstractmethod
    def zeros_like(self, data: T) -> T:
        """Return zero-filled tensor with same shape/dtype."""

    @abstractmethod
    def ones_like(self, data: T) -> T:
        """Return one-filled tensor with same shape/dtype."""
