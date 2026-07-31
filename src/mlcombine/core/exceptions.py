"""mlcombine exception hierarchy."""

from __future__ import annotations


class MLCombineException(Exception):
    """Base exception for all mlcombine framework errors."""


class ConfigurationError(MLCombineException, ValueError):
    """Raised when required configuration is missing or invalid."""


class DatasetNotFoundError(MLCombineException, FileNotFoundError):
    """Raised when a dataset file does not exist."""


class EmptyDatasetError(MLCombineException, ValueError):
    """Raised when a loaded dataset is empty."""


class RowExplosionError(MLCombineException, ValueError):
    """Raised when a join unexpectedly multiplies rows."""


class UnsupportedBackendError(MLCombineException, ValueError):
    """Raised when a requested backend has no registered provider."""


__all__ = [
    "ConfigurationError",
    "DatasetNotFoundError",
    "EmptyDatasetError",
    "MLCombineException",
    "RowExplosionError",
    "UnsupportedBackendError",
]
