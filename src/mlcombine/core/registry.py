"""Central extension registry — single point for registering plugins, handlers, and providers.

Usage::

    from mlcombine.core.registry import registry

    # register a custom pipeline step
    @registry.step("my_step")
    class MyStep(BaseStep):
        ...

    # register a tensor adapter (e.g. for JAX)
    @registry.tensor_adapter("jax")
    class JaxAdapter(BaseAdapter[JaxArray]):
        ...

    # register a model provider
    registry.model_provider("my_backend", MyProvider())

    # register a feature type handler (e.g. "audio")
    @registry.feature_handler("audio")
    class AudioHandler(FeatureHandler):
        def detect(self, series):
            return series.astype(str).str.contains(r'\\.(mp3|wav)$').mean() > 0.5
        def preprocess(self, series, config=None):
            return series
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, overload

import pandas as pd

from mlcombine.core.evaluator import BaseArchitectureValidator
from mlcombine.core.enums import TensorBackendType
from mlcombine.core.protocols import (
    ActivationProtocol,
    LayerBuilderProtocol,
    SupportedModel,
)

logger = logging.getLogger(__name__)


# ── FeatureHandler ABC ────────────────────────────────────────────────────


class FeatureHandler(ABC):
    """Abstract base for custom column-type detection and preprocessing.

    Register a subclass via ``registry.feature_handler(type_name)``.
    The *type_name* is used as the detected type string for the column.
    """

    @abstractmethod
    def detect(self, series: pd.Series) -> bool:
        """Return ``True`` if this handler applies to *series*."""
        ...

    def preprocess(self, series: pd.Series, config: dict[str, Any] | None = None) -> pd.Series:
        """Transform *series* before passing it to the model (default: no-op)."""
        return series


# ── StepRegistry ─────────────────────────────────────────────────────────


class StepRegistry:
    """Manages pipeline-step registration with optional ``before``/``after`` hints."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str | None = None,
        *,
        before: str | None = None,
        after: str | None = None,
        package: str | None = None,
        module: str | None = None,
    ) -> Callable[[type], type]:
        """Decorator to register a :class:`BaseStep` subclass."""

        def _wrapper(cls: type) -> type:
            n = name or cls.__name__
            meta: dict[str, Any] = {"class": cls, "before": before, "after": after}
            if package:
                meta["package"] = package
            if module:
                meta["module"] = module
            self._items[n] = meta
            hints = []
            if before:
                hints.append(f"before={before}")
            if after:
                hints.append(f"after={after}")
            if hints:
                logger.info("Step %-30s registered (%s)", n, ", ".join(hints))
            else:
                logger.info("Step %-30s registered", n)
            return cls

        return _wrapper

    def get(self, name: str) -> type | None:
        """Look up a registered step by name."""
        meta = self._items.get(name)
        return meta["class"] if meta else None

    def get_meta(self, name: str) -> dict[str, Any] | None:
        """Return full metadata dict for the named step (or ``None``)."""
        return self._items.get(name)

    @property
    def names(self) -> list[str]:
        """Return a list of all registered step names."""
        return list(self._items)

    @property
    def data(self) -> dict[str, dict[str, Any]]:
        """Expose raw data (for ``clear()`` and testing)."""
        return self._items

    def clear(self) -> None:
        self._items.clear()


# ── TensorAdapterRegistry ────────────────────────────────────────────────


class TensorAdapterRegistry:
    """Manages tensor-backend adapter registrations."""

    def __init__(self) -> None:
        self._items: dict[str, type] = {}

    def register(self, backend: TensorBackendType | str) -> Callable[[type], type]:
        """Decorator to register a :class:`BaseAdapter` subclass for *backend*."""

        def _wrapper(cls: type) -> type:
            self._items[backend] = cls
            logger.info("TensorAdapter %-25s registered", f"{backend}")
            return cls

        return _wrapper

    def get(self, backend: TensorBackendType | str) -> type | None:
        return self._items.get(backend)

    @property
    def adapters(self) -> dict[str, type]:
        return dict(self._items)

    @property
    def data(self) -> dict[str, type]:
        """Expose raw data (for ``clear()`` and testing)."""
        return self._items

    def clear(self) -> None:
        self._items.clear()


# ── ModelProviderRegistry ────────────────────────────────────────────────


class ModelProviderRegistry:
    """Manages model-provider function registrations with install metadata."""

    def __init__(self) -> None:
        self._providers: dict[str, Callable[..., SupportedModel]] = {}
        self._meta: dict[str, dict[str, str | None]] = {}

    def register(
        self,
        name: str,
        *,
        package: str | None = None,
        module: str | None = None,
    ) -> Callable[[Callable[..., SupportedModel]], Callable[..., SupportedModel]]:
        """Decorator to register a model-provider function.

        Args:
            name: Provider name (e.g. ``"catboost"``).
            package: PyPI package name for auto-install.
            module: Provider module to reload after install.
        """
        meta: dict[str, str | None] = {"package": package, "module": module}

        def _wrapper(fn: Callable[..., SupportedModel]) -> Callable[..., SupportedModel]:
            self._providers[name] = fn
            self._meta[name] = meta
            logger.info("ModelProvider %-25s registered (package=%s, module=%s)", name, package, module)
            return fn

        return _wrapper

    def get_provider(self, name: str) -> Callable[..., SupportedModel] | None:
        return self._providers.get(name)

    def get_meta(self, name: str) -> dict[str, str | None] | None:
        """Return install metadata for a registered provider (or ``None``)."""
        return self._meta.get(name)

    @property
    def providers(self) -> dict[str, Callable[..., SupportedModel]]:
        return dict(self._providers)

    @property
    def providers_data(self) -> dict[str, Callable[..., SupportedModel]]:
        """Expose raw providers dict (for ``clear()`` and testing)."""
        return self._providers

    @property
    def meta_data(self) -> dict[str, dict[str, str | None]]:
        """Expose raw meta dict (for ``clear()`` and testing)."""
        return self._meta

    def clear(self) -> None:
        self._providers.clear()
        self._meta.clear()


# ── ArchitectureValidatorRegistry ──────────────────────────────────────────


class ArchitectureValidatorRegistry:
    """Manages architecture validator class registrations with install metadata."""

    def __init__(self) -> None:
        self._validators: dict[str, type[BaseArchitectureValidator]] = {}
        self._meta: dict[str, dict[str, str | None]] = {}

    def register(
        self,
        name: str,
        *,
        package: str | None = None,
        module: str | None = None,
    ) -> Callable[[type[BaseArchitectureValidator]], type[BaseArchitectureValidator]]:
        """Decorator to register an architecture validator class.

        Args:
            name: Validator name (e.g. ``"cv"``, ``"holdout"``).
            package: PyPI package name for auto-install.
            module: Validator module to reload after install.
        """
        meta: dict[str, str | None] = {"package": package, "module": module}

        def _wrapper(cls: type[BaseArchitectureValidator]) -> type[BaseArchitectureValidator]:
            self._validators[name] = cls
            self._meta[name] = meta
            logger.info("ArchitectureValidator %-25s registered (package=%s, module=%s)", name, package, module)
            return cls

        return _wrapper

    def get_validator(self, name: str) -> type[BaseArchitectureValidator] | None:
        return self._validators.get(name)

    def get_meta(self, name: str) -> dict[str, str | None] | None:
        """Return install metadata for a registered validator (or ``None``)."""
        return self._meta.get(name)

    @property
    def validators(self) -> dict[str, type[BaseArchitectureValidator]]:
        return dict(self._validators)

    @property
    def validator_names(self) -> list[str]:
        return list(self._validators)

    def clear(self) -> None:
        self._validators.clear()
        self._meta.clear()


# ── MetricRegistry ─────────────────────────────────────────────────────────


class MetricRegistry:
    """Registry for named metric functions with default keyword arguments.

    Usage::

        # decorator
        @registry.metric("f1", average="weighted")
        def f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average="weighted")

        # direct register
        registry.metric("rmse", root_mean_squared_error)

    Lookup::

        fn, kwargs = registry.metric.get("f1")
        value = fn(y_true, y_pred, **kwargs)
    """

    def __init__(self) -> None:
        self._metrics: dict[str, tuple[Callable[..., float], dict[str, object]]] = {}

    def __call__(self, name: str, **default_kwargs: object) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Decorator: register a callable as a named metric.

        Parameters in *default_kwargs* are passed to the callable at
        evaluation time.
        """

        def _wrapper(fn: Callable[..., float]) -> Callable[..., float]:
            self._metrics[name] = (fn, default_kwargs)
            logger.info("Metric %-25s registered", name)
            return fn

        return _wrapper

    def register(self, name: str, fn: Callable[..., float], **default_kwargs: object) -> None:
        """Register *fn* as metric *name* with optional default kwargs."""
        self._metrics[name] = (fn, default_kwargs)
        logger.info("Metric %-25s registered", name)

    def get(self, name: str) -> tuple[Callable[..., float], dict[str, object]] | None:
        return self._metrics.get(name)

    @property
    def metric_names(self) -> list[str]:
        return list(self._metrics)

    def clear(self) -> None:
        self._metrics.clear()


# ── ExtensionRegistry (facade) ───────────────────────────────────────────


class ExtensionRegistry:
    """Central container for all user-registered extensions.

    Access the global instance via ``from mlcombine.core.registry import registry``.
    """

    def __init__(self) -> None:
        self._steps = StepRegistry()
        self._feature_handlers: dict[str, type[FeatureHandler]] = {}
        self._tensor_adapters = TensorAdapterRegistry()
        self._model_providers = ModelProviderRegistry()
        self._arch_validators = ArchitectureValidatorRegistry()
        self._backbones: dict[str, dict[str, type | Callable[..., Any]]] = {}
        self._layer_builders: dict[str, LayerBuilderProtocol] = {}
        self._activations: dict[str, type[ActivationProtocol]] = {}
        self._metrics = MetricRegistry()

    # ── steps ─────────────────────────────────────────────────────────────

    def step(
        self,
        name: str | None = None,
        *,
        before: str | None = None,
        after: str | None = None,
        package: str | None = None,
        module: str | None = None,
    ) -> Callable[[type], type]:
        return self._steps.register(name, before=before, after=after, package=package, module=module)

    def get_step(self, name: str) -> type | None:
        return self._steps.get(name)

    def get_step_meta(self, name: str) -> dict[str, Any] | None:
        return self._steps.get_meta(name)

    @property
    def step_names(self) -> list[str]:
        return self._steps.names

    # ── feature handlers ──────────────────────────────────────────────────

    def feature_handler(self, type_name: str) -> Callable[[type[FeatureHandler]], type[FeatureHandler]]:
        def _wrapper(cls: type[FeatureHandler]) -> type[FeatureHandler]:
            self._feature_handlers[type_name] = cls
            logger.info("FeatureHandler %-25s registered", type_name)
            return cls

        return _wrapper

    def get_feature_handler(self, type_name: str) -> type[FeatureHandler] | None:
        return self._feature_handlers.get(type_name)

    @property
    def feature_handler_types(self) -> dict[str, type[FeatureHandler]]:
        return dict(self._feature_handlers)

    # ── tensor adapters ───────────────────────────────────────────────────

    def tensor_adapter(self, backend: TensorBackendType | str) -> Callable[[type], type]:
        return self._tensor_adapters.register(backend)

    def get_tensor_adapter(self, backend: TensorBackendType | str) -> type | None:
        return self._tensor_adapters.get(backend)

    @property
    def tensor_adapters(self) -> dict[str, type]:
        return self._tensor_adapters.adapters

    # ── model providers ───────────────────────────────────────────────────

    def model_provider(
        self,
        name: str,
        *,
        package: str | None = None,
        module: str | None = None,
    ) -> Callable[[Callable[..., SupportedModel]], Callable[..., SupportedModel]]:
        return self._model_providers.register(name, package=package, module=module)

    def get_model_provider(self, name: str) -> Callable[..., SupportedModel] | None:
        return self._model_providers.get_provider(name)

    def get_model_provider_meta(self, name: str) -> dict[str, str | None] | None:
        return self._model_providers.get_meta(name)

    @property
    def model_providers(self) -> dict[str, Callable[..., SupportedModel]]:
        return self._model_providers.providers

    # ── architecture validators ────────────────────────────────────────────

    def architecture_validator(
        self,
        name: str,
        *,
        package: str | None = None,
        module: str | None = None,
    ) -> Callable[[type[BaseArchitectureValidator]], type[BaseArchitectureValidator]]:
        return self._arch_validators.register(name, package=package, module=module)

    def get_architecture_validator(self, name: str) -> type[BaseArchitectureValidator] | None:
        return self._arch_validators.get_validator(name)

    def get_architecture_validator_meta(self, name: str) -> dict[str, str | None] | None:
        return self._arch_validators.get_meta(name)

    @property
    def architecture_validator_names(self) -> list[str]:
        return self._arch_validators.validator_names

    # ── layer builders ────────────────────────────────────────────────────

    def layer_builder(self, name: str) -> Callable[[Callable[..., LayerBuilderProtocol]], Callable[..., LayerBuilderProtocol]]:
        def _wrapper(fn: Callable[..., LayerBuilderProtocol]) -> Callable[..., LayerBuilderProtocol]:
            self._layer_builders[name] = fn
            logger.info("LayerBuilder %-25s registered", name)
            return fn

        return _wrapper

    def get_layer_builder(self, name: str) -> LayerBuilderProtocol | None:
        return self._layer_builders.get(name)

    @property
    def layer_builders(self) -> dict[str, LayerBuilderProtocol]:
        return dict(self._layer_builders)

    # ── activation functions ──────────────────────────────────────────────

    def activation(self, name: str) -> Callable[[type[ActivationProtocol]], type[ActivationProtocol]]:
        def _wrapper(cls: type[ActivationProtocol]) -> type[ActivationProtocol]:
            self._activations[name] = cls
            logger.info("Activation %-25s registered", name)
            return cls

        return _wrapper

    def get_activation(self, name: str) -> type[ActivationProtocol] | None:
        return self._activations.get(name)

    @property
    def activations(self) -> dict[str, type[ActivationProtocol]]:
        return dict(self._activations)

    # ── metrics ────────────────────────────────────────────────────────────

    @property
    def metric(self) -> MetricRegistry:
        return self._metrics

    # ── backbones ─────────────────────────────────────────────────────────

    @overload
    def backbone(
        self,
        provider: str,
        name: str,
        model_cls: type | Callable[..., Any],
    ) -> type | Callable[..., Any]: ...

    @overload
    def backbone(
        self,
        provider: str,
        name: str,
        model_cls: None = None,
    ) -> Callable[[type | Callable[..., Any]], type | Callable[..., Any]]: ...

    def backbone(
        self,
        provider: str,
        name: str,
        model_cls: type | Callable[..., Any] | None = None,
    ) -> Callable[[type | Callable[..., Any]], type | Callable[..., Any]] | type | Callable[..., Any]:
        def _wrapper(cls: type | Callable[..., Any]) -> type | Callable[..., Any]:
            self._backbones.setdefault(provider, {})[name] = cls
            logger.info("Backbone %-25s registered for provider %s", f"{provider}/{name}", provider)
            return cls

        if model_cls is not None:
            return _wrapper(model_cls)
        return _wrapper

    def get_backbone(self, provider: str, name: str) -> type | Callable[..., Any] | None:
        return self._backbones.get(provider, {}).get(name)

    @property
    def backbones(self) -> dict[str, dict[str, Any]]:
        return {p: dict(b) for p, b in self._backbones.items()}

    # ── lifecycle ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset all registrations — useful in tests."""
        self._steps.clear()
        self._feature_handlers.clear()
        self._tensor_adapters.clear()
        self._model_providers.clear()
        self._arch_validators.clear()
        self._backbones.clear()
        self._layer_builders.clear()
        self._activations.clear()
        self._metrics.clear()


# Global singleton
registry = ExtensionRegistry()


__all__ = [
    "ArchitectureValidatorRegistry",
    "ExtensionRegistry",
    "FeatureHandler",
    "StepRegistry",
    "TensorAdapterRegistry",
    "registry",
]
