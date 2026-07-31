"""ModelBlueprint — lazy model provider call that can be deferred / mutated.

A ``ModelBlueprint`` encapsulates everything needed to build a model instance
via a registered provider function.  It supports:

* ``build()`` — actually calls the provider function and returns an instance.
* ``with_params(**overrides)`` — returns a copy with merged params (for per-fold
  tuning in CV, per-trial overrides in hyperparameter search, etc.).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mlcombine.core.registry import registry
from mlcombine.core.exceptions import UnsupportedBackendError
from mlcombine.core.protocols import SupportedModel


class ModelBlueprint:
    """Deferred model construction from a registered provider + params + deps.

    Building a blueprint does **not** call the provider function — only
    ``.build()`` does.  This allows meta-providers (ensemble, CV, uplift)
    to receive blueprints and decide when/how to materialise instances.
    """

    def __init__(
        self,
        provider: str,
        params: dict[str, Any] | None = None,
        *,
        model: ModelBlueprint | None = None,
        models: list[ModelBlueprint] | None = None,
        task_type: str = "regression",
        num_classes: int | None = None,
        input_size: int | None = None,
    ) -> None:
        self.provider = provider
        self.params = dict(params or {})
        self.model = model
        self.models = list(models) if models else []
        self.task_type = task_type
        self.num_classes = num_classes
        self.input_size = input_size

    def build(self, **overrides: Any) -> SupportedModel:
        """Call the provider function with merged params.

        Dependencies (``model`` / ``models``) are passed **as-is** (blueprints)
        so meta-providers (ensemble, uplift) can defer materialisation
        to ``fit()`` time.  Leaf providers that lack deps never see them.

        Returns:
            A model instance (``SupportedModel``) from the provider.
        """
        provider_fn = registry.get_model_provider(self.provider)
        if provider_fn is None:
            raise UnsupportedBackendError(f"Unsupported model provider: {self.provider!r}. Available: {list(registry.model_providers)}")

        kwargs: dict[str, Any] = dict(self.params)
        kwargs.setdefault("task_type", self.task_type)
        if self.num_classes is not None:
            kwargs.setdefault("num_classes", self.num_classes)
        if self.input_size is not None:
            kwargs.setdefault("input_size", self.input_size)

        # Dependencies are passed as blueprints — meta-providers call
        # ``.build()`` themselves; leaf providers ignore them.
        if self.model is not None:
            kwargs["model"] = self.model
        if self.models:
            kwargs["models"] = list(self.models)

        kwargs.update(overrides)
        return provider_fn(**kwargs)

    def with_params(self, **overrides: Any) -> ModelBlueprint:
        """Return a **shallow copy** with merged ``params``.

        This is the primary extension point for CV folds, optuna trials,
        or any scenario where a slightly different config is needed
        without mutating the original blueprint.
        """
        new = deepcopy(self)
        new.params.update(overrides)
        return new
