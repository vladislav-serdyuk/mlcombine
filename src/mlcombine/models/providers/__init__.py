"""Providers sub-package — concrete provider functions and wrappers.

All provider functions register themselves via ``@registry.model_provider(...)``
when imported.  The wrappers (``SklearnWrapper``, ``CatBoostWrapper``, …) are
internal helpers that adapt framework models to ``MLModelProtocol``.
"""

# Import each provider module so its @registry.model_provider() decorator fires
from mlcombine.models.providers.catboost import catboost_provider  # noqa: F401
from mlcombine.models.providers.lightgbm import lightgbm_provider  # noqa: F401
from mlcombine.models.providers.sklearn import sklearn_provider  # noqa: F401
from mlcombine.models.providers.pytorch import pytorch_provider  # noqa: F401
from mlcombine.models.providers.hybrid import hybrid_provider  # noqa: F401

from mlcombine.core.protocols import MLModelProtocol, SupportedModel

__all__ = [
    "MLModelProtocol",
    "SupportedModel",
]
