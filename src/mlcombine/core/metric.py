"""Metric functions and defaults shared by all evaluators.

Metrics are registered via ``@registry.metric(name, **default_kwargs)``;
lookup is delegated to ``registry.metric.get(name)``.  Each metric declares
its optimization ``direction`` (``MetricDirection``) so that consumers like
the optuna tuner know whether to minimize or maximize.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from mlcombine.core.enums import MetricDirection
from mlcombine.core.registry import registry

MAX = MetricDirection.MAXIMIZE
MIN = MetricDirection.MINIMIZE

# ── Register built-in metrics ─────────────────────────────────────────────

registry.metric("accuracy", direction=MAX)(accuracy_score)
registry.metric("f1", average="weighted", direction=MAX)(f1_score)
registry.metric("f1_macro", average="macro", direction=MAX)(f1_score)
registry.metric("precision", average="weighted", zero_division=0, direction=MAX)(precision_score)
registry.metric("recall", average="weighted", zero_division=0, direction=MAX)(recall_score)
registry.metric("logloss", direction=MIN)(log_loss)
registry.metric("auc", multi_class="ovo", average="weighted", direction=MAX)(roc_auc_score)
registry.metric("rmse", direction=MIN)(root_mean_squared_error)
registry.metric("mse", direction=MIN)(mean_squared_error)
registry.metric("mae", direction=MIN)(mean_absolute_error)
registry.metric("mape", direction=MIN)(mean_absolute_percentage_error)

# ── Task→metric defaults ──────────────────────────────────────────────────

DEFAULT_METRICS: dict[str, list[str]] = {
    "regression": ["rmse", "mae"],
    "classification": ["f1", "accuracy"],
    "multitask": ["f1", "accuracy"],
}


__all__ = [
    "DEFAULT_METRICS",
]
