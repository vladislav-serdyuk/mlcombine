"""Tests for the optuna/tuner meta-provider."""

import numpy as np
import pandas as pd
import pytest

from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.types import ModelNode
from mlcombine.core.builder import ModelBuilder


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.uniform(0, 1, n)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    return X, y


@pytest.fixture
def builder() -> ModelBuilder:
    return ModelBuilder()


@pytest.mark.optuna
class TestTuner:
    def test_tuner_builds_and_fits(self, builder):
        nodes = [
            ModelNode(
                provider="tuner",
                params={
                    "n_trials": 5,
                    "target_provider": "sklearn",
                    "target_params": {"backbone": "random_forest"},
                    "search_space": {
                        "n_estimators": {"type": "int", "low": 50, "high": 150, "step": 50},
                    },
                },
            )
        ]
        bp = builder.build_all(nodes, task_type="classification", num_classes=2)
        assert isinstance(bp, ModelBlueprint)
        model = bp.build()
        X, y = _toy_data()
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_tuner_improves_score(self, builder):
        """Check that tuning finds a reasonable solution."""
        nodes = [
            ModelNode(
                provider="tuner",
                params={
                    "n_trials": 10,
                    "target_provider": "sklearn",
                    "target_params": {"backbone": "gradient_boosting"},
                    "search_space": {
                        "n_estimators": {"type": "int", "low": 50, "high": 150, "step": 50},
                    },
                },
            )
        ]
        model = builder.build_all(nodes, task_type="classification", num_classes=2).build()
        X, y = _toy_data()
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(np.mean(preds == y.to_numpy()))
        assert accuracy > 0.7

    def test_tuner_predict_before_fit_raises(self, builder):
        nodes = [
            ModelNode(
                provider="tuner",
                params={
                    "n_trials": 2,
                    "target_provider": "sklearn",
                    "target_params": {"backbone": "random_forest"},
                    "search_space": {"n_estimators": {"type": "int", "low": 50, "high": 100, "step": 50}},
                },
            )
        ]
        model = builder.build_all(nodes, task_type="classification").build()
        X, _ = _toy_data()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_tuner_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported model provider"):
            from mlcombine.models.meta.tuner import TunerWrapper

            t = TunerWrapper(
                target_provider="nonexistent",
                target_params={},
                search_space={},
                n_trials=1,
            )
            X, y = _toy_data()
            t.fit(X, y)
