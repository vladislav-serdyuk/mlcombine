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

    def test_tuner_study_direction_regression_minimizes(self):
        """Regression metrics (mae/rmse) must be minimized by optuna."""
        from mlcombine.models.meta.tuner import TunerWrapper

        t = TunerWrapper(
            target_provider="sklearn",
            target_params={},
            search_space={"n_estimators": {"type": "int", "low": 10, "high": 20}},
            n_trials=1,
            task_type="regression",
            tune_metric="mae",
        )
        assert t._study_direction() == "minimize"

    def test_tuner_study_direction_classification_maximizes(self):
        from mlcombine.models.meta.tuner import TunerWrapper

        t = TunerWrapper(
            target_provider="sklearn",
            target_params={},
            search_space={},
            n_trials=1,
            task_type="classification",
            tune_metric="f1",
        )
        assert t._study_direction() == "maximize"

    def test_tuner_study_direction_from_registry(self):
        """Custom metric direction in the registry drives the study direction."""
        from mlcombine.core.registry import registry
        from mlcombine.models.meta.tuner import TunerWrapper

        @registry.metric("custom_min_metric", direction="minimize")
        def _fn(y_true, y_pred):
            return 0.0

        try:
            t = TunerWrapper(
                target_provider="sklearn",
                target_params={},
                search_space={},
                n_trials=1,
                task_type="regression",
                tune_metric="custom_min_metric",
            )
            assert t._study_direction() == "minimize"
        finally:
            registry.metric._metrics.pop("custom_min_metric", None)

    def test_tuner_regression_improves_mae(self, builder):
        """Regression tuning with tune_metric=mae fits and improves MAE."""
        rng = np.random.default_rng(7)
        n = 200
        X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.uniform(0, 1, n)})
        y = pd.Series(3 * X["f1"] + 0.5 * X["f2"] + rng.normal(0, 0.1, n))
        nodes = [
            ModelNode(
                provider="tuner",
                params={
                    "n_trials": 3,
                    "target_provider": "sklearn",
                    "target_params": {"backbone": "gradient_boosting"},
                    "search_space": {"n_estimators": {"type": "int", "low": 20, "high": 60, "step": 20}},
                    "evaluator": "cv",
                    "evaluator_params": {"n_folds": 2},
                    "tune_metric": "mae",
                },
            )
        ]
        model = builder.build_all(nodes, task_type="regression").build()
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        mae = float(np.mean(np.abs(preds - y.to_numpy())))
        assert mae < 1.0
