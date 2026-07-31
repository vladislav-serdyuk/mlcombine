"""Tests for ModelBuilder and provider functions."""

import pytest

from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.types import ModelNode, UnsupportedBackendError
from mlcombine.core.builder import ModelBuilder


@pytest.fixture
def builder():
    return ModelBuilder()


class TestModelBuilder:
    def _build_and_materialise(self, builder, nodes, **kwargs):
        """Helper: build blueprint then materialise."""
        bp = builder.build_all(nodes, **kwargs)
        return bp.build()

    def test_create_sklearn_random_forest(self, builder):
        nodes = [ModelNode(provider="sklearn", params={"backbone": "random_forest"})]
        bp = builder.build_all(nodes, task_type="regression")
        assert isinstance(bp, ModelBlueprint)
        model = bp.build()
        from mlcombine.models.providers.sklearn import SklearnWrapper

        assert isinstance(model, SklearnWrapper)
        assert "RandomForestRegressor" in type(model._model).__name__

    def test_create_sklearn_classification(self, builder):
        nodes = [ModelNode(provider="sklearn", params={"backbone": "random_forest"})]
        model = self._build_and_materialise(builder, nodes, task_type="classification", num_classes=2)
        assert "RandomForestClassifier" in type(model._model).__name__

    @pytest.mark.parametrize(
        "backbone,expected_class",
        [
            ("gradient_boosting", "GradientBoosting"),
            ("svm", "SVR"),
            ("mlp", "MLPRegressor"),
        ],
    )
    def test_sklearn_backbone_dispatch(self, builder, backbone, expected_class):
        nodes = [ModelNode(provider="sklearn", params={"backbone": backbone})]
        model = self._build_and_materialise(builder, nodes, task_type="regression")
        assert expected_class in type(model._model).__name__, f"{backbone} should produce {expected_class}"

    def test_sklearn_logistic_regression_classification(self, builder):
        nodes = [ModelNode(provider="sklearn", params={"backbone": "logistic_regression"})]
        model = self._build_and_materialise(builder, nodes, task_type="classification", num_classes=2)
        assert "LogisticRegression" in type(model._model).__name__

    def test_unsupported_backend(self, builder):
        nodes = [ModelNode(provider="__nonexistent__")]
        bp = builder.build_all(nodes)
        with pytest.raises(UnsupportedBackendError):
            bp.build()

    def test_fit_and_predict_roundtrip(self, builder):
        import numpy as np

        nodes = [ModelNode(provider="sklearn", params={"backbone": "random_forest"})]
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (30, 4))
        y = rng.normal(5, 2, 30)
        model = self._build_and_materialise(builder, nodes, task_type="regression")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert isinstance(preds, np.ndarray)
