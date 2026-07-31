import numpy as np
import pandas as pd
import pytest

from mlcombine.core.types import ModelNode
from mlcombine.core.builder import ModelBuilder
from mlcombine.models.meta import TLearner, SLearner


@pytest.fixture
def base_model():
    builder = ModelBuilder()
    nodes = [ModelNode(provider="sklearn", params={"backbone": "random_forest"})]
    return builder.build_all(nodes, task_type="classification", num_classes=2)


@pytest.fixture
def uplift_data():
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.uniform(0, 1, n)})
    y = pd.Series(rng.integers(0, 2, n))
    t = pd.Series(rng.integers(0, 2, n))
    return X, y, t


class TestTLearner:
    def test_init(self, base_model):
        t = TLearner(base_model)
        assert not t.is_fitted

    def test_fit_requires_treatment(self, base_model, uplift_data):
        X, y, _ = uplift_data
        t = TLearner(base_model)
        with pytest.raises(ValueError, match="treatment is required"):
            t.fit(X, y)

    def test_fit_and_predict(self, base_model, uplift_data):
        X, y, t = uplift_data
        learner = TLearner(base_model)
        learner.fit(X, y, treatment=t)
        assert learner.is_fitted
        assert learner.model_treatment is not None
        assert learner.model_control is not None
        preds = learner.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_conforms_to_mlmodelprotocol(self, base_model, uplift_data):
        t = TLearner(base_model)
        assert hasattr(t, "fit") and callable(t.fit)
        assert hasattr(t, "predict") and callable(t.predict)
        import inspect

        sig = inspect.signature(t.fit)
        assert "treatment" in {p.name for p in sig.parameters.values() if p.default is not p.empty}

    def test_predict_before_fit_raises(self, base_model, uplift_data):
        X, _, _ = uplift_data
        t = TLearner(base_model)
        with pytest.raises(RuntimeError, match="fitted"):
            t.predict(X)

    def test_accepts_treatment_kwarg(self, base_model, uplift_data):
        X, y, t = uplift_data
        learner = TLearner(base_model)
        learner.fit(X, y, treatment=t)
        preds = learner.predict(X)
        assert len(preds) == len(X)


class TestSLearner:
    def test_init(self, base_model):
        s = SLearner(base_model)
        assert not s.is_fitted

    def test_fit_requires_treatment(self, base_model, uplift_data):
        X, y, _ = uplift_data
        s = SLearner(base_model)
        with pytest.raises(ValueError, match="treatment is required"):
            s.fit(X, y)

    def test_fit_and_predict(self, base_model, uplift_data):
        X, y, t = uplift_data
        learner = SLearner(base_model)
        learner.fit(X, y, treatment=t)
        assert learner.is_fitted
        preds = learner.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_conforms_to_mlmodelprotocol(self, base_model, uplift_data):
        s = SLearner(base_model)
        assert hasattr(s, "fit") and callable(s.fit)
        assert hasattr(s, "predict") and callable(s.predict)
        import inspect

        sig = inspect.signature(s.fit)
        assert "treatment" in {p.name for p in sig.parameters.values() if p.default is not p.empty}

    def test_accepts_treatment_kwarg(self, base_model, uplift_data):
        X, y, t = uplift_data
        learner = SLearner(base_model)
        learner.fit(X, y, treatment=t)
        preds = learner.predict(X)
        assert len(preds) == len(X)
