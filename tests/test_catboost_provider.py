"""Tests for the CatBoost provider objective/loss handling (no real training)."""

import pytest

from mlcombine.core.enums import ModelObjective, TaskType
from mlcombine.models.providers import catboost as cb_module
from mlcombine.models.providers.catboost import catboost_provider


@pytest.fixture
def fake_catboost(monkeypatch):
    captured: dict[str, dict] = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured["regressor"] = kwargs

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured["classifier"] = kwargs

    monkeypatch.setattr(cb_module, "_CATBOOST_AVAILABLE", True)
    monkeypatch.setattr(cb_module, "CatBoostRegressor", FakeRegressor)
    monkeypatch.setattr(cb_module, "CatBoostClassifier", FakeClassifier)
    return captured


class TestCatboostProviderObjective:
    def test_string_objective_sets_loss_and_metric(self, fake_catboost):
        catboost_provider(objective="MAE")
        kw = fake_catboost["regressor"]
        assert kw["loss_function"] == "MAE"
        assert kw["eval_metric"] == "MAE"

    def test_enum_objective_keeps_default_loss(self, fake_catboost):
        catboost_provider(objective=ModelObjective.MAPE)
        kw = fake_catboost["regressor"]
        assert kw["loss_function"] == "RMSE"
        assert kw["eval_metric"] == "MAPE"

    def test_explicit_eval_metric_wins(self, fake_catboost):
        catboost_provider(objective="MAE", eval_metric="MAPE")
        kw = fake_catboost["regressor"]
        assert kw["loss_function"] == "MAE"
        assert kw["eval_metric"] == "MAPE"

    def test_classification_string_objective(self, fake_catboost):
        catboost_provider(task_type=TaskType.CLASSIFICATION, num_classes=2, objective="Logloss")
        kw = fake_catboost["classifier"]
        assert kw["loss_function"] == "Logloss"
        assert kw["eval_metric"] == "Accuracy"

    def test_unsupported_string_ignored(self, fake_catboost):
        catboost_provider(objective="FancyLoss")
        kw = fake_catboost["regressor"]
        assert kw["loss_function"] == "RMSE"
        assert kw["eval_metric"] == "RMSE"
