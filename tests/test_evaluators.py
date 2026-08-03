from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlcombine.core.metric import DEFAULT_METRICS
from mlcombine.core.enums import MetricDirection
from mlcombine.core.registry import registry
from mlcombine.core.types import MLCombineConfig, PipelineContext
from mlcombine.evaluators.holdout import HoldoutArchitectureValidator
from mlcombine.evaluators.cv import CVEvaluator
from mlcombine.steps.evaluate import EvaluateStep


# ── Metric registry and DEFAULT_METRICS tests ─────────────────────────────


class TestMetricRegistry:
    def test_all_metrics_have_fn(self):
        for name in registry.metric.metric_names:
            entry = registry.metric.get(name)
            assert entry is not None, f"metric {name!r} has no entry"
            fn, kwargs = entry
            assert fn is not None, f"metric {name!r} has no function"
            assert isinstance(kwargs, dict)

    def test_all_metrics_have_direction(self):
        for name in registry.metric.metric_names:
            meta = registry.metric.get_meta(name)
            assert meta is not None, f"metric {name!r} has no meta"
            assert meta["direction"] in (MetricDirection.MINIMIZE, MetricDirection.MAXIMIZE)

    def test_builtin_metric_directions(self):
        for name in ("rmse", "mse", "mae", "mape", "logloss"):
            assert registry.metric.get_meta(name)["direction"] == MetricDirection.MINIMIZE
        for name in ("accuracy", "f1", "f1_macro", "precision", "recall", "auc"):
            assert registry.metric.get_meta(name)["direction"] == MetricDirection.MAXIMIZE

    def test_custom_metric_with_direction(self):
        @registry.metric("test_dir_metric", direction="minimize")
        def _fn(y_true, y_pred):
            return 0.0

        try:
            meta = registry.metric.get_meta("test_dir_metric")
            assert meta is not None
            assert meta["direction"] == MetricDirection.MINIMIZE
            # get() still returns (fn, kwargs) only
            fn, kwargs = registry.metric.get("test_dir_metric")
            assert fn is _fn
            assert kwargs == {}
        finally:
            registry.metric._metrics.pop("test_dir_metric", None)

    def test_custom_metric_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="valid MetricDirection"):
            registry.metric("bad_dir_metric", direction="sideways")(lambda y_true, y_pred: 0.0)

    def test_default_metrics_are_known(self):
        known = set(registry.metric.metric_names)
        for task_key, metrics in DEFAULT_METRICS.items():
            for m in metrics:
                assert m in known, f"DEFAULT_METRICS.{task_key} has unknown metric {m!r}"

    def test_mae_returns_non_negative(self):
        fn, kwargs = registry.metric.get("mae")
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        result = float(fn(y_true, y_pred, **kwargs))
        assert result >= 0

    def test_rmse_works_now(self):
        fn, kwargs = registry.metric.get("rmse")
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        result = float(fn(y_true, y_pred, **kwargs))
        assert result >= 0

    def test_accuracy_perfect(self):
        fn, kwargs = registry.metric.get("accuracy")
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        assert float(fn(y_true, y_pred, **kwargs)) == 1.0

    def test_logloss_requires_proba(self):
        fn, kwargs = registry.metric.get("logloss")
        y_true = np.array([0, 1, 0])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        result = float(fn(y_true, y_pred_proba, **kwargs))
        assert result > 0


# ── HoldoutArchitectureValidator tests ───────────────────────────────────────


@pytest.fixture
def reg_data() -> tuple[pd.DataFrame, str]:
    n = 100
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(0, 10, n),
            "target": rng.normal(5, 2, n),
        }
    )
    return df, "target"


@pytest.fixture
def cls_data() -> tuple[pd.DataFrame, str]:
    n = 100
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.uniform(0, 10, n),
            "target": rng.choice([0, 1], size=n),
        }
    )
    return df, "target"


@pytest.fixture
def eval_context(reg_data) -> PipelineContext:
    df, target_col = reg_data
    ctx = PipelineContext()
    ctx.data.train_df = df
    ctx.data.target_col = target_col
    ctx.data.task_type = "regression"
    return ctx


@pytest.fixture
def cls_eval_context(cls_data) -> PipelineContext:
    df, target_col = cls_data
    ctx = PipelineContext()
    ctx.data.train_df = df
    ctx.data.target_col = target_col
    ctx.data.task_type = "classification"
    return ctx


def _make_cls_model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500, random_state=42)


def _make_reg_model():
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


class TestHoldoutArchitectureValidator:
    def test_regression_holdout(self, eval_context: PipelineContext):
        validator = HoldoutArchitectureValidator(cfg=None)
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        results = validator.validate(bp, eval_context)
        assert isinstance(results, dict)
        assert len(results) > 0
        assert "rmse" in results
        assert results["rmse"] >= 0

    def test_classification_holdout(self, cls_eval_context: PipelineContext):
        validator = HoldoutArchitectureValidator(cfg=None)
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="classification",
        )
        results = validator.validate(bp, cls_eval_context)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_stratified_split(self, cls_eval_context: PipelineContext):
        validator = HoldoutArchitectureValidator(cfg=None, stratified=True, val_fraction=0.3, random_state=42)
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="classification",
        )
        results = validator.validate(bp, cls_eval_context)
        assert len(results) > 0

    def test_empty_train_df(self):
        ctx = PipelineContext()
        ctx.data.train_df = None
        validator = HoldoutArchitectureValidator()
        assert validator.validate(None, ctx) == {}

    def test_missing_target_col(self, reg_data):
        df, _ = reg_data
        ctx = PipelineContext()
        ctx.data.train_df = df
        ctx.data.target_col = "nonexistent"
        validator = HoldoutArchitectureValidator()
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        assert validator.validate(bp, ctx) == {}

    def test_invalid_val_fraction(self, eval_context: PipelineContext):
        validator = HoldoutArchitectureValidator(cfg=None, val_fraction=0)
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        assert validator.validate(bp, eval_context) == {}

    def test_is_required_default(self):
        assert HoldoutArchitectureValidator.is_required(None) is True

    def test_params_stored(self):
        validator = HoldoutArchitectureValidator(cfg=None, val_fraction=0.3, random_state=7)
        assert validator._val_fraction == 0.3
        assert validator._random_state == 7


# ── CVEvaluator tests ────────────────────────────────────────────────────


class TestCVEvaluator:
    def test_regression_cv(self, eval_context: PipelineContext):
        evaluator = CVEvaluator(cfg=None, n_folds=3)
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        model_bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        results = evaluator.validate(model_bp, eval_context)
        assert isinstance(results, dict)
        assert len(results) > 0
        assert "rmse" in results
        # OOF predictions stored on context
        assert eval_context.artifacts.oof_preds is not None
        assert len(eval_context.artifacts.oof_preds) == len(eval_context.data.train_df)

    def test_classification_cv(self, cls_eval_context: PipelineContext):
        evaluator = CVEvaluator(cfg=None, n_folds=3)
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        model_bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="classification",
        )
        results = evaluator.validate(model_bp, cls_eval_context)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_custom_n_folds(self, eval_context: PipelineContext):
        evaluator = CVEvaluator(cfg=None, n_folds=2)
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        model_bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        results = evaluator.validate(model_bp, eval_context)
        assert len(results) > 0

    def test_with_custom_metrics(self, eval_context: PipelineContext):
        evaluator = CVEvaluator(cfg=None, n_folds=3, metrics=["mae", "rmse"])
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        model_bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        results = evaluator.validate(model_bp, eval_context)
        assert "mae" in results
        assert "rmse" in results

    def test_empty_train_df(self):
        ctx = PipelineContext()
        ctx.data.train_df = None
        evaluator = CVEvaluator()
        results = evaluator.validate(None, ctx)
        assert results == {}

    def test_missing_target_col(self, reg_data):
        df, _ = reg_data
        ctx = PipelineContext()
        ctx.data.train_df = df
        ctx.data.target_col = "nonexistent"
        evaluator = CVEvaluator()
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        model_bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        results = evaluator.validate(model_bp, ctx)
        assert results == {}

    def test_target_encoding(self):
        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame(
            {
                "cat": pd.Categorical(np.random.choice(["a", "b", "c"], n)),
                "num": rng.normal(0, 1, n),
                "target": rng.normal(5, 2, n),
            }
        )
        ctx = PipelineContext()
        ctx.data.train_df = df
        ctx.data.target_col = "target"
        ctx.data.task_type = "regression"

        evaluator = CVEvaluator(
            cfg=None,
            n_folds=3,
            target_encode_cols=["cat"],
            target_encode_smoothing=5.0,
        )
        from mlcombine.core.schemas.blueprint import ModelBlueprint

        model_bp = ModelBlueprint(
            provider="sklearn",
            params={"backbone": "random_forest"},
            task_type="regression",
        )
        results = evaluator.validate(model_bp, ctx)
        assert len(results) > 0

    def test_is_required_default(self):
        assert CVEvaluator.is_required(None) is True


# ── EvaluateStep tests ────────────────────────────────────────────────────


class TestEvaluateStep:
    def _make_cfg(self) -> MLCombineConfig:
        cfg = MLCombineConfig(
            **{
                "data": {"train_df": "train.csv", "test_df": "test.csv", "target_col": "target"},
                "model": [{"provider": "sklearn", "params": {"backbone": "random_forest"}}],
            }
        )
        return cfg

    def test_insample_fallback(self, eval_context: PipelineContext):
        cfg = self._make_cfg()
        cfg.data.train_df = eval_context.data.train_df
        cfg.data.target_col = eval_context.data.target_col

        step = EvaluateStep(cfg=cfg)
        model = _make_reg_model()
        x = eval_context.data.train_df.drop(columns=["target"])
        y = eval_context.data.train_df["target"]
        model.fit(x, y)
        eval_context.artifacts.model = model

        result = step.run(eval_context)
        assert result.artifacts.evaluation_results is not None
        assert len(result.artifacts.evaluation_results) > 0

    def test_no_data_skip(self):
        cfg = self._make_cfg()
        ctx = PipelineContext()
        step = EvaluateStep(cfg=cfg)
        assert step.run(ctx) is ctx

    def test_metrics_from_step_config(self, eval_context: PipelineContext):
        cfg = self._make_cfg()
        cfg.data.train_df = eval_context.data.train_df
        cfg.data.target_col = eval_context.data.target_col
        cfg.step_config.evaluate = {"metrics": ["mae"]}

        step = EvaluateStep(cfg=cfg)
        model = _make_reg_model()
        x = eval_context.data.train_df.drop(columns=["target"])
        y = eval_context.data.train_df["target"]
        model.fit(x, y)
        eval_context.artifacts.model = model

        result = step.run(eval_context)
        assert result.artifacts.evaluation_results is not None
        assert "mae" in result.artifacts.evaluation_results
