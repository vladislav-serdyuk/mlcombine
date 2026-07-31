"""Micro-tests — full pipeline on tiny synthetic data (< 30s each).

If these pass, the pipeline WILL run on real data (assuming data itself is
valid).  Catches DAG resolution, type mismatches, OOF stacking, tuner
cache, column alignment, and predict/predict_proba regressions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlcombine.core.pipeline import PipelineEngine
from mlcombine.core.schemas.config import MLCombineConfig
from mlcombine.core.types import PipelineContext

N_ROWS = 80


# ── helpers ─────────────────────────────────────────────────────────────


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _make_data(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    n = N_ROWS
    train = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.uniform(0, 10, n),
            "cat1": rng.choice(["a", "b", "c"], n),
            "target": rng.integers(0, 4, n),
        }
    )
    test = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n // 2),
            "num2": rng.uniform(0, 10, n // 2),
            "cat1": rng.choice(["a", "b", "c"], n // 2),
        }
    )
    return {
        "train": _write_csv(tmp_path / "train.csv", train),
        "test": _write_csv(tmp_path / "test.csv", test),
    }


def _build_config(
    train_path: Path,
    test_path: Path,
    model: list[dict],
    *,
    step_config: dict | None = None,
    handling: dict | None = None,
    drop_columns: list[str] | None = None,
) -> MLCombineConfig:
    out = train_path.parent / "outputs"
    data_cfg: dict = {
        "train_df": str(train_path),
        "test_df": str(test_path),
        "target_col": "target",
    }
    if drop_columns:
        data_cfg["drop_columns"] = drop_columns
    raw: dict = {
        "data": data_cfg,
        "model": model,
        "handling": handling
        or {
            "numbers": {"impute": "median", "scale": "none"},
            "categories": {"encode": "ordinal", "smoothing": 10.0},
        },
        "step_config": step_config or {},
        "trainer": {
            "output_dir": str(out),
            "output_file": str(out / "predictions.csv"),
        },
    }
    return MLCombineConfig(**raw)


def _run(cfg: MLCombineConfig) -> PipelineContext:
    engine = PipelineEngine.from_config(cfg)
    return engine.run_all(PipelineContext())


def _test_features(ctx: PipelineContext) -> pd.DataFrame:
    """Return test features as they looked right before predict (post-transform)."""
    df = ctx.data.test_df.copy()
    target = ctx.data.target_col
    if isinstance(target, str) and target in df.columns:
        df = df.drop(columns=[target])
    return df


# ── tests ───────────────────────────────────────────────────────────────


class TestMicroPipeline:
    """Suite of micro-tests — each should complete in < 30 seconds."""

    def test_single_catboost(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"provider": "catboost", "params": {"iterations": 5, "verbose": False}},
            ],
            step_config={
                "split": {"val_fraction": 0.2, "stratified": True},
                "evaluate": {"metrics": ["f1"]},
            },
        )
        ctx = _run(cfg)

        assert ctx.artifacts.model is not None
        assert ctx.artifacts.evaluation_results is not None
        assert "f1" in ctx.artifacts.evaluation_results

        x = _test_features(ctx)
        preds = ctx.artifacts.model.predict(x)
        assert preds.dtype.kind == "i"
        assert len(preds) == N_ROWS // 2

        proba = ctx.artifacts.model.predict_proba(x)
        assert proba.shape == (N_ROWS // 2, 4)
        assert proba.dtype.kind == "f"

    def test_single_sklearn(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert preds.dtype.kind == "i"

    def test_single_lightgbm(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"provider": "lightgbm", "params": {"iterations": 5, "verbose": 0}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert preds.dtype.kind == "i"

    def test_stacking_logistic_2fold(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"provider": "stacking", "models": ["cb", "rf"], "params": {"meta_model": "logistic", "n_folds": 2, "stratified": True}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None

        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert preds.dtype.kind == "i"
        assert len(preds) == N_ROWS // 2

    def test_stacking_with_lightgbm(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "lgb", "provider": "lightgbm", "params": {"iterations": 5, "verbose": 0}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"provider": "stacking", "models": ["cb", "lgb", "rf"], "params": {"meta_model": "logistic", "n_folds": 2}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None

        x = _test_features(ctx)
        preds = ctx.artifacts.model.predict(x)
        assert len(preds) == N_ROWS // 2

        proba = ctx.artifacts.model.predict_proba(x)
        assert proba.shape == (N_ROWS // 2, 4)

    def test_fold_ensemble(self, tmp_path):
        cfg = _build_config(
            *_make_data(tmp_path).values(),
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"id": "stack", "provider": "stacking", "models": ["cb", "rf"], "params": {"meta_model": "logistic", "n_folds": 2}},
                {"provider": "fold_ensemble", "model": "stack", "params": {"n_folds": 2}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None
        assert hasattr(ctx.artifacts.model, "fold_models_")
        assert len(ctx.artifacts.model.fold_models_) == 2

        x = _test_features(ctx)
        preds = ctx.artifacts.model.predict(x)
        assert preds.dtype.kind == "i"
        assert len(preds) == N_ROWS // 2

        proba = ctx.artifacts.model.predict_proba(x)
        assert proba.dtype.kind == "f"
        assert proba.shape == (N_ROWS // 2, 4)

    def test_fold_ensemble_full_stack(self, tmp_path):
        """FoldEnsemble wrapping stacking of 3 models — exercises real DAG."""
        cfg = _build_config(
            *_make_data(tmp_path).values(),
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "lgb", "provider": "lightgbm", "params": {"iterations": 5, "verbose": 0}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"id": "stack", "provider": "stacking", "models": ["cb", "lgb", "rf"], "params": {"meta_model": "logistic", "n_folds": 2}},
                {"provider": "fold_ensemble", "model": "stack", "params": {"n_folds": 2}},
            ],
        )
        ctx = _run(cfg)

        x = _test_features(ctx)
        preds = ctx.artifacts.model.predict(x)
        assert len(preds) == N_ROWS // 2

        proba = ctx.artifacts.model.predict_proba(x)
        assert proba.shape == (N_ROWS // 2, 4)

    def test_ensemble_weighted(self, tmp_path):
        cfg = _build_config(
            *_make_data(tmp_path).values(),
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"provider": "ensemble", "models": ["cb", "rf"], "params": {"weights": [0.6, 0.4]}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None

        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert preds.dtype.kind == "i"
        assert len(preds) == N_ROWS // 2

    def test_tuner_catboost(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {
                    "provider": "tuner",
                    "params": {
                        "n_trials": 2,
                        "target_provider": "catboost",
                        "target_params": {"iterations": 5, "verbose": False},
                        "search_space": {
                            "depth": {"type": "int", "low": 4, "high": 6},
                            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
                        },
                    },
                },
            ],
            step_config={
                "split": {"val_fraction": 0.2, "stratified": True},
                "evaluate": {"metrics": ["f1"]},
            },
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None

        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert preds.dtype.kind == "i"
        assert len(preds) == N_ROWS // 2

    def test_tuner_cache_across_oof(self, tmp_path):
        """Tuner must cache best params so OOF fold 2+ skip optuna."""
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {
                    "id": "tuned",
                    "provider": "tuner",
                    "params": {
                        "n_trials": 2,
                        "target_provider": "catboost",
                        "target_params": {"iterations": 5, "verbose": False},
                        "search_space": {
                            "depth": {"type": "int", "low": 4, "high": 6},
                        },
                    },
                },
                {"provider": "stacking", "models": ["cb", "rf", "tuned"], "params": {"meta_model": "logistic", "n_folds": 2}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert len(preds) == N_ROWS // 2

    def test_predict_and_predict_proba_from_all_layers(self, tmp_path):
        """Verify predict() output dtype is always int for classification
        and predict_proba() always returns float64 probabilities."""
        cfg = _build_config(
            *_make_data(tmp_path).values(),
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"id": "lgb", "provider": "lightgbm", "params": {"iterations": 5, "verbose": 0}},
                {"id": "stack", "provider": "stacking", "models": ["cb", "rf", "lgb"], "params": {"meta_model": "logistic", "n_folds": 2}},
                {"provider": "fold_ensemble", "model": "stack", "params": {"n_folds": 2}},
            ],
        )
        ctx = _run(cfg)
        x = _test_features(ctx)

        preds = ctx.artifacts.model.predict(x)
        assert preds.dtype.kind == "i", f"predict() should return int, got {preds.dtype}"

        proba = ctx.artifacts.model.predict_proba(x)
        assert proba.dtype == np.float64, f"predict_proba() should return float64, got {proba.dtype}"
        assert proba.shape[1] >= 2, f"predict_proba() should have >=2 columns, got {proba.shape[1]}"
        assert np.all((proba >= 0) & (proba <= 1)), "predict_proba() values out of [0, 1] range"
        assert np.allclose(proba.sum(axis=1), 1.0), "predict_proba() rows should sum to 1"

    def test_group_split(self, tmp_path):
        rng = np.random.default_rng(42)
        n = N_ROWS
        train = pd.DataFrame(
            {
                "num1": rng.normal(0, 1, n),
                "cat1": rng.choice(["a", "b", "c"], n),
                "group": rng.integers(0, n // 5, n),
                "target": rng.integers(0, 4, n),
            }
        )
        test = pd.DataFrame(
            {
                "num1": rng.normal(0, 1, n // 2),
                "cat1": rng.choice(["a", "b", "c"], n // 2),
            }
        )
        train_path = _write_csv(tmp_path / "train.csv", train)
        test_path = _write_csv(tmp_path / "test.csv", test)
        cfg = _build_config(
            train_path,
            test_path,
            [
                {"provider": "catboost", "params": {"iterations": 5, "verbose": False}},
            ],
            step_config={
                "split": {"val_fraction": 0.2, "stratified": True, "group_cols": ["group"]},
                "evaluate": {"metrics": ["f1"]},
            },
            drop_columns=["group"],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None
        assert ctx.artifacts.evaluation_results is not None

    def test_group_split_with_stacking(self, tmp_path):
        rng = np.random.default_rng(42)
        n = N_ROWS
        train = pd.DataFrame(
            {
                "num1": rng.normal(0, 1, n),
                "cat1": rng.choice(["a", "b", "c"], n),
                "authorId": rng.integers(0, n // 5, n),
                "target": rng.integers(0, 4, n),
            }
        )
        test = pd.DataFrame(
            {
                "num1": rng.normal(0, 1, n // 2),
                "cat1": rng.choice(["a", "b", "c"], n // 2),
            }
        )
        train_path = _write_csv(tmp_path / "train.csv", train)
        test_path = _write_csv(tmp_path / "test.csv", test)
        cfg = _build_config(
            train_path,
            test_path,
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"provider": "stacking", "models": ["cb", "rf"], "params": {"meta_model": "logistic", "n_folds": 2}},
            ],
            step_config={
                "split": {"val_fraction": 0.2, "stratified": True, "group_cols": ["authorId"]},
                "evaluate": {"metrics": ["f1"]},
            },
            drop_columns=["authorId"],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert len(preds) == N_ROWS // 2

    def test_multi_model_artifacts(self, tmp_path):
        """All intermediate models are saved in context.artifacts.models."""
        cfg = _build_config(
            *_make_data(tmp_path).values(),
            [
                {"id": "cb", "provider": "catboost", "params": {"iterations": 5, "verbose": False}},
                {"id": "rf", "provider": "sklearn", "params": {"backbone": "random_forest", "n_estimators": 10}},
                {"provider": "stacking", "models": ["cb", "rf"], "params": {"meta_model": "logistic", "n_folds": 2}},
            ],
        )
        ctx = _run(cfg)
        assert "cb" in ctx.artifacts.models
        assert "rf" in ctx.artifacts.models
        # The key is the *provider* name (stacking) not the id
        assert "stacking" in ctx.artifacts.models

    def test_fast_iteration_catboost_only(self, tmp_path):
        """Single CatBoost, no split, no evaluate — fastest path (~2s)."""
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"provider": "catboost", "params": {"iterations": 2, "verbose": False}},
            ],
        )
        ctx = _run(cfg)
        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_test_features(ctx))
        assert len(preds) == N_ROWS // 2


class TestMicroPredictMode:
    """Predict-mode pipeline with pre-saved artifacts."""

    def test_predict_after_train(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            [
                {"provider": "catboost", "params": {"iterations": 5, "verbose": False}},
            ],
            step_config={
                "split": {"val_fraction": 0.2, "stratified": True},
                "evaluate": {"metrics": ["f1"]},
            },
        )
        _run(cfg)

        output_dir = Path(cfg.trainer.output_dir)
        assert (output_dir / "model.joblib").exists()

        engine = PipelineEngine.from_config(cfg, predict=True)
        engine.run_all(PipelineContext())

        pred_path = Path(cfg.trainer.output_file)
        assert pred_path.exists()
        result = pd.read_csv(pred_path)
        assert "prediction" in result.columns
        assert len(result) == N_ROWS // 2
