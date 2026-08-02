"""End-to-end tests — full pipeline from CSV to predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlcombine.core.pipeline import PipelineEngine
from mlcombine.core.types import (
    DatasetNotFoundError,
    EmptyDatasetError,
    MLCombineConfig,
    PipelineContext,
    UnsupportedBackendError,
)


# ── helpers ──────────────────────────────────────────────────────────


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _make_numeric_data(tmp_path: Path) -> dict[str, Path]:
    """Regression dataset — all numeric, no categoricals."""
    rng = np.random.default_rng(42)
    n = 100
    train = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.uniform(0, 10, n),
            "num3": rng.poisson(3, n),
            "target": rng.normal(5, 2, n),
        }
    )
    test = train.copy()
    test["target"] = rng.normal(5, 2, n)
    return {
        "train": _write_csv(tmp_path / "train.csv", train),
        "test": _write_csv(tmp_path / "test.csv", test),
    }


def _make_uplift_data(tmp_path: Path) -> dict[str, Path]:
    """Binary treatment column — all numeric."""
    rng = np.random.default_rng(42)
    n = 100
    train = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.uniform(0, 1, n),
            "treatment": rng.integers(0, 2, n),
            "target": rng.normal(5, 2, n),
        }
    )
    test = train.copy()
    test["target"] = rng.normal(5, 2, n)
    return {
        "train": _write_csv(tmp_path / "train.csv", train),
        "test": _write_csv(tmp_path / "test.csv", test),
    }


def _build_config(
    train_path: Path,
    test_path: Path,
    *,
    provider: str = "sklearn",
    backbone: str = "random_forest",
    target_col: str = "target",
    treatment_col: str | None = None,
    uplift_provider: str | None = None,
    extra: dict | None = None,
) -> MLCombineConfig:
    raw: dict = {
        "data": {
            "train_df": str(train_path),
            "test_df": str(test_path),
            "target_col": target_col,
        },
        "model": [
            {"provider": provider, "params": {"backbone": backbone, "objective": "RMSE"}},
        ],
        "handling": {
            "numbers": {"impute": "median", "scale": "standard"},
            "categories": {"encode": "ordinal", "smoothing": 10.0},
        },
        "trainer": {
            "output_dir": str(train_path.parent / "outputs"),
            "output_file": str(train_path.parent / "outputs" / "predictions.csv"),
        },
    }
    if treatment_col is not None:
        raw["data"]["treatment_col"] = treatment_col
    if uplift_provider is not None:
        raw["model"].append(
            {
                "provider": uplift_provider,
                "model": "sklearn",
            }
        )
    if extra:
        raw.update(extra)
    return MLCombineConfig(**raw)


def _run_train(cfg: MLCombineConfig) -> PipelineContext:
    engine = PipelineEngine.from_config(cfg)
    return engine.run_all(PipelineContext())


def _features_for_predict(ctx: PipelineContext) -> pd.DataFrame:
    """Return test_df with target (and treatment) columns removed, matching train."""
    df = ctx.data.test_df.copy()
    treat = ctx.data.treatment_col
    target = ctx.data.target_col
    to_drop: list[str] = []
    if isinstance(target, str):
        to_drop.append(target)
    if treat is not None:
        to_drop.append(treat)
    if to_drop:
        df = df.drop(columns=[c for c in to_drop if c in df.columns])
    return df


# ── scenarios ────────────────────────────────────────────────────────


class TestE2ETrainRegression:
    """Full train pipeline — all-numeric data."""

    def test_train_creates_model(self, tmp_path):
        paths = _make_numeric_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"])
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        assert ctx.data.train_df is not None
        assert ctx.data.test_df is not None

        # Predict on feature subset (exclude target)
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 100

    def test_train_saves_artifacts(self, tmp_path):
        paths = _make_numeric_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"])
        ctx = _run_train(cfg)

        output_dir = Path(cfg.trainer.output_dir)
        assert (output_dir / "model.joblib").exists()
        assert ctx.artifacts.model is not None

    def test_train_with_missing_values(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "num1": rng.normal(0, 1, n),
                "num2": rng.uniform(0, 10, n),
                "target": rng.normal(5, 2, n),
            }
        )
        df.loc[::5, "num1"] = None
        df.loc[::7, "num2"] = None
        train = _write_csv(tmp_path / "train.csv", df)
        test = _write_csv(tmp_path / "test.csv", df.copy())

        cfg = _build_config(train, test)
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert len(preds) == 100


class TestE2EPredict:
    """End-to-end train → predict flow."""

    def test_predict_creates_csv(self, tmp_path):
        paths = _make_numeric_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"])
        _run_train(cfg)

        # Predict pipeline — load saved model, drop target, predict
        pred_cfg = _build_config(paths["train"], paths["test"])
        engine = PipelineEngine.from_config(pred_cfg, predict=True)
        engine.run_all(PipelineContext())

        pred_path = Path(cfg.trainer.output_file)
        assert pred_path.exists()
        df = pd.read_csv(pred_path)
        assert "target" in df.columns
        assert len(df) == 100


class TestE2EUplift:
    """Uplift pipeline with treatment column."""

    @pytest.mark.parametrize("method", ["t_learner", "s_learner"])
    def test_uplift_train(self, tmp_path, method):
        paths = _make_uplift_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            treatment_col="treatment",
            uplift_provider=method,
        )
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        assert ctx.data.test_df is not None

        # Predict on feature subset (exclude target + treatment)
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 100

    def test_uplift_predict(self, tmp_path):
        paths = _make_uplift_data(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            treatment_col="treatment",
            uplift_provider="t_learner",
        )
        ctx = _run_train(cfg)
        assert ctx.artifacts.model is not None

        # quick sanity — predict before saving
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert len(preds) == 100


class TestE2EBackboneDispatch:
    """Different sklearn backbones produce different model types."""

    def test_gradient_boosting(self, tmp_path):
        paths = _make_numeric_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], backbone="gradient_boosting")
        ctx = _run_train(cfg)
        assert "GradientBoostingRegressor" in type(ctx.artifacts.model._model).__name__

    def test_svm(self, tmp_path):
        paths = _make_numeric_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], backbone="svm")
        ctx = _run_train(cfg)
        assert "SVR" in type(ctx.artifacts.model._model).__name__


class TestE2EErrors:
    """Error paths — file not found, empty dataset, invalid config."""

    def test_dataset_not_found(self, tmp_path):
        cfg = _build_config(tmp_path / "nonexistent.csv", tmp_path / "nonexistent.csv")
        with pytest.raises(DatasetNotFoundError):
            _run_train(cfg)

    def test_empty_dataset(self, tmp_path):
        df = pd.DataFrame({"a": [], "b": []})
        train = _write_csv(tmp_path / "empty.csv", df)
        cfg = _build_config(train, train)
        with pytest.raises(EmptyDatasetError):
            _run_train(cfg)

    def test_extra_config_field_rejected(self, tmp_path):
        paths = _make_numeric_data(tmp_path)
        with pytest.raises(ValidationError):
            _build_config(paths["train"], paths["test"], extra={"invalid_key": "value"})

    def test_unsupported_backend(self):
        cfg = MLCombineConfig(
            data={"train_df": "/none", "test_df": "/none", "target_col": "t"},
            model=[{"provider": "__nonexistent__"}],
        )
        from mlcombine.core.builder import ModelBuilder

        bp = ModelBuilder().build_all(cfg.model)
        with pytest.raises(UnsupportedBackendError):
            bp.build()

    def test_uplift_without_treatment(self, tmp_path):
        paths = _make_uplift_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], uplift_provider="t_learner")
        with pytest.raises((ValueError, RuntimeError)):
            _run_train(cfg)
