from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlcombine.core.pipeline import PipelineEngine
from mlcombine.core.types import MLCombineConfig, PipelineContext
from mlcombine.steps.cross_encoder import CrossEncoderStep


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _make_data(tmp_path: Path, n: int = 100) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.uniform(0, 10, n),
            "cat1": np.random.choice(["a", "b", "c"], n),
            "text_long": [f"long product description with many words and details for testing {i}" for i in range(n)],
            "target": rng.integers(0, 4, n),
        }
    )
    df.loc[::7, "text_long"] = None
    return {
        "train": _write_csv(tmp_path / "train.csv", df),
        "test": _write_csv(tmp_path / "test.csv", df.copy()),
    }


def _build_config(
    train_path: Path,
    test_path: Path,
    *,
    with_split: bool = True,
    with_cross_encoder: bool = False,
    ce_cache_dir: str | None = None,
) -> MLCombineConfig:
    raw: dict = {
        "data": {
            "train_df": str(train_path),
            "test_df": str(test_path),
            "target_col": "target",
        },
        "model": [
            {
                "provider": "catboost",
                "params": {
                    "backbone": "gradient_boosting",
                    "iterations": 10,
                    "verbose": False,
                },
            },
        ],
        "handling": {
            "numbers": {"impute": "median", "scale": "none"},
            "categories": {"encode": "ordinal", "smoothing": 10.0},
        },
        "step_config": {},
        "trainer": {
            "output_dir": str(train_path.parent / "outputs"),
            "output_file": str(train_path.parent / "outputs" / "predictions.csv"),
        },
    }
    if with_split:
        raw["step_config"]["split"] = {"val_fraction": 0.2, "stratified": True}
        raw["step_config"]["evaluate"] = {"metrics": ["f1"]}
    if with_cross_encoder:
        raw["step_config"]["cross_encoder"] = {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "pairs": [["text_long", "text_long"]],
            "batch_size": 32,
            "predict_chunk": 200,
            "device": "cpu",
        }
        if ce_cache_dir is not None:
            raw["step_config"]["cross_encoder"]["cache_dir"] = ce_cache_dir
    return MLCombineConfig(**raw)


def _run_train(cfg: MLCombineConfig) -> PipelineContext:
    engine = PipelineEngine.from_config(cfg)
    return engine.run_all(PipelineContext())


def _features_for_predict(ctx: PipelineContext) -> pd.DataFrame:
    df = ctx.data.test_df.copy()
    target = ctx.data.target_col
    if isinstance(target, str) and target in df.columns:
        df = df.drop(columns=[target])
    return df


# ── tests ─────────────────────────────────────────────────────────────


class TestE2EMini:
    """Mini end-to-end test covering core functionality."""

    def test_cross_encoder_lazy_torch_import(self, tmp_path, monkeypatch):
        """torch must not be imported at module level — a config without
        cross_encoder must work even when torch is missing."""
        paths = _make_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], with_split=True)
        ctx = _run_train(cfg)
        assert ctx.artifacts.model is not None

        step = CrossEncoderStep(cfg)
        assert step.is_required(cfg) is False
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(RuntimeError, match="requires 'torch'"):
            step._load_model()

    def test_catboost_classification_with_holdout(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], with_split=True)
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        assert ctx.data.train_df is not None
        assert ctx.data.holdout_df is not None
        assert ctx.data.test_df is not None

        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 100

        proba = ctx.artifacts.model.predict_proba(_features_for_predict(ctx))
        assert proba.shape == (100, 4)

    def test_catboost_classification_without_split(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], with_split=False)
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        assert ctx.data.holdout_df is None

        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert len(preds) == 100

    def test_catboost_text_features_nan_in_holdout(self, tmp_path):
        n = 80
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "num1": rng.normal(0, 1, n),
                "text_col": [f"long product description with many words for testing purposes {i}" for i in range(n)],
                "target": rng.integers(0, 3, n),
            }
        )
        df.loc[::5, "text_col"] = None
        train = _write_csv(tmp_path / "train.csv", df.iloc[:60])
        test = _write_csv(tmp_path / "test.csv", df.iloc[60:])

        cfg = _build_config(train, test, with_split=False)
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert len(preds) == 20

    def test_train_save_and_predict(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], with_split=True)
        _run_train(cfg)

        output_dir = Path(cfg.trainer.output_dir)
        assert (output_dir / "model.joblib").exists()

        engine = PipelineEngine.from_config(cfg, predict=True)
        engine.run_all(PipelineContext())

        pred_path = Path(cfg.trainer.output_file)
        assert pred_path.exists()
        result = pd.read_csv(pred_path)
        assert "prediction" in result.columns
        assert len(result) == 100

    def test_evaluate_step_holdout_metrics(self, tmp_path):
        paths = _make_data(tmp_path)
        cfg = _build_config(paths["train"], paths["test"], with_split=True)
        ctx = _run_train(cfg)

        assert ctx.artifacts.evaluation_results is not None
        assert "f1" in ctx.artifacts.evaluation_results
