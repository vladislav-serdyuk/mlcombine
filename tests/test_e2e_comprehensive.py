from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlcombine.core.pipeline import PipelineEngine
from mlcombine.core.types import MLCombineConfig, PipelineContext

HAS_CATBOOST = True
try:
    import catboost  # noqa: F401
except ImportError:
    HAS_CATBOOST = False

HAS_LIGHTGBM = True
try:
    import lightgbm  # noqa: F401
except ImportError:
    HAS_LIGHTGBM = False

HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:
    HAS_TORCH = False


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _make_dataset(tmp_path: Path, n: int = 100) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.uniform(0, 10, n),
            "cat1": np.random.choice(["a", "b", "c"], n),
            "text_short": [f"item {i}" for i in range(n)],
            "text_long": [f"long product description with many words and details for testing {i}" for i in range(n)],
            "target": rng.integers(0, 4, n),
        }
    )
    df.loc[::7, "text_long"] = None
    return {
        "train": _write_csv(tmp_path / "train.csv", df),
        "test": _write_csv(tmp_path / "test.csv", df.copy()),
    }


def _make_regression_data(tmp_path: Path, n: int = 100) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.uniform(0, 10, n),
            "cat1": np.random.choice(["a", "b", "c"], n),
            "target": rng.normal(5, 2, n),
        }
    )
    return {
        "train": _write_csv(tmp_path / "train.csv", df),
        "test": _write_csv(tmp_path / "test.csv", df.copy()),
    }


def _build_config(
    train_path: Path,
    test_path: Path,
    *,
    provider: str = "sklearn",
    params: dict | None = None,
    with_split: bool = False,
    with_all_steps: bool = False,
) -> MLCombineConfig:
    params = params or {}
    if provider == "catboost":
        params.setdefault("backbone", "gradient_boosting")
        params.setdefault("iterations", 5)
        params.setdefault("verbose", False)
    elif provider == "lightgbm":
        params.setdefault("backbone", "gradient_boosting")
        params.setdefault("verbose", -1)
    elif provider == "pytorch":
        params.setdefault("num_classes", 4)
    elif provider == "sklearn":
        params.setdefault("backbone", "random_forest")

    raw: dict = {
        "data": {
            "train_df": str(train_path),
            "test_df": str(test_path),
            "target_col": "target",
        },
        "model": [
            {"provider": provider, "params": params},
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
    if with_all_steps:
        raw["step_config"].update(
            {
                "column_length": {"columns": ["text_long"]},
                "same_value": {"pairs": [["num1", "num2"]], "drop_source": False},
                "diff_ratio": {"pairs": [["num1", "num2"]]},
                "text_overlap": {"pairs": [["text_long", "text_long"]], "token_pattern": "[a-z]+"},
                "split": {"val_fraction": 0.2, "stratified": True},
                "evaluate": {"metrics": ["f1"]},
            }
        )
    return MLCombineConfig(**raw)


def _run_train(cfg: MLCombineConfig) -> PipelineContext:
    return PipelineEngine.from_config(cfg).run_all(PipelineContext())


def _features_for_predict(ctx: PipelineContext) -> pd.DataFrame:
    df = ctx.data.test_df.copy()
    target = ctx.data.target_col
    if isinstance(target, str) and target in df.columns:
        df = df.drop(columns=[target])
    return df


# ── provider parametrization ──────────────────────────────────────────

_PROVIDER_CONFIGS = [
    pytest.param("sklearn", {"backbone": "random_forest"}, id="sklearn"),
    pytest.param(
        "catboost",
        {"backbone": "gradient_boosting", "iterations": 5, "verbose": False},
        marks=pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed"),
        id="catboost",
    ),
    pytest.param("pytorch", {"num_classes": 4}, marks=pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"), id="pytorch"),
]


# ── tests ─────────────────────────────────────────────────────────────


class TestE2EProviders:
    """Parametrized test: train+predict for every model provider."""

    @pytest.mark.parametrize("provider,params", _PROVIDER_CONFIGS)
    def test_train_and_predict(self, tmp_path, provider, params):
        is_classification = provider != "sklearn" or params.get("backbone") != "random_forest"
        if is_classification:
            paths = _make_dataset(tmp_path)
        else:
            paths = _make_regression_data(tmp_path)

        cfg = _build_config(paths["train"], paths["test"], provider=provider, params=params)
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 100

        output_dir = Path(cfg.trainer.output_dir)
        assert (output_dir / "model.joblib").exists()

    @pytest.mark.parametrize("provider,params", _PROVIDER_CONFIGS)
    def test_save_and_predict(self, tmp_path, provider, params):
        is_classification = provider != "sklearn" or params.get("backbone") != "random_forest"
        if is_classification:
            paths = _make_dataset(tmp_path)
        else:
            paths = _make_regression_data(tmp_path)

        cfg = _build_config(paths["train"], paths["test"], provider=provider, params=params)
        _run_train(cfg)

        engine = PipelineEngine.from_config(cfg, predict=True)
        engine.run_all(PipelineContext())

        pred_path = Path(cfg.trainer.output_file)
        assert pred_path.exists()
        result = pd.read_csv(pred_path)
        assert "target" in result.columns
        assert len(result) == 100

    @pytest.mark.parametrize("provider,params", _PROVIDER_CONFIGS)
    def test_with_holdout(self, tmp_path, provider, params):
        is_classification = provider != "sklearn" or params.get("backbone") != "random_forest"
        if is_classification:
            paths = _make_dataset(tmp_path)
        else:
            paths = _make_regression_data(tmp_path)

        cfg = _build_config(paths["train"], paths["test"], provider=provider, params=params, with_split=True)
        ctx = _run_train(cfg)

        assert ctx.data.holdout_df is not None
        assert ctx.artifacts.evaluation_results is not None

    @pytest.mark.parametrize("provider,params", _PROVIDER_CONFIGS)
    def test_text_feature_nan(self, tmp_path, provider, params):
        if provider == "sklearn":
            pytest.skip("sklearn does not support text features")
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

        cfg = _build_config(train, test, provider=provider, params=params, with_split=False)
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert len(preds) == 20


class TestE2EGPU:
    """GPU-specific tests."""

    def test_catboost_text_features_nan_with_gpu(self, tmp_path):
        HAS_GPU = False
        try:
            from catboost import CatBoostClassifier

            m = CatBoostClassifier(task_type="GPU", iterations=1, verbose=False)
            m.fit([[1.0], [2.0], [3.0]], [0, 1, 0])
            HAS_GPU = True
        except Exception:
            pass
        if not HAS_GPU:
            pytest.skip("CatBoost GPU not available")

        n = 80
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "leftItemId": rng.integers(1_000_000_000_000_000_000, 9_000_000_000_000_000_000, n, dtype=np.int64),
                "content": [f"long product description with many words for testing purposes {i}" for i in range(n)],
                "target": rng.integers(0, 3, n),
            }
        )
        df.loc[::5, "content"] = None
        train = _write_csv(tmp_path / "train.csv", df)

        cfg = MLCombineConfig(
            **{
                "data": {
                    "train_df": str(train),
                    "test_df": str(train),
                    "target_col": "target",
                },
                "model": [
                    {
                        "provider": "catboost",
                        "params": {
                            "backbone": "gradient_boosting",
                            "iterations": 5,
                            "verbose": False,
                            "gpu": True,
                        },
                    },
                ],
                "handling": {
                    "numbers": {"impute": "none", "scale": "none"},
                    "categories": {"encode": "none"},
                },
                "trainer": {
                    "output_dir": str(tmp_path / "outputs"),
                },
            }
        )
        ctx = _run_train(cfg)
        assert ctx.artifacts.model is not None
        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert len(preds) == 80


class TestE2EAllSteps:
    """All feature-engineering steps enabled simultaneously."""

    def test_all_steps_together(self, tmp_path):
        paths = _make_dataset(tmp_path)
        cfg = _build_config(
            paths["train"],
            paths["test"],
            provider="catboost",
            params={
                "backbone": "gradient_boosting",
                "iterations": 5,
                "verbose": False,
            },
            with_all_steps=True,
        )
        ctx = _run_train(cfg)

        assert ctx.artifacts.model is not None
        train_df = ctx.data.train_df
        assert train_df is not None
        assert "text_long_len" in train_df.columns
        assert "num1_same" in train_df.columns
        assert "num1_diff" in train_df.columns
        assert "num1_ratio" in train_df.columns
        assert "text_long_word_overlap" in train_df.columns
        assert ctx.data.holdout_df is not None
        assert ctx.artifacts.evaluation_results is not None
        assert "f1" in ctx.artifacts.evaluation_results

        preds = ctx.artifacts.model.predict(_features_for_predict(ctx))
        assert len(preds) == 100


class TestE2ECLI:
    """CLI entry point — mlcombine train / predict."""

    def _write_mini_config(self, path: Path, train_csv: str, test_csv: str, output_dir: str, output_file: str) -> Path:
        config_content = f"""
data:
  train_df: "{train_csv}"
  test_df: "{test_csv}"
  target_col: "target"
model:
  - provider: "catboost"
    params:
      backbone: "gradient_boosting"
      iterations: 5
      verbose: false
handling:
  numbers:
    impute: "none"
    scale: "none"
  categories:
    encode: "ordinal"
trainer:
  output_dir: "{output_dir}"
  output_file: "{output_file}"
"""
        path.write_text(config_content)
        return path

    def test_cli_train(self, tmp_path):
        from click.testing import CliRunner
        from mlcombine.cli.main import cli

        paths = _make_dataset(tmp_path)
        output_dir = tmp_path / "outputs"
        output_file = output_dir / "predictions.csv"
        config_path = self._write_mini_config(
            tmp_path / "mlcombine.yaml",
            str(paths["train"]),
            str(paths["test"]),
            str(output_dir),
            str(output_file),
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["train", "-c", str(config_path), "-v"])

        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert (output_dir / "model.joblib").exists(), "model.joblib not created"

    def test_cli_train_predict_cycle(self, tmp_path):
        from click.testing import CliRunner
        from mlcombine.cli.main import cli

        paths = _make_dataset(tmp_path)
        output_dir = tmp_path / "outputs"
        output_file = output_dir / "predictions.csv"
        config_path = self._write_mini_config(
            tmp_path / "mlcombine.yaml",
            str(paths["train"]),
            str(paths["test"]),
            str(output_dir),
            str(output_file),
        )

        runner = CliRunner()

        train_result = runner.invoke(cli, ["train", "-c", str(config_path), "-v"])
        assert train_result.exit_code == 0, f"Train failed:\n{train_result.output}"
        assert (output_dir / "model.joblib").exists()

        predict_result = runner.invoke(cli, ["predict", "-c", str(config_path), "-v"])
        assert predict_result.exit_code == 0, f"Predict failed:\n{predict_result.output}"
        assert output_file.exists()
        df = pd.read_csv(output_file)
        assert "target" in df.columns
        assert len(df) == 100
