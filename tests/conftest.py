import numpy as np
import pandas as pd
import pytest

from mlcombine.core.types import MLCombineConfig, PipelineContext


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.uniform(0, 10, n),
            "cat1": pd.Categorical(np.random.choice(["a", "b", "c"], n)),
            "cat2": pd.Categorical(np.random.choice(["x", "y"], n)),
            "target": rng.normal(5, 2, n),
        }
    )


@pytest.fixture
def sample_df_with_missing() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.uniform(0, 10, n),
            "cat1": pd.Categorical(np.random.choice(["a", "b", "c"], n)),
            "target": rng.normal(5, 2, n),
        }
    )
    df.loc[::5, "num1"] = None
    df.loc[::7, "num2"] = None
    return df


@pytest.fixture
def model_cfg() -> None:
    return None


@pytest.fixture
def pipeline_context() -> PipelineContext:
    return PipelineContext()


@pytest.fixture
def minimal_config_dict() -> dict:
    return {
        "data": {
            "train_df": "train.csv",
            "test_df": "test.csv",
            "target_col": "target",
        },
        "model": [
            {"provider": "sklearn", "params": {"objective": "MAPE"}},
        ],
    }


@pytest.fixture
def minimal_config(minimal_config_dict: dict) -> MLCombineConfig:
    return MLCombineConfig(**minimal_config_dict)


@pytest.fixture
def sklearn_cfg() -> MLCombineConfig:
    return MLCombineConfig(
        **{
            "data": {
                "train_df": "train.csv",
                "test_df": "test.csv",
                "target_col": "target",
            },
            "model": [
                {"provider": "sklearn", "params": {"backbone": "random_forest", "objective": "MAPE"}},
            ],
        }
    )
