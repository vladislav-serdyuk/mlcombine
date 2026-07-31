import numpy as np
import pytest

from mlcombine.core.types import (
    FeatureType,
    MLCombineConfig,
    PipelineContext,
    PipelineData,
)
from mlcombine.steps.preprocess import EncodeScaleStep, ImputeStep


def _impute_cfg(strategy: str = "median") -> MLCombineConfig:
    return MLCombineConfig(
        **{
            "data": {"train_df": "train.csv", "test_df": "test.csv", "target_col": "target"},
            "handling": {"numbers": {"impute": strategy}},
            "model": [{"provider": "sklearn"}],
        }
    )


def _encode_cfg(encode: str = "ordinal", scale: str = "standard", target_col: str | None = "target") -> MLCombineConfig:
    raw: dict[str, object] = {
        "data": {"train_df": "train.csv", "test_df": "test.csv", "target_col": target_col or "target"},
        "handling": {
            "categories": {"encode": encode, "smoothing": 10.0},
            "numbers": {"impute": "median", "scale": scale},
        },
        "model": [{"provider": "sklearn"}],
    }
    return MLCombineConfig(**raw)


class TestImputeStep:
    def test_impute_median(self, sample_df_with_missing):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df_with_missing))
        step = ImputeStep(_impute_cfg("median"))
        ctx = step.run(ctx)
        assert ctx.data.train_df is not None
        assert ctx.data.train_df["num1"].isna().sum() == 0
        assert ctx.data.train_df["num2"].isna().sum() == 0

    def test_impute_no_nans(self, sample_df):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df.copy()))
        step = ImputeStep(_impute_cfg("mean"))
        before = sample_df["num1"].iloc[0]
        ctx = step.run(ctx)
        assert ctx.data.train_df["num1"].iloc[0] == before

    def test_impute_train_test(self, sample_df_with_missing):
        train = sample_df_with_missing.copy()
        test = sample_df_with_missing.copy()
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=test))
        step = ImputeStep(_impute_cfg("median"))
        ctx = step.run(ctx)
        assert ctx.data.train_df is not None
        assert ctx.data.test_df is not None
        assert ctx.data.test_df["num1"].isna().sum() == 0

    def test_fit_transform_with_detected_types(self, sample_df_with_missing):
        ctx = PipelineContext(
            data=PipelineData(
                train_df=sample_df_with_missing,
                detected_types={"num1": FeatureType.NUMBER, "num2": FeatureType.NUMBER, "cat1": FeatureType.CATEGORY, "target": FeatureType.NUMBER},
            )
        )
        step = ImputeStep(_impute_cfg("median"))
        ctx = step.run(ctx)
        assert ctx.data.train_df["num1"].isna().sum() == 0
        # categorical column should be untouched
        assert ctx.data.train_df["cat1"].dtype.name == "category"


class TestEncodeScaleStep:
    def test_ordinal_encode_and_scale(self, sample_df):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df.copy()))
        step = EncodeScaleStep(_encode_cfg())
        ctx = step.run(ctx)
        df = ctx.data.train_df
        assert df is not None
        assert df["cat1"].dtype in (np.float64, np.int64, float, int)
        assert abs(df["num1"].mean()) < 0.5  # standard scaled → near zero mean

    def test_no_scale(self, sample_df):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df.copy()))
        step = EncodeScaleStep(_encode_cfg(scale="none"))
        ctx = step.run(ctx)
        df = ctx.data.train_df
        assert df is not None

    def test_transform_test(self, sample_df):
        train = sample_df.copy()
        test = sample_df.copy()
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=test))
        step = EncodeScaleStep(_encode_cfg())
        ctx = step.run(ctx)
        assert ctx.data.test_df is not None
        assert ctx.data.test_df["cat1"].dtype in (np.float64, np.int64, float, int)

    def test_target_encoding_raises(self, sample_df):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df.copy()))
        step = EncodeScaleStep(_encode_cfg(encode="target"))
        with pytest.raises(NotImplementedError):
            ctx = step.run(ctx)

    def test_handle_unknown_category(self, sample_df):
        train = sample_df.copy()
        test = sample_df.copy()
        test["cat1"] = test["cat1"].cat.add_categories("unseen_category")
        test.loc[0, "cat1"] = "unseen_category"
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=test))
        step = EncodeScaleStep(_encode_cfg(scale="none"))
        ctx = step.run(ctx)
        assert ctx.data.test_df is not None
        # unknown category should get -1
        assert ctx.data.test_df["cat1"].iloc[0] == -1
