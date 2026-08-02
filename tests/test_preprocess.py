import numpy as np
import pytest

from mlcombine.core.types import (
    FeatureType,
    MLCombineConfig,
    PipelineContext,
    PipelineData,
)
from mlcombine.steps.preprocess import EncodeScaleStep, ImputeStep


def _impute_cfg(strategy: str = "median", columns: dict[str, object] | None = None) -> MLCombineConfig:
    raw: dict[str, object] = {
        "data": {"train_df": "train.csv", "test_df": "test.csv", "target_col": "target"},
        "handling": {"numbers": {"impute": strategy}},
        "model": [{"provider": "sklearn"}],
    }
    if columns:
        raw["handling"] = {"numbers": {"impute": strategy}, "columns": columns}
    return MLCombineConfig(**raw)


def _encode_cfg(
    encode: str = "ordinal",
    scale: str = "standard",
    target_col: str | None = "target",
    columns: dict[str, object] | None = None,
) -> MLCombineConfig:
    raw: dict[str, object] = {
        "data": {"train_df": "train.csv", "test_df": "test.csv", "target_col": target_col or "target"},
        "handling": {
            "categories": {"encode": encode, "smoothing": 10.0},
            "numbers": {"impute": "median", "scale": scale},
        },
        "model": [{"provider": "sklearn"}],
    }
    if columns:
        raw["handling"] = {
            "categories": {"encode": encode, "smoothing": 10.0},
            "numbers": {"impute": "median", "scale": scale},
            "columns": columns,
        }
    return MLCombineConfig(**raw)


def _column_cfg(
    encode: str | None = None,
    impute: str | None = None,
    scale: str | None = None,
    fill_value: float | None = None,
) -> dict[str, object]:
    raw: dict[str, object] = {}
    if encode is not None:
        raw["encode"] = encode
    if impute is not None:
        raw["impute"] = impute
    if scale is not None:
        raw["scale"] = scale
    if fill_value is not None:
        raw["fill_value"] = fill_value
    return raw


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

    def test_column_impute_override(self, sample_df_with_missing):
        train = sample_df_with_missing.copy()
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=train.copy()))
        step = ImputeStep(_impute_cfg("none", columns={"num1": _column_cfg(impute="median")}))
        ctx = step.run(ctx)
        assert ctx.data.train_df is not None
        assert ctx.data.test_df is not None
        assert ctx.data.train_df["num1"].isna().sum() == 0
        assert ctx.data.test_df["num1"].isna().sum() == 0
        # num2 has no override → global none → untouched
        assert ctx.data.train_df["num2"].isna().sum() > 0

    def test_column_impute_constant_fill_value(self, sample_df_with_missing):
        train = sample_df_with_missing.copy()
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=train.copy()))
        step = ImputeStep(_impute_cfg("none", columns={"num1": _column_cfg(impute="constant", fill_value=42.0)}))
        ctx = step.run(ctx)
        assert ctx.data.train_df is not None
        assert ctx.data.train_df["num1"].isna().sum() == 0
        assert (ctx.data.train_df["num1"].dropna() == 42.0).any() or (ctx.data.train_df["num1"] == 42.0).all()
        assert ctx.data.train_df["num1"].isin([42.0]).sum() > 0

    def test_is_required_with_column_override(self):
        cfg = MLCombineConfig(
            **{
                "data": {"train_df": "train.csv", "test_df": "test.csv", "target_col": "target"},
                "handling": {
                    "numbers": {"impute": "none"},
                    "columns": {"num1": {"impute": "median"}},
                },
                "model": [{"provider": "sklearn"}],
            }
        )
        assert ImputeStep.is_required(cfg)
        cfg2 = MLCombineConfig(
            **{
                "data": {"train_df": "train.csv", "test_df": "test.csv", "target_col": "target"},
                "handling": {"numbers": {"impute": "none"}},
                "model": [{"provider": "sklearn"}],
            }
        )
        assert not ImputeStep.is_required(cfg2)


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

    def test_onehot_encode(self, sample_df):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df.copy()))
        step = EncodeScaleStep(_encode_cfg(encode="onehot", scale="none"))
        ctx = step.run(ctx)
        df = ctx.data.train_df
        assert df is not None
        assert "cat1" not in df.columns
        n_categories = len(sample_df["cat1"].cat.categories)
        onehot_cols = [c for c in df.columns if c.startswith("cat1_")]
        assert len(onehot_cols) == n_categories
        # exactly one active column per row
        assert (df[onehot_cols].sum(axis=1) == 1).all()

    def test_onehot_unknown_category(self, sample_df):
        train = sample_df.copy()
        test = sample_df.copy()
        test["cat1"] = test["cat1"].cat.add_categories("unseen_category")
        test.loc[0, "cat1"] = "unseen_category"
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=test))
        step = EncodeScaleStep(_encode_cfg(encode="onehot", scale="none"))
        ctx = step.run(ctx)
        assert ctx.data.test_df is not None
        onehot_cols = [c for c in ctx.data.test_df.columns if c.startswith("cat1_")]
        # unknown category → all zero columns
        assert ctx.data.test_df[onehot_cols].iloc[0].sum() == 0

    def test_onehot_with_nan(self, sample_df):
        train = sample_df.copy()
        train.loc[0, "cat1"] = None
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=train.copy()))
        step = EncodeScaleStep(_encode_cfg(encode="onehot", scale="none"))
        ctx = step.run(ctx)
        assert ctx.data.train_df is not None
        onehot_cols = [c for c in ctx.data.train_df.columns if c.startswith("cat1_")]
        # NaN gets its own category → exactly one active column per row
        assert (ctx.data.train_df[onehot_cols].sum(axis=1) == 1).all()

    def test_column_encode_override_onehot(self, sample_df):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df.copy()))
        step = EncodeScaleStep(_encode_cfg(encode="ordinal", scale="none", columns={"cat1": _column_cfg(encode="onehot")}))
        ctx = step.run(ctx)
        df = ctx.data.train_df
        assert df is not None
        onehot_cols = [c for c in df.columns if c.startswith("cat1_")]
        assert len(onehot_cols) == len(sample_df["cat1"].cat.categories)
        assert "cat1" not in df.columns
        # cat2 keeps the global ordinal strategy
        assert "cat2" in df.columns
        assert df["cat2"].dtype in (np.float64, np.int64, float, int)

    def test_column_encode_none_skips(self, sample_df):
        ctx = PipelineContext(data=PipelineData(train_df=sample_df.copy()))
        step = EncodeScaleStep(_encode_cfg(encode="ordinal", scale="none", columns={"cat1": _column_cfg(encode="none")}))
        ctx = step.run(ctx)
        df = ctx.data.train_df
        assert df is not None
        # cat1 untouched (still categorical), cat2 encoded ordinally
        assert df["cat1"].dtype.name == "category"
        assert df["cat2"].dtype in (np.float64, np.int64, float, int)

    def test_column_scale_none(self, sample_df):
        train = sample_df.copy()
        train["num1"] = train["num1"] * 100  # huge scale → contrast after scaling
        ctx = PipelineContext(data=PipelineData(train_df=train))
        step = EncodeScaleStep(_encode_cfg(scale="standard", columns={"num1": _column_cfg(scale="none")}))
        ctx = step.run(ctx)
        df = ctx.data.train_df
        assert df is not None
        # num1 unscaled → std ~100; num2 scaled → std ≈ 1
        assert df["num1"].std() > 50
        assert abs(df["num2"].std() - 1.0) < 0.3
        # target column is never scaled
        assert df["target"].std() > 1.5

    def test_predict_mode_with_column_overrides(self, sample_df):
        train = sample_df.copy()
        test = sample_df.copy()
        test["cat1"] = test["cat1"].cat.add_categories("unseen")
        test.loc[0, "cat1"] = "unseen"
        ctx = PipelineContext(data=PipelineData(train_df=train, test_df=test))
        step = EncodeScaleStep(_encode_cfg(encode="ordinal", scale="none", columns={"cat1": _column_cfg(encode="onehot")}))
        ctx = step.run(ctx)

        predict_ctx = PipelineContext(data=PipelineData(test_df=test))
        predict_ctx.artifacts.encoders = ctx.artifacts.encoders
        predict_ctx.artifacts.scalers = ctx.artifacts.scalers
        predict_ctx.artifacts.scaler_features = ctx.artifacts.scaler_features
        predict_ctx.data.detected_types = {"cat1": FeatureType.CATEGORY, "cat2": FeatureType.CATEGORY}
        predict_step = EncodeScaleStep(_encode_cfg(encode="ordinal", scale="none", columns={"cat1": _column_cfg(encode="onehot")}), predict=True)
        predict_ctx = predict_step.run(predict_ctx)

        assert predict_ctx.data.test_df is not None
        onehot_cols = [c for c in predict_ctx.data.test_df.columns if c.startswith("cat1_")]
        assert len(onehot_cols) == len(sample_df["cat1"].cat.categories)
        # unknown category on predict → all zeros, not a crash
        assert predict_ctx.data.test_df[onehot_cols].iloc[0].sum() == 0
