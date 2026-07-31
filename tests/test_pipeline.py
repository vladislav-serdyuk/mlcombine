import pytest

from mlcombine.core.pipeline import PipelineEngine, _resolve_order
from mlcombine.core.types import MLCombineConfig


@pytest.fixture
def train_cfg() -> MLCombineConfig:
    return MLCombineConfig(
        **{
            "data": {
                "train_df": "train.csv",
                "test_df": "test.csv",
                "target_col": "target",
            },
            "model": [{"provider": "sklearn", "params": {"backbone": "random_forest", "objective": "MAPE"}}],
        }
    )


@pytest.fixture
def predict_cfg() -> MLCombineConfig:
    return MLCombineConfig(
        **{
            "data": {
                "train_df": "train.csv",
                "test_df": "test.csv",
                "target_col": "target",
            },
            "model": [{"provider": "sklearn"}],
        }
    )


class TestPipelineEngineConstruction:
    def test_train_pipeline_steps(self, train_cfg):
        engine = PipelineEngine.from_config(train_cfg)
        names = [name for name, _ in engine.steps]
        assert names == [
            "PrepareEnvironmentStep",
            "DataLoaderStep",
            "TypeDetectStep",
            "FeatureGenerationStep",
            "ImputeStep",
            "EncodeScaleStep",
            "CreateModelStep",
            "ModelFitStep",
            "EvaluateStep",
            "SaveArtifactsStep",
        ]

    def test_predict_pipeline_steps(self, predict_cfg):
        engine = PipelineEngine.from_config(predict_cfg, predict=True)
        names = [name for name, _ in engine.steps]
        assert names == [
            "PrepareEnvironmentStep",
            "DataLoaderStep",
            "TypeDetectStep",
            "FeatureGenerationStep",
            "LoadArtifactsStep",
            "ImputeStep",
            "EncodeScaleStep",
            "AlignFeaturesStep",
            "ModelPredictStep",
            "DropTargetColumnsStep",
            "SavePredictionsStep",
        ]

    def test_engine_name_uniqueness(self, train_cfg):
        engine = PipelineEngine.from_config(train_cfg)
        duplicate = engine._steps[0][1]
        engine._add_step(duplicate)
        names = [name for name, _ in engine._steps]
        assert len(names) == len(set(names))


class TestResolveOrder:
    """Tests for _resolve_order — step ordering via before/after and YAML."""

    def test_default_base_order(self, train_cfg):
        order = _resolve_order(train_cfg)
        assert order[0] == "PrepareEnvironmentStep"
        assert "PrepareDatasetStep" in order
        assert "SavePredictionsStep" in order  # present but filtered by mode
        assert order.index("PrepareEnvironmentStep") == 0

    def test_pipeline_order_overrides(self, train_cfg):
        train_cfg.pipeline.order = ["TypeDetectStep", "DataLoaderStep", "PrepareDatasetStep"]
        order = _resolve_order(train_cfg)
        assert order == ["TypeDetectStep", "DataLoaderStep", "PrepareDatasetStep"]

    def test_custom_step_before_inserts_correctly(self):
        """A step registered with before='TypeDetectStep' appears right before it."""
        from mlcombine.core.registry import ExtensionRegistry

        reg = ExtensionRegistry()

        @reg.step("custom_validate", before="TypeDetectStep")
        class FakeStep:
            pass

        from mlcombine.core.types import MLCombineConfig

        cfg = MLCombineConfig(
            **{
                "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
                "model": [{"provider": "sklearn"}],
            }
        )
        result = _resolve_order(cfg, _registry=reg)
        assert "custom_validate" in result
        tidx = result.index("TypeDetectStep")
        cidx = result.index("custom_validate")
        assert cidx + 1 == tidx  # custom_validate is right before TypeDetectStep

    def test_custom_step_after_inserts_correctly(self):
        """A step registered with after='DataLoaderStep' appears right after it."""
        from mlcombine.core.registry import ExtensionRegistry

        reg = ExtensionRegistry()

        @reg.step("custom_stats", after="DataLoaderStep")
        class FakeStep:
            pass

        from mlcombine.core.types import MLCombineConfig

        cfg = MLCombineConfig(
            **{
                "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
                "model": [{"provider": "sklearn"}],
            }
        )
        import mlcombine.core.pipeline as pipeline_mod

        original_registry = pipeline_mod.registry
        pipeline_mod.registry = reg
        try:
            result = _resolve_order(cfg)
            didx = result.index("DataLoaderStep")
            cidx = result.index("custom_stats")
            assert cidx == didx + 1
        finally:
            pipeline_mod.registry = original_registry

    def test_registry_step_metadata(self):
        """Verify that before/after are stored in step metadata."""
        from mlcombine.core.registry import ExtensionRegistry

        reg = ExtensionRegistry()

        @reg.step("my_step", before="SomeStep")
        class FakeStep:
            pass

        meta = reg.get_step_meta("my_step")
        assert meta is not None
        assert meta["before"] == "SomeStep"
        assert meta["after"] is None
        assert meta["class"] is FakeStep
