"""Tests for the ExtensionRegistry, FeatureHandler, and their integration points."""

import pandas as pd
import pytest

from mlcombine.core.registry import ExtensionRegistry, FeatureHandler, registry


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestExtensionRegistry:
    def test_singleton_instance(self):
        from mlcombine.core.registry import registry as r1
        from mlcombine.core.registry import registry as r2

        assert r1 is r2

    def test_step_decorator(self):
        reg = ExtensionRegistry()

        @reg.step("my_step")
        class FakeStep:
            pass

        assert reg.get_step("my_step") is FakeStep

    def test_step_decorator_default_name(self):
        reg = ExtensionRegistry()

        @reg.step()
        class MyCustomStep:
            pass

        assert reg.get_step("MyCustomStep") is MyCustomStep

    def test_feature_handler_decorator(self):
        reg = ExtensionRegistry()

        @reg.feature_handler("audio")
        class AudioHandler(FeatureHandler):
            def detect(self, series):
                return series.astype(str).str.contains(r"\.(mp3|wav)$").mean() > 0.5

        assert reg.get_feature_handler("audio") is AudioHandler
        assert "audio" in reg.feature_handler_types

    def test_feature_handler_detect(self):
        reg = ExtensionRegistry()

        @reg.feature_handler("geo")
        class GeoHandler(FeatureHandler):
            def detect(self, series):
                return bool(series.astype(str).str.contains(r"^-?\d+\.\d+,-?\d+\.\d+$").mean() > 0.5)

        handler_cls = reg.get_feature_handler("geo")
        assert handler_cls is not None
        handler = handler_cls()
        s = pd.Series(["12.34,-56.78", "0.0,1.0", "foo"])
        assert handler.detect(s) is True
        s2 = pd.Series(["apple", "banana", "cherry"])
        assert handler.detect(s2) is False

    def test_tensor_adapter_decorator(self):
        reg = ExtensionRegistry()

        @reg.tensor_adapter("jax")
        class JaxAdapter:
            pass

        assert reg.get_tensor_adapter("jax") is JaxAdapter

    def test_model_provider_decorator(self):
        reg = ExtensionRegistry()

        @reg.model_provider("my_backend")
        def my_provider(**kwargs):
            return "model"

        provider = reg.get_model_provider("my_backend")
        assert provider is not None
        assert provider(**{"model": None}) == "model"

    def test_layer_builder_decorator(self):
        reg = ExtensionRegistry()

        @reg.layer_builder("custom_layer")
        def build_custom(cfg, prev_dim, **kw):
            return object()

        assert reg.get_layer_builder("custom_layer") is build_custom
        assert "custom_layer" in reg.layer_builders

    def test_activation_decorator(self):
        reg = ExtensionRegistry()

        @reg.activation("custom_act")
        class CustomAct:
            pass

        assert reg.get_activation("custom_act") is CustomAct
        assert "custom_act" in reg.activations


class TestModelFactoryRegistry:
    def test_custom_provider_via_registry(self):
        @registry.model_provider("custom_ml")
        def custom_provider(**kwargs):
            return "custom_model"

        from mlcombine.core.builder import ModelBuilder
        from mlcombine.core.types import ModelNode

        builder = ModelBuilder()
        nodes = [ModelNode(provider="custom_ml")]
        model = builder.build_all(nodes).build()
        assert model == "custom_model"
        # Clean up
        registry._model_providers.providers_data.pop("custom_ml", None)
        registry._model_providers.meta_data.pop("custom_ml", None)

    def test_custom_provider_overrides_builtin(self):
        from mlcombine.core.registry import registry as global_registry

        orig_provider = global_registry.get_model_provider("sklearn")
        orig_meta = global_registry._model_providers._meta.get("sklearn")

        @global_registry.model_provider("sklearn")
        def fake_sklearn(**kwargs):
            return "overridden"

        from mlcombine.core.builder import ModelBuilder
        from mlcombine.core.types import ModelNode

        builder = ModelBuilder()
        nodes = [ModelNode(provider="sklearn")]
        model = builder.build_all(nodes).build()
        assert model == "overridden"

        if orig_provider is not None:
            global_registry._model_providers._providers["sklearn"] = orig_provider
            if orig_meta is not None:
                global_registry._model_providers._meta["sklearn"] = orig_meta
        else:
            global_registry._model_providers._providers.pop("sklearn", None)
            global_registry._model_providers._meta.pop("sklearn", None)


class TestTypeDetectFeatureHandler:
    def test_custom_feature_type_detected(self):
        from mlcombine.steps.type_detect import TypeDetectStep
        from mlcombine.core.types import MLCombineConfig

        @registry.feature_handler("emoji")
        class EmojiHandler(FeatureHandler):
            def detect(self, series):
                return series.astype(str).str.contains("[\U0001f600-\U0001f64f]", regex=True).mean() > 0.3

        df = pd.DataFrame(
            {
                "text_col": ["hello 😊", "world 😎", "foo", "bar", "baz 😍", "qux"],
                "target": [1, 2, 3, 4, 5, 6],
            }
        )
        cfg = MLCombineConfig(
            **{
                "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "target"},
                "model": [{"provider": "sklearn"}],
            }
        )
        step = TypeDetectStep(cfg)
        from mlcombine.core.types import PipelineContext

        ctx = PipelineContext()
        ctx.data.train_df = df
        result = step.run(ctx)
        assert result.data.detected_types["text_col"] == "emoji"

    def test_custom_feature_handler_preprocess(self):
        """Test that handlers with preprocess can be invoked."""

        @registry.feature_handler("upper")
        class UpperHandler(FeatureHandler):
            def detect(self, series):
                return bool(series.astype(str).str.isupper().mean() > 0.5)

            def preprocess(self, series, config=None):
                return series.str.lower()

        handler_cls = registry.get_feature_handler("upper")
        assert handler_cls is not None
        handler = handler_cls()
        s = pd.Series(["HELLO", "WORLD", "foo"])
        assert handler.detect(s) is True
        processed = handler.preprocess(s)
        assert (processed == pd.Series(["hello", "world", "foo"])).all()


class TestTensorAdapter:
    def test_custom_adapter_fallback(self):
        """Built-in NUMPY still works when no custom adapter is registered."""
        from mlcombine.core.tensor.unified import _resolve_adapter
        from mlcombine.core.enums import TensorBackendType

        adapter = _resolve_adapter(TensorBackendType.NUMPY)
        from mlcombine.core.tensor.backend_numpy import NumpyAdapter

        assert isinstance(adapter, NumpyAdapter)


class TestGlobalRegistrySeeding:
    """Verify that built-in modules seed the global registry on import."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch — skipped")
    def test_builtin_layer_builders_available(self):
        assert "linear" in registry.layer_builders
        assert "dropout" in registry.layer_builders
        assert "add" in registry.layer_builders

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch — skipped")
    def test_builtin_activations_available(self):
        assert "relu" in registry.activations
        assert "tanh" in registry.activations
        assert "gelu" in registry.activations


class TestPluginLoading:
    def test_plugin_config_field_default(self):
        from mlcombine.core.types import MLCombineConfig

        cfg = MLCombineConfig(
            **{
                "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
                "model": [{"provider": "sklearn"}],
            }
        )
        assert cfg.plugins == []

    def test_plugin_config_field(self):
        from mlcombine.core.types import MLCombineConfig

        cfg = MLCombineConfig(
            **{
                "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
                "plugins": ["./my_plugin.py"],
                "model": [{"provider": "sklearn"}],
            }
        )
        assert cfg.plugins == ["./my_plugin.py"]

    def test_load_plugin_from_file(self):
        from mlcombine.core.pipeline import _load_plugins

        # Clean up before
        registry._feature_handlers.pop("test_plugin_type", None)
        registry._steps.data.pop("test_plugin_step", None)

        _load_plugins(["tests/test_plugin_example.py"])

        assert registry.get_feature_handler("test_plugin_type") is not None
        assert registry.get_step("test_plugin_step") is not None

        # Clean up after
        registry._feature_handlers.pop("test_plugin_type", None)
        registry._steps.data.pop("test_plugin_step", None)

    def test_load_plugin_raises_on_missing_file(self):
        from mlcombine.core.pipeline import _load_plugins

        with pytest.raises(FileNotFoundError):
            _load_plugins(["nonexistent.py"])

    def test_load_plugin_from_module(self):
        from mlcombine.core.pipeline import _load_plugins
        import os
        import shutil
        import sys
        import tempfile

        tmpdir = tempfile.mkdtemp()
        plugin_path = os.path.join(tmpdir, "tmp_plugin_mod.py")
        with open(plugin_path, "w") as f:
            f.write("from mlcombine.core.registry import registry; registry._feature_handlers['tmp_mod_type'] = object()")
        sys_path = list(sys.path)
        sys.path.insert(0, tmpdir)
        try:
            _load_plugins(["tmp_plugin_mod"])
            assert registry._feature_handlers.get("tmp_mod_type") is not None
        finally:
            registry._feature_handlers.pop("tmp_mod_type", None)
            sys.path = sys_path
            shutil.rmtree(tmpdir)
