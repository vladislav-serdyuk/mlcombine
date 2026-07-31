"""Tests for ``build_sequential()`` — dynamic neural network construction from YAML layer descriptions."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlcombine.models.providers.pytorch import (  # noqa: E402
    build_sequential,
    build_model,
    PyTorchWrapper,
    _DEFAULT_LAYERS,
)


class TestBuildSequential:
    """Unit tests for ``build_sequential()`` and the layer builder registry."""

    def test_default_layers_build_mlp(self):
        """_DEFAULT_LAYERS should produce a valid MLP matching old SimpleNN shape."""
        model = build_sequential(_DEFAULT_LAYERS, input_size=100, out_dim=1)
        x = torch.randn(4, 100)
        out = model(x)
        assert out.shape == (4, 1)

    def test_custom_mlp(self):
        layers = [
            {"type": "linear", "out_features": 64},
            {"type": "relu"},
            {"type": "linear", "out_features": 32},
            {"type": "relu"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=50, out_dim=5)
        x = torch.randn(2, 50)
        out = model(x)
        assert out.shape == (2, 5)

    def test_unsqueeze_squeeze(self):
        layers = [
            {"type": "linear", "out_features": 16},
            {"type": "unsqueeze", "dim": 2},
            {"type": "squeeze", "dim": 2},
            {"type": "linear", "out_features": 8},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=10, out_dim=3)
        x = torch.randn(2, 10)
        out = model(x)
        assert out.shape == (2, 3)

    def test_cnn_with_flatten(self):
        layers = [
            {"type": "unsqueeze", "dim": 2},
            {"type": "conv1d", "out_channels": 8, "kernel_size": 3, "padding": 1},
            {"type": "relu"},
            {"type": "max_pool1d", "kernel_size": 1},
            {"type": "flatten"},
            {"type": "linear", "out_features": 16},
            {"type": "relu"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=20, out_dim=2)
        x = torch.randn(2, 20)
        out = model(x)
        assert out.shape == (2, 2)

    def test_dropout_and_batchnorm(self):
        layers = [
            {"type": "linear", "out_features": 64},
            {"type": "batch_norm1d"},
            {"type": "relu"},
            {"type": "dropout", "p": 0.3},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=32, out_dim=4)
        model.train()
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 4)

    def test_layer_norm(self):
        layers = [
            {"type": "linear", "out_features": 32},
            {"type": "layer_norm"},
            {"type": "relu"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=16, out_dim=2)
        x = torch.randn(3, 16)
        out = model(x)
        assert out.shape == (3, 2)

    def test_unknown_layer_type_raises(self):
        with pytest.raises(ValueError, match="Unknown layer type"):
            build_sequential([{"type": "nonexistent"}], input_size=10, out_dim=1)

    def test_avg_pool(self):
        layers = [
            {"type": "unsqueeze", "dim": 2},
            {"type": "conv1d", "out_channels": 4, "kernel_size": 3, "padding": 1},
            {"type": "relu"},
            {"type": "avg_pool1d", "kernel_size": 1},
            {"type": "flatten"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=12, out_dim=3)
        x = torch.randn(2, 12)
        out = model(x)
        assert out.shape == (2, 3)

    def test_gelu_tanh_sigmoid(self):
        layers = [
            {"type": "linear", "out_features": 16},
            {"type": "gelu"},
            {"type": "linear", "out_features": 8},
            {"type": "tanh"},
            {"type": "linear", "out_features": 4},
            {"type": "sigmoid"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=8, out_dim=1)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 1)


class TestPyTorchWrapper:
    """Wrapper works with build_sequential models end-to-end."""

    def test_wrapper_fit_and_predict_with_custom_mlp(self):
        layers = [
            {"type": "linear", "out_features": 16},
            {"type": "relu"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=5, out_dim=1)
        wrapper = PyTorchWrapper(model, task_type="regression", out_dim=1)

        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        wrapper.fit(X, y, epochs=5, lr=0.01)
        preds = wrapper.predict(X)
        assert preds.shape == (10, 1)

    def test_wrapper_classification(self):
        layers = [
            {"type": "linear", "out_features": 16},
            {"type": "relu"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=4, out_dim=2)
        wrapper = PyTorchWrapper(model, task_type="classification", out_dim=2)

        X = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=10).astype(np.float32)
        wrapper.fit(X, y, epochs=5, lr=0.01)
        preds = wrapper.predict(X)
        assert preds.shape == (10, 1) or preds.shape == (10,)

    def test_wrapper_predict_proba(self):
        layers = [
            {"type": "linear", "out_features": 8},
            {"type": "relu"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=3, out_dim=2)
        wrapper = PyTorchWrapper(model, task_type="classification", out_dim=2)

        X = np.random.randn(5, 3).astype(np.float32)
        y = np.random.randint(0, 2, size=5).astype(np.float32)
        wrapper.fit(X, y, epochs=3, lr=0.01)

        proba = wrapper.predict_proba(X)
        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_identity_layer(self):
        layers = [
            {"type": "linear", "out_features": 10},
            {"type": "identity"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=10, out_dim=2)
        x = torch.randn(2, 10)
        out = model(x)
        assert out.shape == (2, 2)


class TestTransformerLayers:
    """Transformer blocks in ``build_sequential()``."""

    def test_single_transformer_encoder_layer(self):
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 32},
            {"type": "positional_encoding", "d_model": 32},
            {"type": "transformer_encoder_layer", "d_model": 32, "nhead": 4, "dim_feedforward": 64},
            {"type": "squeeze", "dim": 1},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=20, out_dim=3)
        x = torch.randn(2, 20)
        out = model(x)
        assert out.shape == (2, 3)

    def test_stacked_transformer_encoder(self):
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 16},
            {"type": "positional_encoding", "d_model": 16},
            {"type": "transformer_encoder", "d_model": 16, "nhead": 4, "num_layers": 3, "dim_feedforward": 64},
            {"type": "squeeze", "dim": 1},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=10, out_dim=2)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 2)

    def test_multihead_attention_self_attn(self):
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 16},
            {"type": "multihead_attention", "embed_dim": 16, "num_heads": 4},
            {"type": "squeeze", "dim": 1},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=8, out_dim=1)
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 1)

    def test_pos_enc_inferred_d_model(self):
        """d_model defaults to prev_dim when not specified."""
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 24},
            {"type": "positional_encoding"},
            {"type": "squeeze", "dim": 1},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=12, out_dim=4)
        x = torch.randn(2, 12)
        out = model(x)
        assert out.shape == (2, 4)

    def test_pos_enc_and_transformer_with_sequence_length(self):
        """Multi-token sequence: (batch, seq_len, d_model) input with unsqueeze(1) gives seq_len=1."""
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 8},
            {"type": "positional_encoding", "d_model": 8},
            {"type": "transformer_encoder", "d_model": 8, "nhead": 2, "num_layers": 2},
            {"type": "squeeze", "dim": 1},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=6, out_dim=2)
        x = torch.randn(5, 6)
        out = model(x)
        assert out.shape == (5, 2)

    def test_transformer_with_gelu_activation(self):
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 16},
            {"type": "transformer_encoder_layer", "d_model": 16, "nhead": 2, "activation": "gelu"},
            {"type": "squeeze", "dim": 1},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=8, out_dim=3)
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 3)

    def test_transformer_prev_dim_tracking(self):
        """A linear after a transformer with d_model=32 should produce (batch, 32 -> out_dim)."""
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 32},
            {"type": "transformer_encoder", "d_model": 32, "nhead": 4},
            {"type": "multihead_attention", "embed_dim": 32, "num_heads": 4},
            {"type": "squeeze", "dim": 1},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=16, out_dim=5)
        x = torch.randn(2, 16)
        out = model(x)
        assert out.shape == (2, 5)


class TestActivationInLayer:
    """Activation as a parameter of any layer."""

    def test_linear_with_activation(self):
        layers = [
            {"type": "linear", "out_features": 16, "activation": "relu"},
            {"type": "linear"},
        ]
        model = build_sequential(layers, input_size=8, out_dim=3)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 3)

    def test_different_activations(self):
        for act in ("tanh", "sigmoid", "gelu"):
            layers = [
                {"type": "linear", "out_features": 8, "activation": act},
                {"type": "linear"},
            ]
            model = build_sequential(layers, input_size=4, out_dim=2)
            x = torch.randn(2, 4)
            out = model(x)
            assert out.shape == (2, 2)


class TestLayerGraph:
    """_LayerGraph DAG engine with add, concat, pooling."""

    def test_add_two_branches(self):
        layers = [
            {"name": "a", "type": "linear", "out_features": 16},
            {"name": "b", "type": "linear", "out_features": 16},
            {"name": "c", "type": "add", "inputs": ["a", "b"]},
            {"name": "d", "type": "linear"},
        ]
        model = build_model(layers, input_size=10, out_dim=3)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_concat_two_branches(self):
        layers = [
            {"name": "a", "type": "linear", "out_features": 8},
            {"name": "b", "type": "linear", "out_features": 4},
            {"name": "c", "type": "concat", "inputs": ["a", "b"], "dim": 1},
            {"name": "d", "type": "linear"},
        ]
        model = build_model(layers, input_size=6, out_dim=2)
        x = torch.randn(3, 6)
        out = model(x)
        assert out.shape == (3, 2)

    def test_take_last(self):
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 16},
            {"name": "pool", "type": "take_last"},
            {"type": "linear"},
        ]
        model = build_model(layers, input_size=8, out_dim=4)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 4)

    def test_mean_pool(self):
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": 16},
            {"name": "pool", "type": "mean_pool"},
            {"type": "linear"},
        ]
        model = build_model(layers, input_size=8, out_dim=4)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 4)

    def test_block_with_constants(self):
        """Full pipeline: constants + blocks + graph features."""
        layers = [
            {"type": "unsqueeze", "dim": 1},
            {"type": "linear", "out_features": "${d_model}"},
            {"type": "transformer_encoder", "d_model": "${d_model}", "nhead": "${nhead}"},
            {"name": "pool", "type": "take_last"},
            {"type": "linear", "activation": "sigmoid"},
        ]
        model = build_model(
            layers,
            input_size=8,
            out_dim=1,
            constants={"d_model": 16, "nhead": 4},
        )
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 1)
