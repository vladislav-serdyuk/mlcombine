"""Tests for non-torch utilities: constants interpolation, block expansion, validation."""

import pytest
import yaml
from pathlib import Path

from mlcombine.models.providers.pytorch import (
    interpolate_constants,
    expand_blocks,
    load_blocks,
    validate_config,
    _load_blocks_from_file,
    _validate_layer_cfg,
)


class TestInterpolateConstants:
    def test_simple_replacement(self):
        result = interpolate_constants(
            {"type": "linear", "out_features": "${d_model}"},
            {"d_model": 64},
        )
        assert result == {"type": "linear", "out_features": "64"}

    def test_nested_replacement(self):
        result = interpolate_constants(
            {"layers": [{"type": "linear", "out_features": "${d}"}]},
            {"d": 128},
        )
        assert result == {"layers": [{"type": "linear", "out_features": "128"}]}

    def test_list_replacement(self):
        layers = [
            {"type": "linear", "out_features": "${a}"},
            {"type": "linear", "out_features": "${b}"},
        ]
        result = interpolate_constants(layers, {"a": 32, "b": 64})
        assert result[0]["out_features"] == "32"
        assert result[1]["out_features"] == "64"

    def test_unknown_constant_raises(self):
        with pytest.raises(ValueError, match="Unknown constant"):
            interpolate_constants({"a": "${missing}"}, {"present": 1})

    def test_no_constants_returns_unchanged(self):
        obj = {"a": "${x}", "b": 1}
        result = interpolate_constants(obj)
        assert result == obj

    def test_mixed_types(self):
        result = interpolate_constants(
            {
                "type": "linear",
                "out_features": "${d}",
                "bias": "${bias}",
            },
            {"d": 64, "bias": False},
        )
        assert result == {"type": "linear", "out_features": "64", "bias": "False"}

    def test_multiple_vars_in_same_string(self):
        result = interpolate_constants(
            {"name": "${prefix}_${suffix}"},
            {"prefix": "enc", "suffix": "block"},
        )
        assert result == {"name": "enc_block"}


class TestExpandBlocks:
    def test_no_blocks_passthrough(self):
        layers = [{"type": "linear", "out_features": 32}]
        assert expand_blocks(layers, {}, {}) == layers

    def test_simple_block_ref(self):
        blocks = {
            "dense": {
                "params": {"units": 64},
                "layers": [
                    {"type": "linear", "out_features": "${units}"},
                    {"type": "relu"},
                ],
            }
        }
        layers = [
            {"type": "block", "ref": "dense", "params": {"units": 128}},
            {"type": "linear"},
        ]
        result = expand_blocks(layers, blocks, {})
        assert len(result) == 3
        assert result[0]["type"] == "linear"
        assert result[0]["out_features"] == "128"
        assert result[1]["type"] == "relu"
        assert result[2]["type"] == "linear"

    def test_block_with_input_reference(self):
        """@input resolves to the preceding sequential layer; own names get prefixed."""
        blocks = {
            "residual_block": {
                "layers": [
                    {"name": "fc", "type": "linear", "out_features": 32, "activation": "relu"},
                    {"name": "drop", "type": "dropout", "p": 0.1},
                    {"name": "add1", "type": "add", "inputs": ["@input", "drop"]},
                ]
            }
        }
        layers = [
            {"name": "proj", "type": "linear", "out_features": 32},
            {"type": "block", "ref": "residual_block"},
            {"type": "linear"},
        ]
        result = expand_blocks(layers, blocks, {})
        # proj, then residual_block expanded:
        #   residual_block_fc, residual_block_drop, residual_block_add1
        assert result[1]["name"] == "residual_block_fc"
        assert result[2]["name"] == "residual_block_drop"
        assert result[3]["name"] == "residual_block_add1"
        assert result[3]["type"] == "add"
        # @input → "proj" (preceding sequential layer)
        # "drop" → "residual_block_drop" (own name, prefixed)
        assert result[3]["inputs"] == ["proj", "residual_block_drop"]

    def test_block_repeat(self):
        blocks = {
            "mlp": {
                "layers": [
                    {"type": "linear", "out_features": 32},
                    {"type": "relu"},
                ]
            }
        }
        layers = [{"type": "block", "ref": "mlp", "repeat": 3}, {"type": "linear"}]
        result = expand_blocks(layers, blocks, {})
        # 3 * (linear + relu) + 1 linear = 7
        assert len(result) == 7
        assert result[0]["type"] == "linear"
        assert result[6]["type"] == "linear"

    def test_unknown_block_ref_raises(self):
        with pytest.raises(ValueError, match="Unknown block"):
            expand_blocks([{"type": "block", "ref": "nonexistent"}], {}, {})

    def test_nested_block(self):
        blocks = {
            "act": {"layers": [{"type": "relu"}]},
            "fc": {
                "layers": [
                    {"type": "linear", "out_features": 16},
                    {"type": "block", "ref": "act"},
                ]
            },
        }
        layers = [{"type": "block", "ref": "fc"}, {"type": "linear"}]
        result = expand_blocks(layers, blocks, {})
        assert len(result) == 3
        assert result[0]["type"] == "linear"
        assert result[1]["type"] == "relu"
        assert result[2]["type"] == "linear"

    def test_block_repeat_chains_inputs(self):
        """When repeat>1, each instance's output feeds the next via @input."""
        blocks = {
            "dense": {
                "layers": [
                    {"name": "fc", "type": "linear", "out_features": 16, "activation": "relu"},
                    {"name": "add1", "type": "add", "inputs": ["@input", "fc"]},
                ]
            }
        }
        layers = [
            {"name": "proj", "type": "linear", "out_features": 16},
            {"type": "block", "ref": "dense", "repeat": 2},
            {"type": "linear"},
        ]
        result = expand_blocks(layers, blocks, {})
        # 0: proj, 1: dense_0_fc, 2: dense_0_add1, 3: dense_1_fc, 4: dense_1_add1, 5: linear
        # dense_0: @input = "proj" → dense_0_add1 gets ["proj", "dense_0_fc"]
        # dense_1: @input = "dense_0_add1" → dense_1_add1 gets ["dense_0_add1", "dense_1_fc"]
        assert result[2]["type"] == "add"
        assert result[2]["inputs"] == ["proj", "dense_0_fc"]
        assert result[4]["type"] == "add"
        assert result[4]["inputs"] == ["dense_0_add1", "dense_1_fc"]

    def test_block_params_override_defaults(self):
        blocks = {
            "dense": {
                "params": {"units": 32},
                "layers": [{"type": "linear", "out_features": "${units}"}],
            }
        }
        layers = [
            {"type": "block", "ref": "dense", "params": {"units": 64}},
        ]
        result = expand_blocks(layers, blocks, {})
        assert result[0]["out_features"] == "64"


class TestValidateLayerConfig:
    def test_valid_layers_pass(self):
        _validate_layer_cfg(
            [{"type": "linear"}, {"type": "relu"}, {"type": "linear"}],
            {},
            {},
        )

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown type"):
            _validate_layer_cfg(
                [{"type": "nonexistent"}],
                {},
                {},
            )

    def test_all_known_types_validate(self):
        known = [
            "linear",
            "relu",
            "tanh",
            "sigmoid",
            "gelu",
            "dropout",
            "batch_norm1d",
            "layer_norm",
            "conv1d",
            "max_pool1d",
            "avg_pool1d",
            "flatten",
            "identity",
            "unsqueeze",
            "squeeze",
            "positional_encoding",
            "multihead_attention",
            "transformer_encoder_layer",
            "transformer_encoder",
            "add",
            "concat",
            "take_last",
            "take_first",
            "mean_pool",
            "max_pool_seq",
            "sum_pool_seq",
        ]
        for t in known:
            _validate_layer_cfg([{"type": t}], {}, {})
            # Should not raise


class TestValidateConfig:
    def test_interpolation_detects_missing_constants(self):
        with pytest.raises(ValueError, match="Unknown constant"):
            validate_config(
                [{"type": "linear", "out_features": "${d_model}"}],
                {},
                {},
            )

    def test_unknown_block_raises(self):
        with pytest.raises(ValueError, match="Unknown block"):
            validate_config(
                [{"type": "block", "ref": "nope"}],
                {"existing": {"layers": [{"type": "linear"}]}},
                {},
            )

    def test_valid_config_passes(self):
        validate_config(
            [{"type": "linear", "out_features": 32}, {"type": "linear"}],
            {},
            {},
        )
        # Should not raise


class TestLoadBlocks:
    def test_empty_blocks(self, tmp_path: Path):
        assert load_blocks(None, None, None) == {}

    def test_inline_blocks_returned(self):
        blocks = {"my_block": {"layers": [{"type": "relu"}]}}
        result = load_blocks(blocks, None, None)
        assert "my_block" in result

    def test_yaml_file_loading(self, tmp_path: Path):
        block_file = tmp_path / "blocks" / "test.yaml"
        block_file.parent.mkdir()
        block_file.write_text(
            yaml.dump(
                {
                    "blocks": {
                        "test_block": {
                            "layers": [{"type": "relu"}],
                        }
                    }
                }
            )
        )

        result = load_blocks({}, [str(tmp_path / "blocks")], tmp_path)
        assert "test_block" in result

    def test_include_resolution(self, tmp_path: Path):
        inner = tmp_path / "inner.yaml"
        inner.write_text(
            yaml.dump(
                {
                    "blocks": {"inner_block": {"layers": [{"type": "tanh"}]}},
                }
            )
        )

        outer = tmp_path / "outer.yaml"
        outer.write_text(
            yaml.dump(
                {
                    "include": ["./inner.yaml"],
                    "blocks": {"outer_block": {"layers": [{"type": "relu"}]}},
                }
            )
        )

        result = _load_blocks_from_file(str(outer))
        assert "inner_block" in result
        assert "outer_block" in result

    def test_circular_include_raises(self, tmp_path: Path):
        a_file = tmp_path / "a.yaml"
        b_file = tmp_path / "b.yaml"
        a_file.write_text(
            yaml.dump(
                {
                    "include": ["./b.yaml"],
                    "blocks": {"a": {"layers": [{"type": "relu"}]}},
                }
            )
        )
        b_file.write_text(
            yaml.dump(
                {
                    "include": ["./a.yaml"],
                    "blocks": {"b": {"layers": [{"type": "relu"}]}},
                }
            )
        )

        with pytest.raises(ValueError, match="Circular include"):
            _load_blocks_from_file(str(a_file))

    def test_inline_overrides_file_blocks(self, tmp_path: Path):
        block_file = tmp_path / "blocks" / "test.yaml"
        block_file.parent.mkdir()
        block_file.write_text(
            yaml.dump(
                {
                    "blocks": {
                        "dup": {"layers": [{"type": "relu"}]},
                    }
                }
            )
        )

        result = load_blocks(
            {"dup": {"layers": [{"type": "tanh"}]}},
            [str(tmp_path / "blocks")],
            tmp_path,
        )
        assert result["dup"]["layers"][0]["type"] == "tanh"
