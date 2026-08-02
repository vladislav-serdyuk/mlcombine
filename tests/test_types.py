"""Tests for core type models — config parsing, validation, and defaults."""

import pytest
from pydantic import ValidationError

from mlcombine.core.types import (
    MLCombineConfig,
    TrainerConfig,
    PipelineData,
    PipelineArtifacts,
    PipelineContext,
)


class TestMLCombineConfig:
    def test_minimal_parsing(self, minimal_config):
        assert minimal_config.data.train_df == "train.csv"
        assert minimal_config.model[-1].provider == "sklearn"

    def test_auto_default_provider(self):
        raw = {
            "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
            "model": [{"provider": "auto"}],
        }
        cfg = MLCombineConfig(**raw)
        assert cfg.model[-1].provider == "auto"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            MLCombineConfig(
                **{
                    "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
                    "nonexistent": True,
                    "model": [{"provider": "sklearn"}],
                }
            )

    def test_version_field_accepted(self):
        raw = {
            "version": "1.0",
            "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
            "model": [{"provider": "sklearn"}],
        }
        cfg = MLCombineConfig(**raw)
        assert cfg.version == "1.0"

    def test_removed_fields_rejected(self):
        with pytest.raises(ValidationError):
            MLCombineConfig(
                **{
                    "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t"},
                    "validation": {"strategy": "kfold", "splits": 3},
                    "model": [{"provider": "sklearn"}],
                }
            )

    def test_handling_config_wired(self, minimal_config):
        assert minimal_config.handling.numbers.impute.value == "median"
        assert minimal_config.handling.numbers.scale.value == "robust"

    def test_trainer_default_paths(self):
        cfg = TrainerConfig()
        assert cfg.output_dir == "./outputs"
        assert cfg.output_file == "./outputs/result"
        assert not hasattr(cfg, "strict_inference_isolation")

    def test_uplift_provider_config(self):
        raw = {
            "data": {"train_df": "t.csv", "test_df": "t.csv", "target_col": "t", "treatment_col": "treat"},
            "model": [
                {"provider": "sklearn", "params": {"backbone": "random_forest"}},
                {"provider": "t_learner", "model": "sklearn"},
            ],
        }
        cfg = MLCombineConfig(**raw)
        assert cfg.data.treatment_col == "treat"
        assert len(cfg.model) == 2

        assert cfg.model[-1].provider == "t_learner"


class TestPipelineData:
    def test_defaults(self):
        d = PipelineData()
        assert d.train_df is None
        assert d.test_df is None
        assert d.treatment_col is None

    def test_treatment_col(self):
        d = PipelineData(treatment_col="treat")
        assert d.treatment_col == "treat"


class TestPipelineArtifacts:
    def test_defaults(self):
        a = PipelineArtifacts()
        assert a.model is None
        assert a.imputers == {}


class TestPipelineContext:
    def test_defaults(self):
        ctx = PipelineContext()
        assert ctx.data.train_df is None
        assert ctx.artifacts.model is None

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            PipelineContext(**{"data": {}, "extra": True})
