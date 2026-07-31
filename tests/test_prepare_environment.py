"""Tests for PrepareEnvironmentStep — unit tests with mocks."""

from unittest.mock import MagicMock, patch

from mlcombine.core.types import MLCombineConfig, PipelineContext
from mlcombine.steps.prepare_environment import PrepareEnvironmentStep


def _cfg(provider: str = "sklearn", auto_install: bool = False) -> MLCombineConfig:
    return MLCombineConfig(
        **{
            "data": {
                "train_df": "train.csv",
                "test_df": "test.csv",
                "target_col": "target",
            },
            "model": [{"provider": provider}],
            "environment": {"auto_install": auto_install},
        }
    )


class TestPrepareEnvironmentStep:
    """Step-level unit tests — all external calls are mocked."""

    def test_skips_when_auto_install_false(self):
        step = PrepareEnvironmentStep(_cfg(auto_install=False))
        ctx = PipelineContext()
        result = step.run(ctx)
        assert result is ctx

    def test_skips_when_provider_is_sklearn(self):
        step = PrepareEnvironmentStep(_cfg("sklearn", auto_install=True))
        ctx = PipelineContext()
        result = step.run(ctx)
        assert result is ctx

    def test_skips_when_provider_is_auto(self):
        step = PrepareEnvironmentStep(_cfg("auto", auto_install=True))
        ctx = PipelineContext()
        result = step.run(ctx)
        assert result is ctx

    @patch("mlcombine.steps.prepare_environment.importlib.import_module", side_effect=[ImportError("missing"), MagicMock()])
    @patch("mlcombine.steps.prepare_environment.shutil.which", return_value="/usr/bin/uv")
    @patch("mlcombine.steps.prepare_environment.subprocess.run")
    @patch("mlcombine.steps.prepare_environment.importlib.reload")
    def test_installs_catboost_via_uv(
        self,
        mock_reload: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        step = PrepareEnvironmentStep(_cfg("catboost", auto_install=True))
        step.run(PipelineContext())
        mock_run.assert_called_once_with(
            ["uv", "pip", "install", "catboost"],
            check=True,
            capture_output=True,
        )
        mock_reload.assert_called_once()

    @patch("mlcombine.steps.prepare_environment.importlib.import_module", side_effect=[ImportError("missing"), MagicMock()])
    @patch("mlcombine.steps.prepare_environment.shutil.which", return_value="/usr/bin/uv")
    @patch("mlcombine.steps.prepare_environment.subprocess.run")
    @patch("mlcombine.steps.prepare_environment.importlib.reload")
    def test_installs_lightgbm_via_uv(
        self,
        mock_reload: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        step = PrepareEnvironmentStep(_cfg("lightgbm", auto_install=True))
        step.run(PipelineContext())
        mock_run.assert_called_once_with(
            ["uv", "pip", "install", "lightgbm"],
            check=True,
            capture_output=True,
        )
        mock_reload.assert_called_once()

    @patch("mlcombine.steps.prepare_environment.importlib.import_module", side_effect=[ImportError("missing"), MagicMock()])
    @patch("mlcombine.steps.prepare_environment.shutil.which", return_value="/usr/bin/uv")
    @patch("mlcombine.steps.prepare_environment.subprocess.run")
    @patch("mlcombine.steps.prepare_environment.importlib.reload")
    def test_installs_torch_via_uv_for_pytorch(
        self,
        mock_reload: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        step = PrepareEnvironmentStep(_cfg("pytorch", auto_install=True))
        step.run(PipelineContext())
        mock_run.assert_called_once_with(
            ["uv", "pip", "install", "torch"],
            check=True,
            capture_output=True,
        )
        mock_reload.assert_called_once()

    @patch("mlcombine.steps.prepare_environment.importlib.import_module", side_effect=[ImportError("missing"), MagicMock()])
    @patch("mlcombine.steps.prepare_environment.shutil.which", return_value="/usr/bin/uv")
    @patch("mlcombine.steps.prepare_environment.subprocess.run")
    @patch("mlcombine.steps.prepare_environment.importlib.reload")
    def test_installs_torch_via_uv_for_hybrid(
        self,
        mock_reload: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        step = PrepareEnvironmentStep(_cfg("hybrid", auto_install=True))
        step.run(PipelineContext())
        mock_run.assert_called_once_with(
            ["uv", "pip", "install", "torch"],
            check=True,
            capture_output=True,
        )
        mock_reload.assert_called_once()

    @patch("mlcombine.steps.prepare_environment.importlib.import_module", side_effect=[ImportError("missing"), MagicMock()])
    @patch("mlcombine.steps.prepare_environment.shutil.which", return_value=None)
    @patch("mlcombine.steps.prepare_environment.subprocess.run")
    @patch("mlcombine.steps.prepare_environment.importlib.reload")
    def test_falls_back_to_pip_when_uv_missing(
        self,
        mock_reload: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        step = PrepareEnvironmentStep(_cfg("catboost", auto_install=True))
        step.run(PipelineContext())
        mock_run.assert_called_once_with(
            ["pip", "install", "catboost"],
            check=True,
            capture_output=True,
        )
        mock_reload.assert_called_once()
