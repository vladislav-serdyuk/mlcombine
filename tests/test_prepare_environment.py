from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

from mlcombine.core.types import MLCombineConfig, PipelineContext
from mlcombine.steps.prepare_environment import PrepareEnvironmentStep

_COMPLETED = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"installed ok\n", stderr=b"")


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

    def test_import_works_without_heavy_backends(self):
        """The whole package must import cleanly without catboost/lightgbm/torch
        so that PrepareEnvironmentStep gets a chance to install them."""
        code = textwrap.dedent(
            """
            import sys

            class _Block:
                def find_spec(self, name, path=None, target=None):
                    if (
                        name in ("catboost", "lightgbm", "torch")
                        or name.startswith("catboost.")
                        or name.startswith("lightgbm.")
                        or name.startswith("torch.")
                    ):
                        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
                    return None

            sys.meta_path.insert(0, _Block())
            import mlcombine
            import mlcombine.models.providers
            from mlcombine.models.providers.catboost import catboost_provider
            from mlcombine.models.providers.lightgbm import lightgbm_provider
            try:
                catboost_provider()
            except ImportError:
                pass
            try:
                lightgbm_provider()
            except ImportError:
                pass
            print("IMPORT_OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "IMPORT_OK" in result.stdout

    @patch("mlcombine.steps.prepare_environment.importlib.import_module", side_effect=[ImportError("missing"), MagicMock()])
    @patch("mlcombine.steps.prepare_environment.shutil.which", return_value="/usr/bin/uv")
    @patch(
        "mlcombine.steps.prepare_environment.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b"  Downloaded catboost-1.2\n  Installed catboost-1.2\n", stderr=b""),
    )
    @patch("mlcombine.steps.prepare_environment.importlib.reload")
    def test_install_output_is_logged(
        self,
        mock_reload: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
        mock_import: MagicMock,
        caplog,
    ) -> None:
        step = PrepareEnvironmentStep(_cfg("catboost", auto_install=True))
        with caplog.at_level("INFO"):
            step.run(PipelineContext())
        messages = [r.message for r in caplog.records]
        assert any("uv:   Downloaded catboost-1.2" in m for m in messages)
        assert any("uv:   Installed catboost-1.2" in m for m in messages)

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
    @patch("mlcombine.steps.prepare_environment.subprocess.run", return_value=_COMPLETED)
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
    @patch("mlcombine.steps.prepare_environment.subprocess.run", return_value=_COMPLETED)
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
    @patch("mlcombine.steps.prepare_environment.subprocess.run", return_value=_COMPLETED)
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
    @patch("mlcombine.steps.prepare_environment.subprocess.run", return_value=_COMPLETED)
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
