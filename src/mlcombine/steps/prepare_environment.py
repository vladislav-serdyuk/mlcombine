"""PrepareEnvironmentStep — ensures required packages are installed.

Runs early (first in _BASE_ORDER).  Installs:
  - Backend packages for all model providers (via registry metadata)
  - Data-framework packages (e.g. ``pyarrow`` or ``fastparquet``) when
    any data file reference ends in ``.parquet``.

The step is a no-op when ``cfg.environment.auto_install`` is ``False``.
"""

from __future__ import annotations

import importlib
import logging
import shutil
import subprocess

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)

_PARQUET_DEPS = ["fastparquet", "pyarrow"]


def _log_install_output(prefix: str, output: bytes) -> None:
    """Log pip/uv install output line-by-line through the Python logger."""
    text = output.decode(errors="replace")
    for line in text.splitlines():
        if line.strip():
            logger.info("%s: %s", prefix, line)


def _install_package(package: str) -> None:
    """Install *package* via uv pip install or pip (fallback)."""
    if shutil.which("uv"):
        logger.info("Installing %s via uv pip install", package)
        try:
            result = subprocess.run(
                ["uv", "pip", "install", package],
                check=True,
                capture_output=True,
            )
            _log_install_output("uv", result.stdout)
        except subprocess.CalledProcessError as e:
            _log_install_output("uv", e.stdout)
            _log_install_output("uv (error)", e.stderr)
            logger.error("uv pip install failed for %s", package)
    else:
        logger.info("Installing %s via pip", package)
        try:
            result = subprocess.run(
                ["pip", "install", package],
                check=True,
                capture_output=True,
            )
            _log_install_output("pip", result.stdout)
        except subprocess.CalledProcessError as e:
            _log_install_output("pip", e.stdout)
            _log_install_output("pip (error)", e.stderr)
            logger.error("pip install failed for %s", package)


def _is_parquet_path(value: object) -> bool:
    return isinstance(value, str) and value.endswith(".parquet")


def _collect_parquet_paths(cfg: MLCombineConfig) -> set[str]:
    """Collect all parquet file references from config (data + step_config)."""
    paths: set[str] = set()
    if cfg.data.train_df.endswith(".parquet"):
        paths.add(cfg.data.train_df)
    if cfg.data.test_df.endswith(".parquet"):
        paths.add(cfg.data.test_df)

    def _walk(obj: object) -> None:
        if isinstance(obj, str) and obj.endswith(".parquet"):
            paths.add(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    for val in cfg.step_config.model_dump(exclude_none=True).values():
        _walk(val)
    return paths


def _ensure_package(import_name: str, install_name: str | None = None) -> None:
    """Try to import; if missing, install and retry."""
    try:
        importlib.import_module(import_name)
    except ImportError:
        _install_package(install_name or import_name)
        importlib.import_module(import_name)


@registry.step("PrepareEnvironmentStep")
class PrepareEnvironmentStep(BaseStep[PipelineContext]):
    """Ensures backend packages for all model nodes are installed.

    Side Effects:
        - Installs Python packages into the current environment.
        - Reloads provider modules so module-level imports pick up the new
          packages.

    This step is a no-op when ``auto_install`` is ``False``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self._model_nodes = cfg.model
        self._auto_install = cfg.environment.auto_install
        self._cfg = cfg

    def run(self, context: PipelineContext) -> PipelineContext:
        """Check and install backend packages for all model providers."""
        if not self._auto_install:
            return context

        # 1. Data-format dependencies
        parquet_paths = _collect_parquet_paths(self._cfg)
        if parquet_paths:
            logger.info("Parquet files detected: %s — ensuring parquet engine", parquet_paths)
            for dep in _PARQUET_DEPS:
                try:
                    _ensure_package(dep)
                    break
                except ImportError:
                    continue
            else:
                logger.warning("Could not install any parquet engine (pyarrow / fastparquet)")

        # 2. Model-provider dependencies
        seen: set[str] = set()
        for node in self._model_nodes:
            provider = node.provider
            if provider in seen:
                continue
            seen.add(provider)

            meta = registry.get_model_provider_meta(provider)
            if meta is None:
                continue

            pkg = meta.get("package")
            if pkg is None:
                continue

            try:
                importlib.import_module(pkg)
            except ImportError:
                _install_package(pkg)
                mod_name = meta.get("module")
                if mod_name:
                    mod = importlib.import_module(mod_name)
                    importlib.reload(mod)

        # 3. Step dependencies (e.g. cross-encoder package)
        for step_name in registry.step_names:
            smeta = registry.get_step_meta(step_name)
            if smeta is None:
                continue
            step_cls: type = smeta["class"]
            is_required = getattr(step_cls, "is_required", lambda _: True)
            if not is_required(self._cfg):
                continue
            pkg = smeta.get("package")
            if pkg is None:
                continue
            mod_name = smeta.get("module")
            if mod_name is None:
                mod_name = pkg
            _ensure_package(mod_name, install_name=pkg)

        return context
