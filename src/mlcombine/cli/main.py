"""Ultra-thin Click CLI for mlcombine.

All heavy imports are deferred to command bodies so ``--help`` is instant.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import click
from typing import Any

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _find_config_file(config_path: str | None = None) -> Path:
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path
        raise click.ClickException(f"Config file not found: {config_path}")
    for name in ["mlcombine.yaml", "config.yaml"]:
        path = Path(name)
        if path.exists():
            return path
    raise click.ClickException("Configuration file not found. Use --config or place mlcombine.yaml/config.yaml in the current directory.")


def _parse_config(path: Path) -> Any:
    import yaml
    from mlcombine.core.schemas.config import MLCombineConfig

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise click.ClickException("Config file must contain a YAML mapping")
        return MLCombineConfig(**raw)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error loading config {path}: {e}")


@click.group()
def cli() -> None:
    """MLCombine - Declarative Low-Code/No-Code framework for ML competitions."""


@cli.command()
@click.option("--config", "-c")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--force-prepare-dataset", is_flag=True)
def train(config: str | None, verbose: bool, force_prepare_dataset: bool) -> None:
    """Run the full training pipeline."""
    _setup_logging(verbose)
    from mlcombine.core.pipeline import PipelineEngine
    from mlcombine.core.types import PipelineContext
    from mlcombine.core.exceptions import ConfigurationError, DatasetNotFoundError, EmptyDatasetError

    try:
        cfg = _parse_config(_find_config_file(config))
        if force_prepare_dataset:
            cfg.data.force_prepare_dataset = True
        logger.info("Starting mlcombine pipeline")
        engine = PipelineEngine.from_config(cfg)
        context = engine.run_all(PipelineContext())
        logger.info("Training pipeline completed. Model: %s", type(context.artifacts.model).__name__)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except click.ClickException, ConfigurationError, DatasetNotFoundError, EmptyDatasetError, ValueError:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        logger.debug(traceback.format_exc())
        raise click.ClickException(f"Training pipeline failed: {e}")


@cli.command()
@click.option("--config", "-c")
@click.option("--weights", "-w")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--force-prepare-dataset", is_flag=True)
def predict(config: str | None, weights: str | None, verbose: bool, force_prepare_dataset: bool) -> None:
    """Run prediction pipeline with strict inference isolation."""
    _setup_logging(verbose)
    from mlcombine.core.pipeline import PipelineEngine
    from mlcombine.core.types import PipelineContext
    from mlcombine.core.exceptions import ConfigurationError, DatasetNotFoundError, EmptyDatasetError

    try:
        cfg = _parse_config(_find_config_file(config))
        if force_prepare_dataset:
            cfg.data.force_prepare_dataset = True
        logger.info("Starting mlcombine prediction pipeline")
        engine = PipelineEngine.from_config(cfg, predict=True, weights=weights)
        engine.run_all(PipelineContext())
        logger.info("Prediction pipeline completed successfully")
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except click.ClickException, ConfigurationError, DatasetNotFoundError, EmptyDatasetError:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        logger.debug(traceback.format_exc())
        raise click.ClickException(f"Prediction pipeline failed: {e}")


if __name__ == "__main__":
    cli()
