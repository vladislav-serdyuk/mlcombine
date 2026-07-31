"""Pipeline context, base step, pipeline engine, and built-in training/save steps.

Provides from_config() factory on PipelineEngine and ModelFitStep / SaveArtifactsStep
so CLI never builds steps manually.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import time
from pathlib import Path

import mlcombine._registration  # noqa: F401  # side-effect: @registry.* decorators fire
from mlcombine.core.registry import ExtensionRegistry, registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)

# ── default step order ──────────────────────────────────────────

_BASE_ORDER: list[str] = [
    "PrepareEnvironmentStep",
    "PrepareDatasetStep",
    "DataLoaderStep",
    "TypeDetectStep",
    "FeatureGenerationStep",
    "LoadArtifactsStep",
    "SplitStep",
    "ImputeStep",
    "EncodeScaleStep",
    "CreateModelStep",
    "ModelFitStep",
    "EvaluateStep",
    "SaveArtifactsStep",
    "AlignFeaturesStep",
    "ModelPredictStep",
    "DropTargetColumnsStep",
    "SavePredictionsStep",
]


def _resolve_order(cfg: MLCombineConfig, _registry: ExtensionRegistry | None = None) -> list[str]:
    """Compute the final list of step names for the pipeline.

    1. If ``cfg.pipeline.order`` is set, use it verbatim.
    2. Otherwise, start from ``_BASE_ORDER`` and insert any registered step
       that declares ``before`` / ``after`` and is not already in the list.

    ``after`` steps are inserted in registration order (first registered = closest
    to target).  ``before`` steps are inserted in **reverse** registration order
    (last registered = closest to target) so that the step imported last runs
    immediately before the target.
    """
    reg = _registry or registry
    if cfg.pipeline.order:
        return list(cfg.pipeline.order)

    result = list(_BASE_ORDER)

    # Steps with ``before`` — process in reverse so last registered = closest to target
    for name in reversed(reg.step_names):
        meta = reg.get_step_meta(name)
        if meta is None:
            continue
        before = meta.get("before")
        if before is None:
            continue
        if name in result:
            continue
        try:
            idx = result.index(before)
        except ValueError:
            logger.warning("Step '%s' references unknown target '%s' — skipping", name, before)
            continue
        result.insert(idx, name)

    # Steps with ``after`` — process in forward order so first registered runs first
    for name in reg.step_names:
        meta = reg.get_step_meta(name)
        if meta is None:
            continue
        after = meta.get("after")
        if after is None:
            continue
        if name in result:
            continue
        try:
            idx = result.index(after)
        except ValueError:
            logger.warning("Step '%s' references unknown target '%s' — skipping", name, after)
            continue
        result.insert(idx + 1, name)

    return result


class PipelineEngine[ContextType: PipelineContext]:
    """Orchestrates a sequence of BaseStep instances, timing each execution."""

    def __init__(self) -> None:
        """Initialize an empty pipeline engine with no registered steps."""
        self._steps: list[tuple[str, BaseStep[ContextType]]] = []

    @property
    def steps(self) -> list[tuple[str, BaseStep[ContextType]]]:
        """Return a read-only copy of the registered step list."""
        return list(self._steps)

    # ── factory ────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: MLCombineConfig,
        *,
        predict: bool = False,
        weights: str | None = None,
    ) -> PipelineEngine[PipelineContext]:
        """Build a fully-configured PipelineEngine from an MLCombineConfig.

        Steps are resolved from the global ``registry`` by name.  The
        execution order is driven by ``_BASE_ORDER`` plus any ``before`` /
        ``after`` hints from ``@registry.step()`` decorators, or explicitly
        via ``cfg.pipeline.order``.

        Args:
            cfg: Parsed pipeline configuration.
            predict: If ``True`` only steps with ``predict = True`` are added.
            weights: Override path for prediction output file.

        Side Effects:
            - Loads plugin modules which may register additional steps.
            - Instantiates each step via its uniform ``__init__(cfg, ...)``.

        """
        engine: PipelineEngine[PipelineContext] = PipelineEngine()

        # Load plugins before resolving any steps
        _load_plugins(cfg.plugins)

        order = _resolve_order(cfg)
        skip = set(cfg.pipeline.skip)

        step_names: list[str] = []

        for name in order:
            if name in skip:
                continue
            meta = registry.get_step_meta(name)
            if meta is None:
                available = sorted(registry.step_names)
                raise ValueError(f"Step '{name}' is not registered. Available steps: {available}")

            step_cls: type[BaseStep[PipelineContext]] = meta["class"]

            if predict and not step_cls.predict:
                continue
            if not predict and not step_cls.train:
                continue
            if not step_cls.is_required(cfg):
                continue

            step = step_cls(cfg, predict=predict, weights=weights)
            engine._add_step(step, name=name)
            step_names.append(name)

        logger.info("Pipeline steps: %s", " → ".join(step_names))
        return engine

    # ── step graph construction ────────────────────────────────

    def _resolve_name(self, base_name: str) -> str:
        """Generate a unique step name by appending ``_<counter>``.

        Args:
            base_name: Preferred step name.

        Returns:
            A name not present in ``self.steps``.  If *base_name* is already
            taken, a monotonically increasing numeric suffix separated by an
            underscore is appended.  Existing trailing digits are treated as
            part of the base name (e.g. ``nn_model_2026`` → ``nn_model_2026_1``).

        Side Effects:
            None (pure computation over ``self.steps``).

        """
        existing_names = {name for name, _ in self._steps}
        if base_name not in existing_names:
            return base_name

        prefix = base_name + "_"
        counter = 1
        while f"{prefix}{counter}" in existing_names:
            counter += 1
        return f"{prefix}{counter}"

    def _add_step(self, step: BaseStep[ContextType], name: str | None = None) -> None:
        """Append a step to the end of the pipeline (internal)."""
        if name is None:
            name = step.__class__.__name__
        name = self._resolve_name(name)
        self._steps.append((name, step))

    # ── execution ──────────────────────────────────────────────

    def run_all(self, initial_context: ContextType) -> ContextType:
        """Execute all registered steps sequentially.

        Args:
            initial_context: A fully-initialised pipeline context.  Must not be None.

        Returns:
            The mutated context after all steps have run.

        Side Effects:
            - Mutates *initial_context* in place as each step runs.
            - Logs timing for every step.

        """
        context = initial_context

        for name, step in self.steps:
            start_time = time.time()
            logger.info(f"Running step: {name}")
            try:
                context = step.run(context)
                elapsed = time.time() - start_time
                logger.info(f"Step {name} completed in {elapsed:.2f} seconds")
            except KeyboardInterrupt:
                logger.warning("Pipeline interrupted by user during step: %s", name)
                raise
            except Exception as e:
                logger.error(f"Error in step {name}: {e}")
                raise

        return context


# ── plugin loading ──────────────────────────────────────────────


def _load_plugins(plugins: list[str]) -> None:
    """Import each plugin in *plugins* so that ``@registry.*`` decorators fire.

    Supports two forms:
      * Path to a ``.py`` file (absolute or relative to CWD).
      * Dotted Python module name (e.g. ``my_package.my_plugin``).

    Args:
        plugins: List of plugin paths or module names.

    Raises:
        FileNotFoundError: If a ``.py`` file does not exist.
        ModuleNotFoundError: If a module name cannot be imported.

    """
    if not plugins:
        return

    for entry in plugins:
        path = Path(entry)
        if path.suffix == ".py":
            if not path.exists():
                # Try resolving relative to CWD
                resolved = Path.cwd() / entry
                if not resolved.exists():
                    raise FileNotFoundError(f"Plugin file not found: {entry} (tried {resolved})")
                path = resolved
            abs_path = str(path.resolve())
            module_name = path.stem
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load plugin from {abs_path}")
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            logger.info("Loaded plugin from %s", abs_path)
        else:
            importlib.import_module(entry)
            logger.info("Loaded plugin module %s", entry)


__all__ = [
    "PipelineContext",
    "PipelineEngine",
]
