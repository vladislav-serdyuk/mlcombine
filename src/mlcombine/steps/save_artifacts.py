"""SaveArtifactsStep — saves trained model and creates output directory on disk."""

from __future__ import annotations

import logging
from pathlib import Path

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("SaveArtifactsStep")
class SaveArtifactsStep(BaseStep[PipelineContext]):
    """Creates output directory and saves the trained model to disk.

    Side Effects:
        - Creates directories and writes model file via joblib.
    """

    train = True
    predict = False

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        """Initialize with config — extracts output directory."""
        self._output_dir = cfg.trainer.output_dir

    def run(self, context: PipelineContext) -> PipelineContext:
        """Create output directory and save the model(s)."""
        out = Path(self._output_dir)
        out.mkdir(parents=True, exist_ok=True)

        import joblib

        for node_id, model in context.artifacts.models.items():
            safe_name = node_id.replace("/", "_").replace(".", "_")
            model_path = out / f"model_{safe_name}.joblib"
            joblib.dump(model, model_path)
            logger.info("Model '%s' saved to %s", node_id, model_path)

        model = context.artifacts.model
        if model is not None:
            model_path = out / "model.joblib"
            joblib.dump(model, model_path)
            logger.info("Model saved to %s", model_path)

        if context.artifacts.feature_names:
            import json

            fn_path = out / "feature_names.json"
            with open(fn_path, "w", encoding="utf-8") as f:
                json.dump(context.artifacts.feature_names, f)
            logger.info("Feature names saved to %s", fn_path)

        if context.artifacts.imputer is not None:
            joblib.dump(context.artifacts.imputer, out / "imputer.joblib")
            logger.info("Imputer saved")
        if context.artifacts.encoders:
            joblib.dump(context.artifacts.encoders, out / "encoders.joblib")
            logger.info("Encoders saved")
        if context.artifacts.scaler is not None:
            joblib.dump(context.artifacts.scaler, out / "scaler.joblib")
            logger.info("Scaler saved")

        return context
