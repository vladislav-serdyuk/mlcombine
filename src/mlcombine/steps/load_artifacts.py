"""LoadArtifactsStep — loads trained models from disk (joblib or torch)."""

from __future__ import annotations

import logging
from pathlib import Path

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


def _load_joblib(path: Path) -> object:
    import joblib

    return joblib.load(path)


@registry.step("LoadArtifactsStep")
class LoadArtifactsStep(BaseStep[PipelineContext]):
    """Loads trained model(s) from disk into context.artifacts.

    Supports joblib (.joblib, .pkl) and PyTorch (.pt, .pth).
    If the directory contains ``model_<id>.joblib`` files, all are loaded
    into ``artifacts.models``.  The legacy ``model.joblib`` is loaded into
    ``artifacts.model`` for backward compatibility.
    """

    train = False
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        """Initialize with config — extracts path from weights or output_dir."""
        self._path = weights or cfg.trainer.output_dir

    def _load_file(self, path: Path) -> object:
        if path.suffix in (".joblib", ".pkl"):
            return _load_joblib(path)
        if path.suffix in (".pt", ".pth"):
            try:
                import torch

                return torch.load(path, map_location="cpu")
            except ImportError:
                raise ImportError("torch is required to load .pt/.pth files")
        raise ValueError(f"Unsupported model file extension: {path.suffix}")

    def run(self, context: PipelineContext) -> PipelineContext:
        """Load model(s) from disk and store on context."""
        p = Path(self._path)

        if p.is_dir():
            # Load all multi-model files: model_<id>.joblib
            model_files = sorted(p.glob("model_*.joblib"))
            for mf in model_files:
                node_id = mf.stem[len("model_") :]
                context.artifacts.models[node_id] = self._load_file(mf)
                logger.info("Model '%s' loaded: %s", node_id, mf)

            # Legacy single model
            legacy = p / "model.joblib"
            if legacy.exists():
                context.artifacts.model = self._load_file(legacy)  # type: ignore[assignment]
                logger.info("Model loaded: %s", legacy)
            elif not model_files:
                candidates = list(p.glob("model.*"))
                if candidates:
                    context.artifacts.model = self._load_file(candidates[0])  # type: ignore[assignment]
                else:
                    raise FileNotFoundError(f"No model file found in {p}")
        else:
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            context.artifacts.model = self._load_file(p)  # type: ignore[assignment]

        if p.is_dir():
            fn_path = p / "feature_names.json"
        else:
            fn_path = p.parent / "feature_names.json"
        if fn_path.exists():
            import json

            with open(fn_path, encoding="utf-8") as f:
                context.artifacts.feature_names = json.load(f)
            logger.info("Feature names loaded: %d features", len(context.artifacts.feature_names))

        if p.is_dir():
            artifacts_dir = p
        else:
            artifacts_dir = p.parent

        import joblib

        imputer_path = artifacts_dir / "imputer.joblib"
        if imputer_path.exists():
            context.artifacts.imputer = joblib.load(imputer_path)
            logger.info("Imputer loaded")

        encoders_path = artifacts_dir / "encoders.joblib"
        if encoders_path.exists():
            context.artifacts.encoders = joblib.load(encoders_path)
            logger.info("Encoders loaded: %d columns", len(context.artifacts.encoders))

        scaler_path = artifacts_dir / "scaler.joblib"
        if scaler_path.exists():
            context.artifacts.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")

        return context
