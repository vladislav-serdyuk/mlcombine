"""TextEmbeddingStep — replaces text columns with sentence-transformer embeddings.

Requires ``sentence-transformers`` and ``torch`` to be installed.
When they are not available, the step logs a warning and passes through.

Auto-inserted before ``CreateModelStep`` when ``step_config.text_embedding`` is set.

Usage::

    step_config:
      text_embedding:
        model_name: "intfloat/multilingual-e5-small"
        device: "cuda"
        batch_size: 64
        max_length: 512
        cache_dir: "./embedding_cache"
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import (
    BaseStep,
    FeatureType,
    MLCombineConfig,
    PipelineContext,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_TEXT_EMBEDDER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer

    _TEXT_EMBEDDER_AVAILABLE = True
except ImportError:
    pass


@registry.step("TextEmbeddingStep", before="CreateModelStep")
class TextEmbeddingStep(BaseStep[PipelineContext]):
    """Replace text columns with dense embeddings from a sentence-transformer model.

    Detects columns marked as ``FeatureType.TEXT`` in the pipeline context and
    replaces each with ``n`` numeric columns (one per embedding dimension).

    Side Effects:
        - Replaces text columns with embedding columns in ``train_df`` and ``test_df``.
        - Updates ``detected_types`` to mark new columns as ``FeatureType.NUMBER``.
    """

    train = True
    predict = True

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        return getattr(cfg.step_config, "text_embedding", None) is not None

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        te = getattr(cfg.step_config, "text_embedding", None) or {}
        self._model_name = te.get("model_name", "intfloat/multilingual-e5-small")
        self._device = te.get("device", "cuda")
        self._batch_size = te.get("batch_size", 64)
        self._max_length = te.get("max_length", 512)
        self._cache_dir = te.get("cache_dir", None)

        self._model: SentenceTransformer | None = None
        self._embedding_dim: int = 0

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not _TEXT_EMBEDDER_AVAILABLE:
            raise ImportError("sentence-transformers is required for TextEmbeddingStep. Install with: uv add sentence-transformers")
        logger.info("Loading sentence-transformer model: %s (device=%s)", self._model_name, self._device)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._embedding_dim = self._model.get_embedding_dimension() or 384

    def _cache_key(self, texts: list[str]) -> str:
        """SHA-256 hash of joined texts + model config for cache lookup."""
        raw = "".join(texts) + self._model_name + str(self._max_length)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> np.ndarray | None:
        if self._cache_dir is None:
            return None
        path = Path(self._cache_dir) / f"{key}.npy"
        if path.exists():
            logger.info("Loading cached embeddings from %s", path)
            return np.load(path)  # type: ignore[no-any-return]
        return None

    def _save_cache(self, key: str, emb: np.ndarray) -> None:
        if self._cache_dir is None:
            return
        path = Path(self._cache_dir) / f"{key}.npy"
        os.makedirs(path.parent, exist_ok=True)
        np.save(path, emb)
        logger.info("Cached embeddings to %s", path)

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, bool]:
        if not texts:
            return np.empty((0, 0)), False
        cache_key = self._cache_key(texts)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached, True
        self._load_model()
        assert self._model is not None
        emb = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._save_cache(cache_key, emb)
        return emb, False

    def _transform_features(self, df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
        if not text_cols:
            return df

        df = df.copy()
        for col in text_cols:
            if col not in df.columns:
                continue
            vals = df[col].fillna("").astype(str).tolist()
            prefix = "e5" if "e5" in self._model_name else "emb"
            emb, was_cached = self._embed(vals)
            emb_cols = [f"{prefix}_{col}_{i}" for i in range(emb.shape[1])]
            emb_df = pd.DataFrame(emb, columns=emb_cols, index=df.index)
            df = pd.concat([df.drop(columns=[col]), emb_df], axis=1)
            action = "Applied cached" if was_cached else "Computed"
            logger.info("%s embeddings for '%s': %d dims", action, col, emb.shape[1])

        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        detected = context.data.detected_types
        if detected is None:
            return context

        text_cols = [col for col, ft in detected.items() if ft == FeatureType.TEXT]
        if not text_cols:
            return context

        self._load_model()
        emb_dim = self._model.get_embedding_dimension() or 384 if self._model else 384
        self._embedding_dim = emb_dim

        if context.data.train_df is not None:
            context.data.train_df = self._transform_features(context.data.train_df, text_cols)
        if context.data.holdout_df is not None:
            context.data.holdout_df = self._transform_features(context.data.holdout_df, text_cols)
        if context.data.test_df is not None:
            context.data.test_df = self._transform_features(context.data.test_df, text_cols)

        # Update detected types: remove TEXT, add NUMBER for embedding columns
        for col in text_cols:
            detected.pop(col, None)
            prefix = "e5" if "e5" in self._model_name else "emb"
            for i in range(emb_dim):
                detected[f"{prefix}_{col}_{i}"] = FeatureType.NUMBER

        return context


__all__ = [
    "TextEmbeddingStep",
]
