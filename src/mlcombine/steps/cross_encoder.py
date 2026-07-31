"""CrossEncoderStep — compute text-pair relevance scores via a cross-encoder.

Uses ``sentence-transformers`` CrossEncoder models to score the relevance
between paired text columns.  Adds a ``{col}_cross_encoder_score`` column
for each configured pair.

Configure via ``step_config.cross_encoder``::

    cross_encoder:
      model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      pairs:
        - ["left_title", "right_title"]
        - ["left_content", "right_content"]
      batch_size: 64
      max_length: 128       # optional — max tokens per text; default no limit
      predict_chunk: 500    # optional — rows per predict call; default 500
      drop_source: false    # optional — drop source text columns after scoring
      device: null           # auto-detect; set to "cuda" or "cpu" to override
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_PREDICT_CHUNK_DEFAULT = 512


@registry.step("CrossEncoderStep", before="SplitStep", package="sentence-transformers", module="sentence_transformers")
class CrossEncoderStep(BaseStep[PipelineContext]):
    """Add cross-encoder similarity scores for text column pairs.

    Side Effects:
        - Adds new numeric columns to ``train_df`` and ``test_df``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        ce = getattr(cfg.step_config, "cross_encoder", None) or {}
        self._predict: bool = predict
        self._model_name = ce.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._pairs = ce.get("pairs", [])
        self._batch_size = ce.get("batch_size", 64)
        self._max_length = ce.get("max_length", None)
        self._predict_chunk = ce.get("predict_chunk", _PREDICT_CHUNK_DEFAULT)
        self._drop_source = ce.get("drop_source", False)
        self._device = ce.get("device", None)
        self._cache_dir = ce.get("cache_dir", None)
        self._model: CrossEncoder | None = None

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        ce = getattr(cfg.step_config, "cross_encoder", None) or {}
        return bool(ce.get("pairs", []))

    def _load_model(self) -> CrossEncoder:
        try:
            torch.set_num_threads(1)
        except Exception as e:
            logger.debug("Could not set torch num_threads: %s", e)
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, object] = dict(model_name_or_path=self._model_name, device=self._device)
        if self._max_length is not None:
            kwargs["max_length"] = self._max_length
        return CrossEncoder(**kwargs)

    def _clear_model(self) -> None:
        self._model = None
        gc.collect()
        try:
            torch.set_num_threads(1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("GPU cleanup error: %s", e)

    def _cache_key(
        self,
        col_a: str,
        col_b: str,
        texts_a: list[str],
        texts_b: list[str],
    ) -> str:
        h = hashlib.sha256()
        h.update(self._model_name.encode("utf-8"))
        h.update(col_a.encode("utf-8"))
        h.update(col_b.encode("utf-8"))
        h.update(str(self._max_length).encode("utf-8"))
        for ta, tb in zip(texts_a, texts_b):
            h.update(str(ta).encode("utf-8", errors="replace"))
            h.update(str(tb).encode("utf-8", errors="replace"))
        return h.hexdigest()

    def _load_cache(self, key: str) -> np.ndarray | None:
        if self._cache_dir is None:
            return None
        path = Path(self._cache_dir) / f"{key}.npy"
        if path.exists():
            logger.info("Loading cached cross-encoder scores from %s", path)
            return np.load(path)  # type: ignore[no-any-return]
        return None

    def _save_cache(self, key: str, scores: np.ndarray) -> None:
        if self._cache_dir is None:
            return
        path = Path(self._cache_dir) / f"{key}.npy"
        os.makedirs(path.parent, exist_ok=True)
        np.save(path, scores)
        logger.info("Cached cross-encoder scores to %s", path)

    def _drop_source_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop: set[str] = set()
        for pair in self._pairs:
            if len(pair) == 2:
                for c in pair:
                    if c in df.columns:
                        cols_to_drop.add(c)
        if cols_to_drop:
            logger.info("Dropping source columns: %s", sorted(cols_to_drop))
            df = df.drop(columns=list(cols_to_drop))
        return df

    def _score_pair(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        col_name: str,
    ) -> np.ndarray:
        """Compute cross-encoder scores for a single text pair across all rows."""
        if self._model is None:
            self._model = self._load_model()

        texts_a = df[col_a].fillna("").astype(str).tolist()
        texts_b = df[col_b].fillna("").astype(str).tolist()
        cache_key = self._cache_key(col_a, col_b, texts_a, texts_b)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached[: len(df)]

        n = len(df)
        all_chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, n, self._predict_chunk),
            desc=f"CrossEncoder [{col_name}]",
            unit="chunk",
        ):
            end = min(start + self._predict_chunk, n)
            chunk_texts = [
                (
                    str(texts_a[j]) if texts_a[j] is not None and not (isinstance(texts_a[j], float) and np.isnan(texts_a[j])) else "",
                    str(texts_b[j]) if texts_b[j] is not None and not (isinstance(texts_b[j], float) and np.isnan(texts_b[j])) else "",
                )
                for j in range(start, end)
            ]
            with torch.inference_mode():
                scores = self._model.predict(chunk_texts, batch_size=self._batch_size, show_progress_bar=False)
            all_chunks.append(np.asarray(scores))
            gc.collect()

        result = np.concatenate(all_chunks, axis=0)
        self._save_cache(cache_key, result)
        return result

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._pairs:
            return df

        for pair in self._pairs:
            if len(pair) != 2:
                continue
            col_a, col_b = pair[0], pair[1]
            if col_a not in df.columns or col_b not in df.columns:
                logger.warning("Columns %s or %s not found — skipping", col_a, col_b)
                continue
            col_name = f"{col_a}_cross_encoder_score"
            if col_name in df.columns:
                logger.info("Column %s already exists — skipping pair", col_name)
                continue

            scores = self._score_pair(df, col_a, col_b, col_name)
            if scores.ndim == 1:
                df[col_name] = scores
            else:
                df[col_name] = scores[:, 0]

        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.data.train_df is not None and not self._predict:
            context.data.train_df = self._transform_features(context.data.train_df)
            if self._drop_source:
                context.data.train_df = self._drop_source_cols(context.data.train_df)
            self._clear_model()
        if context.data.test_df is not None:
            if self._predict:
                context.data.test_df = self._transform_features(context.data.test_df)
                if self._drop_source:
                    context.data.test_df = self._drop_source_cols(context.data.test_df)
                self._clear_model()
        return context
