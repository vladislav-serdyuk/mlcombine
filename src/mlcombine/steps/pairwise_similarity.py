"""PairwiseSimilarityStep — bi-encoder cosine similarity for text column pairs.

Uses ``sentence-transformers`` models to embed each text independently,
then computes cosine similarity per pair.  Much lighter on memory than
a cross-encoder (no pairwise attention).

Configure via ``step_config.pairwise_similarity``::

    pairwise_similarity:
      model_name: "all-MiniLM-L6-v2"
      pairs:
        - ["left_title", "right_title"]
        - ["left_content", "right_content"]
      max_length: 64
      batch_size: 64
      predict_chunk: 512
      drop_source: false
      device: null
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_PREDICT_CHUNK_DEFAULT = 512


@registry.step("PairwiseSimilarityStep", before="SplitStep", package="sentence-transformers", module="sentence_transformers")
class PairwiseSimilarityStep(BaseStep[PipelineContext]):
    """Add cosine similarity scores for text column pairs using a bi-encoder.

    Side Effects:
        - Adds new numeric columns to ``train_df`` and ``test_df``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        ps = getattr(cfg.step_config, "pairwise_similarity", None) or {}
        self._predict_flag: bool = predict
        self._model_name = ps.get("model_name", "all-MiniLM-L6-v2")
        self._pairs = ps.get("pairs", [])
        self._batch_size = ps.get("batch_size", 64)
        self._max_length = ps.get("max_length", None)
        self._predict_chunk = ps.get("predict_chunk", _PREDICT_CHUNK_DEFAULT)
        self._drop_source = ps.get("drop_source", False)
        self._device = ps.get("device", None)
        self._model: SentenceTransformer | None = None

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        ps = getattr(cfg.step_config, "pairwise_similarity", None) or {}
        return bool(ps.get("pairs", []))

    def _load_model(self) -> SentenceTransformer:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("PairwiseSimilarityStep requires 'torch' — install it with 'pip install torch' or add 'torch' to project dependencies.") from e
        try:
            torch.set_num_threads(1)
        except Exception as e:
            logger.debug("Could not set torch num_threads: %s", e)
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, object] = dict(model_name_or_path=self._model_name, device=self._device)
        return SentenceTransformer(**kwargs)

    def _clear_model(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            torch.set_num_threads(1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("GPU cleanup error: %s", e)

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

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._pairs:
            return df

        if self._model is None:
            self._model = self._load_model()
        import torch

        valid_pairs: list[tuple[str, str, str]] = []
        for pair in self._pairs:
            if len(pair) != 2:
                continue
            col_a, col_b = pair[0], pair[1]
            if col_a not in df.columns or col_b not in df.columns:
                logger.warning("Columns %s or %s not found — skipping", col_a, col_b)
                continue
            col_name = f"{col_a}_sim_score"
            if col_name in df.columns:
                logger.info("Column %s already exists — skipping pair", col_name)
                continue
            valid_pairs.append((col_a, col_b, col_name))

        if not valid_pairs:
            return df

        pair_scores: list[list[np.ndarray]] = [[] for _ in valid_pairs]
        n = len(df)

        for start in tqdm(
            range(0, n, self._predict_chunk),
            desc="BiEncoder",
            unit="chunk",
        ):
            end = min(start + self._predict_chunk, n)

            for pi, (col_a, col_b, _) in enumerate(valid_pairs):
                col_a_vals = df[col_a].values[start:end]
                col_b_vals = df[col_b].values[start:end]

                texts_a = [str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "" for v in col_a_vals]
                texts_b = [str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "" for v in col_b_vals]

                with torch.inference_mode():
                    emb_a = self._model.encode(texts_a, batch_size=self._batch_size, show_progress_bar=False, convert_to_tensor=True)
                    emb_b = self._model.encode(texts_b, batch_size=self._batch_size, show_progress_bar=False, convert_to_tensor=True)
                    sim = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=1).cpu().numpy()

                pair_scores[pi].append(sim)

            gc.collect()

        for pi, (_, _, col_name) in enumerate(valid_pairs):
            df[col_name] = np.concatenate(pair_scores[pi], axis=0)

        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.data.train_df is not None and not self._predict_flag:
            context.data.train_df = self._transform_features(context.data.train_df)
            if self._drop_source:
                context.data.train_df = self._drop_source_cols(context.data.train_df)
            self._clear_model()
        if context.data.test_df is not None:
            if self._predict_flag:
                context.data.test_df = self._transform_features(context.data.test_df)
                if self._drop_source:
                    context.data.test_df = self._drop_source_cols(context.data.test_df)
                self._clear_model()
        return context
