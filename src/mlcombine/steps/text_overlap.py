"""TextOverlapStep — Jaccard word/char-ngram overlap for column pairs.

Configure via ``step_config.text_overlap``::

    text_overlap:
      pairs:
        - ["left_title", "right_title"]
        - ["left_content", "right_content"]
      char_ngram: 3       # 0 to disable char n-gram
      token_pattern: "[а-яёa-z]+"
      drop_source: false   # delete source text columns after adding features
"""

from __future__ import annotations

import logging
import re
from collections.abc import Hashable
import numpy as np
import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN_PATTERN = r"[а-яёa-z]+"


def _ngram_hashes(t: str, n: int) -> frozenset[int]:
    """Rolling hash of char n-grams — no substring allocation."""
    if len(t) < n:
        return frozenset()
    base = 127
    mod = (1 << 61) - 1
    power = 1
    for _ in range(n - 1):
        power = (power * base) % mod
    h = 0
    for i in range(n):
        h = (h * base + ord(t[i])) % mod
    hashes = {h}
    for i in range(n, len(t)):
        h = ((h - ord(t[i - n]) * power) * base + ord(t[i])) % mod
        hashes.add(h)
    return frozenset(hashes)


@registry.step("TextOverlapStep", before="SplitStep")
class TextOverlapStep(BaseStep[PipelineContext]):
    """Add Jaccard word overlap (and optionally char n-gram overlap) columns.

    Side Effects:
        - Adds new numeric columns to ``train_df`` and ``test_df``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        to = getattr(cfg.step_config, "text_overlap", None) or {}
        self._pairs = to.get("pairs", [])
        self._char_ngram = to.get("char_ngram", 0)
        self._token_pattern = to.get("token_pattern", _DEFAULT_TOKEN_PATTERN)
        self._drop_source = to.get("drop_source", False)
        self._compiled_re = re.compile(self._token_pattern)
        self._predict = predict

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        to = getattr(cfg.step_config, "text_overlap", None) or {}
        return bool(to.get("pairs", []))

    @staticmethod
    def _jaccard(a: frozenset[Hashable] | set[Hashable], b: frozenset[Hashable] | set[Hashable]) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    def _add_overlap(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = [(p[0], p[1]) for p in self._pairs if len(p) == 2 and p[0] in df.columns and p[1] in df.columns]
        if not pairs:
            return df

        cols = sorted({c for p in pairs for c in p})
        col_arr = {c: df[c].values for c in cols}
        rows = len(df)

        n = len(pairs)
        word_results = [np.empty(rows, dtype=np.float64) for _ in range(n)]
        if self._char_ngram > 0:
            char_results = [np.empty(rows, dtype=np.float64) for _ in range(n)]
        else:
            char_results = None

        for i in range(rows):
            for pi, (a, b) in enumerate(pairs):
                va = col_arr[a][i]
                vb = col_arr[b][i]
                text_a = str(va).lower() if va is not None and va is not np.nan else ""
                text_b = str(vb).lower() if vb is not None and vb is not np.nan else ""

                tokens_a = frozenset(self._compiled_re.findall(text_a))
                tokens_b = frozenset(self._compiled_re.findall(text_b))
                word_results[pi][i] = self._jaccard(tokens_a, tokens_b)

                if char_results is not None:
                    na = _ngram_hashes(text_a, self._char_ngram)
                    nb = _ngram_hashes(text_b, self._char_ngram)
                    char_results[pi][i] = self._jaccard(na, nb)

        for pi, (a, b) in enumerate(pairs):
            df[f"{a}_word_overlap"] = word_results[pi]
            if char_results is not None:
                df[f"{a}_char_overlap"] = char_results[pi]

        if self._drop_source:
            cols_to_drop = {c for p in pairs for c in p if c in df}
            if cols_to_drop:
                df = df.drop(columns=list(cols_to_drop))

        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        if not self._predict and context.data.train_df is not None:
            context.data.train_df = self._add_overlap(context.data.train_df)
        if context.data.test_df is not None:
            context.data.test_df = self._add_overlap(context.data.test_df)
        if self._pairs:
            logger.info("Added overlap features for %d text column pairs (char_ngram=%d)", len(self._pairs), self._char_ngram)
        return context
