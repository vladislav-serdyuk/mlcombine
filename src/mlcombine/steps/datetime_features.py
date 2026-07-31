"""DateTimeFeatureStep — extracts numeric features from datetime columns.

Converts each ``FeatureType.DATETIME`` column into:
  - year, month, day, dayofweek, hour, minute, second
  - sin/cos of circular features: month (12), dayofweek (7), hour (24)

Usage::

    pipeline:
      order:
        - TypeDetectStep
        - DateTimeFeatureStep
        - ImputeStep
        ...
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import (
    BaseStep,
    FeatureMap,
    FeatureType,
    MLCombineConfig,
    PipelineContext,
)

logger = logging.getLogger(__name__)


def _sin_cos(x: pd.Series, period: int, name: str) -> pd.DataFrame:
    x_num = x.astype(float)
    return pd.DataFrame(
        {
            f"{name}_sin": np.sin(2 * np.pi * x_num / period),
            f"{name}_cos": np.cos(2 * np.pi * x_num / period),
        },
        index=x.index,
    )


@registry.step("DateTimeFeatureStep")
class DateTimeFeatureStep(BaseStep[PipelineContext]):
    """Extract numeric features from datetime columns.

    Side Effects:
        - Replaces datetime columns with numeric feature columns (year, month, day,
          dayofweek, hour, minute, second, plus sin/cos of circular periods).
        - Updates ``detected_types`` to mark new columns as ``FeatureType.NUMBER``.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        pass

    def _transform_features(self, df: pd.DataFrame, detected: FeatureMap) -> pd.DataFrame:
        dt_cols = [col for col, ft in detected.items() if ft == FeatureType.DATETIME and col in df.columns]
        if not dt_cols:
            return df

        df = df.copy()
        for col in dt_cols:
            parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
            new_cols: list[str] = []

            for attr, period in [("month", 12), ("dayofweek", 7), ("hour", 24)]:
                vals = getattr(parsed.dt, attr)
                sin_cos = _sin_cos(vals, period, f"dt_{col}_{attr}")
                for c in sin_cos.columns:
                    df[c] = sin_cos[c]
                    detected[c] = FeatureType.NUMBER
                    new_cols.append(c)

            for attr in ["year", "day", "minute", "second"]:
                vals = getattr(parsed.dt, attr, None)
                if vals is not None:
                    name = f"dt_{col}_{attr}"
                    df[name] = vals.astype(float)
                    detected[name] = FeatureType.NUMBER
                    new_cols.append(name)

            df.drop(columns=[col], inplace=True)
            detected.pop(col, None)
            logger.info("Extracted %d datetime features from '%s'", len(new_cols), col)

        return df

    def run(self, context: PipelineContext) -> PipelineContext:
        detected = context.data.detected_types
        if detected is None:
            return context

        if context.data.train_df is not None:
            context.data.train_df = self._transform_features(context.data.train_df, detected)
        if context.data.test_df is not None:
            context.data.test_df = self._transform_features(context.data.test_df, detected)

        return context


__all__ = [
    "DateTimeFeatureStep",
]
